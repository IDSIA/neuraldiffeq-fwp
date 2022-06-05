import copy
import json
import math
import numpy as np
import os
import pathlib
import sklearn.metrics
import torch
import tqdm

import models

here = pathlib.Path(__file__).resolve().parent


def _add_weight_regularisation(loss_fn, regularise_parameters, scaling=0.03):
    def new_loss_fn(pred_y, true_y):
        total_loss = loss_fn(pred_y, true_y)
        for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                total_loss = total_loss + scaling * parameter.norm()
        return total_loss
    return new_loss_fn


class _SqueezeEnd(models.BaseModel):
    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).squeeze(-1)


def _count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


class _AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]


def _evaluate_metrics(dataloader, model, times, loss_fn, num_classes, device, kwargs):
    with torch.no_grad():
        total_accuracy = 0
        total_confusion = torch.zeros(num_classes, num_classes).numpy()  # occurs all too often
        total_dataset_size = 0
        total_loss = 0
        true_y_cpus = []
        pred_y_cpus = []

        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch
            batch_size = true_y.size(0)
            pred_y = model(times, coeffs, lengths, **kwargs)

            if num_classes == 2:
                thresholded_y = (pred_y > 0).to(true_y.dtype)
            else:
                thresholded_y = torch.argmax(pred_y, dim=1)
            true_y_cpu = true_y.detach().cpu()
            pred_y_cpu = pred_y.detach().cpu()
            if num_classes == 2:
                # Assume that our datasets aren't so large that this breaks
                true_y_cpus.append(true_y_cpu)
                pred_y_cpus.append(pred_y_cpu)
            thresholded_y_cpu = thresholded_y.detach().cpu()

            total_accuracy += (thresholded_y == true_y).sum().to(pred_y.dtype)
            total_confusion += sklearn.metrics.confusion_matrix(
                true_y_cpu, thresholded_y_cpu, labels=range(num_classes))
            total_dataset_size += batch_size
            total_loss += loss_fn(pred_y, true_y) * batch_size

        total_loss /= total_dataset_size  # assume 'mean' reduction in the loss function
        total_accuracy /= total_dataset_size
        metrics = _AttrDict(
            accuracy=total_accuracy.item(), confusion=total_confusion,
            dataset_size=total_dataset_size, loss=total_loss.item())

        if num_classes == 2:
            true_y_cpus = torch.cat(true_y_cpus, dim=0)
            pred_y_cpus = torch.cat(pred_y_cpus, dim=0)
            metrics.auroc = sklearn.metrics.roc_auc_score(
                true_y_cpus, pred_y_cpus)
            metrics.average_precision = sklearn.metrics.average_precision_score(true_y_cpus, pred_y_cpus)
        return metrics


class _SuppressAssertions:
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is AssertionError:
            self.tqdm_range.write('Caught AssertionError: ' + str(exc_val))
            return True


def _train_loop(train_dataloader, val_dataloader, model, times, optimizer,
                loss_fn, max_epochs, num_classes, device, kwargs, step_mode, *,
                use_wandb=False, epoch_per_metric=10,
                val_based_stopping=False, auroc_based_stopping=False,
                loginf=None):
    if use_wandb:
        import wandb
    model.train()
    best_model = model
    best_train_loss = math.inf
    best_train_accuracy = 0
    best_val_loss = math.inf
    best_val_accuracy = 0
    best_train_accuracy_epoch = 0
    best_train_loss_epoch = 0
    best_val_accuracy_epoch = 0
    best_val_loss_epoch = 0
    if auroc_based_stopping:  # val based
        best_val_auroc = 0
        best_val_auroc_epoch = 0
    history = []
    breaking = False

    if step_mode:
        plateau_terminate = 100
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2)
    else:
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=1, mode='max')

    tqdm_range = tqdm.tqdm(range(max_epochs))
    tqdm_range.write('Starting training for model:\n\n' + str(model) + '\n\n')
    for epoch in tqdm_range:
        if breaking:
            break
        for batch in train_dataloader:
            batch = tuple(b.to(device) for b in batch)
            if breaking:
                break
            with _SuppressAssertions(tqdm_range):
                *train_coeffs, train_y, lengths = batch
                pred_y = model(times, train_coeffs, lengths, **kwargs)
                loss = loss_fn(pred_y, train_y)
                loss.backward()
                optimizer.step()
                model.reset_grad()
                # optimizer.zero_grad()

        if epoch % epoch_per_metric == 0 or epoch == max_epochs - 1:
            model.eval()
            train_metrics = _evaluate_metrics(
                train_dataloader, model, times, loss_fn, num_classes, device,
                kwargs)
            val_metrics = _evaluate_metrics(
                val_dataloader, model, times, loss_fn, num_classes, device,
                kwargs)
            model.train()

            if train_metrics.loss * 1.0001 < best_train_loss:
                best_train_loss = train_metrics.loss
                best_train_loss_epoch = epoch

            if train_metrics.accuracy > best_train_accuracy * 1.001:
                best_train_accuracy = train_metrics.accuracy
                best_train_accuracy_epoch = epoch

            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                best_val_loss_epoch = epoch

            if auroc_based_stopping:
                assert num_classes == 2
                if val_metrics.auroc > best_val_auroc:
                    best_val_auroc = val_metrics.auroc
                    best_val_auroc_epoch = epoch

            if val_metrics.accuracy > best_val_accuracy:
                best_val_accuracy = val_metrics.accuracy
                best_val_accuracy_epoch = epoch
                # so that we don't have three copies of a model simultaneously
                del best_model
                best_model = copy.deepcopy(model)

            if use_wandb:
                wandb.log({
                    "train_loss": train_metrics.loss,
                    "train_acc": train_metrics.accuracy,
                    "val_loss": val_metrics.loss,
                    "val_acc": val_metrics.accuracy
                })
                if num_classes == 2:
                    wandb.log({
                        "train_auroc": train_metrics.auroc,
                        "train_precision": train_metrics.average_precision,
                        "val_auroc": val_metrics.auroc,
                        "val_precision": val_metrics.average_precision,
                    })
            if num_classes == 2:
                log_str = (f'Epoch: {epoch} '
                           f'Train loss: {train_metrics.loss:.3f} '
                           f'Train accuracy: {train_metrics.accuracy:.3f} '
                           f'Train AUC: {train_metrics.auroc:.3f} '
                           'Train precision: '
                           f'{train_metrics.average_precision:.3f} '
                           f'Val loss: {val_metrics.loss:.3f} '
                           f'Val accuracy: {val_metrics.accuracy:.3f} '
                           f'Val AUC: {val_metrics.auroc:.3f} '
                           'Val precision: '
                           f'{val_metrics.average_precision:.3f}')
            else:
                log_str = (f'Epoch: {epoch} '
                           f'Train loss: {train_metrics.loss:.3f} '
                           f'Train accuracy: {train_metrics.accuracy:.3f} '
                           f'Val loss: {val_metrics.loss:.3f} '
                           f'Val accuracy: {val_metrics.accuracy:.3f}')
            if loginf is not None:
                loginf(log_str + '\n')
            tqdm_range.write(log_str)
            if step_mode:
                scheduler.step(train_metrics.loss)
            else:
                scheduler.step(val_metrics.accuracy)
            history.append(
                _AttrDict(
                    epoch=epoch, train_metrics=train_metrics,
                    val_metrics=val_metrics))
            if auroc_based_stopping:
                if epoch > best_val_auroc_epoch + plateau_terminate:
                    log_str = (f'Breaking because of no improvement in val AUC'
                               f' for {plateau_terminate} epochs.')
                    tqdm_range.write(log_str)
                    if loginf is not None:
                        loginf(log_str)
                    breaking = True            
            elif val_based_stopping:
                if epoch > best_val_loss_epoch + plateau_terminate:
                    log_str = (f'Breaking because of no improvement in val '
                               f'loss for {plateau_terminate} epochs.')
                    tqdm_range.write(log_str)
                    if loginf is not None:
                        loginf(log_str)
                    breaking = True
                if epoch > best_val_accuracy_epoch + plateau_terminate:
                    log_str = (f'Breaking because of no improvement in val '
                               f'accuracy for {plateau_terminate} epochs.')
                    tqdm_range.write(log_str)
                    if loginf is not None:
                        loginf(log_str)
                    breaking = True
            else:
                if epoch > best_train_loss_epoch + plateau_terminate:
                    log_str = (f'Breaking because of no improvement in train '
                               f'loss for {plateau_terminate} epochs.')
                    tqdm_range.write(log_str)
                    if loginf is not None:
                        loginf(log_str)
                    breaking = True
                if epoch > best_train_accuracy_epoch + plateau_terminate:
                    log_str = (f'Breaking because of no improvement in train '
                               f'accuracy for {plateau_terminate} epochs.')
                    tqdm_range.write(log_str)
                    if loginf is not None:
                        loginf(log_str)
                    breaking = True

    zip_iter = zip(model.parameters(), best_model.parameters())
    for parameter, best_parameter in zip_iter:
        parameter.data = best_parameter.data

    return history


class _TensorEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            super(_TensorEncoder, self).default(o)


def _save_results(name, result):
    loc = here / 'results' / name
    if not os.path.exists(loc):
        os.mkdir(loc)
    num = -1
    for filename in os.listdir(loc):
        try:
            num = max(num, int(filename))
        except ValueError:
            pass
    result_to_save = result.copy()
    del result_to_save['train_dataloader']
    del result_to_save['val_dataloader']
    del result_to_save['test_dataloader']
    result_to_save['model'] = str(result_to_save['model'])

    num += 1
    with open(loc / str(num), 'w') as f:
        json.dump(result_to_save, f, cls=_TensorEncoder)


def main(name, times, train_dataloader, val_dataloader, test_dataloader,
         device, make_model, num_classes, max_epochs,
         lr, kwargs, step_mode, pos_weight=torch.tensor(1), use_wandb=False,
         epoch_per_metric=10, val_based_stopping=False,
         auroc_based_stopping=False, loginf=None):
    times = times.to(device)
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None

    model, regularise_parameters = make_model()
    if loginf is not None:
        loginf(f"Model: {model}")
        loginf(f"Number of trainable params: {model.num_params()}")
    else:
        print(f"Number of trainable params: {model.num_params()}")

    if num_classes == 2:
        model = _SqueezeEnd(model)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.functional.cross_entropy
    loss_fn = _add_weight_regularisation(loss_fn, regularise_parameters)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = _train_loop(
        train_dataloader, val_dataloader, model, times, optimizer, loss_fn,
        max_epochs, num_classes, device, kwargs, step_mode,
        use_wandb=use_wandb, epoch_per_metric=epoch_per_metric,
        val_based_stopping=val_based_stopping,
        auroc_based_stopping=auroc_based_stopping, loginf=loginf)

    model.eval()
    train_metrics = _evaluate_metrics(
        train_dataloader, model, times, loss_fn, num_classes, device, kwargs)
    val_metrics = _evaluate_metrics(
        val_dataloader, model, times, loss_fn, num_classes, device, kwargs)
    test_metrics = _evaluate_metrics(
        test_dataloader, model, times, loss_fn, num_classes, device, kwargs)

    if device != 'cpu':
        memory_usage = (
            torch.cuda.max_memory_allocated(device) - baseline_memory)
    else:
        memory_usage = None

    result = _AttrDict(times=times,
                       memory_usage=memory_usage,
                       baseline_memory=baseline_memory,
                       num_classes=num_classes,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       test_dataloader=test_dataloader,
                       model=model.to('cpu'),
                       parameters=_count_parameters(model),
                       history=history,
                       train_metrics=train_metrics,
                       val_metrics=val_metrics,
                       test_metrics=test_metrics)
    if name is not None:
        _save_results(name, result)
    return result


def make_model(name, input_channels, output_channels, hidden_channels,
               hidden_hidden_channels, num_hidden_layers, use_intensity,
               initial, num_heads=8, trafo_ff_dim=None, dropout=0.0,
               use_v_laynorm=False, delta_ode_post_tahn=False):
    if name == 'ncde':
        def make_model():
            vector_field = models.FinalTanh(
                input_channels=input_channels, hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE(
                func=vector_field, input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'bigfwcde' or name == 'big_fw_cde':
        def make_model():
            model = models.BigMapFastWeightCDE(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
                output_channels=output_channels,
                dropout=dropout,
                trafo_ff_dim=trafo_ff_dim, initial=initial)
            return model, model
    elif name == 'hebbcde' or name == 'hebb_cde':
        def make_model():
            model = models.FastWeightCDE(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                num_heads=num_heads,
                trafo_ff_dim=trafo_ff_dim,
                dropout=dropout,
                learning_rule='hebb',
                initial=initial,
                use_v_laynorm=use_v_laynorm)
            return model, model
    elif name == 'ojacde' or name == 'oja_cde':
        def make_model():
            model = models.FastWeightCDE(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                num_heads=num_heads,
                trafo_ff_dim=trafo_ff_dim,
                dropout=dropout,
                learning_rule='oja',
                initial=initial,
                use_v_laynorm=use_v_laynorm)
            return model, model
    elif name == 'deltacde' or name == 'delta_cde':
        def make_model():
            model = models.FastWeightCDEv2(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                num_heads=num_heads,
                trafo_ff_dim=trafo_ff_dim,
                dropout=dropout,
                learning_rule='delta',
                initial=initial,
                delta_post_tahn=delta_ode_post_tahn,
                use_v_laynorm=use_v_laynorm)
            return model, model
    elif name == 'hebbcdedxonly' or name == 'hebb_cde_dx_only':
        def make_model():
            model = models.FastWeightCDEv2(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                num_heads=num_heads,
                trafo_ff_dim=trafo_ff_dim,
                dropout=dropout,
                learning_rule='hebb_dx_only',
                initial=initial,
                use_v_laynorm=use_v_laynorm)
            return model, model
    elif name == 'ojacdedxonly' or name == 'oja_cde_dx_only':
        def make_model():
            model = models.FastWeightCDEv2(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                num_heads=num_heads,
                trafo_ff_dim=trafo_ff_dim,
                dropout=dropout,
                learning_rule='oja_dx_only',
                initial=initial,
                use_v_laynorm=use_v_laynorm)
            return model, model
    elif name == 'deltacdedxonly' or name == 'delta_cde_dx_only':
        def make_model():
            model = models.FastWeightCDEv2(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                num_heads=num_heads,
                trafo_ff_dim=trafo_ff_dim,
                dropout=dropout,
                learning_rule='delta_dx_only',
                initial=initial,
                delta_post_tahn=delta_ode_post_tahn,
                use_v_laynorm=use_v_laynorm)
            return model, model
    elif name == 'hebbcdev2' or name == 'hebb_cde_v2':
        def make_model():
            model = models.FastWeightCDEv2(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                num_heads=num_heads,
                trafo_ff_dim=trafo_ff_dim,
                dropout=dropout,
                learning_rule='hebb',
                initial=initial,
                use_v_laynorm=use_v_laynorm)
            return model, model
    elif name == 'hebbcdev3' or name == 'hebb_cde_v3':
        def make_model():
            model = models.FastWeightCDEv3(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                num_heads=num_heads,
                trafo_ff_dim=trafo_ff_dim,
                dropout=dropout,
                learning_rule='hebb',
                initial=initial)
            return model, model
    elif name == 'hebbode' or name == 'hebb_ode':
        def make_model():
            model = models.FastWeightODE(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                num_heads=num_heads,
                trafo_ff_dim=trafo_ff_dim,
                dropout=dropout,
                learning_rule='hebb',
                initial=initial)
            return model, model
    elif name == 'ojaode' or name == 'oja_ode':
        def make_model():
            model = models.FastWeightODE(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                num_heads=num_heads,
                trafo_ff_dim=trafo_ff_dim,
                dropout=dropout,
                learning_rule='oja',
                initial=initial)
            return model, model
    elif name == 'deltaode' or name == 'delta_ode':
        def make_model():
            model = models.FastWeightODE(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                num_heads=num_heads,
                trafo_ff_dim=trafo_ff_dim,
                dropout=dropout,
                learning_rule='delta',
                initial=initial,
                delta_post_tahn=delta_ode_post_tahn)
            return model, model
    elif name == 'mygruode':
        def make_model():
            vector_field = models._GRU_ODE(
                input_channels=input_channels, hidden_channels=hidden_channels)
            model = models.DirectRecurrentODE(
                func=vector_field, input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'gruode':
        def make_model():
            vector_field = models.GRU_ODE(
                input_channels=input_channels, hidden_channels=hidden_channels)
            model = models.NeuralCDE(
                func=vector_field, input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'dt':
        def make_model():
            model = models.GRU_dt(
                input_channels=input_channels, hidden_channels=hidden_channels,
                output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'decay':
        def make_model():
            model = models.GRU_D(
                input_channels=input_channels, hidden_channels=hidden_channels,
                output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'odernn':
        def make_model():
            model = models.ODERNN(
                input_channels=input_channels, hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
                output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    else:
        raise ValueError(f"Unrecognised model name: {name}")
    return make_model
