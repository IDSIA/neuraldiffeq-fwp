import common
import datasets


def main(dataset_name, model_name, hidden_channels, hidden_hidden_channels,
         num_hidden_layers, *, num_heads=8, trafo_ff_dim=None, dropout=0.0,
         cde_use_v_laynorm=False, delta_ode_post_tahn=False,
         missing_rate=0.3, device='cuda', max_epochs=1000, batch_size=32,
         base_lr=0.001, grad_scale=1, dry_run=False, use_wandb=False,
         epoch_per_metric=10, val_based_stopping=False,
         auroc_based_stopping=False, loginf=None,
         **kwargs):  # kwargs passed on to cdeint

    lr = base_lr * (batch_size / 32)

    # Need the intensity data to know how long to evolve for in between observations, but the model doesn't otherwise
    # use it because of use_intensity=False below.
    intensity_data = True if model_name in ('odernn', 'dt', 'decay') else False

    (times, train_dataloader, val_dataloader, test_dataloader, num_classes,
     input_channels) = datasets.uea.get_data(
         dataset_name, missing_rate, device, intensity=intensity_data,
         batch_size=batch_size, loginf=loginf)

    if num_classes == 2:
        output_channels = 1
    else:
        output_channels = num_classes

    make_model = common.make_model(
        model_name, input_channels, output_channels, hidden_channels,
        hidden_hidden_channels, num_hidden_layers, use_intensity=False,
        initial=True, num_heads=num_heads, trafo_ff_dim=trafo_ff_dim,
        dropout=dropout, use_v_laynorm=cde_use_v_laynorm,
        delta_ode_post_tahn=delta_ode_post_tahn)

    def new_make_model():
        model, regularise = make_model()
        model.linear.weight.register_hook(lambda grad: grad_scale * grad)
        model.linear.bias.register_hook(lambda grad: grad_scale * grad)
        return model, regularise

    if dry_run:
        name = None
    else:
        name = dataset_name + str(int(missing_rate * 100))
    return common.main(
        name, times, train_dataloader, val_dataloader, test_dataloader, device,
        new_make_model, num_classes, max_epochs, lr, kwargs, step_mode=False,
        use_wandb=use_wandb, epoch_per_metric=epoch_per_metric,
        val_based_stopping=val_based_stopping,
        auroc_based_stopping=auroc_based_stopping, loginf=loginf)


def run_all(group, device, dataset_name,
            model_names=('ncde', 'odernn', 'dt', 'decay', 'gruode')):
    if group == 1:
        missing_rate = 0.3
    elif group == 2:
        missing_rate = 0.5
    elif group == 3:
        missing_rate = 0.7
    else:
        raise ValueError
    model_kwargs = dict(ncde=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
                        odernn=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
                        dt=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None),
                        decay=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None),
                        gruode=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None))
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(dataset_name, missing_rate, device, model_name=model_name, **model_kwargs[model_name])
