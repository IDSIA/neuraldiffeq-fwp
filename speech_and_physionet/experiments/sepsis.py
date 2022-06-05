import torch
import torch.nn as nn

import common
import datasets

from models import BaseModel


class InitialValueNetwork(BaseModel):
    def __init__(self, intensity, hidden_channels, model, model_name,
                 num_heads=None, input_channels=None):
        super(InitialValueNetwork, self).__init__()
        # 256 was fixed in the original code/paper: we keep this fixed as
        # a part of input processing
        # 5: age, ...
        self.linear1 = torch.nn.Linear(7 if intensity else 5, 256)
        if model_name in (
            'ncde', 'gruode', 'mygruode', 'odernn', 'dt', 'decay'):
            # models in the original paper
            self.linear_map = True
            self.linear2 = torch.nn.Linear(256, hidden_channels)
        elif num_heads is None:  # big map case
            self.linear_map = True
            self.num_heads = None
            self.linear2 = torch.nn.Linear(
                256, hidden_channels * input_channels)
        else:
            assert model_name != 'bigfwcde'
            self.linear_map = False
            self.num_heads = num_heads
            self.head_dim = hidden_channels // num_heads
            # key, value, learning rate
            self.kvl_linear = nn.Linear(
                256, num_heads * (2 * self.head_dim + 1), bias=True)

        self.model = model
        self.model_name = model_name

    def forward(self, times, coeffs, final_index, **kwargs):
        *coeffs, static = coeffs
        z0 = self.linear1(static)
        z0 = z0.relu()
        if self.linear_map:
            z0 = self.linear2(z0)
        else:
            bsz = static.shape[0]
            kvl = self.kvl_linear(z0)
            kvl = kvl.view(bsz, self.num_heads, 2 * self.head_dim + 1)
            ks, vs, lrs = torch.split(kvl, 2 * (self.head_dim,) + (1,), -1)

            # reshape to merge batch and head dims
            ks = ks.reshape(bsz * self.num_heads, self.head_dim)
            vs = vs.reshape(bsz * self.num_heads, self.head_dim)
            lrs = lrs.reshape(bsz * self.num_heads, 1)

            lrs = torch.sigmoid(lrs)
            ks = torch.softmax(ks, dim=-1)
            vs = lrs * torch.tanh(vs)  # tanh is optional

            # weight update term 'bi, bj->bij'
            z0 = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
            # (B * heads, head_dim, head_dim)

            # reshape to vector
            z0 = z0.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
            z0 = z0.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return self.model(times, coeffs, final_index, z0=z0, **kwargs)


def main(intensity, model_name, hidden_channels, hidden_hidden_channels,
         num_hidden_layers, *,
         num_heads=8, trafo_ff_dim=None, cde_use_v_laynorm=False, dropout=0.0,
         delta_ode_post_tahn=False,
         batch_size=1024, device='cuda', max_epochs=200, pos_weight=10,
         base_lr=0.0001, grad_scale=100, dry_run=False, use_wandb=False,
         epoch_per_metric=10, val_based_stopping=False,
         auroc_based_stopping=False, loginf=None,
         **kwargs):  # kwargs passed on to cdeint

    lr = base_lr * (batch_size / 32)

    static_intensity = intensity
    # these models use the intensity for their evolution. They won't explicitly use it as an input unless we include it
    # via the use_intensity parameter, though.
    time_intensity = intensity or (model_name in ('odernn', 'dt', 'decay'))

    (times, train_dataloader, val_dataloader, test_dataloader
     ) = datasets.sepsis.get_data(
         static_intensity, time_intensity, batch_size, loginf)

    input_channels = 1 + (1 + time_intensity) * 34
    make_model = common.make_model(
        model_name, input_channels, 1, hidden_channels, hidden_hidden_channels,
        num_hidden_layers, use_intensity=intensity, initial=False,
        num_heads=num_heads, trafo_ff_dim=trafo_ff_dim, dropout=dropout,
        use_v_laynorm=cde_use_v_laynorm,
        delta_ode_post_tahn=delta_ode_post_tahn)

    def new_make_model():
        model, regularise = make_model()
        model.linear.weight.register_hook(lambda grad: grad_scale * grad)
        model.linear.bias.register_hook(lambda grad: grad_scale * grad)
        return InitialValueNetwork(
            intensity, hidden_channels, model, model_name, num_heads=num_heads,
            input_channels=input_channels), regularise

    if dry_run:
        name = None
    else:
        intensity_str = '_intensity' if intensity else '_nointensity'
        name = 'sepsis' + intensity_str
    num_classes = 2
    return common.main(
        name, times, train_dataloader, val_dataloader, test_dataloader, device,
        new_make_model, num_classes, max_epochs, lr, kwargs,
        pos_weight=torch.tensor(pos_weight), step_mode=True,
        use_wandb=use_wandb, epoch_per_metric=epoch_per_metric,
        val_based_stopping=val_based_stopping,
        auroc_based_stopping=auroc_based_stopping, loginf=loginf)


def run_all(intensity, device, model_names=('ncde', 'odernn', 'dt', 'decay', 'gruode')):
    model_kwargs = dict(ncde=dict(hidden_channels=49, hidden_hidden_channels=49, num_hidden_layers=4),
                        odernn=dict(hidden_channels=128, hidden_hidden_channels=128, num_hidden_layers=4),
                        dt=dict(hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None),
                        decay=dict(hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None),
                        gruode=dict(hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None))
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(intensity, device, model_name=model_name, **model_kwargs[model_name])
