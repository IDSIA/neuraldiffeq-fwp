import pathlib
import sys
import torch
import torch.nn as nn

here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..' / '..'))

import torchdiffeq
import controldiffeq

from .learning_rules import (
    LearningRuleVectorFieldWrapper, CDELearningRuleVectorFieldWrapper,
    TransformerFFlayers, CDEDirectMapVectorField,
    HebbUpdateVectorField, OjaUpdateVectorField, DeltaUpdateVectorField,
    CDEHebbUpdateVectorField, CDEHebbUpdateVectorFieldv2,
    CDEOjaUpdateVectorField, CDEDeltaUpdateVectorField,
    DxOnlyCDEHebbUpdateVectorField, DxOnlyCDEOjaUpdateVectorField,
    DxOnlyCDEDeltaUpdateVectorField)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    # return number of parameters
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_grad(self):
        # More efficient than optimizer.zero_grad() according to:
        # Szymon Migacz "PYTORCH PERFORMANCE TUNING GUIDE" at GTC-21.
        # - doesn't execute memset for every parameter
        # - memory is zeroed-out by the allocator in a more efficient way
        # - backward pass updates gradients with "=" operator (write) (unlike
        # zero_grad() which would result in "+=").
        # In PyT >= 1.7, one can do `model.zero_grad(set_to_none=True)`
        for p in self.parameters():
            p.grad = None

    def print_params(self):
        for p in self.named_parameters():
            print(p)


class NeuralCDE(BaseModel):
    """A Neural CDE model. Provides a wrapper around the lower-level cdeint function, to get a flexible Neural CDE
    model.

    Specifically, considering the CDE
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where X is determined by the data, and given some terminal time t_N, then this model first computes z_{t_N}, then
    performs a linear function on it, and then outputs the result.

    It's known that linear functions on CDEs are universal approximators, so this is a very general type of model.
    """
    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        if isinstance(func, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(NeuralCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.func = func
        self.initial = initial
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """

        # Extract the sizes of the batch dimensions from the coefficients
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device)
            else:
                z0 = self.initial_network(cubic_spline.evaluate(times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # continuing adventures in ugly hacks
                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)

        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat([times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # Actually solve the CDE
        z_t = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=t,
                                   **kwargs)

        # Organise the output
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # Linear map and return
        pred_y = self.linear(z_t)
        return pred_y


# Note that this relies on the first channel being time
class ContinuousRNNConverter(BaseModel):
    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousRNNConverter, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model

        out_base = torch.zeros(self.input_channels + self.hidden_channels, self.input_channels)
        for i in range(self.input_channels):
            out_base[i, i] = 1
        self.register_buffer('out_base', out_base)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(self.input_channels, self.hidden_channels)

    def forward(self, z):
        # z is a tensor of shape (..., input_channels + hidden_channels)
        x = z[..., :self.input_channels]
        h = z[..., self.input_channels:]
        # In theory the hidden state must lie in this region. And most of the time it does anyway! Very occasionally
        # it escapes this and breaks everything, though. (Even when using adaptive solvers or small step sizes.) Which
        # is kind of surprising given how similar the GRU-ODE is to a standard negative exponential problem, we'd
        # expect to get absolute stability without too much difficulty. Maybe there's a bug in the implementation
        # somewhere, but not that I've been able to find... (and h does only escape this region quite rarely.)
        h = h.clamp(-1, 1)
        # model_out is a tensor of shape (..., hidden_channels)
        model_out = self.model(x, h)
        batch_dims = model_out.shape[:-1]
        out = self.out_base.repeat(*batch_dims, 1, 1).clone()
        out[..., self.input_channels:, 0] = model_out
        return out


class DirectRecurrentODE(BaseModel):
    """A Direct Recurrent ODE approach.
    
    Take control signals as an input to the vector field.
    """
    def __init__(self, func, input_channels, hidden_channels, output_channels,
                 initial=True):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        super(DirectRecurrentODE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.func = func
        self.initial = initial
        # map initial input to init hidden state
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """

        # Extract the sizes of the batch dimensions from the coefficients
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            z0 = self.initial_network(cubic_spline.evaluate(times[0]))

        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(
                sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat(
                [times[0].unsqueeze(0), times[sorted_final_index],
                 times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # Actually solve the ODE
        z_t = controldiffeq.recodeint(X=cubic_spline.evaluate,
                                      z0=z0,
                                      func=self.func,
                                      t=t,
                                      **kwargs)

        # Organise the output
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # Linear map and return
        pred_y = self.linear(z_t)
        return pred_y


class FastWeightODE(BaseModel):
    """A Direct Recurrent ODE approach for FWP.
    
    Take control signals as an input to the vector field.
    """
    def __init__(self, input_channels, hidden_channels, output_channels,
                 num_heads=8, trafo_ff_dim=None, learning_rule='hebb',
                 dropout=0.0, initial=True, delta_post_tahn=False):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        super(FastWeightODE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.trafo_ff_dim = trafo_ff_dim
        self.learning_rule = learning_rule
        assert hidden_channels % num_heads == 0
        self.head_dim = hidden_channels // num_heads

        if learning_rule == 'oja':
            self.update_net = OjaUpdateVectorField(
                input_channels, hidden_channels, num_heads)
        elif learning_rule == 'delta':
            self.update_net = DeltaUpdateVectorField(
                input_channels, hidden_channels, num_heads,
                post_tahn=delta_post_tahn)    
        else:
            assert learning_rule == 'hebb'
            self.update_net = HebbUpdateVectorField(
                input_channels, hidden_channels, num_heads)
           
        self.initial = initial
        # map initial input to init hidden state
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

        self.query_net = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if trafo_ff_dim is None:
            trafo_ff_dim = 4 * hidden_channels

        self.ff_block = TransformerFFlayers(
            trafo_ff_dim, hidden_channels, dropout=dropout)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """

        # Extract the sizes of the batch dimensions from the coefficients
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        X = cubic_spline.evaluate

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            z0 = self.update_net(X(times[0]))

        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(
                sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat(
                [times[0].unsqueeze(0), times[sorted_final_index],
                 times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # Actually solve the ODE
        vector_field = LearningRuleVectorFieldWrapper(X, self.update_net)
        z_t = torchdiffeq.odeint_adjoint(
            func=vector_field, y0=z0, t=t, **kwargs)

        # Organise the output
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # reshape to get weight matrix
        bsz = z0.shape[0]
        # z_t is a weight matrix, but needs to be reshaped to be so
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        z_t = z_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)

        # get query from the last time step
        qs = self.update_net.input_proj(X(times[-1]))
        qs = self.update_net.input_layer_norm(qs)
        qs = self.query_net(qs)
        qs = qs.view(bsz, self.num_heads, self.head_dim)
        qs = qs.view(bsz * self.num_heads, self.head_dim)

        qs = torch.softmax(qs, dim=-1)

        # 'bij, bj->bi'
        z_t = torch.bmm(z_t, qs.unsqueeze(2)).squeeze()
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim)
        z_t = z_t.reshape(bsz, self.num_heads * self.head_dim)

        # out projection
        z_t = self.out_proj(z_t)

        if self.dropout is not None:
            z_t = self.dropout(z_t)

        # Transformer FF layers:
        z_t = self.ff_block(z_t)

        # Linear map and return
        pred_y = self.linear(z_t)
        return pred_y


# Use x:keys, dx/dt:values, Hebb or Oja
class FastWeightCDE(BaseModel):
    """A Direct Recurrent *C*DE approach for FWP.
    
    Take control signals as an input to the vector field.
    """
    def __init__(self, input_channels, hidden_channels, output_channels,
                 num_heads=8, trafo_ff_dim=None, learning_rule='hebb',
                 dropout=0.0, initial=True, use_v_laynorm=False):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        super(FastWeightCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.trafo_ff_dim = trafo_ff_dim
        self.learning_rule = learning_rule
        self.use_v_laynorm = use_v_laynorm
        assert hidden_channels % num_heads == 0
        self.head_dim = hidden_channels // num_heads

        if learning_rule == 'oja':
            self.update_net = CDEOjaUpdateVectorField(
                input_channels, hidden_channels, num_heads, use_v_laynorm)
        else:
            assert learning_rule == 'hebb'
            self.update_net = CDEHebbUpdateVectorField(
                input_channels, hidden_channels, num_heads, use_v_laynorm)

        self.initial = initial
        # map initial input to init hidden state
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

        self.query_net = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if trafo_ff_dim is None:
            trafo_ff_dim = 4 * hidden_channels

        self.ff_block = TransformerFFlayers(
            trafo_ff_dim, hidden_channels, dropout=dropout)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """

        # Extract the sizes of the batch dimensions from the coefficients
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        X = cubic_spline.evaluate
        dX_dt = cubic_spline.derivative

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            z0 = self.update_net(X(times[0]), dX_dt(times[0]))

        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(
                sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat(
                [times[0].unsqueeze(0), times[sorted_final_index],
                 times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # Actually solve the ODE
        vector_field = CDELearningRuleVectorFieldWrapper(
            X, dX_dt, self.update_net)
        z_t = torchdiffeq.odeint_adjoint(
            func=vector_field, y0=z0, t=t, **kwargs)

        # Organise the output
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # reshape to get weight matrix
        bsz = z0.shape[0]
        # z_t is a weight matrix, but needs to be reshaped to be so
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        z_t = z_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)

        # CDE specific: transpose z_t as we use x as keys, dx as values
        z_t = z_t.transpose(1, 2)

        # get query from the last time step
        qs = self.update_net.input_proj(X(times[-1])) 
        qs = self.update_net.input_layer_norm(qs)
        qs = self.query_net(qs)
        qs = qs.view(bsz, self.num_heads, self.head_dim)
        qs = qs.view(bsz * self.num_heads, self.head_dim)

        qs = torch.softmax(qs, dim=-1)

        # 'bij, bj->bi'
        z_t = torch.bmm(z_t, qs.unsqueeze(2)).squeeze()
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim)
        z_t = z_t.reshape(bsz, self.num_heads * self.head_dim)

        # out projection
        z_t = self.out_proj(z_t)

        if self.dropout is not None:
            z_t = self.dropout(z_t)

        # Transformer FF layers:
        z_t = self.ff_block(z_t)

        # Linear map and return
        pred_y = self.linear(z_t)
        return pred_y


# v2: Use dx to generate key, Hebb or Delta
class FastWeightCDEv2(BaseModel):
    """A Direct Recurrent *C*DE approach for FWP.
    
    Take control signals as an input to the vector field.
    """
    def __init__(self, input_channels, hidden_channels, output_channels,
                 num_heads=8, trafo_ff_dim=None, learning_rule='hebb',
                 dropout=0.0, initial=True, use_v_laynorm=False,
                 delta_post_tahn=False):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        super(FastWeightCDEv2, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.trafo_ff_dim = trafo_ff_dim
        self.learning_rule = learning_rule
        self.use_v_laynorm = use_v_laynorm
        assert hidden_channels % num_heads == 0
        self.head_dim = hidden_channels // num_heads

        if learning_rule == 'delta':
            self.update_net = CDEDeltaUpdateVectorField(
                input_channels, hidden_channels, num_heads, use_v_laynorm,
                delta_post_tahn)
        elif learning_rule == 'hebb_dx_only':
            self.update_net = DxOnlyCDEHebbUpdateVectorField(
                input_channels, hidden_channels, num_heads, use_v_laynorm)
        elif learning_rule == 'oja_dx_only':
            self.update_net = DxOnlyCDEOjaUpdateVectorField(
                input_channels, hidden_channels, num_heads, use_v_laynorm)
        elif learning_rule == 'delta_dx_only':
            self.update_net = DxOnlyCDEDeltaUpdateVectorField(
                input_channels, hidden_channels, num_heads, use_v_laynorm,
                delta_post_tahn)
        else:
            assert learning_rule == 'hebb'
            self.update_net = CDEHebbUpdateVectorFieldv2(
                input_channels, hidden_channels, num_heads, use_v_laynorm)
           
        self.initial = initial
        # map initial input to init hidden state
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

        self.query_net = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if trafo_ff_dim is None:
            trafo_ff_dim = 4 * hidden_channels

        self.ff_block = TransformerFFlayers(
            trafo_ff_dim, hidden_channels, dropout=dropout)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """

        # Extract the sizes of the batch dimensions from the coefficients
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        X = cubic_spline.evaluate
        dX_dt = cubic_spline.derivative

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            z0 = self.update_net(X(times[0]), dX_dt(times[0]))

        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(
                sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat(
                [times[0].unsqueeze(0), times[sorted_final_index],
                 times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # Actually solve the ODE
        vector_field = CDELearningRuleVectorFieldWrapper(
            X, dX_dt, self.update_net)
        z_t = torchdiffeq.odeint_adjoint(
            func=vector_field, y0=z0, t=t, **kwargs)

        # Organise the output
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # reshape to get weight matrix
        bsz = z0.shape[0]
        # z_t is a weight matrix, but needs to be reshaped to be so
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        z_t = z_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)

        # no need to reshape for v2
        # CDE specific: transpose z_t as we use x as keys, dx as values
        # z_t = z_t.transpose(1, 2)

        # get query from dx the last time step
        qs = self.update_net.input_proj(dX_dt(times[-1])) 
        qs = self.update_net.input_layer_norm(qs)
        qs = self.query_net(qs)
        qs = qs.view(bsz, self.num_heads, self.head_dim)
        qs = qs.view(bsz * self.num_heads, self.head_dim)

        qs = torch.softmax(qs, dim=-1)

        # 'bij, bj->bi'
        z_t = torch.bmm(z_t, qs.unsqueeze(2)).squeeze()
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim)
        z_t = z_t.reshape(bsz, self.num_heads * self.head_dim)

        # out projection
        z_t = self.out_proj(z_t)

        if self.dropout is not None:
            z_t = self.dropout(z_t)

        # Transformer FF layers:
        z_t = self.ff_block(z_t)

        # Linear map and return
        pred_y = self.linear(z_t)
        return pred_y


# v3: same as v2 but with transposing the FWM for ablation
class FastWeightCDEv3(BaseModel):
    """A Direct Recurrent *C*DE approach for FWP.
    
    Take control signals as an input to the vector field.
    """
    def __init__(self, input_channels, hidden_channels, output_channels,
                 num_heads=8, trafo_ff_dim=None, learning_rule='hebb',
                 dropout=0.0, initial=True):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        super(FastWeightCDEv3, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.trafo_ff_dim = trafo_ff_dim
        self.learning_rule = learning_rule
        assert hidden_channels % num_heads == 0
        self.head_dim = hidden_channels // num_heads

        if learning_rule == 'oja':
            self.update_net = OjaUpdateVectorField(
                input_channels, hidden_channels, num_heads)
        elif learning_rule == 'delta':
            self.update_net = DeltaUpdateVectorField(
                input_channels, hidden_channels, num_heads)    
        else:
            assert learning_rule == 'hebb'
            self.update_net = CDEHebbUpdateVectorFieldv2(
                input_channels, hidden_channels, num_heads)
           
        self.initial = initial
        # map initial input to init hidden state
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

        self.query_net = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if trafo_ff_dim is None:
            trafo_ff_dim = 4 * hidden_channels

        self.ff_block = TransformerFFlayers(
            trafo_ff_dim, hidden_channels, dropout=dropout)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """

        # Extract the sizes of the batch dimensions from the coefficients
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        X = cubic_spline.evaluate
        dX_dt = cubic_spline.derivative

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            z0 = self.update_net(X(times[0]), dX_dt(times[0]))

        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(
                sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat(
                [times[0].unsqueeze(0), times[sorted_final_index],
                 times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # Actually solve the ODE
        vector_field = CDELearningRuleVectorFieldWrapper(
            X, dX_dt, self.update_net)
        z_t = torchdiffeq.odeint_adjoint(
            func=vector_field, y0=z0, t=t, **kwargs)

        # Organise the output
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # reshape to get weight matrix
        bsz = z0.shape[0]
        # z_t is a weight matrix, but needs to be reshaped to be so
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        z_t = z_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)

        # no need to reshape for v2
        # CDE specific: transpose z_t as we use x as keys, dx as values
        z_t = z_t.transpose(1, 2)

        # get query from dx the last time step
        qs = self.update_net.input_proj(dX_dt(times[-1])) 
        qs = self.update_net.input_layer_norm(qs)
        qs = self.query_net(qs)
        qs = qs.view(bsz, self.num_heads, self.head_dim)
        qs = qs.view(bsz * self.num_heads, self.head_dim)

        qs = torch.softmax(qs, dim=-1)

        # 'bij, bj->bi'
        z_t = torch.bmm(z_t, qs.unsqueeze(2)).squeeze()
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim)
        z_t = z_t.reshape(bsz, self.num_heads * self.head_dim)

        # out projection
        z_t = self.out_proj(z_t)

        if self.dropout is not None:
            z_t = self.dropout(z_t)

        # Transformer FF layers:
        z_t = self.ff_block(z_t)

        # Linear map and return
        pred_y = self.linear(z_t)
        return pred_y


class BigMapFastWeightCDE(BaseModel):
    """A Direct Recurrent *C*DE approach for FWP.
    
    Take control signals as an input to the vector field.
    """
    def __init__(self, input_channels, hidden_channels, output_channels,
                 hidden_hidden_channels=None, num_hidden_layers=1,
                 dropout=0.0, trafo_ff_dim=None, initial=True):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        super(BigMapFastWeightCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
           
        self.initial = initial
        # map initial input to init hidden state
        self.initial_network = torch.nn.Linear(
            input_channels, input_channels * hidden_channels)
        self.update_net = CDEDirectMapVectorField(
                input_channels, hidden_channels,
                hidden_hidden_channels, num_hidden_layers)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

        if trafo_ff_dim is None:
            trafo_ff_dim = 4 * hidden_channels

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.ff_block = TransformerFFlayers(
            trafo_ff_dim, hidden_channels, dropout=dropout)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """

        # Extract the sizes of the batch dimensions from the coefficients
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        X = cubic_spline.evaluate
        dX_dt = cubic_spline.derivative

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            z0 = self.initial_network(X(times[0]))

        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(
                sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat(
                [times[0].unsqueeze(0), times[sorted_final_index],
                 times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # Actually solve the ODE
        vector_field = CDELearningRuleVectorFieldWrapper(
            X, dX_dt, self.update_net)
        z_t = torchdiffeq.odeint_adjoint(
            func=vector_field, y0=z0, t=t, **kwargs)

        # Organise the output
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # reshape to get weight matrix
        bsz = z0.shape[0]
        # z_t is a weight matrix, but needs to be reshaped to be so
        z_t = z_t.reshape(bsz, self.hidden_channels, self.input_channels)
        x_final = X(times[-1])
        z_t = torch.bmm(z_t, x_final.unsqueeze(2)).squeeze()

        if self.dropout is not None:
            z_t = self.dropout(z_t)

        # Transformer FF layers:
        z_t = self.ff_block(z_t)

        # Linear map and return
        pred_y = self.linear(z_t)
        return pred_y
