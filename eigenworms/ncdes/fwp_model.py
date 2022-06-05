# FWP models
import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint, odeint
from .learning_rules import (
    LearningRuleVectorFieldWrapper, CDELearningRuleVectorFieldWrapper,
    MultiLayerLearningRuleVectorFieldWrapper,
    TransformerFFlayers, CDEDirectMapVectorField,
    HebbUpdateVectorField, OjaUpdateVectorField, DeltaUpdateVectorField,
    CDEHebbUpdateVectorField, CDEHebbUpdateVectorFieldv2,
    CDEOjaUpdateVectorField, CDEDeltaUpdateVectorField,
    DxOnlyCDEHebbUpdateVectorField, DxOnlyCDEOjaUpdateVectorField,
    DxOnlyCDEDeltaUpdateVectorField)

from .rdeint import set_options, _GetLogsignature


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


class FastWeightODE(BaseModel):
    """A Direct Recurrent ODE approach for FWP.
    
    Take control signals as an input to the vector field.
    """
    def __init__(self, initial_dim, input_channels, hidden_channels, output_channels,
                 num_heads=8, trafo_ff_dim=None, learning_rule='hebb',
                 dropout=0.0, delta_post_tahn=True, query_with_init=False,
                 scale_query_net=False, scale_ff=False, scale_out_proj=False):
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
        self.initial_dim = initial_dim
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.trafo_ff_dim = trafo_ff_dim
        self.learning_rule = learning_rule
        self.query_with_init = query_with_init
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
           
        # map initial input to init hidden state
        self.initial_network = torch.nn.Linear(initial_dim, input_channels)
        self.final_linear = torch.nn.Linear(hidden_channels, output_channels)

        self.scale_query_net = scale_query_net
        if scale_query_net:
            self.final_query_net = nn.Linear(hidden_channels, hidden_channels)
        else:
            self.query_net = nn.Linear(hidden_channels, hidden_channels)

        self.scale_out_proj = scale_out_proj
        if scale_out_proj:
            self.final_out_proj = nn.Linear(hidden_channels, hidden_channels)
        else:
            self.out_proj = nn.Linear(hidden_channels, hidden_channels)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if trafo_ff_dim is None:
            trafo_ff_dim = 4 * hidden_channels

        self.scale_ff = scale_ff
        if scale_ff:
            self.final_ff_block = TransformerFFlayers(
                trafo_ff_dim, hidden_channels, dropout=dropout)
        else:
            self.ff_block = TransformerFFlayers(
                trafo_ff_dim, hidden_channels, dropout=dropout)

    def extra_repr(self):
        return (f"input_channels={self.input_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"output_channels={self.output_channels}")

    def forward(self, inputs, method='rk4', adjoint=True, return_sequences=False):
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
        assert len(inputs) == 2, "`inputs` must be a 2-tuple containing `(inital_values, logsig)`."
        initial, logsig = inputs
        # z0 = self.initial_linear(initial)
        initial_proj = self.initial_network(initial)
        z0 = self.update_net(initial_proj)

        # Method to get the logsig value
        logsig_getter = _GetLogsignature(logsig)
        # Set options
        t, options, = set_options(logsig, return_sequences=return_sequences)

        # Actually solve the ODE
        vector_field = LearningRuleVectorFieldWrapper(
            logsig_getter, self.update_net)
        odeint_func = odeint_adjoint if adjoint else odeint
        z_t = odeint_func(
            func=vector_field, t=t, y0=z0, method=method, options=options)

        # reshape to get weight matrix
        z_t = z_t[-1]
        bsz = z0.shape[0]
        # z_t is a weight matrix, but needs to be reshaped to be so
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        z_t = z_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)

        # get query from the last time step
        if self.query_with_init:
            qs = self.update_net.input_proj(initial_proj)
        else:
            qs = self.update_net.input_proj(logsig_getter[t[-1]])

        qs = self.update_net.input_layer_norm(qs)

        if self.scale_query_net:
            qs = self.final_query_net(qs)
        else:
            qs = self.query_net(qs)

        qs = qs.view(bsz, self.num_heads, self.head_dim)
        qs = qs.view(bsz * self.num_heads, self.head_dim)

        qs = torch.softmax(qs, dim=-1)

        # 'bij, bj->bi'
        z_t = torch.bmm(z_t, qs.unsqueeze(2)).squeeze()
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim)
        z_t = z_t.reshape(bsz, self.num_heads * self.head_dim)

        # out projection
        if self.scale_out_proj:
            z_t = self.final_out_proj(z_t)
        else:
            z_t = self.out_proj(z_t)

        if self.dropout is not None:
            z_t = self.dropout(z_t)

        # Transformer FF layers:
        if self.scale_ff:
            z_t = self.final_ff_block(z_t)
        else:
            z_t = self.ff_block(z_t)

        # Linear map and return
        pred_y = self.final_linear(z_t)
        return pred_y


# v2: Use dx to generate key, Hebb or Delta
class FastWeightCDEv2(BaseModel):
    """A Direct Recurrent *C*DE approach for FWP.
    
    Take control signals as an input to the vector field.
    """
    def __init__(self, initial_dim, input_channels, hidden_channels,
                 output_channels, num_heads=16, trafo_ff_dim=None,
                 learning_rule='oja', query_with_init=False,
                 scale_query_net=False, scale_ff=False, scale_out_proj=False,
                 dropout=0.0, use_v_laynorm=True):
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
        self.initial_dim = initial_dim
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.trafo_ff_dim = trafo_ff_dim
        self.learning_rule = learning_rule
        self.use_v_laynorm = use_v_laynorm
        self.query_with_init = query_with_init
        assert hidden_channels % num_heads == 0
        self.head_dim = hidden_channels // num_heads

        if learning_rule == 'oja':
            self.update_net = DxOnlyCDEOjaUpdateVectorField(
                input_channels, hidden_channels, num_heads, use_v_laynorm)
        elif learning_rule == 'delta':
            self.update_net = DxOnlyCDEDeltaUpdateVectorField(
                input_channels, hidden_channels, num_heads, use_v_laynorm)
        else:
            assert learning_rule == 'hebb'
            self.update_net = DxOnlyCDEHebbUpdateVectorField(
                input_channels, hidden_channels, num_heads, use_v_laynorm)

        # map initial input to init hidden state
        self.initial_network = torch.nn.Linear(initial_dim, input_channels)

        self.scale_query_net = scale_query_net
        if scale_query_net:
            self.final_query_net = nn.Linear(hidden_channels, hidden_channels)
        else:
            self.query_net = nn.Linear(hidden_channels, hidden_channels)

        self.scale_out_proj = scale_out_proj
        if scale_out_proj:
            self.final_out_proj = nn.Linear(hidden_channels, hidden_channels)
        else:
            self.out_proj = nn.Linear(hidden_channels, hidden_channels)

        self.final_linear = torch.nn.Linear(hidden_channels, output_channels)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if trafo_ff_dim is None:
            trafo_ff_dim = 4 * hidden_channels

        self.scale_ff = scale_ff
        if scale_ff:
            self.final_ff_block = TransformerFFlayers(
                trafo_ff_dim, hidden_channels, dropout=dropout)
        else:
            self.ff_block = TransformerFFlayers(
                trafo_ff_dim, hidden_channels, dropout=dropout)

    def extra_repr(self):
        return (f"input_channels={self.input_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"output_channels={self.output_channels}")

    def forward(self, inputs, method='rk4', adjoint=True, return_sequences=False):
        """
        Arguments:
            times: The times of the observations for the input path X,
                e.g. as passed as an argument to
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

        assert len(inputs) == 2, "`inputs` must be a 2-tuple containing `(inital_values, logsig)`."
        initial, logsig = inputs

        initial_proj = self.initial_network(initial)
        z0 = self.update_net(initial_proj, initial_proj)

        # Method to get the logsig value
        logsig_getter = _GetLogsignature(logsig)
        # Set options
        t, options, = set_options(logsig, return_sequences=return_sequences)

        # Actually solve the ODE
        vector_field = CDELearningRuleVectorFieldWrapper(
            logsig_getter, logsig_getter, self.update_net)

        odeint_func = odeint_adjoint if adjoint else odeint
        z_t = odeint_func(
            func=vector_field, t=t, y0=z0, method=method, options=options)

        # reshape to get weight matrix
        z_t = z_t[-1]
        bsz = z0.shape[0]
        # z_t is a weight matrix, but needs to be reshaped to be so
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        z_t = z_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)

        # no need to reshape for v2
        # CDE specific: transpose z_t as we use x as keys, dx as values
        # z_t = z_t.transpose(1, 2)
        # get query from dx the last time step
        if self.query_with_init:
            qs = self.update_net.input_proj(initial_proj)
        else:
            qs = self.update_net.input_proj(logsig_getter[t[-1]])
        qs = self.update_net.input_layer_norm(qs)
        if self.scale_query_net:
            qs = self.final_query_net(qs)
        else:
            qs = self.query_net(qs)
        qs = qs.view(bsz, self.num_heads, self.head_dim)
        qs = qs.view(bsz * self.num_heads, self.head_dim)

        qs = torch.softmax(qs, dim=-1)

        # 'bij, bj->bi'
        z_t = torch.bmm(z_t, qs.unsqueeze(2)).squeeze()
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim)
        z_t = z_t.reshape(bsz, self.num_heads * self.head_dim)

        # out projection
        if self.scale_out_proj:
            z_t = self.final_out_proj(z_t)
        else:
            z_t = self.out_proj(z_t)

        if self.dropout is not None:
            z_t = self.dropout(z_t)

        # Transformer FF layers:
        if self.scale_ff:
            z_t = self.final_ff_block(z_t)
        else:
            z_t = self.ff_block(z_t)

        # Linear map and return
        pred_y = self.final_linear(z_t)
        return pred_y


# Hard coded for two-layer models. Can be easily extended to support arbitrary depth.
class MultiLayerFastWeightODE(BaseModel):
    """A Direct Recurrent ODE approach for FWP.
    
    Take control signals as an input to the vector field.
    """
    def __init__(self, initial_dim, input_channels, hidden_channels, output_channels,
                 num_heads=8, trafo_ff_dim=None, learning_rule='hebb',
                 dropout=0.0, delta_post_tahn=True, query_with_init=False,
                 scale_query_net=False, scale_ff=False, scale_out_proj=False,
                 residual=False):
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
        super(MultiLayerFastWeightODE, self).__init__()
        self.initial_dim = initial_dim
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.trafo_ff_dim = trafo_ff_dim
        self.learning_rule = learning_rule
        self.query_with_init = query_with_init
        assert hidden_channels % num_heads == 0
        self.head_dim = hidden_channels // num_heads
        self.total_fw_size = num_heads * self.head_dim * self.head_dim
        self.residual = residual

        if learning_rule == 'oja':
            self.update_net = OjaUpdateVectorField(
                input_channels, hidden_channels, num_heads)
        elif learning_rule == 'delta':
            self.update_net = DeltaUpdateVectorField(
                input_channels, hidden_channels, num_heads,
                post_tahn=delta_post_tahn)
            self.update_net2 = DeltaUpdateVectorField(
                input_channels, hidden_channels, num_heads,
                post_tahn=delta_post_tahn, skip_input_proj=True)
        else:
            assert learning_rule == 'hebb'
            self.update_net = HebbUpdateVectorField(
                input_channels, hidden_channels, num_heads)
           
        # map initial input to init hidden state
        self.initial_network = torch.nn.Linear(initial_dim, input_channels)
        self.final_linear = torch.nn.Linear(hidden_channels, output_channels)

        self.scale_query_net = scale_query_net
        if scale_query_net:
            self.final_query_net = nn.Linear(hidden_channels, hidden_channels)
        else:
            self.query_net = nn.Linear(hidden_channels, hidden_channels)

        self.scale_out_proj = scale_out_proj
        if scale_out_proj:
            self.final_out_proj = nn.Linear(hidden_channels, hidden_channels)
        else:
            self.out_proj = nn.Linear(hidden_channels, hidden_channels)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if trafo_ff_dim is None:
            trafo_ff_dim = 4 * hidden_channels

        self.scale_ff = scale_ff
        if scale_ff:
            self.final_ff_block = TransformerFFlayers(
                trafo_ff_dim, hidden_channels, dropout=dropout)
        else:
            self.ff_block = TransformerFFlayers(
                trafo_ff_dim, hidden_channels, dropout=dropout)

        # for second layer
        self.mid_ff_block = TransformerFFlayers(
            trafo_ff_dim, hidden_channels, dropout=dropout)
        self.mid_query_net = nn.Linear(hidden_channels, hidden_channels)

    def extra_repr(self):
        return (f"input_channels={self.input_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"output_channels={self.output_channels}")

    def forward(self, inputs, method='rk4', adjoint=True, return_sequences=False):
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
        assert len(inputs) == 2, "`inputs` must be a 2-tuple containing `(inital_values, logsig)`."
        initial, logsig = inputs
        # z0 = self.initial_linear(initial)
        initial_proj = self.initial_network(initial)
        z0 = self.update_net(initial_proj)
        z02 = torch.zeros_like(z0)
        z0 = torch.cat([z0, z02], dim=-1)

        # Method to get the logsig value
        logsig_getter = _GetLogsignature(logsig)
        # Set options
        t, options, = set_options(logsig, return_sequences=return_sequences)

        # Actually solve the ODE
        vector_field = MultiLayerLearningRuleVectorFieldWrapper(
            logsig_getter, self.update_net, self.update_net2,
            self.mid_ff_block, self.mid_query_net, self.total_fw_size,
            self.num_heads, self.head_dim, self.residual)
        odeint_func = odeint_adjoint if adjoint else odeint
        z_t = odeint_func(
            func=vector_field, t=t, y0=z0, method=method, options=options)

        # reshape to get weight matrix
        z_t = z_t[-1]
        # take the last layer:
        z_t, z_t2 = torch.split(
            z_t, [self.total_fw_size, self.total_fw_size], dim=-1)
        bsz = z0.shape[0]
        # z_t is a weight matrix, but needs to be reshaped to be so
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        z_t = z_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)

        z_t2 = z_t2.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        z_t2 = z_t2.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)

        # get query from the last time step
        if self.query_with_init:
            qs_prel = self.update_net.input_proj(initial_proj)
        else:
            qs_prel = self.update_net.input_proj(logsig_getter[t[-1]])

        qs = self.update_net.input_layer_norm(qs_prel)

        if self.scale_query_net:
            assert False
            qs = self.final_query_net(qs)
        else:
            qs = self.mid_query_net(qs)

        qs = qs.view(bsz, self.num_heads, self.head_dim)
        qs = qs.view(bsz * self.num_heads, self.head_dim)
        qs = torch.softmax(qs, dim=-1)

        # 'bij, bj->bi'
        z_t = torch.bmm(z_t, qs.unsqueeze(2)).squeeze()
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim)
        z_t = z_t.reshape(bsz, self.num_heads * self.head_dim)

        if self.residual:
            z_t = z_t + qs_prel

        qs_prel = self.mid_ff_block(z_t)

        qs = self.update_net2.input_layer_norm(qs_prel)
        qs = self.query_net(qs)

        qs = qs.view(bsz, self.num_heads, self.head_dim)
        qs = qs.view(bsz * self.num_heads, self.head_dim)
        qs = torch.softmax(qs, dim=-1)

        z_t = torch.bmm(z_t2, qs.unsqueeze(2)).squeeze()
        z_t = z_t.reshape(bsz, self.num_heads, self.head_dim)
        z_t = z_t.reshape(bsz, self.num_heads * self.head_dim)
        if self.residual:
            z_t = z_t + qs_prel

        # out projection
        if self.scale_out_proj:
            z_t = self.final_out_proj(z_t)
        else:
            z_t = self.out_proj(z_t)

        if self.dropout is not None:
            z_t = self.dropout(z_t)

        # Transformer FF layers:
        if self.scale_ff:
            z_t = self.final_ff_block(z_t)
        else:
            z_t = self.ff_block(z_t)

        # Linear map and return
        pred_y = self.final_linear(z_t)
        return pred_y
