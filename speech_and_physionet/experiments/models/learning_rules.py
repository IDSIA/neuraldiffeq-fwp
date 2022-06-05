# Learning rule vector fields
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearningRuleVectorFieldWrapper(torch.nn.Module):
    def __init__(self, X, rule_func):
        """Generic wrapper for learning rule vector field for ODE.

        Arguments:
            X: control.
            rule_func: learning rule function.
        """
        super(LearningRuleVectorFieldWrapper, self).__init__()
        self.X = X
        self.rule_func = rule_func

    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_x = self.X(t)

        # vector_field is of shape (..., hidden_channels, input_channels)
        out = self.rule_func(control_x, z)

        return out


class HebbUpdateVectorField(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads):
        super(HebbUpdateVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        # maybe one extra layer from input dim to model dim such that we can use residual?
        self.input_proj = nn.Linear(input_channels, hidden_channels)
        self.input_layer_norm = nn.LayerNorm(hidden_channels)

        # slow net producing key and value vectors and a learning rate
        self.kvb_net = nn.Linear(
            hidden_channels, num_heads * (2 * self.head_dim + 1), bias=True)

    # forward output the update term
    def forward(self, x, W=None):  # Hebbian rule does not depend on current value of W
        bsz = x.shape[0]

        out = self.input_proj(x)
        out = self.input_layer_norm(out)
        out = self.kvb_net(out)
        # split into heads
        out = out.view(bsz, self.num_heads, 2 * self.head_dim + 1)
        ks, vs, lrs = torch.split(out, (self.head_dim,) * 2 + (1,), -1)

        # reshape to merge batch and head dims
        ks = ks.reshape(bsz * self.num_heads, self.head_dim)
        vs = vs.reshape(bsz * self.num_heads, self.head_dim)
        lrs = lrs.reshape(bsz * self.num_heads, 1)

        # activations
        lrs = torch.sigmoid(lrs)
        ks = torch.softmax(ks, dim=-1)
        vs = lrs * torch.tanh(vs)

        # weight update term 'bi, bj->bij'
        out = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
        # (B * heads, head_dim, head_dim)

        # reshape to vector
        out = out.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        out = out.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return out


class OjaUpdateVectorField(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads):
        super(OjaUpdateVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.input_proj = nn.Linear(input_channels, hidden_channels)
        self.input_layer_norm = nn.LayerNorm(hidden_channels)

        # slow net producing key and value vectors and a learning rate
        self.kvb_net = nn.Linear(
            hidden_channels, num_heads * (2 * self.head_dim + 1), bias=True)

    # forward output the update term
    def forward(self, x, W=None):
        # x: (B, dim)
        # W: (B, flattened_dim)
        bsz = x.shape[0]

        out = self.input_proj(x)
        out = self.input_layer_norm(out)
        out = self.kvb_net(out)
        # split into heads
        out = out.view(bsz, self.num_heads, 2 * self.head_dim + 1)
        ks, vs, lrs = torch.split(out, (self.head_dim,) * 2 + (1,), -1)

        # reshape to merge batch and head dims
        ks = ks.reshape(bsz * self.num_heads, self.head_dim)
        vs = vs.reshape(bsz * self.num_heads, self.head_dim)
        lrs = lrs.reshape(bsz * self.num_heads, 1)

        # activations
        lrs = torch.sigmoid(lrs)
        vs = torch.tanh(vs)

        if W is not None:
            # Oja:
            W_t = W.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
            W_t = W_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)
            ks_remove = torch.bmm(W_t.transpose(1, 2), vs.unsqueeze(2)).squeeze()
            ks = ks - ks_remove
        ks = torch.softmax(ks, dim=-1) * lrs

        # weight update term 'bi, bj->bij'
        out = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
        # (B * heads, head_dim, head_dim)

        # reshape to vector
        out = out.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        out = out.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return out


class DeltaUpdateVectorField(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads,
                 post_tahn=False):
        super(DeltaUpdateVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.post_tahn = post_tahn

        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.input_proj = nn.Linear(input_channels, hidden_channels)
        self.input_layer_norm = nn.LayerNorm(hidden_channels)

        # slow net producing key and value vectors and a learning rate
        self.kvb_net = nn.Linear(
            hidden_channels, num_heads * (2 * self.head_dim + 1), bias=True)

    # forward output the update term
    def forward(self, x, W=None):
        # x: (B, dim)
        # W: (B, flattened_dim)
        bsz = x.shape[0]

        out = self.input_proj(x)
        out = self.input_layer_norm(out)
        out = self.kvb_net(out)
        # split into heads
        out = out.view(bsz, self.num_heads, 2 * self.head_dim + 1)
        ks, vs, lrs = torch.split(out, (self.head_dim,) * 2 + (1,), -1)

        # reshape to merge batch and head dims
        ks = ks.reshape(bsz * self.num_heads, self.head_dim)
        vs = vs.reshape(bsz * self.num_heads, self.head_dim)
        lrs = lrs.reshape(bsz * self.num_heads, 1)

        # activations
        lrs = torch.sigmoid(lrs)
        ks = torch.softmax(ks, dim=-1)
        if not self.post_tahn:
            vs = torch.tanh(vs)

        if W is not None:
            # Delta rule:
            W_t = W.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
            W_t = W_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)
            vs_remove = torch.bmm(W_t, ks.unsqueeze(2)).squeeze()
            vs = (vs - vs_remove)

        if self.post_tahn:
            vs = torch.tanh(vs)

        vs = lrs * vs

        # weight update term 'bi, bj->bij'
        out = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
        # (B * heads, head_dim, head_dim)

        # reshape to vector
        out = out.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        out = out.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return out


###############################################################################


# A block of residual feed-forward layers in Transformer
class TransformerFFlayers(nn.Module):
    def __init__(self, ff_dim, res_dim, dropout=0.0, use_layernorm=True):
        super(TransformerFFlayers, self).__init__()

        self.res_dim = res_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        self.ff_layers = nn.Sequential(
            nn.Linear(res_dim, ff_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, res_dim),
            nn.Dropout(dropout),
        )

        if use_layernorm:
            self.layer_norm = nn.LayerNorm(res_dim)

    def forward(self, x):
        out = self.layer_norm(x) if self.use_layernorm else x
        out = self.ff_layers(out) + x
        return out


###############################################################################


class CDELearningRuleVectorFieldWrapper(torch.nn.Module):
    def __init__(self, X, dX_dt, rule_func):
        """Generic wrapper for learning rule vector field for CDE.

        unlike LearningRuleVectorFieldWrapper above, takes dX_dt as an input.
        Arguments:
            X: control.
            dX_dt: derivative of control
            rule_func: learning rule function.
        """
        super(CDELearningRuleVectorFieldWrapper, self).__init__()
        self.X = X
        self.dX_dt = dX_dt
        self.rule_func = rule_func

    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_x = self.X(t)
        control_gradient = self.dX_dt(t)

        # vector_field is of shape (..., hidden_channels, input_channels)
        out = self.rule_func(control_x, control_gradient, z)

        return out


# first variant using x as key, fast weight thus has to be transposed for query
class CDEHebbUpdateVectorField(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads,
                 use_v_laynorm=False):
        super(CDEHebbUpdateVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        # self.model_dim = hidden_channels

        self.input_proj = nn.Linear(input_channels, hidden_channels)
        self.input_layer_norm = nn.LayerNorm(hidden_channels)

        # projection matrix for dx_dt
        self.dx_dt_proj = nn.Linear(input_channels, hidden_channels)
        self.use_v_laynorm = use_v_laynorm
        if use_v_laynorm:
            self.v_layer_norm = nn.LayerNorm(hidden_channels)
            self.v_net = nn.Linear(hidden_channels, hidden_channels)

        # slow net producing key and value vectors and a learning rate
        self.kb_net = nn.Linear(
            hidden_channels, num_heads * (self.head_dim + 1), bias=True)

    # forward output the update term
    def forward(self, x, dx_dt, W=None):  # Hebbian rule does not depend on current value of W
        bsz = x.shape[0]

        out = self.input_proj(x)
        out = self.input_layer_norm(out)
        out = self.kb_net(out)
        # split into heads
        out = out.view(bsz, self.num_heads, self.head_dim + 1)
        ks, lrs = torch.split(out, (self.head_dim,) + (1,), -1)

        # reshape to merge batch and head dims
        ks = ks.reshape(bsz * self.num_heads, self.head_dim)
        lrs = lrs.reshape(bsz * self.num_heads, 1)

        # get values from dx_dt
        vs = self.dx_dt_proj(dx_dt)
        if self.use_v_laynorm:
            vs = self.v_net(self.v_layer_norm(vs))
        vs = vs.reshape(bsz * self.num_heads, self.head_dim)
        vs = lrs * vs

        # activations
        lrs = torch.sigmoid(lrs)
        ks = torch.softmax(ks, dim=-1)

        # weight update term 'bi, bj->bij'
        out = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
        # (B * heads, head_dim, head_dim)

        # reshape to vector
        out = out.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        out = out.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return out


# v2: using x as values, dx for keys
class CDEHebbUpdateVectorFieldv2(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads,
                 use_v_laynorm=False):
        super(CDEHebbUpdateVectorFieldv2, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        # self.model_dim = hidden_channels

        self.input_proj = nn.Linear(input_channels, hidden_channels)
        self.input_layer_norm = nn.LayerNorm(hidden_channels)

        # projection matrix for x
        self.vs_proj = nn.Linear(input_channels, hidden_channels)
        self.use_v_laynorm = use_v_laynorm
        if use_v_laynorm:
            self.v_layer_norm = nn.LayerNorm(hidden_channels)
            self.v_net = nn.Linear(hidden_channels, hidden_channels)

        # slow net producing key and value vectors and a learning rate
        self.kb_net = nn.Linear(
            hidden_channels, num_heads * (self.head_dim + 1), bias=True)

    # forward output the update term
    def forward(self, x, dx_dt, W=None):  # no dependency on current value of W
        bsz = x.shape[0]

        out = self.input_proj(dx_dt)
        out = self.input_layer_norm(out)
        out = self.kb_net(out)
        # split into heads
        out = out.view(bsz, self.num_heads, self.head_dim + 1)
        ks, lrs = torch.split(out, (self.head_dim,) + (1,), -1)

        # reshape to merge batch and head dims
        ks = ks.reshape(bsz * self.num_heads, self.head_dim)
        lrs = lrs.reshape(bsz * self.num_heads, 1)

        # get values from x
        vs = self.vs_proj(x)
        if self.use_v_laynorm:
            vs = self.v_net(self.v_layer_norm(vs))
        vs = vs.reshape(bsz * self.num_heads, self.head_dim)
        vs = lrs * vs

        # activations
        lrs = torch.sigmoid(lrs)
        ks = torch.softmax(ks, dim=-1)

        # weight update term 'bi, bj->bij'
        out = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
        # (B * heads, head_dim, head_dim)

        # reshape to vector
        out = out.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        out = out.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return out


class CDEOjaUpdateVectorField(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads,
                 use_v_laynorm=False):
        super(CDEOjaUpdateVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        # self.model_dim = hidden_channels

        self.input_proj = nn.Linear(input_channels, hidden_channels)
        self.input_layer_norm = nn.LayerNorm(hidden_channels)

        # projection matrix for dx_dt
        self.dx_dt_proj = nn.Linear(input_channels, hidden_channels)
        self.use_v_laynorm = use_v_laynorm
        if use_v_laynorm:
            self.v_layer_norm = nn.LayerNorm(hidden_channels)
            self.v_net = nn.Linear(hidden_channels, hidden_channels)

        # slow net producing key and value vectors and a learning rate
        self.kb_net = nn.Linear(
            hidden_channels, num_heads * (self.head_dim + 1), bias=True)

    # forward output the update term
    def forward(self, x, dx_dt, W=None):
        bsz = x.shape[0]

        out = self.input_proj(x)
        out = self.input_layer_norm(out)
        out = self.kb_net(out)
        # split into heads
        out = out.view(bsz, self.num_heads, self.head_dim + 1)
        ks, lrs = torch.split(out, (self.head_dim,) + (1,), -1)

        # reshape to merge batch and head dims
        ks = ks.reshape(bsz * self.num_heads, self.head_dim)
        lrs = lrs.reshape(bsz * self.num_heads, 1)

        # get values from dx_dt
        vs = self.dx_dt_proj(dx_dt)
        if self.use_v_laynorm:
            vs = self.v_net(self.v_layer_norm(vs))
        vs = vs.reshape(bsz * self.num_heads, self.head_dim)

        # activations
        lrs = torch.sigmoid(lrs)
        vs = torch.tanh(vs)

        if W is not None:
            # Oja:
            W_t = W.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
            W_t = W_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)
            ks_remove = torch.bmm(W_t.transpose(1, 2), vs.unsqueeze(2)).squeeze()
            ks = ks - ks_remove
        ks = torch.softmax(ks, dim=-1) * lrs

        # weight update term 'bi, bj->bij'
        out = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
        # (B * heads, head_dim, head_dim)

        # reshape to vector
        out = out.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        out = out.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return out


class CDEDeltaUpdateVectorField(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads,
                 use_v_laynorm=False, post_tahn=False):
        super(CDEDeltaUpdateVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.post_tahn = post_tahn

        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.input_proj = nn.Linear(input_channels, hidden_channels)
        self.input_layer_norm = nn.LayerNorm(hidden_channels)

        # projection matrix for dx_dt
        self.vs_proj = nn.Linear(input_channels, hidden_channels)
        self.use_v_laynorm = use_v_laynorm
        if use_v_laynorm:
            self.v_layer_norm = nn.LayerNorm(hidden_channels)
            self.v_net = nn.Linear(hidden_channels, hidden_channels)

        # slow net producing key and value vectors and a learning rate
        self.kb_net = nn.Linear(
            hidden_channels, num_heads * (self.head_dim + 1), bias=True)

    # forward output the update term
    def forward(self, x, dx_dt, W=None):
        bsz = x.shape[0]

        out = self.input_proj(dx_dt)
        out = self.input_layer_norm(out)
        out = self.kb_net(out)
        # split into heads
        out = out.view(bsz, self.num_heads, self.head_dim + 1)
        ks, lrs = torch.split(out, (self.head_dim,) + (1,), -1)

        # reshape to merge batch and head dims
        ks = ks.reshape(bsz * self.num_heads, self.head_dim)
        lrs = lrs.reshape(bsz * self.num_heads, 1)

        # get values from dx_dt
        vs = self.vs_proj(x)
        if self.use_v_laynorm:
            vs = self.v_net(self.v_layer_norm(vs))
        vs = vs.reshape(bsz * self.num_heads, self.head_dim)

        # activations
        lrs = torch.sigmoid(lrs)
        ks = torch.softmax(ks, dim=-1)
        if not self.post_tahn:
            vs = torch.tanh(vs)

        if W is not None:
            # Delta rule:
            W_t = W.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
            W_t = W_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)
            vs_remove = torch.bmm(W_t, ks.unsqueeze(2)).squeeze()
            vs = vs - vs_remove

        if self.post_tahn:
            vs = torch.tanh(vs)

        vs = lrs * vs

        # weight update term 'bi, bj->bij'
        out = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
        # (B * heads, head_dim, head_dim)

        # reshape to vector
        out = out.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        out = out.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return out


# Use dx/dt to generate both keys and values ##################################


class DxOnlyCDEHebbUpdateVectorField(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads,
                 use_v_laynorm=False):
        super(DxOnlyCDEHebbUpdateVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.input_proj = nn.Linear(input_channels, hidden_channels)
        self.input_layer_norm = nn.LayerNorm(hidden_channels)

        # projection matrix for dx_dt
        self.vs_proj = nn.Linear(input_channels, hidden_channels)
        self.use_v_laynorm = use_v_laynorm
        if use_v_laynorm:
            self.v_layer_norm = nn.LayerNorm(hidden_channels)
            self.v_net = nn.Linear(hidden_channels, hidden_channels)

        # slow net producing key and value vectors and a learning rate
        self.kb_net = nn.Linear(
            hidden_channels, num_heads * (self.head_dim + 1), bias=True)

    # forward output the update term
    # Hebbian rule does not depend on current value of W
    def forward(self, x, dx_dt, W=None):
        bsz = dx_dt.shape[0]

        out = self.input_proj(dx_dt)
        out = self.input_layer_norm(out)
        out = self.kb_net(out)
        # split into heads
        out = out.view(bsz, self.num_heads, self.head_dim + 1)
        ks, lrs = torch.split(out, (self.head_dim,) + (1,), -1)

        # reshape to merge batch and head dims
        ks = ks.reshape(bsz * self.num_heads, self.head_dim)
        lrs = lrs.reshape(bsz * self.num_heads, 1)

        # get values from dx_dt
        vs = self.vs_proj(dx_dt)
        if self.use_v_laynorm:
            vs = self.v_net(self.v_layer_norm(vs))
        vs = vs.reshape(bsz * self.num_heads, self.head_dim)

        # activations
        lrs = torch.sigmoid(lrs)
        ks = torch.softmax(ks, dim=-1)
        vs = torch.tanh(vs)

        if W is not None:
            # Delta rule:
            W_t = W.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
            W_t = W_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)
            vs_remove = torch.bmm(W_t, ks.unsqueeze(2)).squeeze()
            vs = lrs * (vs - vs_remove)
        else:
            vs = lrs * vs

        # weight update term 'bi, bj->bij'
        out = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
        # (B * heads, head_dim, head_dim)

        # reshape to vector
        out = out.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        out = out.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return out


class DxOnlyCDEOjaUpdateVectorField(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads,
                 use_v_laynorm=False):
        super(DxOnlyCDEOjaUpdateVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        # self.model_dim = hidden_channels

        self.input_proj = nn.Linear(input_channels, hidden_channels)
        self.input_layer_norm = nn.LayerNorm(hidden_channels)

        # projection matrix for dx_dt
        self.dx_dt_proj = nn.Linear(input_channels, hidden_channels)
        self.use_v_laynorm = use_v_laynorm
        if use_v_laynorm:
            self.v_layer_norm = nn.LayerNorm(hidden_channels)
            self.v_net = nn.Linear(hidden_channels, hidden_channels)

        # slow net producing key and value vectors and a learning rate
        self.kb_net = nn.Linear(
            hidden_channels, num_heads * (self.head_dim + 1), bias=True)

    # forward output the update term
    def forward(self, x, dx_dt, W=None):
        bsz = dx_dt.shape[0]

        out = self.input_proj(dx_dt)
        out = self.input_layer_norm(out)
        out = self.kb_net(out)
        # split into heads
        out = out.view(bsz, self.num_heads, self.head_dim + 1)
        ks, lrs = torch.split(out, (self.head_dim,) + (1,), -1)

        # reshape to merge batch and head dims
        ks = ks.reshape(bsz * self.num_heads, self.head_dim)
        lrs = lrs.reshape(bsz * self.num_heads, 1)

        # get values from dx_dt
        vs = self.dx_dt_proj(dx_dt)
        if self.use_v_laynorm:
            vs = self.v_net(self.v_layer_norm(vs))
        vs = vs.reshape(bsz * self.num_heads, self.head_dim)

        # activations
        lrs = torch.sigmoid(lrs)
        vs = torch.tanh(vs)

        if W is not None:
            # Oja:
            W_t = W.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
            W_t = W_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)
            ks_remove = torch.bmm(W_t.transpose(1, 2), vs.unsqueeze(2)).squeeze()
            ks = ks - ks_remove
        ks = torch.softmax(ks, dim=-1) * lrs

        # weight update term 'bi, bj->bij'
        out = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
        # (B * heads, head_dim, head_dim)

        # reshape to vector
        out = out.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        out = out.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return out


class DxOnlyCDEDeltaUpdateVectorField(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads,
                 use_v_laynorm=False, post_tahn=False):
        super(DxOnlyCDEDeltaUpdateVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.post_tahn = post_tahn

        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.input_proj = nn.Linear(input_channels, hidden_channels)
        self.input_layer_norm = nn.LayerNorm(hidden_channels)

        # projection matrix for dx_dt
        self.vs_proj = nn.Linear(input_channels, hidden_channels)
        self.use_v_laynorm = use_v_laynorm
        if use_v_laynorm:
            self.v_layer_norm = nn.LayerNorm(hidden_channels)
            self.v_net = nn.Linear(hidden_channels, hidden_channels)

        # slow net producing key and value vectors and a learning rate
        self.kb_net = nn.Linear(
            hidden_channels, num_heads * (self.head_dim + 1), bias=True)

    # forward output the update term
    def forward(self, x, dx_dt, W=None):
        bsz = dx_dt.shape[0]

        out = self.input_proj(dx_dt)
        out = self.input_layer_norm(out)
        out = self.kb_net(out)
        # split into heads
        out = out.view(bsz, self.num_heads, self.head_dim + 1)
        ks, lrs = torch.split(out, (self.head_dim,) + (1,), -1)

        # reshape to merge batch and head dims
        ks = ks.reshape(bsz * self.num_heads, self.head_dim)
        lrs = lrs.reshape(bsz * self.num_heads, 1)

        # get values from dx_dt
        vs = self.vs_proj(dx_dt)
        if self.use_v_laynorm:
            vs = self.v_net(self.v_layer_norm(vs))
        vs = vs.reshape(bsz * self.num_heads, self.head_dim)

        # activations
        lrs = torch.sigmoid(lrs)
        ks = torch.softmax(ks, dim=-1)
        if not self.post_tahn:
            vs = torch.tanh(vs)

        if W is not None:
            # Delta rule:
            W_t = W.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
            W_t = W_t.reshape(bsz * self.num_heads, self.head_dim, self.head_dim)
            vs_remove = torch.bmm(W_t, ks.unsqueeze(2)).squeeze()
            vs = vs - vs_remove

        if self.post_tahn:
            vs = torch.tanh(vs)

        vs = lrs * vs

        # weight update term 'bi, bj->bij'
        out = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
        # (B * heads, head_dim, head_dim)

        # reshape to vector
        out = out.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        out = out.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return out


###############################################################################


# Just produce the entire map from flattened weight matrix
class CDEDirectMapVectorField(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 hidden_hidden_channels=None, num_hidden_layers=1):
        super(CDEDirectMapVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        module_list = []

        if num_hidden_layers == 1:
            module_list.append(nn.Linear(
                hidden_channels * input_channels,
                hidden_channels * input_channels * input_channels))
        else:
            assert hidden_hidden_channels is not None, (
                "set `hidden_hidden_channels` when `num_hidden_layers` > 1")
            module_list.append(
                torch.nn.Linear(hidden_channels * input_channels,
                                hidden_hidden_channels))
            if num_hidden_layers > 2:
                module_list += [torch.nn.Linear(hidden_hidden_channels,
                                                hidden_hidden_channels)
                                for _ in range(num_hidden_layers - 2)]
            module_list.append(
                torch.nn.Linear(
                    hidden_hidden_channels,
                    hidden_channels * input_channels * input_channels))
        assert len(module_list) == self.num_hidden_layers
        self.big_net_layers = torch.nn.ModuleList(module_list)

    # forward output the update term
    def forward(self, x, dx_dt, W=None):  # x will not be used
        bsz = x.shape[0]

        # generate the map from W (B, dim)
        big_map = W
        for i in range(self.num_hidden_layers):
            big_map = self.big_net_layers[i](big_map)
            if i != self.num_hidden_layers - 1:
                big_map = F.relu(big_map)
        big_map = F.tanh(big_map)
        big_map = big_map.reshape(
            bsz, self.hidden_channels * self.input_channels,
            self.input_channels)
        
        out = torch.bmm(big_map, dx_dt.unsqueeze(2)).squeeze()

        return out