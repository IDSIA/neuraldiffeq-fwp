# Learning rules

import torch
import torch.nn as nn


class RecurrentHebbUpdateRule(nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads):
        super(RecurrentHebbUpdateRule, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        assert latent_dim % num_heads == 0
        self.head_dim = latent_dim // num_heads

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.input_layer_norm = nn.LayerNorm(latent_dim)

        self.merge_layer = nn.Linear(latent_dim * 2, latent_dim)

        self.kvb_net = nn.Linear(
            latent_dim, num_heads * (2 * self.head_dim + 1), bias=True)

    def forward(self, x, z, W):
        """
        Update the hidden state W with the new input x (state action embedding)

        W is the flattened fast WM.
        """
        bsz = x.shape[0]
        z = z.tanh()

        out = self.input_proj(x)
        out = self.input_layer_norm(out)
        out = self.merge_layer(torch.cat([out, z], dim=-1))
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

        return out + W


class RecurrentOjaUpdateRule(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads):
        super(RecurrentOjaUpdateRule, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        assert latent_dim % num_heads == 0
        self.head_dim = latent_dim // num_heads

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.input_layer_norm = nn.LayerNorm(latent_dim)

        self.merge_layer = nn.Linear(latent_dim * 2, latent_dim)

        self.kvb_net = nn.Linear(
            latent_dim, num_heads * (2 * self.head_dim + 1), bias=True)

    # forward output the update term
    def forward(self, x, z, W):
        # x: (B, dim)
        # W: (B, flattened_dim)
        bsz = x.shape[0]

        out = self.input_proj(x)
        out = self.input_layer_norm(out)
        out = self.merge_layer(torch.cat([out, z], dim=-1))
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
            W_t = W_t.reshape(
                bsz * self.num_heads, self.head_dim, self.head_dim)
            ks_remove = torch.bmm(
                W_t.transpose(1, 2), vs.unsqueeze(2)).squeeze()
            ks = ks - ks_remove
        ks = torch.softmax(ks, dim=-1) * lrs

        # weight update term 'bi, bj->bij'
        out = torch.bmm(vs.unsqueeze(2), ks.unsqueeze(1))
        # (B * heads, head_dim, head_dim)

        # reshape to vector
        out = out.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
        out = out.reshape(bsz, -1)  # (B, heads * head_dim * head_dim)

        return out + W


class RecurrentDeltaUpdateRule(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads, post_tahn=True):
        super(RecurrentDeltaUpdateRule, self).__init__()

        self.post_tahn = post_tahn
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        assert latent_dim % num_heads == 0
        self.head_dim = latent_dim // num_heads

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.input_layer_norm = nn.LayerNorm(latent_dim)

        self.merge_layer = nn.Linear(latent_dim * 2, latent_dim)

        self.kvb_net = nn.Linear(
            latent_dim, num_heads * (2 * self.head_dim + 1), bias=True)

    # forward output the update term
    def forward(self, x, z, W):
        # x: (B, dim)
        # W: (B, flattened_dim)
        bsz = x.shape[0]

        out = self.input_proj(x)
        out = self.input_layer_norm(out)
        out = self.merge_layer(torch.cat([out, z], dim=-1))
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
            W_t = W_t.reshape(
                bsz * self.num_heads, self.head_dim, self.head_dim)
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

        return out + W
