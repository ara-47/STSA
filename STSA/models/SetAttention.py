import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SetAttention(nn.Module):
    def __init__(self, num_sets, head_dim, dim, iters = 5, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_sets = num_sets
        self.iters = iters
        self.eps = eps
        self.scale = head_dim ** -0.5

        self.sets_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.sets_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)



        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_sets  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_sets = None):
        b, n, d = inputs.shape
        n_s = num_sets if num_sets is not None else self.num_sets
        
        mu = self.sets_mu.expand(b, n_s, -1)
        sigma = self.sets_sigma.expand(b, n_s, -1)
        sets = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            sets_prev = sets

            sets = self.norm_sets(sets)
            q = self.to_q(sets)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            sets = self.gru(
                updates.reshape(-1, d),
                sets_prev.reshape(-1, d)
            )



            sets = sets.reshape(b, -1, d)
            sets = sets + self.fc2(F.relu(self.fc1(self.norm_pre_ff(sets))))


        return sets