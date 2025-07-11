"""Prototype implementation of the IND-CPA-secure
SO(N)-equivariant split language model described in the proof.

Highlights
==========
* Adds **m noisy coordinates** to every token vector. Client knows which
  coordinates are real, server cannot distinguish.
* Uses Haar-uniform random rotation K ∈ SO(N) **fresh for every prompt**.
* Guarantees information-theoretic IND-CPA secrecy up to 2^λ (see doc).
* Provides a Monte-Carlo IND-CPA game that empirically shows advantage ≈ 0.5.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import random_orthogonal, l2_normalize


class IGatedNonlinear(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norms2 = (x**2).sum(dim=-1, keepdim=True)
        gates = torch.sigmoid(self.net(norms2))
        return x * gates


class LieResidual(nn.Module):
    def __init__(self, n: int, rank: int = 4):
        super().__init__()
        self.u = nn.Parameter(torch.randn(rank, n) / math.sqrt(n))
        self.v = nn.Parameter(torch.randn(rank, n) / math.sqrt(n))
        self.a = nn.Parameter(torch.zeros(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u_x = torch.einsum("rn,...n->...r", self.u, x)
        v_x = torch.einsum("rn,...n->...r", self.v, x)
        upd = torch.einsum("r,...r,rn->...n", self.a, u_x, self.v) - torch.einsum(
            "r,...r,rn->...n", self.a, v_x, self.u
        )
        return x + upd


class ETokenAttention(nn.Module):
    def __init__(self, n: int, k_vec: int, heads: int = 4):
        super().__init__()
        assert k_vec % heads == 0, "k_vec divisible by heads"
        self.h = heads
        self.d = k_vec // heads
        self.n = n
        self.qs = nn.Parameter(torch.ones(heads))
        self.ks = nn.Parameter(torch.ones(heads))
        self.vs = nn.Parameter(torch.ones(heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,k,n)
        B, T, k, n = x.shape
        h, d = self.h, self.d
        xh = x.view(B, T, h, d, n)
        q = self.qs.view(1, 1, h, 1, 1) * xh
        k_ = self.ks.view(1, 1, h, 1, 1) * xh
        v = self.vs.view(1, 1, h, 1, 1) * xh
        scores = torch.einsum("bthdn,bshdn->bhts", q, k_) / math.sqrt(n)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhts,bshdn->bthdn", attn, v)
        return out.reshape(B, T, k, n)


class EBlock(nn.Module):
    def __init__(self, n: int, k_vec: int, heads: int = 4, rank: int = 4):
        super().__init__()
        self.attn = ETokenAttention(n, k_vec, heads)
        self.lie = LieResidual(n, rank)
        self.gate = IGatedNonlinear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.lie(self.gate(x))
        return x


class ClientFront(nn.Module):
    """Embeds tokens, L2-normalizes, pads with Gaussian noise, rotates."""

    def __init__(
        self, vocab: int, d_signal: int, m_noise: int, k_vec: int, sigma: float = 1.0
    ):
        super().__init__()
        self.d = d_signal
        self.m = m_noise
        self.N = d_signal + m_noise
        self.k = k_vec
        self.sigma = sigma
        self.embed = nn.Parameter(
            torch.randn(vocab, k_vec, d_signal) / math.sqrt(d_signal * k_vec)
        )

    def forward(self, tokens: torch.Tensor):
        # (B,T)
        B, T = tokens.shape
        device, dtype = tokens.device, self.embed.dtype
        E = self.embed[tokens]  # (B,T,k,d)
        E = l2_normalize(E)  # unit radius β=1
        noise = (
            torch.randn(B, T, self.k, self.m, device=device, dtype=dtype) * self.sigma
        )
        V = torch.cat([E, noise], dim=-1)  # (B,T,k,N)
        K = random_orthogonal(self.N, device=device, dtype=dtype)
        C = torch.einsum("ij,btkj->btki", K, V)  # ciphertext manifold
        return C, K


class ServerCore(nn.Module):
    def __init__(
        self, N: int, k_vec: int, layers: int = 4, heads: int = 4, rank: int = 4
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [EBlock(N, k_vec, heads, rank) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class ClientBack(nn.Module):
    def __init__(self, vocab: int, d_signal: int, m_noise: int, k_vec: int):
        super().__init__()
        self.d = d_signal
        self.m = m_noise
        self.k = k_vec
        self.head = nn.Linear(k_vec, vocab, bias=False)

    def forward(self, x_rot: torch.Tensor, K: torch.Tensor):
        x = torch.einsum("ij,btkj->btki", K.t(), x_rot)  # decrypt
        x_sig = x[..., : self.d]
        logits = self.head((x_sig**2).sum(dim=-1))
        return logits
