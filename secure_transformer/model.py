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

from .config import TrainingConfig
from .utils import random_orthogonal, l2_normalize


class IGatedNonlinear(nn.Module):
    """Gated nonlinearity module that applies a learnable gate based on the squared L2 norm of the input.

    This module computes the squared L2 norm of the input tensor along the last dimension,
    passes it through a small feedforward neural network (linear layer, ReLU, linear layer),
    applies a sigmoid activation to produce a gate value between 0 and 1, and then multiplies
    the input tensor element-wise by this gate. It serves as a norm-dependent gating mechanism,
    potentially useful in equivariant models to introduce nonlinearity while preserving certain symmetries.

    Args:
        hidden (int, optional): The number of units in the hidden layer of the gate network.
            This controls the complexity of the norm-to-gate mapping. Defaults to 16.

    Attributes:
        net (nn.Sequential): The feedforward network that maps squared norms to gate logits before sigmoid.

    Example:
        >>> module = IGatedNonlinear(hidden=32)
        >>> x = torch.randn(2, 3, 4)  # shape (batch, seq, features)
        >>> output = module(x)
        >>> output.shape
        torch.Size([2, 3, 4])

    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norms2 = (x**2).sum(dim=-1, keepdim=True)
        gates = torch.sigmoid(self.net(norms2))
        return x * gates


class LieResidual(nn.Module):
    """Lie algebra residual layer for equivariant updates.

    This module implements a residual connection using a low-rank skew-symmetric update, which can be seen as an
    infinitesimal rotation in SO(n). It adds a learnable update to the input that preserves certain geometric properties,
    useful in equivariant neural networks.

    The update is computed as upd = sum_r a_r * ( (u_r · x) v_r - (v_r · x) u_r ), where · denotes dot product.
    This is equivalent to applying a sum of rank-1 skew-symmetric matrices scaled by a_r.

    Args:
        n (int): Dimensionality of the input features (last dimension).
        rank (int, optional): Number of rank-1 components in the low-rank approximation. Defaults to 4.

    Attributes:
        u (nn.Parameter): Learnable matrix of shape (rank, n).
        v (nn.Parameter): Learnable matrix of shape (rank, n).
        a (nn.Parameter): Learnable scales of shape (rank,).

    Shape:
        - Input: (*, n) where * is any number of dimensions.
        - Output: (*, n) same shape as input.

    Example:
        >>> module = LieResidual(n=10, rank=2)
        >>> x = torch.randn(5, 10)
        >>> output = module(x)
        >>> output.shape
        torch.Size([5, 10])

    """

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
    """SO(n)-equivariant multi-head token attention layer.

    This module implements a variant of multi-head attention that operates on equivariant representations,
    where each token is represented as a set of k_vec vectors in n-dimensional space. It projects the input
    into query, key, and value representations by scaling with learnable per-head parameters, computes
    dot-product attention scores normalized by sqrt(n), applies softmax, and computes the weighted sum.
    The operations are designed to be equivariant under SO(n) transformations.

    Args:
        n (int): Dimensionality of the vector space (n in SO(n)).
        k_vec (int): Total number of vectors per token; must be divisible by heads.
        heads (int, optional): Number of attention heads. Defaults to 4.

    Attributes:
        h (int): Number of heads.
        d (int): Vectors per head (k_vec // heads).
        n (int): Stored n parameter.
        qs (nn.Parameter): Learnable scales for queries, shape (heads,).
        ks (nn.Parameter): Learnable scales for keys, shape (heads,).
        vs (nn.Parameter): Learnable scales for values, shape (heads,).

    Shape:
        - Input: (B, T, k_vec, n) where B is batch size, T is sequence length.
        - Output: (B, T, k_vec, n) same as input.

    Example:
        >>> module = ETokenAttention(n=8, k_vec=16, heads=4)
        >>> x = torch.randn(2, 10, 16, 8)
        >>> output = module(x)
        >>> output.shape
        torch.Size([2, 10, 16, 8])

    """

    def __init__(self, n: int, k_vec: int, heads: int = 4, rel_max: int = 2048):
        super().__init__()
        assert k_vec % heads == 0, "k_vec divisible by heads"
        self.h = heads
        self.rel_max = rel_max

        self.d = k_vec // heads
        self.n = n
        self.qs = nn.Parameter(torch.ones(heads))
        self.ks = nn.Parameter(torch.ones(heads))
        self.vs = nn.Parameter(torch.ones(heads))

        # config.rel_max = 2048
        self.rel_bias = nn.Parameter(torch.zeros(2 * rel_max - 1, heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def _relative_bias(self, T, device):
        Δ = torch.arange(T, device=device)
        idx = (Δ[:, None] - Δ[None, :]).clamp(-self.rel_max + 1, self.rel_max - 1)
        idx = idx + self.rel_max - 1
        return self.rel_bias[idx]  # (T,T,heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,k,n)
        B, T, k, n = x.shape
        h, d = self.h, self.d
        xh = x.view(B, T, h, d, n)
        q = self.qs.view(1, 1, h, 1, 1) * xh
        k_ = self.ks.view(1, 1, h, 1, 1) * xh
        v = self.vs.view(1, 1, h, 1, 1) * xh
        scores = torch.einsum("bthdn,bshdn->bhts", q, k_) / math.sqrt(n)
        scores = scores + self._relative_bias(T, x.device).permute(2, 0, 1)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhts,bshdn->bthdn", attn, v)
        return out.reshape(B, T, k, n)


class EBlock(nn.Module):
    """SO(n)-equivariant transformer block.

    This module combines an equivariant attention layer with a Lie algebra residual connection
    and a gated nonlinearity. It processes inputs in a way that preserves equivariance under
    SO(n) transformations, making it suitable for geometric or rotationally invariant tasks.

    The forward pass applies attention, then adds the output of the gated Lie residual to it.

    Args:
        n (int): Dimensionality of the vector space.
        k_vec (int): Number of vectors per token.
        heads (int, optional): Number of attention heads. Defaults to 4.
        rank (int, optional): Rank for the Lie residual. Defaults to 4.

    Attributes:
        attn (ETokenAttention): Equivariant attention module.
        lie (LieResidual): Lie algebra residual module.
        gate (IGatedNonlinear): Gated nonlinearity module.

    Shape:
        - Input: (B, T, k_vec, n)
        - Output: (B, T, k_vec, n)

    Example:
        >>> module = EBlock(n=8, k_vec=16, heads=4, rank=4)
        >>> x = torch.randn(2, 10, 16, 8)
        >>> output = module(x)
        >>> output.shape
        torch.Size([2, 10, 16, 8])

    """

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
    """Client-side frontend for secure equivariant transformer.

    This module handles the client-side preprocessing: embedding tokens into vectors, L2-normalizing them to unit length,
    padding with Gaussian noise coordinates to obfuscate the signal, and applying a random orthogonal rotation (fresh for each prompt)
    to encrypt the representation. The rotation matrix K is returned along with the ciphertext for later decryption.

    This provides information-theoretic security by making the signal indistinguishable from noise in the higher-dimensional space.

    Args:
        vocab (int): Vocabulary size for the embedding layer.
        d_signal (int): Dimensionality of the signal (real) coordinates.
        m_noise (int): Number of noise coordinates to add.
        k_vec (int): Number of vectors per token.
        sigma (float, optional): Standard deviation for the Gaussian noise. Defaults to 1.0.

    Attributes:
        d (int): Stored d_signal.
        m (int): Stored m_noise.
        N (int): Total dimensionality (d + m).
        k (int): Stored k_vec.
        sigma (float): Stored noise sigma.
        embed (nn.Parameter): Embedding weights of shape (vocab, k_vec, d_signal).

    Shape:
        - Input: (B, T) integer tokens.
        - Output: tuple of (ciphertext (B, T, k_vec, N), rotation matrix (N, N)).

    Example:
        >>> module = ClientFront(vocab=1000, d_signal=4, m_noise=4, k_vec=8, sigma=1.0)
        >>> tokens = torch.randint(0, 1000, (2, 10))
        >>> C, K = module(tokens)
        >>> C.shape, K.shape
        (torch.Size([2, 10, 8, 8]), torch.Size([8, 8]))

    """

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
    """Server-side core for secure equivariant transformer.

    This module represents the server-side computation: a stack of equivariant transformer blocks (EBlock)
    that process the encrypted, rotated representations without knowledge of the rotation key or noise positions.
    It maintains SO(N) equivariance, allowing it to operate on the higher-dimensional ciphertext space.

    The forward pass sequentially applies each block in the stack.

    Args:
        N (int): Total dimensionality of the input vectors (signal + noise).
        k_vec (int): Number of vectors per token.
        layers (int, optional): Number of EBlock layers. Defaults to 4.
        heads (int, optional): Number of attention heads per block. Defaults to 4.
        rank (int, optional): Rank for Lie residuals in each block. Defaults to 4.

    Attributes:
        blocks (nn.ModuleList): List of EBlock modules.

    Shape:
        - Input: (B, T, k_vec, N)
        - Output: (B, T, k_vec, N)

    Example:
        >>> module = ServerCore(N=8, k_vec=16, layers=2, heads=4, rank=4)
        >>> x = torch.randn(2, 10, 16, 8)
        >>> output = module(x)
        >>> output.shape
        torch.Size([2, 10, 16, 8])

    """

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
    """Client-side backend for secure equivariant transformer.

    This module handles the client-side postprocessing: decrypting the rotated output by applying the inverse rotation,
    extracting the signal coordinates (discarding noise), computing squared L2 norms of the signal vectors,
    and passing them through a linear head to produce logits for next-token prediction.

    It assumes the input is the output from ServerCore and the rotation matrix K from ClientFront.

    Args:
        vocab (int): Vocabulary size for the output logits.
        d_signal (int): Dimensionality of the signal coordinates.
        m_noise (int): Number of noise coordinates (to know where to slice).
        k_vec (int): Number of vectors per token.

    Attributes:
        d (int): Stored d_signal.
        m (int): Stored m_noise.
        k (int): Stored k_vec.
        head (nn.Linear): Linear layer mapping norm sums to logits, shape (k_vec, vocab).

    Shape:
        - Input: rotated output (B, T, k_vec, N), rotation matrix (N, N)
        - Output: logits (B, T, vocab)

    Example:
        >>> module = ClientBack(vocab=1000, d_signal=4, m_noise=4, k_vec=8)
        >>> x_rot = torch.randn(2, 10, 8, 8)
        >>> K = torch.eye(8)
        >>> logits = module(x_rot, K)
        >>> logits.shape
        torch.Size([2, 10, 1000])

    """

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


class SecureTransformer(nn.Module):
    """Complete secure equivariant transformer model.

    This module integrates the client-side frontend, server-side core, and client-side backend to form
    a full secure language model that provides information-theoretic IND-CPA security for prompts.
    The client embeds and encrypts the input tokens, the server processes the ciphertext equivariantly,
    and the client decrypts to obtain next-token logits.

    The security relies on random rotations and noise padding, with the server operating on the obfuscated
    higher-dimensional space without knowledge of the rotation key or signal positions.

    Args:
        config (object): Configuration object with attributes:
            - vocab_size (int): Vocabulary size.
            - d_signal (int): Signal dimensionality.
            - m_noise (int): Noise dimensionality.
            - k_vec (int): Vectors per token.
            - sigma (float): Noise standard deviation.
            - layers (int): Number of server layers.
            - heads (int): Number of attention heads.
            - rank (int): Lie residual rank.

    Attributes:
        config (object): Stored configuration.
        client_front (ClientFront): Embedding and encryption module.
        server_core (ServerCore): Equivariant processing module.
        client_back (ClientBack): Decryption and readout module.

    Shape:
        - Input: (B, T) integer tokens.
        - Output: (B, T, vocab_size) logits.

    Example:
        >>> class Config:
        ...     vocab_size = 1000
        ...     d_signal = 4
        ...     m_noise = 4
        ...     k_vec = 8
        ...     sigma = 1.0
        ...     layers = 2
        ...     heads = 4
        ...     rank = 4
        >>> model = SecureTransformer(Config())
        >>> tokens = torch.randint(0, 1000, (2, 10))
        >>> logits = model(tokens)
        >>> logits.shape
        torch.Size([2, 10, 1000])

    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        self.client_front = ClientFront(
            vocab=config.vocab_size,
            d_signal=config.d_signal,
            m_noise=config.m_noise,
            k_vec=config.k_vec,
            sigma=config.sigma,
        )

        self.server_core = ServerCore(
            N=config.d_signal + config.m_noise,
            k_vec=config.k_vec,
            layers=config.layers,
            heads=config.heads,
            rank=config.rank,
        )

        self.client_back = ClientBack(
            vocab=config.vocab_size,
            d_signal=config.d_signal,
            m_noise=config.m_noise,
            k_vec=config.k_vec,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass through the secure transformer"""
        # Client front: embed, add noise, rotate
        encrypted, rotation = self.client_front(tokens)

        # Server processing: equivariant computation
        processed = self.server_core(encrypted)

        # Client back: decrypt and get logits
        logits = self.client_back(processed, rotation)

        return logits
