import torch
import pytest

from secure_transformer.model import ClientFront, ServerCore, ClientBack, IGatedNonlinear, ETokenAttention, LieResidual
from secure_transformer.utils import l2_normalize, random_orthogonal


def rand_tensor(*shape, device="cpu", dtype=torch.float32):
    return torch.randn(shape, device=device, dtype=dtype)


def equivariant_forward(layer, x, atol=1e-6):
    *batch, n = x.shape
    K = random_orthogonal(n, device=x.device, dtype=x.dtype)
    lhs = layer(torch.einsum("ij,...j->...i", K, x))
    rhs = torch.einsum("ij,...j->...i", K, layer(x))
    return torch.allclose(lhs, rhs, atol=atol)


def equivariant_backward(layer, x, atol=1e-5):
    x = x.clone().requires_grad_()
    *batch, n = x.shape
    K = random_orthogonal(n, device=x.device, dtype=x.dtype)
    y1 = layer(torch.einsum("ij,...j->...i", K, x)).sum()
    y2 = torch.einsum("ij,...j->...i", K, layer(x)).sum()
    (y1 - y2).abs().backward()
    return torch.allclose(x.grad, torch.zeros_like(x.grad), atol=atol)


Ns = [32, 64]
ks = [4, 8]
heads_options = [2, 4]
rank_options = [4, 8]


@pytest.mark.parametrize("n,rank", [(n, r) for n in Ns for r in rank_options])
def test_lie_residual_equivariance(n, rank):
    B, C = 3, 5
    x = rand_tensor(B, C, n)
    layer = LieResidual(n, rank)
    assert equivariant_forward(layer, x)
    assert equivariant_backward(layer, x)


@pytest.mark.parametrize("n", Ns)
def test_gate_equivariance(n):
    x = rand_tensor(7, 3, n)
    gate = IGatedNonlinear()
    assert equivariant_forward(gate, x)
    assert equivariant_backward(gate, x)


@pytest.mark.parametrize("n,k,h", [(n, k, h) for n in Ns for k in ks for h in heads_options if k % h == 0])
def test_token_attention_equivariance(n, k, h):
    x = rand_tensor(2, 6, k, n)
    attn = ETokenAttention(n, k, h)
    assert equivariant_forward(attn, x)


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("n,k", [(n, k) for n in Ns for k in ks])
def test_server_core_equivariance(n, k, depth):
    x = rand_tensor(2, 5, k, n)
    core = ServerCore(n, k, layers=depth)
    *_, N = x.shape
    K = random_orthogonal(N)
    y_enc = core(torch.einsum("ij,btkj->btki", K, x))
    y_plain = torch.einsum("ij,btkj->btki", K, core(x))
    assert torch.allclose(y_enc, y_plain, atol=1e-5)


@pytest.mark.parametrize("d,m", [(32, 16), (64, 64)])
def test_round_trip(d, m):
    B, T, k = 3, 7, 8
    front = ClientFront(120, d, m, k)
    server = ServerCore(d + m, k, layers=2)
    back = ClientBack(120, d, m, k)

    tokens = torch.randint(0, 120, (B, T))
    C, K = front(tokens)
    logits_secure = back(server(C), K)

    zeros = torch.zeros(B, T, k, m)
    plain = torch.cat([l2_normalize(front.embed[tokens]), zeros], dim=-1)
    logits_plain = back(server(plain), torch.eye(d + m))

    assert torch.allclose(logits_secure, logits_plain, atol=1e-5)
