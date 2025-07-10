import torch
import pytest

from secure_transformer.model import (
    ClientFront,
    ServerCore,
    ClientBack,
    IGatedNonlinear,
    ETokenAttention,
    LieResidual,
    EBlock,
)
from secure_transformer.utils import l2_normalize, random_orthogonal


def rand_tensor(*shape, device="cpu", dtype=torch.float32):
    return torch.randn(shape, device=device, dtype=dtype)


def equivariant_forward(layer, x, atol=1e-4):
    """Test forward equivariance: layer(K @ x) ≈ K @ layer(x)"""
    *batch, n = x.shape
    K = random_orthogonal(n, device=x.device, dtype=x.dtype)
    lhs = layer(torch.einsum("ij,...j->...i", K, x))
    rhs = torch.einsum("ij,...j->...i", K, layer(x))

    error = (lhs - rhs).abs().max()
    if error > atol:
        print(f"Error: {error}")
        print(f"LHS: {lhs.mean()=}, {lhs.std()=}")
        print(f"RHS: {rhs.mean()=}, {rhs.std()=}")
        print(f"X: {x.mean()=}, {x.std()=}")
        print(f"K: {K.mean()=}, {K.std()=}")

    return torch.allclose(lhs, rhs, atol=atol, equal_nan=True)


# NOTE(@000alen): tolerances are maybe too high, but I'm not sure how to make them tighter.
def equivariant_backward(layer, x, atol=1e-4):
    """Test backward equivariance: gradients should also be equivariant"""
    x = x.clone().requires_grad_()
    *batch, n = x.shape
    K = random_orthogonal(n, device=x.device, dtype=x.dtype)
    y1 = layer(torch.einsum("ij,...j->...i", K, x)).sum()
    y2 = torch.einsum("ij,...j->...i", K, layer(x)).sum()
    (y1 - y2).abs().backward()

    error = (x.grad - torch.zeros_like(x.grad)).abs().max()
    if error > atol:
        print(f"Error: {error}")
        print(f"X.grad: {x.grad.mean()=}, {x.grad.std()=}")
        print(f"K: {K.mean()=}, {K.std()=}")

    return torch.allclose(x.grad, torch.zeros_like(x.grad), atol=atol, equal_nan=True)


# Test parameters
Ns = [32, 64, 128]
ks = [4, 8, 16]
heads_options = [2, 4, 8]
rank_options = [4, 8, 16]
hidden_options = [8, 16, 32]


@pytest.mark.parametrize("n,rank", [(n, r) for n in Ns for r in rank_options])
def test_lie_residual_equivariance(n, rank):
    """Test SO(n) equivariance of LieResidual layer"""
    B, C = 3, 5
    x = rand_tensor(B, C, n)
    layer = LieResidual(n, rank)

    # Test forward equivariance
    assert equivariant_forward(
        layer, x
    ), f"Forward equivariance failed for n={n}, rank={rank}"

    # Test backward equivariance
    assert equivariant_backward(
        layer, x
    ), f"Backward equivariance failed for n={n}, rank={rank}"


@pytest.mark.parametrize("n,hidden", [(n, h) for n in Ns for h in hidden_options])
def test_igated_nonlinear_equivariance(n, hidden):
    """Test SO(n) equivariance of IGatedNonlinear layer"""
    B, C = 2, 4
    x = rand_tensor(B, C, n)
    layer = IGatedNonlinear(hidden)

    # Test forward equivariance
    assert equivariant_forward(
        layer, x
    ), f"Forward equivariance failed for n={n}, hidden={hidden}"

    # Test backward equivariance
    assert equivariant_backward(
        layer, x
    ), f"Backward equivariance failed for n={n}, hidden={hidden}"


@pytest.mark.parametrize(
    "n,k,h", [(n, k, h) for n in Ns for k in ks for h in heads_options if k % h == 0]
)
def test_etoken_attention_equivariance(n, k, h):
    """Test SO(n) equivariance of ETokenAttention layer"""
    B, T = 2, 6
    x = rand_tensor(B, T, k, n)
    layer = ETokenAttention(n, k, h)

    # Test forward equivariance
    assert equivariant_forward(
        layer, x
    ), f"Forward equivariance failed for n={n}, k={k}, h={h}"

    # Test backward equivariance
    assert equivariant_backward(
        layer, x
    ), f"Backward equivariance failed for n={n}, k={k}, h={h}"


@pytest.mark.parametrize(
    "n,k,h,rank",
    [
        (n, k, h, r)
        for n in [32, 64]
        for k in [4, 8]
        for h in [2, 4]
        for r in [4, 8]
        if k % h == 0
    ],
)
def test_eblock_equivariance(n, k, h, rank):
    """Test SO(n) equivariance of EBlock (combination of attention + residual + gate)"""
    B, T = 2, 5
    x = rand_tensor(B, T, k, n)
    layer = EBlock(n, k, h, rank)

    # Test forward equivariance
    assert equivariant_forward(
        layer, x
    ), f"Forward equivariance failed for n={n}, k={k}, h={h}, rank={rank}"

    # Test backward equivariance
    assert equivariant_backward(
        layer, x
    ), f"Backward equivariance failed for n={n}, k={k}, h={h}, rank={rank}"


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("n,k", [(n, k) for n in [32, 64] for k in [4, 8]])
def test_server_core_equivariance(n, k, depth):
    """Test SO(n) equivariance of ServerCore (stack of EBlocks)"""
    B, T = 2, 5
    x = rand_tensor(B, T, k, n)
    layer = ServerCore(n, k, layers=depth)

    # Test forward equivariance
    assert equivariant_forward(
        layer, x
    ), f"Forward equivariance failed for n={n}, k={k}, depth={depth}"

    # Test backward equivariance
    assert equivariant_backward(
        layer, x
    ), f"Backward equivariance failed for n={n}, k={k}, depth={depth}"


def test_equivariance_with_different_batch_sizes():
    """Test equivariance holds across different batch sizes"""
    n, k, h, rank = 32, 8, 4, 4

    layers = [
        IGatedNonlinear(),
        LieResidual(n, rank),
        ETokenAttention(n, k, h),
        EBlock(n, k, h, rank),
    ]

    batch_sizes = [1, 3, 7]
    seq_lens = [1, 4, 10]

    for layer in layers:
        for B in batch_sizes:
            if isinstance(layer, (ETokenAttention, EBlock)):
                for T in seq_lens:
                    x = rand_tensor(B, T, k, n)
                    assert equivariant_forward(
                        layer, x
                    ), f"Batch size test failed for {type(layer).__name__} with B={B}, T={T}"
            else:
                x = rand_tensor(
                    B, 3, n
                )  # Simple case for IGatedNonlinear and LieResidual
                assert equivariant_forward(
                    layer, x
                ), f"Batch size test failed for {type(layer).__name__} with B={B}"


def test_equivariance_with_extreme_values():
    """Test equivariance with extreme input values"""
    n, k, h, rank = 32, 8, 4, 4
    B, T = 2, 3

    layers = [
        IGatedNonlinear(),
        LieResidual(n, rank),
        ETokenAttention(n, k, h),
        EBlock(n, k, h, rank),
    ]

    # Test with very small values
    for layer in layers:
        if isinstance(layer, (ETokenAttention, EBlock)):
            x_small = torch.full((B, T, k, n), 1e-6)
            assert equivariant_forward(
                layer, x_small, atol=1e-5
            ), f"Small values test failed for {type(layer).__name__}"
        else:
            x_small = torch.full((B, T, n), 1e-6)
            assert equivariant_forward(
                layer, x_small, atol=1e-5
            ), f"Small values test failed for {type(layer).__name__}"

    # Test with large values
    for layer in layers:
        if isinstance(layer, (ETokenAttention, EBlock)):
            x_large = torch.full((B, T, k, n), 10.0)
            assert equivariant_forward(
                layer, x_large, atol=1e-4
            ), f"Large values test failed for {type(layer).__name__}"
        else:
            x_large = torch.full((B, T, n), 10.0)
            assert equivariant_forward(
                layer, x_large, atol=1e-4
            ), f"Large values test failed for {type(layer).__name__}"


@pytest.mark.parametrize("d,m", [(32, 16), (64, 32), (64, 64)])
def test_round_trip_equivariance(d, m):
    """Test end-to-end equivariance through client-server-client pipeline"""
    B, T, k = 3, 7, 8
    vocab = 120

    front = ClientFront(vocab, d, m, k)
    server = ServerCore(d + m, k, layers=2)
    back = ClientBack(vocab, d, m, k)

    tokens = torch.randint(0, vocab, (B, T))
    
    # Secure path: embed + noise + rotation
    C, K = front(tokens)
    logits_secure = back(server(C), K)

    # Proper equivariance test: use same noise pattern without rotation
    # Extract the embeddings and noise that were actually used
    E = l2_normalize(front.embed[tokens])  # (B,T,k,d)
    
    # Decrypt to get the original V = [signal, noise] that was rotated
    V = torch.einsum("ij,btkj->btki", K.t(), C)
    
    # Plain path: same V but no rotation (identity transformation)
    logits_plain = back(server(V), torch.eye(d + m))

    error = (logits_secure - logits_plain).abs().max()
    
    # Use slightly higher tolerance for larger matrices due to accumulated floating point errors
    tolerance = 2e-4 if (d + m) >= 128 else 1e-4
    
    if error > tolerance:
        print(f"Error: {error}")
        print(f"Logits secure: {logits_secure.mean()=}, {logits_secure.std()=}")
        print(f"Logits plain: {logits_plain.mean()=}, {logits_plain.std()=}")
        print(f"C: {C.mean()=}, {C.std()=}")
        print(f"K: {K.mean()=}, {K.std()=}")

    assert torch.allclose(
        logits_secure, logits_plain, atol=tolerance
    ), f"Round trip test failed for d={d}, m={m} with error {error:.6f}"


def test_equivariance_invariance_under_composition():
    """Test that composition of equivariant layers is still equivariant"""
    n, k, h, rank = 32, 8, 4, 4
    B, T = 2, 4

    # Test composition of individual layers
    layer1 = LieResidual(n, rank)
    layer2 = IGatedNonlinear()
    layer3 = LieResidual(n, rank // 2)

    def composite_layer(x):
        return layer3(layer2(layer1(x)))

    x = rand_tensor(B, T, n)
    assert equivariant_forward(
        composite_layer, x
    ), "Composition of equivariant layers failed"

    # Test EBlock which is internally a composition
    eblock = EBlock(n, k, h, rank)
    x_block = rand_tensor(B, T, k, n)
    assert equivariant_forward(eblock, x_block), "EBlock composition test failed"
