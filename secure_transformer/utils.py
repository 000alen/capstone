import torch


def random_orthogonal(n: int, *, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Sample Haar(SO(n)) via QR of a Gaussian matrix."""
    A = torch.randn(n, n, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    # force det +1
    if torch.det(Q) < 0:
        Q[0].neg_()
    return Q


def l2_normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)
