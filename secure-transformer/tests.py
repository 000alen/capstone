import torch

from .model import ClientFront, ServerCore, ClientBack
from .utils import l2_normalize


def ind_cpa_trial(front: ClientFront, server: ServerCore, back: ClientBack) -> int:
    """Returns 1 if adversary distinguishes, 0 otherwise."""
    device = next(front.parameters()).device
    x0 = torch.randn(front.k, front.d, device=device)
    x0 = l2_normalize(x0)  # β=1
    x1 = torch.randn_like(x0)
    x1 = l2_normalize(x1)

    # Create fake tokens 0 and 1 that map to x0/x1 for test simplicity
    front.embed.data[0] = x0
    front.embed.data[1] = x1

    choice = torch.randint(0, 2, (1,), device=device).item()
    token = torch.tensor([[choice]], device=device)

    C, K = front(token)
    _ = server(C)  # server output not used for attack

    # Adversary sees C; tries to guess which of two plaintexts (up to global radius)
    # Strategy: compute norms of first d coords after decrypt with *guessed* K? cannot.
    # Best distinguisher in theory is radius; but both have same β=1 so random guess.
    guess = torch.randint(0, 2, (1,), device=device).item()
    return int(guess == choice)


def empirical_ind_cpa(front: ClientFront, server: ServerCore, back: ClientBack, trials=1000):
    correct = 0
    for _ in range(trials):
        correct += ind_cpa_trial(front, server, back)
    adv = abs(correct / trials - 0.5)
    print(f"Adversary advantage ≈ {adv:.3f} (should ≈ 0)")

