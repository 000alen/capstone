import torch

from .model import ClientFront, ServerCore, ClientBack
from .utils import l2_normalize
from .tests import empirical_ind_cpa


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, vocab = 2, 8, 100
    d, m, k = 128, 128, 8

    front = ClientFront(vocab, d, m, k)
    server = ServerCore(d + m, k, layers=2)
    back = ClientBack(vocab, d, m, k)

    # sanity pass (forward‑backward equality)
    tokens = torch.randint(0, vocab, (B, T))
    enc, K = front(tokens)
    out_server = server(enc)
    logits_secure = back(out_server, K)

    # Plaintext baseline (noise zeros, identity rotation)
    zeros = torch.zeros(B, T, k, m)
    plain = torch.cat([l2_normalize(front.embed[tokens]), zeros], dim=-1)
    logits_plain = back(server(plain), torch.eye(d + m))
    print("max |secure-plain| logits diff:", (logits_secure - logits_plain).abs().max().item())

    # empirical IND‑CPA game
    empirical_ind_cpa(front, server, back, trials=500)
