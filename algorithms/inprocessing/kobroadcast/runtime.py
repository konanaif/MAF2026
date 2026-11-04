"""Seed and sequence-padding helpers for broadcast classification."""
import random
import numpy as np
import torch

def set_random_seed(random_seed) :
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def pad_seqs(seqs, pad_val, max_length=256):
    _max_length = max([len(s) for s in seqs])
    max_length = _max_length if max_length is None else min(_max_length, max_length)

    padded_seqs = []
    for seq in seqs:
        seq = seq[:max_length]
        pads = [pad_val] * (max_length - len(seq))
        padded_seqs.append(seq + pads)

    return torch.tensor(padded_seqs)
