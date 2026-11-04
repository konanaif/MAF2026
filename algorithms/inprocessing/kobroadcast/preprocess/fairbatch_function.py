"""CE-risk resampling for overlapping negative-class groups."""
import math
from typing import Iterable, Mapping
import torch

def _labels_and_memberships(labels, memberships):
    if len(labels) != len(memberships) or not len(labels):
        raise ValueError("nonempty labels and memberships must have the same length")
    ys = []
    for y in labels:
        if y in ("nontoxic", "toxic"):
            ys.append(int(y == "toxic"))
        elif type(y) is int and y in (0, 1):
            ys.append(y)
        else:
            raise ValueError("labels must be integer 0/1 or nontoxic/toxic")
    members = []
    for gs in memberships:
        if isinstance(gs, str) or not isinstance(gs, Iterable):
            raise ValueError("each membership must be an iterable of group strings")
        gs = list(gs)
        if not all(isinstance(g, str) and g for g in gs):
            raise ValueError("group names must be nonempty strings")
        members.append(set(gs))
    groups = sorted({g for y, gs in zip(ys, members) if y == 0 for g in gs})
    matrix = torch.tensor([[float(y == 0 and g in gs) for g in groups] for y, gs in zip(ys, members)], dtype=torch.float64)
    if not groups:
        matrix = torch.empty((len(ys), 0), dtype=torch.float64)
    return torch.tensor(ys, dtype=torch.long), groups, matrix

def fairbatch_negative_weights(labels, memberships, group_risks: Mapping[str, float | None], *, strength=0.3, max_weight=5.0):
    """FairBatch-inspired *adaptation*, not the original FairBatch algorithm.

    Resample only inside the negative class based on full-train overlapping
    negative-group CE risks. Positive weights stay 1; negative weights sum to
    their original class count. The caller must draw exactly the original
    positive and negative counts separately to preserve class mass exactly.
    A single global WeightedRandomSampler preserves that mass only in
    expectation. No metadata or predictions from dev/test are accepted here.
    """
    if not math.isfinite(strength) or strength < 0 or not math.isfinite(max_weight) or max_weight < 1:
        raise ValueError("invalid resampling hyperparameters")
    ys, groups, matrix = _labels_and_memberships(labels, memberships)
    weights = torch.ones(len(ys), dtype=torch.float64)
    risk_values = []
    for g in groups:
        risk = group_risks.get(g)
        if risk is None:
            risk = 0.0
        if not isinstance(risk, (int, float)) or not math.isfinite(risk) or risk < 0:
            raise ValueError("group risks must be nonnegative finite floats or None")
        risk_values.append(float(risk))
    if not groups or not any(risk_values) or strength == 0:
        return weights
    negative = ys == 0
    # Fixed denominators avoid counting overlap as mutually exclusive groups.
    risk = matrix @ (torch.tensor(risk_values, dtype=torch.float64) / matrix.sum(dim=0))
    if risk[negative].sum() > 0:
        risk /= risk[negative].mean()
        weights[negative] = (1 + strength * risk[negative]).clamp(max=max_weight)
        weights[negative] /= weights[negative].mean()
    return weights
