"""Training-supported atomic and low-order intersection groups.

Adapted from fairbench/methods/dr_wrapper.py at upstream 79a9eea.
Store the predicates as well as masks so new rows use the same group definition.
"""
from itertools import combinations, product
from numbers import Integral

import numpy as np


def validate_sensitive(values):
    values = np.asarray(values)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Sensitive attributes must be a nonempty (n, q) matrix.")
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("DRAF requires explicitly binarized sensitive attributes (0/1).")
    return values.astype(np.int8)


class SubgroupFamily:
    def __init__(self, min_fraction, max_order, max_groups):
        if not 0 < min_fraction <= 0.5:
            raise ValueError("min_fraction must be in (0, 0.5].")
        if isinstance(max_groups, bool) or not isinstance(max_groups, Integral) or max_groups < 1:
            raise ValueError("max_groups must be a positive integer.")
        self.min_fraction = float(min_fraction)
        self.max_order = None if max_order is None else tuple(sorted(set(max_order)))
        if self.max_order is not None and any(isinstance(k, bool) or not isinstance(k, Integral) or k < 1
                                              for k in self.max_order):
            raise ValueError("Subgroup orders must be positive integers.")
        self.max_groups = int(max_groups)

    def fit_transform(self, sensitive):
        sensitive = validate_sensitive(sensitive)
        n, self.n_attributes_ = sensitive.shape
        self.min_count_ = max(1, int(np.ceil(self.min_fraction * n)))
        self.rules_ = []
        masks, seen = [], set()
        self.truncated_ = False

        def add(rule):
            mask = self._mask(sensitive, rule)
            count = int(mask.sum())
            if min(count, n - count) < self.min_count_:
                return False
            key = mask.tobytes()
            if key in seen:
                return False
            if len(masks) == self.max_groups:
                self.truncated_ = True
                return True
            seen.add(key)
            masks.append(mask)
            self.rules_.append(tuple(rule))
            return False

        # Upstream puts complete atomic cells first, ordered by their bit code.
        patterns = sorted(set(map(tuple, sensitive)),
                          key=lambda row: sum(int(v) << j for j, v in enumerate(row)))
        for pattern in patterns:
            if add(tuple(enumerate(pattern))):
                break
        # Preserve upstream Apriori pruning: both a marginal and its complement
        # must reach T before that attribute can enter a generated intersection.
        eligible_attributes = [j for j in range(self.n_attributes_)
                               if min(int(sensitive[:, j].sum()),
                                      n - int(sensitive[:, j].sum())) >= self.min_count_]
        if not self.truncated_ and self.max_order:
            for order in self.max_order:
                if order > len(eligible_attributes):
                    continue
                for cols in combinations(eligible_attributes, order):
                    for pattern in product((0, 1), repeat=order):
                        if add(tuple(zip(cols, pattern))):
                            break
                    if self.truncated_:
                        break
                if self.truncated_:
                    break
        return np.column_stack(masks).astype(np.float32) if masks else np.zeros((n, 0), np.float32)

    @staticmethod
    def _mask(sensitive, rule):
        mask = np.ones(len(sensitive), dtype=bool)
        for column, value in rule:
            mask &= sensitive[:, column] == value
        return mask

    def transform(self, sensitive):
        if not hasattr(self, "rules_"):
            raise RuntimeError("Fit the subgroup family first.")
        sensitive = validate_sensitive(sensitive)
        if sensitive.shape[1] != self.n_attributes_:
            raise ValueError("Sensitive attribute columns must match training.")
        if not self.rules_:
            return np.zeros((len(sensitive), 0), np.float32)
        return np.column_stack([self._mask(sensitive, rule) for rule in self.rules_]).astype(np.float32)

    def describe(self, names):
        return [{str(names[j]): int(v) for j, v in rule} for rule in self.rules_]
