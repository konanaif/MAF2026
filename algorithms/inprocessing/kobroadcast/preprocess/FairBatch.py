"""Negative-class CE-risk resampling for overlapping groups."""
from ..common import BroadcastAlgorithm


class FairBatch(BroadcastAlgorithm):
    method = 'fairbatch'
