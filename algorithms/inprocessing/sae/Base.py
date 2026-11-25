"""Unweighted cross-entropy baseline."""
from .common import BroadcastAlgorithm


class Base(BroadcastAlgorithm):
    method = 'base'
