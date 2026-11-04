"""Cross-entropy training with stochastic weight averaging."""
from .common import BroadcastAlgorithm


class SWA(BroadcastAlgorithm):
    method = 'base_swa'
