"""Common interface for the four broadcast algorithms."""
from pathlib import Path


class BroadcastAlgorithm:
    method = None

    def __init__(self, model_name, cfg, paths, output, seed, device,
                 rank=128, strength=.8, checkpoint=None):
        self.model_name = model_name
        self.cfg = cfg
        self.paths = paths
        self.output = Path(output)
        self.seed = seed
        self.device = device
        self.rank = rank
        self.strength = strength
        self.checkpoint = Path(checkpoint) if checkpoint else None

    def train(self, method):
        from .experiment import train_checkpoint
        directory = self.output / self.model_name / 'checkpoints' / method
        return train_checkpoint(self.model_name, method, self.cfg, self.paths,
                                directory, self.seed, self.device)

    def compute_metrics(self, checkpoint):
        from .experiment import evaluate_checkpoint
        directory = self.output / self.model_name / 'evaluation' / self.method
        return evaluate_checkpoint(self.model_name, self.method, checkpoint, self.paths,
                                   directory, self.device, self.rank, self.strength)

    def run(self):
        return self.compute_metrics(self.train(self.method))
