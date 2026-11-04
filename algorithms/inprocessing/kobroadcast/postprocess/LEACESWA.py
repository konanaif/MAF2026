"""Apply a TRAIN-fitted conditional LEACE transform to a fixed SWA model."""
from ..common import BroadcastAlgorithm


class LEACESWA(BroadcastAlgorithm):
    method = 'leace_swa'

    def run(self):
        checkpoint = self.checkpoint
        if checkpoint is None:
            checkpoint = self.output / self.model_name / 'checkpoints' / 'base_swa'
            if not checkpoint.exists():
                checkpoint = self.train('base_swa')
        return self.compute_metrics(checkpoint)
