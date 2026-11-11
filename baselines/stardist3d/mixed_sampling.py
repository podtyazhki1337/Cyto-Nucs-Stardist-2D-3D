"""StarDist-3D trained on its volumes plus extra ones (e.g. synthetic volumes) with a fixed share of the training crops
from the extra volumes, as CytoNucs' training.extra_train_dirs / extra_train_share mix them. StarDist itself walks
through its volumes in shuffled order, each as often as the others, so the share would follow the numbers of volumes;
here every crop of a training batch comes from an extra volume with probability `share`, from a main one otherwise
(each picked uniformly). Validation data are not affected.
"""
import numpy as np
from stardist.models import model3d
from stardist.models.model3d import StarDist3D, StarDistData3D


def mixed_data_class(n_main, n_total, share, seed=42):
    """StarDistData3D whose batches over n_total volumes (the first n_main the main ones) take `share` of the crops from
    the others; a generator over another number of volumes (validation) samples as StarDist does."""
    class MixedStarDistData3D(StarDistData3D):
        def batch(self, i):
            if self.data_size != n_total:
                return super().batch(i)
            rng = np.random.default_rng((seed, int(i)))            # the same crops for the same step
            extra = rng.random(self.batch_size) < share
            return np.where(extra, rng.integers(n_main, n_total, self.batch_size), rng.integers(0, n_main, self.batch_size))
    return MixedStarDistData3D


class MixedStarDist3D(StarDist3D):
    """StarDist3D trained on X + extra volumes (passed after the main ones) with a share of the crops from the extra."""

    def __init__(self, config, name=None, basedir='.', n_main=None, share=0.0):
        super().__init__(config, name=name, basedir=basedir)
        self.n_main, self.share = n_main, float(share)

    def train(self, X, Y, *args, **kwargs):
        model3d.StarDistData3D = mixed_data_class(self.n_main, len(X), self.share)   # StarDist3D.train builds it by name
        try:
            return super().train(X, Y, *args, **kwargs)
        finally:
            model3d.StarDistData3D = StarDistData3D
