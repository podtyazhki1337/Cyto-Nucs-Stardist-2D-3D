"""Memory-light replacement of StarDist's get_valid_inds for the baselines.

StarDist (and the HydraStarDist fork of its data class) keeps every valid patch centre of every training image in
memory: a full index grid for background patches and the foreground centres as int64 arrays, hundreds of MB for one
CT volume. The replacement draws the same distribution without that cache: a centre uniformly among the valid
centres, with probability foreground_prob among those whose patch contains foreground (maximum filter of the
labels over the patch > 0, as StarDist), by rejection sampling on a bit mask of those centres.
"""
import numpy as np
from scipy.ndimage import maximum_filter


def light_valid_inds(labels='Y'):
    """get_valid_inds(self, k, foreground_prob=None) reading the labels self.<labels>[k] (StarDist: Y,
    HydraStarDist: Y1)."""

    def valid_centres(self, k, foreground_prob=None):
        fp = self.foreground_prob if foreground_prob is None else foreground_prob
        y = getattr(self, labels)[k]
        lo = np.array([p // 2 for p in self.patch_size])
        region = tuple(s - p + 1 for s, p in zip(y.shape, self.patch_size))
        if np.random.uniform() < fp:
            masks = self.__dict__.setdefault('_foreground_centres', {})
            if k not in masks:
                cover = maximum_filter((np.asarray(y) > 0).astype(np.uint8), self.maxfilter_patch_size, mode='constant')
                inside = cover[tuple(slice(l, l + r) for l, r in zip(lo, region))].astype(bool)
                masks[k] = (np.packbits(inside.ravel()), int(inside.sum()), None)
            packed, count, listed = masks[k]
            if count:
                flat = np.empty(0, np.int64)
                for _ in range(20):
                    draw = np.random.randint(int(np.prod(region)), size=512)
                    flat = draw[((packed[draw >> 3] >> (7 - (draw & 7))) & 1).astype(bool)]
                    if len(flat):
                        break
                if not len(flat):                               # very little foreground: list its centres once
                    if listed is None:
                        listed = np.flatnonzero(np.unpackbits(packed)[:int(np.prod(region))])
                        masks[k] = (packed, count, listed)
                    flat = listed
                return tuple(c + l for c, l in zip(np.unravel_index(flat, region), lo))
        return tuple(np.random.randint(l, l + r, size=512) for l, r in zip(lo, region))

    return valid_centres
