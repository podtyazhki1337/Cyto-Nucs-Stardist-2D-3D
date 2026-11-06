"""Synthetic CytoNucs dataset drawn from the statistics of the annotated data (measure_gt_statistics.py).

Shapes. Every cell is drawn from a donor, a whole annotated cell of the chosen profile, together with
its nuclei: per slice the outline is the donor's radial function r(phi), written as Fourier modes of
log r. The new object keeps the donor's size (times a random factor, the Silverman bandwidth of the
log volumes), its slices along Z, elongation and asymmetry (modes 1-2) and the power of every mode,
so its roughness spectrum is the measured one; the phases of modes 3..24 are shifted at random (the
same shift on every slice keeps the coherence between slices; a slice rougher than the 90th
percentile of the largest slices of the donor objects is scaled down to it: random phases turn
the rare very rough outlines into stars), and the object is rotated (XY orientations of
the annotated objects are uniform) and mirrored. Nuclei keep their offsets from the cell centre,
rotated and mirrored with it.

Nesting. Nucleus within cell, nucleolus within nucleus: a cell holds its nuclei (their voxels and a
NUCLEUS_ZONE px rim are added to its outline where they stick out; a candidate whose nucleus zone
would fall to a placed cell, or that would take the zone of a placed nucleus, is tried elsewhere, so
no nucleus cuts into a neighbour; holes enclosed by one cell in a slice are filled); cells that overlap are split where the
normalised radial distance rho / r(phi) of one is smaller than that of the other, and a nucleus
keeps the voxels of its cell; nucleoli are cut to their nucleus. Every nucleus has its cell in
assignments/<stem>.json, every nucleolus its nucleus in assignments_nucleolus_nucleus/<stem>.json.
Cells, nuclei per cell (multinucleated share) and colours come from the donors of one annotated
volume per synthetic volume.

Nucleoli are not annotated anywhere: their number, size, shape and darkness are assumed
(NUCLEOLUS) and written to split_info.json.

Packing. Cells are added until the cell share of the volume reaches a value drawn from the cell
shares of the non-empty windows of the same size in the annotated data. In dense tissue the cell
outlines are placed smaller in XY (PROFILES cell_scale; nuclei keep their size) and grow into the
gaps afterwards, as rigid donor outlines do not fill space. A new cell starts a new
cluster with the measured probability (clusters per cell), else it is placed next to a placed cell
(those with fewer failed attachments are preferred): in a direction whose angle to the XY plane is
drawn from the nearest-neighbour directions, at a centre distance = gap ratio x (sum of the two
radii along it), the gap ratio drawn from the touching nearest neighbours. A candidate overlapping
placed cells by more than max_overlap of its volume, or whose nuclei would fall into another cell, is
tried elsewhere. Half of the candidates are centred on an empty voxel inside the tissue envelope
(voids get whole cells). Finally cells grow into the background inside the tissue envelope until
the measured envelope density is reached.

Fill (realistic mode). Background far from cells; cell body and nucleus with the colours of their
donor (without the donor's depth factor), times a two-scale multiplicative texture with the
measured correlation (part of the fine scale as streaks, fitted to the measured coherence); membranes as the brighter 1 px rim of every cell body (2 px lines at contacts,
1 px at free borders); nucleolus as its nucleus x NUCLEOLUS['darkness']; depth profile along Z around the tissue
centre; blur = core Gaussian + halo (out-of-focus light) as fitted to the annotated images; noise by
intensity. calibrate() then adjusts core blur, halo weight, class gains, background, noise,
texture (and its streaky share) and membrane brightness until the same measurement on synthetic volumes gives the measured values
(render_params.json). The measured profiles against the distance to an annotated border are not
used for the fill: they mostly show the blur (and the annotation missing the membrane by a pixel).
Randomised mode (--appearance randomized): random class colours and blur, Gaussian noise of
amplitude --noise (fraction of the range 0..255).

Output, as CytoNucsDataGenerator reads it:
  <output>/{train,val,test}/images/<stem>.tiff          (Z, Y, X, 3) uint8 (2D: (Y, X, 3))
  <output>/{train,val,test}/{cytoplasm,nucleus,nucleolus}_masks/<stem>.tiff   uint16, cells hold their nuclei
  <output>/{train,val,test}/assignments/<stem>.json      nucleus -> cell
  <output>/{train,val,test}/assignments_nucleolus_nucleus/<stem>.json
  <output>/{train,val,test}/meta.json, <output>/split_info.json, <output>/render_params.json
A two-level run (model.levels ["nucleus", "cytoplasm"]) ignores nucleolus_masks; a three-level run
(["nucleolus", "nucleus", "cytoplasm"]) reads all three. There is no train_regions folder: set
training.train_region_crops to false.

Usage:
  python dataset_prep/synthetic/make_synthetic_from_stats.py --stats DIR/gt_statistics.json --profile hela \
      --output /storage02/miroslavm/data/Cytonucs_Synthetic/NAME [--counts 60 8 8] [--shape 32 256 256]
  python dataset_prep/synthetic/make_synthetic_from_stats.py ... --preview PNG     (a few volumes and a comparison)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))         # measure_gt_statistics.py
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))     # downscale_split.py of dataset_prep
from measure_gt_statistics import (N_MODES, STEP, Volume, measure_volume, radius_from_modes, sharp_ratio, summarize,  # noqa: E402
                                   tissue_envelope)

# Nucleoli have no annotation: these values are assumptions (HeLa nucleoli: one to four per nucleus,
# about 1-3 um across in nuclei of 10-15 um, poor in DNA and darker in DAPI).
NUCLEOLUS = dict(count_mean=1.0, count_max=3,      # 1 + Poisson(count_mean), at most count_max
                 radius_ratio=(0.18, 0.28),        # equivalent radius / nucleus equivalent XY radius
                 elongation=(1.0, 1.4), roughness=0.5,  # roughness: x the nucleus mode amplitudes
                 darkness=(0.15, 0.35),            # intensity / nucleus intensity (dark enough to be seen after the blur)
                 margin=2.0)                       # px of nucleus around a nucleolus in XY; never in the first or last slice
# cell_scale: XY size of the cell outlines when placed (they grow into the gaps afterwards); 0.75 gives
# the measured cell volumes and tissue density of the HeLa spheroid (whole cells 12.8k voxels against
# 11.6k annotated, density 0.91 against 0.92, measured on 32 x 384 x 384 volumes)
PROFILES = dict(hela=dict(shape=(32, 256, 256), substrate=False, cell_scale=0.75),
                ea=dict(shape=(16, 256, 256), substrate=True),
                hollandi=dict(shape=(512, 512), substrate=False, pooled=True))
PHI_FINE = 2 * np.pi * np.arange(512) / 512
NUCLEUS_ZONE = 2                                   # px of its own cell around every nucleus (XY)
ZONE = np.ones((1, 2 * NUCLEUS_ZONE + 1, 2 * NUCLEUS_ZONE + 1), bool)


# ------------------------------------------------------------------------------------ shapes

def rotate(yx, angle):
    """(y, x) rows rotated by `angle` in the sense of phi = atan2(y, x), as the modes are rotated."""
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(yx, np.float64) @ np.array([[c, -s], [s, c]])


class Shape:
    """A star-convex object: per slice (z relative to the object's first slice) a centre offset
    from the object anchor and the Fourier modes of log r(phi)."""

    def __init__(self, z, offsets, modes):
        self.z = np.asarray(z, int)
        self.offsets = np.asarray(offsets, np.float64).reshape(-1, 2)
        self.modes = np.asarray(modes, np.complex128).reshape(len(self.z), -1)

    @classmethod
    def from_record(cls, rec):
        return cls([s['z'] for s in rec['slices']], [[s['dy'], s['dx']] for s in rec['slices']],
                   [[complex(*m) for m in s['modes']] for s in rec['slices']])

    def transformed(self, scale=1.0, angle=0.0, mirror=False, phases=None, z_scale=None):
        """Scaled, mirrored (y -> -y), rotated by `angle`, with mode phases shifted by `phases` (k = 0..N)."""
        modes, offsets = self.modes.copy(), self.offsets.copy()
        if mirror:
            modes, offsets = np.conj(modes), offsets * [-1, 1]
        k = np.arange(modes.shape[1])
        modes = modes * np.exp(-1j * k * angle)[None]
        if phases is not None:
            modes = modes * np.exp(1j * np.asarray(phases))[None]
        modes[:, 0] = modes[:, 0].real + np.log(scale)
        offsets = rotate(offsets, angle) * scale
        shape = Shape(self.z, offsets, modes)
        return shape.resampled(z_scale if z_scale is not None else 1.0)

    def resampled(self, z_scale):
        """Slices resampled to round(nz * z_scale) slices (linear in the modes and offsets)."""
        nz = int(self.z.max() - self.z.min() + 1)
        new = max(1, int(round(nz * z_scale)))
        if new == nz:
            return self
        src = np.linspace(0, len(self.z) - 1, new) if len(self.z) > 1 else np.zeros(new)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, len(self.z) - 1)
        w = (src - lo)[:, None]
        return Shape(np.arange(new) + self.z.min(), self.offsets[lo] * (1 - w) + self.offsets[hi] * w,
                     self.modes[lo] * (1 - w) + self.modes[hi] * w)

    @property
    def nz(self):
        return int(self.z.max() + 1)

    def radius(self, i):
        return radius_from_modes(self.modes[i], PHI_FINE)

    def extent_xy(self):
        return max(float(self.radius(i).max() + np.abs(self.offsets[i]).max()) for i in range(len(self.z)))

    def capped(self, cap):
        """Modes 3.. of every slice scaled down where their RMS (roughness) exceeds `cap`."""
        modes = self.modes.copy()
        rough = np.sqrt((np.abs(modes[:, 3:]) ** 2 / 2).sum(1))
        modes[:, 3:] *= np.minimum(1.0, cap / np.maximum(rough, 1e-9))[:, None]
        return Shape(self.z, self.offsets, modes)

    def box(self, anchor, shape):
        """Bounding box in the volume of the object placed with its first slice at anchor[0] and its
        XY centre at anchor[1:] (None when it lies outside)."""
        z0, cy, cx = anchor
        R = self.extent_xy() + 1
        box = (slice(max(int(z0), 0), min(int(z0) + self.nz, shape[0])),
               slice(max(int(np.floor(cy - R)), 0), min(int(np.ceil(cy + R)) + 1, shape[1])),
               slice(max(int(np.floor(cx - R)), 0), min(int(np.ceil(cx + R)) + 1, shape[2])))
        return None if any(s.stop <= s.start for s in box) else box

    def render(self, anchor, shape, box=None):
        """Normalised radial distance rho / r(phi) (<= 1 inside) in `box` (default: the object's
        bounding box) of the object placed at `anchor`; returns (box, D), box None when the object
        lies outside the volume."""
        z0, cy, cx = anchor
        box = box or self.box(anchor, shape)
        if box is None:
            return None, None
        D = np.full(tuple(s.stop - s.start for s in box), np.inf, np.float32)
        yy, xx = np.mgrid[box[1], box[2]].astype(np.float64)
        for i, z in enumerate(self.z):
            zz = int(z0) + int(z) - box[0].start
            if not 0 <= zz < D.shape[0]:
                continue
            dy, dx = yy - (cy + self.offsets[i, 0]), xx - (cx + self.offsets[i, 1])
            rho = np.hypot(dy, dx)
            phi = np.arctan2(dy, dx) % (2 * np.pi)
            r = np.interp(phi, PHI_FINE, self.radius(i), period=2 * np.pi)
            D[zz] = (rho / r).astype(np.float32)
        return box, D


def ray_length(mask, centre, direction, spacing_z):
    """Distance from `centre` (local voxel coordinates) to the outermost voxel of `mask` along a
    physical unit direction."""
    t = np.arange(0, 4 * max(mask.shape) * spacing_z + 4, STEP)
    p = np.asarray(centre)[None] + t[:, None] * (np.asarray(direction) / np.array([spacing_z, 1, 1]))[None]
    idx = np.rint(p).astype(int)
    ok = np.all((idx >= 0) & (idx < mask.shape), 1)
    inside = np.zeros(len(t), bool)
    inside[ok] = mask[tuple(idx[ok].T)]
    return float(t[len(t) - 1 - np.argmax(inside[::-1])] + STEP / 2) if inside.any() else 0.0


class Cell:
    """A drawn cell: its shape, its nuclei (shape, anchor offset from the cell anchor), colours."""

    def __init__(self, shape, nuclei, colour, nucleus_colours, donor):
        self.shape, self.nuclei, self.colour, self.nucleus_colours, self.donor = shape, nuclei, colour, nucleus_colours, donor
        self.anchor = None


class Profile:
    """Shape, packing and appearance statistics of one annotated dataset, as the generator uses them."""

    def __init__(self, stats, name):
        self.name = name
        self.source = stats['sources'][name]
        self.summary = stats['datasets'][name]
        self.volumes = stats['volumes'][name]
        self.ndim = int(self.source['ndim'])
        self.spacing_z = float(self.source['spacing_z'])
        self.substrate = PROFILES[name]['substrate']
        self.donors = [(vi, d) for vi, v in enumerate(self.volumes) for d in v['donors']]
        self.by_volume = {vi: [(vi, d) for d in v['donors']] for vi, v in enumerate(self.volumes) if v['donors']}
        if not self.donors:
            raise ValueError(f'{name}: no whole cells to draw from')
        logv = np.log([d['volume'] for _, d in self.donors])
        self.median_volume = float(np.exp(np.median(logv)))
        # Silverman bandwidth of log volume, as a scale jitter of the linear size
        self.scale_sd = float(1.06 * logv.std() * len(logv) ** -0.2 / 3)
        p = [v['packing'] for v in self.volumes]
        self.p_new = float(sum(q['n_clusters'] for q in p) / max(sum(q['n_cells'] for q in p), 1))
        nn = [x for q in p for x in q['nearest']]
        touching = [x for x in nn if x['touching'] and np.isfinite(x['gap_ratio'] or np.nan)]
        self.gap_ratios = np.array([x['gap_ratio'] for x in (touching or nn)])
        self.elevations = np.radians([x['elevation'] for x in nn]) if self.ndim == 3 else np.zeros(1)
        shares = np.array([s for q in p for s in q['window_cell_share']])
        self.window_shares = shares[shares > 0.02] if (shares > 0.02).any() else shares
        # roughness cap per level: 90th percentile of the RMS of modes 3.. of the largest slice of every donor object
        def rough(objs):
            return [float(np.sqrt((np.abs(np.array([complex(*m) for m in max(o['slices'], key=lambda s: s['area'])['modes']][3:])) ** 2
                                   / 2).sum())) for o in objs if o['slices']]
        self.rough_cap = {level: float(np.percentile(r, 90)) if r else 1.0 for level, r in (
            ('cells', rough([d for _, d in self.donors])), ('nuclei', rough([n for _, d in self.donors for n in d['nuclei']])))}
        self.depth = {}
        for vi, v in enumerate(self.volumes):
            dep = v.get('appearance', {}).get('depth')
            if dep and self.ndim == 3:
                rows = [r for r in dep['slices'] if r['cells'] is not None]
                ref = np.average([r['cells'] for r in rows], 0, weights=[r['cell_voxels'] for r in rows])
                self.depth[vi] = (np.array([r['dz'] for r in rows]), np.array([np.asarray(r['cells']) / np.maximum(ref, 1) for r in rows]))
        self.envelope_density = self.summary['packing']['envelope_density'].get('median')
        self.cell_scale = PROFILES[name].get('cell_scale', 1.0)
        self.closing_radius = 0.5 * self.summary['cells']['radius_xy']['median']
        orient = self.summary['cells'].get('orientation', {})
        self.uniform_orientation = orient.get('rayleigh_p', 1.0) > 0.01

    def depth_factor(self, vi, dz):
        """Measured intensity of the cells at depth dz (slices) from the tissue centre / their mean, per channel."""
        if vi not in self.depth or dz is None:
            return 1.0
        zs, f = self.depth[vi]
        return np.array([np.interp(dz, zs, f[:, c]) for c in range(f.shape[1])])

    def draw_cell(self, rng, volume_index=None, like=None):
        """A cell (with its nuclei) from a random donor, of volume `volume_index` when given. With
        `like` (a drawn Cell), a random one of the 8 donors with as many nuclei and the nearest total
        nucleus volume (replacing a cell that found no place without changing what is placed)."""
        pool = self.donors if volume_index is None else self.by_volume[volume_index]
        if like is not None:
            same = [x for x in pool if len(x[1]['nuclei']) == len(like.nuclei)] or pool
            volume = lambda d: sum(n['volume'] for n in d['nuclei'])
            target = volume(like.record)
            pool = sorted(same, key=lambda x: abs(np.log1p(volume(x[1])) - np.log1p(target)))[:8]
        vi, d = pool[int(rng.integers(len(pool)))]
        scale = float(np.exp(rng.normal(0, self.scale_sd)))
        angle = float(rng.uniform(0, 2 * np.pi)) if self.uniform_orientation else 0.0
        mirror = bool(rng.random() < 0.5)

        def phases():
            ph = rng.uniform(0, 2 * np.pi, N_MODES + 1)
            ph[:3] = 0                                   # modes 0-2 (size, asymmetry, elongation) kept
            return ph
        z_scale = scale if self.ndim == 3 else 1.0
        # cells start smaller in XY (self.cell_scale) and grow into the gaps of the tissue (fill_gaps); nuclei keep their size
        k = self.cell_scale
        cell = Shape.from_record(d).transformed(scale * k, angle, mirror, phases(), z_scale).capped(self.rough_cap['cells'])
        nuclei = []
        for n in d['nuclei']:
            offset = rotate([[-n['dy'] if mirror else n['dy'], n['dx']]], angle)[0] * scale * k
            dz = int(round(n['dz'] * cell.nz / max(d['nz'], 1)))
            nuclei.append([Shape.from_record(n).transformed(scale, angle, mirror, phases(), z_scale).capped(self.rough_cap['nuclei']),
                           np.array([dz, *offset])])
        # colours without the depth profile at the donor's depth; the renderer applies it at the new depth
        f = self.depth_factor(vi, d.get('dz_tissue'))
        colour = lambda c: None if c is None else (np.asarray(c) / f).tolist()
        drawn = Cell(cell, nuclei, colour(d.get('colour')), [colour(n.get('colour')) for n in d['nuclei']], (vi, d['label']))
        drawn.donor_z0 = d['z0']
        drawn.record = d
        return drawn


# ----------------------------------------------------------------------------------- packing

def render_cell(cell, anchor, shape3):
    """The cell and its nuclei at the donor's offsets in one box: (box, D, [(donor nucleus index,
    mask)]). D of the cell is lowered inside its nuclei (0.5) and in a NUCLEUS_ZONE px rim around them
    (0.95), so that the cell holds its nuclei with some cytoplasm around; nuclei of one cell do not touch."""
    anchors = [(anchor[0] + off[0], anchor[1] + off[1], anchor[2] + off[2]) for _, off in cell.nuclei]
    boxes = [cell.shape.box(anchor, shape3)] + [n.box(a, shape3) for (n, _), a in zip(cell.nuclei, anchors)]
    boxes = [b for b in boxes if b is not None]           # the cell's own box is not None (checked by the caller)
    box = tuple(slice(min(b[a].start for b in boxes), max(b[a].stop for b in boxes)) for a in range(3))
    _, D = cell.shape.render(anchor, shape3, box)
    taken = np.zeros(D.shape, bool)
    nuclei = []
    for index, ((nshape, _), a) in enumerate(zip(cell.nuclei, anchors)):
        _, Dn = nshape.render(a, shape3, box)
        m = (Dn <= 1) & ~ndi.binary_dilation(taken, np.ones((1, 3, 3), bool))
        if m.sum() >= 4:
            taken |= m
            nuclei.append((index, m))
    if taken.any():
        rim = ndi.binary_dilation(taken, ZONE)
        D = np.where(taken, np.minimum(D, 0.5), np.where(rim, np.minimum(D, 0.95), D)).astype(np.float32)
    return box, D, nuclei


def _first_slice(profile, cell, shape3, rng, centre_z=None):
    """Z of a cell's first slice: 0 in 2D, the donor's slice on a substrate, else around centre_z
    (random when None)."""
    if profile.ndim == 2:
        return 0
    if profile.substrate:
        return int(min(cell.donor_z0, shape3[0] - 1))
    if centre_z is None:
        return int(rng.integers(-(cell.shape.nz // 2), max(shape3[0] - cell.shape.nz // 2, 1)))
    return int(round(centre_z - cell_mask_local(cell)[1][0]))


GAP_SHARE = 0.5                                     # candidates centred in a gap of the tissue (when there are gaps)


def _candidate(profile, cell, placed, fails, shape3, rng, gaps=None):
    """Anchor (first slice, centre y, centre x) of a new cell: a new cluster at a random place; with
    probability GAP_SHARE centred on a random empty voxel inside the tissue envelope (`gaps`), so that
    voids are filled by whole cells and not by growing the neighbours; else next to a placed cell
    (fewer failed attachments, more likely) at the measured gap."""
    sz = profile.spacing_z
    if placed and gaps is not None and len(gaps) and rng.random() < GAP_SHARE:
        g = gaps[int(rng.integers(len(gaps)))]
        mask_new, centre_new = cell_mask_local(cell)
        z0 = int(round(g[0] - centre_new[0])) if profile.ndim == 3 and not profile.substrate else _first_slice(profile, cell, shape3, rng)
        return (z0, g[1] - centre_new[1] + mask_new.shape[1] / 2, g[2] - centre_new[2] + mask_new.shape[2] / 2), None
    if not placed or rng.random() < profile.p_new:
        margin = 0.3 * cell.shape.extent_xy()
        return (_first_slice(profile, cell, shape3, rng), rng.uniform(-margin, shape3[1] + margin),
                rng.uniform(-margin, shape3[2] + margin)), None
    w = 1.0 / (1.0 + np.asarray(fails, np.float64)) ** 2
    j = int(rng.choice(len(placed), p=w / w.sum()))
    other = placed[j]
    az = rng.uniform(0, 2 * np.pi)
    elev = 0.0 if profile.ndim == 2 or profile.substrate else float(rng.choice(profile.elevations)) * rng.choice([-1, 1])
    u = np.array([np.sin(elev), np.cos(elev) * np.sin(az), np.cos(elev) * np.cos(az)])
    mask_new, centre_new = cell_mask_local(cell)
    dist = float(rng.choice(profile.gap_ratios)) * (
        ray_length(other.mask, other.centre_local, u, sz) + ray_length(mask_new, centre_new, -u, sz))
    centre = other.centre + u * dist / np.array([sz, 1, 1])
    return (_first_slice(profile, cell, shape3, rng, centre[0]), centre[1] - centre_new[1] + mask_new.shape[1] / 2,
            centre[2] - centre_new[2] + mask_new.shape[2] / 2), j


def fill_gaps(owner, density, spacing_z, radius, max_steps=20, max_z_steps=1):
    """Grow cells into the background inside the tissue envelope (as measured: closing of the cell
    union with half the median cell radius) until the cell share of the envelope reaches the
    measured density: one pixel in XY per step, and at step spacing_z one slice in Z where a cell
    lies both above and below (at most max_z_steps times: thin gaps between layers of cells close,
    the cells do not get taller)."""
    from skimage.segmentation import expand_labels
    if not (owner > 0).any():
        return owner
    env = tissue_envelope(owner > 0, radius, spacing_z)
    z_every = max(1, int(round(spacing_z)))
    for step in range(1, max_steps + 1):
        if (owner[env] > 0).mean() >= density:
            break
        grown = np.stack([expand_labels(s, 1) for s in owner])
        if owner.shape[0] > 1 and step % z_every == 0 and step // z_every <= max_z_steps:
            below = np.zeros_like(owner)
            above = np.zeros_like(owner)
            below[1:], above[:-1] = owner[:-1], owner[1:]
            vertical = np.where((below > 0) & (above > 0), below, 0)          # one-slice gaps between two cells
            grown = np.where(grown > 0, grown, vertical)
        owner = np.where(env & (owner == 0), grown, owner)
    return owner


def pack_tile(profile, shape, rng, max_overlap=0.3, target=None, max_cells=None, tries=6, max_draws=None, volume_index=None,
              fill=True):
    """Cells and nuclei of one volume (Z, Y, X; 2D: Z = 1). Returns dict(cells, nuclei, assignments
    nucleus -> cell, placed cells, target cell share, cell_colours, nucleus_colours by label)."""
    shape3 = tuple(shape) if len(shape) == 3 else (1,) + tuple(shape)
    target = float(rng.choice(profile.window_shares)) if target is None else target
    best = np.full(shape3, np.inf, np.float32)
    owner = np.zeros(shape3, np.int32)
    nucleus_owner = np.zeros(shape3, np.int32)          # cell number of nuclear voxels
    nucleus_label = np.zeros(shape3, np.int32)
    nucleus_zone = np.zeros(shape3, np.int32)           # nuclei grown by NUCLEUS_ZONE px in XY
    placed, fails, assignments = [], [], {}
    total = float(np.prod(shape3))
    if max_draws is None:                                # about eight draws per cell the target needs
        max_draws = int(50 + 8 * target * total / profile.median_volume)
    covered = 0
    cell, failed = None, 0
    gaps, gaps_at = None, 0
    for _ in range(max_draws):
        if covered / total >= target or (max_cells and len(placed) >= max_cells):
            break
        # A donor is kept for up to 4 x `tries` positions, and then replaced by a similar one (as many
        # nuclei, nearest nucleus volume): drawing a new one after each rejection would favour small
        # cells, small nuclei and cells without nuclei in dense tissue.
        if cell is None:
            cell, failed = profile.draw_cell(rng, volume_index), 0
        elif failed >= 4:
            cell, failed = profile.draw_cell(rng, volume_index, like=cell), 0
        failed += 1
        if placed and (gaps is None or len(placed) >= gaps_at + 10):
            # empty voxels inside the envelope of the cells placed so far, refreshed every 10 cells
            env = tissue_envelope(owner > 0, profile.closing_radius, profile.spacing_z)
            gaps, gaps_at = np.argwhere(env & (owner == 0)), len(placed)
        for _ in range(tries):
            anchor, j = _candidate(profile, cell, placed, fails, shape3, rng, gaps)
            if cell.shape.box(anchor, shape3) is None:
                continue
            box, D, nuclei = render_cell(cell, anchor, shape3)
            inside = D <= 1
            ok = inside.sum() >= 0.25 * cell_mask_local(cell)[0].sum() and (owner[box][inside] > 0).mean() <= max_overlap
            if ok:
                # A nucleus lies in its own cell with NUCLEUS_ZONE px of it around (XY): its zone must be won by
                # the new cell, and the new cell must not win the zone of a placed nucleus. Otherwise a nucleus
                # would cut a hole or a bite into a neighbour.
                wins = inside & (D < best[box])
                zone = np.zeros(D.shape, bool)
                for _, m in nuclei:
                    zone |= ndi.binary_dilation(m, ZONE)
                ok = not (zone & ~wins).any() and not (wins & (nucleus_zone[box] > 0)).any()
            if ok:
                break
            if j is not None:
                fails[j] += 1
        else:
            ok = False
        if not ok:
            continue
        k = len(placed) + 1
        better = inside & (D < best[box])
        covered += int((better & (owner[box] == 0)).sum())
        best[box][better] = D[better]
        owner[box][better] = k
        cell.kept_nuclei = [i for i, _ in nuclei]
        for _, m in nuclei:
            n = len(assignments) + 1
            nucleus_owner[box][m] = k
            nucleus_label[box][m] = n
            nucleus_zone[box][ndi.binary_dilation(m, ZONE)] = k
            assignments[n] = k
        zz, yy, xx = np.nonzero(inside)
        cell.anchor, cell.box, cell.mask = anchor, box, inside
        cell.centre_local = np.array([zz.mean(), yy.mean(), xx.mean()])
        cell.centre = cell.centre_local + [b.start for b in box]
        placed.append(cell)
        fails.append(0)
        cell = None
    # A nucleus keeps its cell's voxels. Pieces of a cell cut off by its neighbours go to the
    # background, except its largest piece and those holding its nuclei.
    owner[nucleus_owner > 0] = nucleus_owner[nucleus_owner > 0]
    for k, box in enumerate(ndi.find_objects(owner), 1):
        if box is None:
            continue
        m = owner[box] == k
        lab, n = ndi.label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            keep = ({int(np.argmax(sizes)) + 1} | set(np.unique(lab[nucleus_owner[box] == k]).tolist())) - {0}
            owner[box][m & ~np.isin(lab, sorted(keep))] = 0
    owner = fill_holes(owner, nucleus_owner)
    if fill and profile.envelope_density:
        owner = fill_holes(fill_gaps(owner, profile.envelope_density, profile.spacing_z, profile.closing_radius), nucleus_owner)
    nucleus_cell = dict(assignments)
    cells, nuclei, assignments, lut, lutn = relabel(owner, nucleus_label, assignments)
    # colours of the donors, by the final labels
    cell_colours, nucleus_colours, order = {}, {}, {}
    for n, k in sorted(nucleus_cell.items()):
        order.setdefault(k, []).append(n)
    for k, cell in enumerate(placed, 1):
        if k < len(lut) and lut[k]:
            cell_colours[int(lut[k])] = cell.colour
        for i, n in enumerate(order.get(k, [])):
            if n < len(lutn) and lutn[n]:
                nucleus_colours[int(lutn[n])] = cell.nucleus_colours[cell.kept_nuclei[i]]
    return dict(cells=cells, nuclei=nuclei, assignments=assignments, placed=placed, target=target,
                cell_colours=cell_colours, nucleus_colours=nucleus_colours)


def fill_holes(owner, nucleus_owner):
    """No cut-outs inside a cell: per slice, background or pieces of other cells enclosed by one cell
    become that cell (not the nuclei of other cells; NUCLEUS_ZONE keeps them out). Pieces of a cell
    that this disconnects in 3D go to the background, except its largest one and those with its nuclei."""
    changed = set()
    for z in range(owner.shape[0]):
        for k, box in enumerate(ndi.find_objects(owner[z]), 1):
            if box is None:
                continue
            m = owner[z][box] == k
            hole = ndi.binary_fill_holes(m) & ~m & (nucleus_owner[z][box] == 0)
            if hole.any():
                changed.update(np.unique(owner[z][box][hole]).tolist())
                owner[z][box][hole] = k
    for k in sorted(changed - {0}):
        box = ndi.find_objects((owner == k).astype(np.uint8))
        if not box or box[0] is None:
            continue
        box = box[0]
        m = owner[box] == k
        lab, n = ndi.label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            keep = ({int(np.argmax(sizes)) + 1} | set(np.unique(lab[nucleus_owner[box] == k]).tolist())) - {0}
            owner[box][m & ~np.isin(lab, sorted(keep))] = 0
    return owner


def cell_mask_local(cell):
    """Mask of a cell's shape in its own box (cached) and its centroid there."""
    if getattr(cell, '_local', None) is None:
        R = cell.shape.extent_xy() + 2
        size = (cell.shape.nz, int(2 * R) + 3, int(2 * R) + 3)
        box, D = cell.shape.render((0, size[1] / 2, size[2] / 2), size)
        m = D <= 1
        zz, yy, xx = np.nonzero(m)
        cell._local = (m, np.array([zz.mean(), yy.mean(), xx.mean()]))
    return cell._local


def relabel(cells, nuclei, assignments):
    """Consecutive labels 1..n by position; nucleus -> cell links follow (nuclei whose cell vanished
    are dropped)."""
    old_c = np.unique(cells)[1:]
    lut = np.zeros(int(cells.max()) + 1, np.int32)
    lut[old_c] = np.arange(1, len(old_c) + 1)
    cells = lut[cells]
    old_n = np.unique(nuclei)[1:]
    keep = [n for n in old_n if assignments[int(n)] < len(lut) and lut[assignments[int(n)]] > 0]
    lutn = np.zeros(int(nuclei.max()) + 1, np.int32)
    lutn[keep] = np.arange(1, len(keep) + 1)
    nuclei = lutn[nuclei]
    pairs = {int(lutn[n]): int(lut[assignments[int(n)]]) for n in keep}
    return cells.astype(np.uint16), nuclei.astype(np.uint16), pairs, lut, lutn


# --------------------------------------------------------------------------------- nucleoli

def add_nucleoli(nuclei, spacing_z, rng, params=NUCLEOLUS):
    """Nucleoli inside the nuclei (assumed parameters). Returns (labels, nucleolus -> nucleus)."""
    out = np.zeros(nuclei.shape, np.uint16)
    links = {}
    planar = nuclei.shape[0] == 1
    for n, box in enumerate(ndi.find_objects(nuclei), 1):
        if box is None:
            continue
        m = nuclei[box] == n
        r_n = np.sqrt(m.reshape(m.shape[0], -1).sum(1).max() / np.pi)
        count = min(1 + rng.poisson(params['count_mean']), params['count_max'])
        dist = np.stack([ndi.distance_transform_edt(np.pad(s, 1))[1:-1, 1:-1] for s in m])   # to the nuclear border in XY
        # strictly inside: at least `margin` px of nucleus around it in XY, and not in the nucleus' first
        # or last slice (when it has three or more)
        interior = dist > params['margin']
        if not planar:
            zs = np.flatnonzero(m.reshape(m.shape[0], -1).any(1))
            if len(zs) >= 3:
                interior[[zs[0], zs[-1]]] = False
        taken = np.zeros_like(m)
        for _ in range(count):
            r = r_n * rng.uniform(*params['radius_ratio'])
            e = rng.uniform(*params['elongation'])
            a, b = r * np.sqrt(e), r / np.sqrt(e)
            rz = r / spacing_z
            ok = interior & (dist >= max(a, 1) + params['margin']) & ~ndi.binary_dilation(taken, np.ones((1, 3, 3), bool), iterations=2)
            if not planar:
                zc = (m.shape[0] - 1) / 2       # centred in Z within the nucleus, +- a quarter of its depth
                zs = np.arange(m.shape[0])
                ok &= (np.abs(zs - zc) <= max(m.shape[0] / 4, 0.5))[:, None, None]
            pts = np.argwhere(ok)
            if not len(pts):
                continue
            c = pts[int(rng.integers(len(pts)))]
            theta = rng.uniform(0, np.pi)
            zz, yy, xx = np.ogrid[:m.shape[0], :m.shape[1], :m.shape[2]]
            dy, dx = yy - c[1], xx - c[2]
            u, v = dy * np.cos(theta) + dx * np.sin(theta), -dy * np.sin(theta) + dx * np.cos(theta)
            phi = np.arctan2(v / b, u / a)
            # mild roughness: low modes with random phases
            k = np.arange(3, 9)
            rough = (params['roughness'] * 0.03 * np.cos(np.multiply.outer(phi, k) + rng.uniform(0, 2 * np.pi, len(k)))).sum(-1)
            rho = np.sqrt((u / a) ** 2 + (v / b) ** 2) * np.exp(-rough)
            if not planar:
                rho = np.sqrt(rho ** 2 + ((zz - c[0]) / max(rz, 0.5)) ** 2)
            blob = (rho <= 1) & interior & ~taken
            if blob.sum() < 3:
                continue
            taken |= blob
            label = len(links) + 1
            out[box][blob] = label
            links[label] = n
    return out, links


# ------------------------------------------------------------------------------------ fill

def _border_distance(labels2d):
    """Per pixel of a label: distance to the nearest pixel of another label or the background
    (1 on the border) and whether the nearest outside pixel belongs to another label."""
    lab = labels2d
    border = np.zeros(lab.shape, bool)
    contact = np.zeros(lab.shape, bool)
    for axis in (0, 1):
        for shift in (1, -1):
            nb = np.roll(lab, shift, axis)
            edge = np.zeros(lab.shape, bool)
            if shift == 1:
                sl = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
            else:
                sl = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
            edge[sl] = True
            diff = (nb != lab) & (lab > 0) & edge
            border |= diff
            contact |= diff & (nb > 0)
    if not border.any():                     # no border in this slice (the image edge is none: cells go on beyond it)
        return np.full(lab.shape, 99, np.float32), np.zeros(lab.shape, bool)
    d, idx = ndi.distance_transform_edt(~border, return_indices=True)
    return (d + 1).astype(np.float32), contact[idx[0], idx[1]]


def texture_field(shape, sigma, rng):
    """Gaussian random field of unit variance, Gaussian correlation with sigma per axis."""
    f = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma, mode='wrap')
    return f / max(float(f.std()), 1e-6)


def fibre_field(shape, sigma, ratio, rng, n_orient=8, smooth=24.0):
    """Unit-variance oriented texture (streaks): white noise filtered per slice with a Gaussian of
    sigma across and ratio x sigma along an orientation that varies smoothly in XY (scale `smooth`
    px), blended from n_orient fixed orientations."""
    from scipy.signal import fftconvolve
    Z, H, W = shape
    noise = rng.standard_normal(shape).astype(np.float32)
    # orientation field from smoothed complex noise (doubled-angle representation)
    c = ndi.gaussian_filter(rng.standard_normal(shape), (1.0 if Z > 1 else 0, smooth, smooth)) + \
        1j * ndi.gaussian_filter(rng.standard_normal(shape), (1.0 if Z > 1 else 0, smooth, smooth))
    theta = (np.angle(c) / 2) % np.pi
    long = sigma * ratio
    half = int(np.ceil(3 * long))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    out = np.zeros(shape, np.float32)
    wsum = np.zeros(shape, np.float32)
    for k in range(n_orient):
        a = np.pi * k / n_orient
        u, v = xx * np.cos(a) + yy * np.sin(a), -xx * np.sin(a) + yy * np.cos(a)
        kernel = np.exp(-u ** 2 / (2 * long ** 2) - v ** 2 / (2 * sigma ** 2))
        kernel /= np.sqrt((kernel ** 2).sum())
        d = np.abs(((theta - a) + np.pi / 2) % np.pi - np.pi / 2)            # angular distance, 0..pi/2
        w = np.clip(1 - d / (np.pi / n_orient), 0, 1).astype(np.float32)     # linear blend of the two nearest
        if not w.any():
            continue
        f = np.stack([fftconvolve(noise[z], kernel, mode='same') for z in range(Z)]).astype(np.float32)
        out += w * f
        wsum += w ** 2
    out /= np.sqrt(np.maximum(wsum, 1e-6))
    return out / max(float(out.std()), 1e-6)


def fit_acf(lags, acf):
    """Two-scale texture: acf(l) ~ w exp(-l^2 / (4 s1^2)) + (1 - w) exp(-l^2 / (4 s2^2)) (Gaussian-
    filtered noise of width s has this correlation). Returns (w, s1, s2) by least squares on a grid."""
    lags, acf = np.asarray(lags, np.float64), np.asarray(acf, np.float64)
    best = None
    for s1 in np.arange(0.5, 6.01, 0.25):
        for s2 in (6, 8, 10, 12, 15, 20, 25, 30, 40, 60):
            a, b = np.exp(-lags ** 2 / (4 * s1 ** 2)), np.exp(-lags ** 2 / (4 * s2 ** 2))
            for w in np.arange(0, 1.001, 0.05):
                err = np.sum((w * a + (1 - w) * b - acf) ** 2)
                if best is None or err < best[0]:
                    best = (err, float(w), float(s1), float(s2))
    return best[1:]


class Look:
    """Rendering parameters of one appearance volume (realistic mode). Starting values come from its
    measured statistics; calibrate() then adjusts the core blur, halo weight, class gains,
    background, noise, texture and border contrast until the same measurement of synthetic volumes
    gives the measured values."""

    def __init__(self, appearance, spacing_z, ndim, depth=None):
        a = appearance
        self.C = len(a['background']['mean'])
        self.spacing_z, self.ndim = spacing_z, ndim
        haze = a.get('haze') or [dict(d0=0, d1=1e9, mean=a['background']['mean'])]
        # the background far from cells; light near them comes from the halo of the blur
        self.background = np.asarray(haze[-1]['mean'], np.float32)
        # Membranes: the 1 px rim of every cell body, so a contact between two cells is a 2 px line and
        # a free border a 1 px line of the same brightness. Fitted to the contrast at contacts; the
        # contrast at free borders depends on where the annotation put the border and is only compared.
        m = a.get('membranes', {})
        self.membrane = np.maximum(3 * np.nan_to_num(np.asarray(m.get('contact', [0.0] * self.C), np.float32)), 0)
        blur = a.get('blur') or dict(sigma_z=0.0, sigma_xy=1.0, halo=8.0, halo_weight=0.0)
        self.core_z = float(blur['sigma_z']) if ndim == 3 else 0.0
        self.core_xy = float(blur['sigma_xy'])
        self.halo = float(blur.get('halo', 8.0))
        self.halo_weight = float(np.clip(blur.get('halo_weight', 0.0), 0, 0.9))
        self.noise = {int(c): np.array(curve, np.float64) for c, curve in a['noise_by_intensity'].items()}
        self.texture = {}
        for name in ('cytoplasm', 'nucleus'):
            o = a[name]
            total = np.asarray(o['texture_rel'], np.float64)
            noise_rel = np.asarray(o.get('noise_sigma', [0] * self.C)) / np.maximum(np.asarray(o['mean']), 1)
            keep = np.clip((total ** 2 - noise_rel ** 2) / np.maximum(total ** 2, 1e-12), 1e-3, 1)   # texture share of the variance
            acf = o.get('texture_acf', {})
            rows = [np.asarray(acf[ax]) / keep[None] for ax in ('x', 'y') if acf.get(ax)]
            if rows:
                n = min(len(r) for r in rows)
                w, s1, s2 = fit_acf(np.arange(1, n + 1), np.mean([r[:n].mean(1) for r in rows], 0))
            else:
                w, s1, s2 = 1.0, 1.0, 10.0
            b = self.core_xy
            s1, s2 = np.sqrt(max(s1 ** 2 - b ** 2, 0.25)), np.sqrt(max(s2 ** 2 - b ** 2, 1.0))
            sz = 0.0
            if ndim == 3 and acf.get('z'):
                r1 = float(np.clip(np.mean(np.asarray(acf['z'])[0] / keep), 0.05, 0.98))
                sz = float(np.sqrt(max(-1 / (4 * np.log(r1)) - self.core_z ** 2, 0.04)))
            amp = np.sqrt(np.maximum(total ** 2 - noise_rel ** 2, 0)).astype(np.float32)
            # fibre: share of the fine texture drawn as streaks (4 x longer than wide), fitted to the coherence
            self.texture[name] = dict(amp=amp, weight=w, fine=(sz, s1, s1), coarse=(sz, s2, s2), fibre=0.0, fibre_ratio=4.0)
        self.depth_dz, self.depth = depth if depth else (None, None)
        self.gain = dict(cytoplasm=np.ones(self.C, np.float32), nucleus=np.ones(self.C, np.float32))

        self.scale = dict(texture=1.0, noise=1.0)

    def depth_factor(self, dz):
        if self.depth_dz is None:
            return np.ones(self.C, np.float32)
        return np.array([np.interp(dz, self.depth_dz, self.depth[:, c]) for c in range(self.C)], np.float32)

    def noise_sigma(self, image, c):
        curve = self.noise.get(c)
        if curve is None or not len(curve):
            return np.zeros_like(image)
        s = np.interp(image, curve[:, 0], curve[:, 1])
        return np.sqrt(np.maximum(s ** 2 - 1 / 12, 0)) * self.scale['noise']   # uint8 rounding adds 1/12

    def blurred(self, sharp):
        """Core Gaussian plus halo (physical sigma, Z sigma = halo / spacing_z) mixed by the halo weight."""
        w = float(np.clip(self.halo_weight, 0, 0.95))
        planar = sharp.shape[0] == 1
        out = np.empty_like(sharp)
        for c in range(sharp.shape[-1]):
            core = ndi.gaussian_filter(sharp[..., c], (0.0 if planar else self.core_z, self.core_xy, self.core_xy), mode='nearest')
            halo = ndi.gaussian_filter(sharp[..., c], (0.0 if planar else self.halo / self.spacing_z, self.halo, self.halo),
                                       mode='nearest') if w > 0 else 0
            out[..., c] = (1 - w) * core + w * halo
        return out

    def texture_of(self, name, shape, rng):
        """Unit-variance two-scale texture field of a class."""
        tx = self.texture[name]
        w = tx['weight']
        fine = texture_field(shape, tx['fine'], rng)
        if tx['fibre'] > 0:
            fine = np.sqrt(1 - tx['fibre']) * fine + np.sqrt(tx['fibre']) * fibre_field(shape, max(tx['fine'][-1], 0.7),
                                                                                          tx['fibre_ratio'], rng)
        f = np.sqrt(w) * fine + np.sqrt(1 - w) * texture_field(shape, tx['coarse'], rng)
        return f / max(float(f.std()), 1e-6)

    def load(self, params):
        """Set the calibrated values written by params() (render_params.json or the preview's .json)."""
        self.core_z, self.core_xy = float(params['core_z']), float(params['core_xy'])
        self.halo, self.halo_weight = float(params['halo']), float(params['halo_weight'])
        self.background = np.asarray(params['background'], np.float32)
        self.gain = {k: np.asarray(v, np.float32) for k, v in params['gain'].items()}
        self.scale = dict(params['scale'])
        self.membrane = np.asarray(params['membrane'], np.float32)
        for k, v in params['texture'].items():
            self.texture[k] = dict(amp=np.asarray(v['amp'], np.float32), weight=v['weight'], fine=tuple(v['fine']),
                                   coarse=tuple(v['coarse']), fibre=v.get('fibre', 0.0), fibre_ratio=v.get('fibre_ratio', 4.0))
        return self

    def params(self):
        return dict(core_z=self.core_z, core_xy=self.core_xy, halo=self.halo, halo_weight=self.halo_weight,
                    background=self.background.tolist(), gain={k: v.tolist() for k, v in self.gain.items()}, scale=dict(self.scale),
                    membrane=self.membrane.tolist(),
                    texture={k: dict(amp=v['amp'].tolist(), weight=v['weight'], fine=list(v['fine']), coarse=list(v['coarse']),
                                     fibre=v['fibre'], fibre_ratio=v['fibre_ratio']) for k, v in self.texture.items()})


def _normalised(labels, factors, base):
    """base[label] * factors / (mean of factors over the label): each object keeps its mean colour."""
    lab = labels.astype(np.int64)
    n = len(base)
    sums = np.stack([np.bincount(lab, factors[:, c], minlength=n) for c in range(factors.shape[1])], 1)
    counts = np.bincount(lab, minlength=n)[:, None]
    norm = np.where(counts > 0, sums / np.maximum(counts, 1), 1)
    return base[lab] * factors / np.maximum(norm[lab], 1e-3)


def render_realistic(cells, nuclei, nucleoli, cell_colours, nucleus_colours, look, rng):
    """Image (Z, Y, X, C) float of a labelled volume with the measured appearance: object colours x
    texture, membranes (2 px lines at the cell borders, contacts and free borders apart), nucleoli
    darker, depth profile, blur (core + halo), noise."""
    Z, C = cells.shape[0], look.C
    sharp = np.empty(cells.shape + (C,), np.float32)
    sharp[:] = look.background
    fg = cells > 0
    d_cell = np.zeros(cells.shape, np.float32)
    contact = np.zeros(cells.shape, bool)
    for z in range(Z):
        if fg[z].any():
            d_cell[z], contact[z] = _border_distance(cells[z].astype(np.int32))
    tex = look.scale['texture']
    body = fg & (nuclei == 0)
    if body.any():
        f = np.maximum(1 + tex * look.texture['cytoplasm']['amp'][None] * look.texture_of('cytoplasm', cells.shape, rng)[body][:, None], 0.05)
        base = np.zeros((int(cells.max()) + 1, C), np.float32)
        for k, col in cell_colours.items():
            base[k] = np.asarray(col) * look.gain['cytoplasm']
        sharp[body] = _normalised(cells[body], f, base)
    nsel = nuclei > 0
    if nsel.any():
        f = np.maximum(1 + tex * look.texture['nucleus']['amp'][None] * look.texture_of('nucleus', cells.shape, rng)[nsel][:, None], 0.05)
        base = np.zeros((int(nuclei.max()) + 1, C), np.float32)
        for k, col in nucleus_colours.items():
            base[k] = np.asarray(col) * look.gain['nucleus']
        sharp[nsel] = _normalised(nuclei[nsel], f, base)
    rim = body & (d_cell <= 1)
    sharp[rim] *= 1 + look.membrane
    if nucleoli is not None and nucleoli.any():
        dark = rng.uniform(*NUCLEOLUS['darkness'], int(nucleoli.max()) + 1).astype(np.float32)
        sel = nucleoli > 0
        sharp[sel] *= dark[nucleoli[sel]][:, None]
    # depth profile around the tissue centre of this volume
    if look.depth_dz is not None and fg.any() and Z > 1:
        zc = float(np.average(np.arange(Z), weights=fg.sum((1, 2)) + 1e-9))
        for z in range(Z):
            sharp[z][fg[z]] *= look.depth_factor(z - zc)
    img = look.blurred(sharp)
    for c in range(C):
        img[..., c] += rng.standard_normal(img.shape[:-1]).astype(np.float32) * look.noise_sigma(img[..., c], c).astype(np.float32)
    return img


def render_randomized(cells, nuclei, nucleoli, rng, noise=0.02, C=3):
    """Image (Z, Y, X, C) float in 0..255 with random class colours and blur, and Gaussian noise of
    amplitude `noise` x 255 (domain randomisation)."""
    while True:
        colours = rng.uniform(0, 1, (4, C))
        if all(np.abs(colours[a] - colours[b]).max() > 0.2 for a, b in ((0, 1), (1, 2), (2, 3))):
            break
    cls = (cells > 0).astype(np.int8) + (nuclei > 0) + (nucleoli > 0 if nucleoli is not None else 0)
    sharp = colours[cls].astype(np.float32)
    for labels in (cells, nuclei):
        jitter = rng.uniform(0.75, 1.25, (int(labels.max()) + 1, C)).astype(np.float32)
        sel = labels > 0
        sharp[sel] *= jitter[labels[sel]]
    if rng.random() < 0.7:
        d, _ = zip(*[_border_distance(c.astype(np.int32)) for c in cells])
        rimmask = (np.stack(d) <= rng.uniform(1, 3)) & (cells > 0)
        sharp[rimmask] += rng.uniform(-0.3, 0.5, C).astype(np.float32)
    T = texture_field(cells.shape, (0.5, 2, 2) if cells.shape[0] > 1 else (0, 2, 2), rng)
    sharp *= (1 + rng.uniform(0, 0.15) * T)[..., None]
    s = rng.uniform(0.5, 2.5)
    sharp = np.stack([ndi.gaussian_filter(sharp[..., c], (s * rng.uniform(0.3, 0.8) if cells.shape[0] > 1 else 0, s, s))
                      for c in range(C)], -1)
    sharp = np.clip(sharp, 0, None) * 255
    return sharp + rng.standard_normal(sharp.shape).astype(np.float32) * noise * 255


# ---------------------------------------------------------------------------------- volumes

def make_looks(profile, params=None):
    """Starting rendering parameters of every appearance volume of a profile; with `params` (a path or
    a dict {volume name: Look.params()}) the calibrated values of an earlier run."""
    looks = {vi: Look(profile.volumes[vi]['appearance'], profile.spacing_z, profile.ndim, profile.depth.get(vi))
             for vi in profile.by_volume}
    if params is not None:
        params = json.loads(Path(params).read_text()) if not isinstance(params, dict) else params
        for vi, look in looks.items():
            look.load(params[profile.volumes[vi]['name']])
    return looks


def make_volume(profile, shape, rng, appearance='realistic', noise=0.02, looks=None, max_overlap=0.3, volume_index=None,
                target=None):
    """One synthetic volume: dict(image uint8 (Z, Y, X, C), cells, nuclei, nucleoli, assignments,
    nucleolus_assignments, appearance volume index, target cell share). 2D volumes have Z = 1.
    Cells, their colours and the appearance come from one annotated volume (random unless given)."""
    vi = int(rng.choice(sorted(profile.by_volume))) if volume_index is None else volume_index
    t = pack_tile(profile, shape, rng, max_overlap=max_overlap, volume_index=vi, target=target)
    nucleoli, links = add_nucleoli(t['nuclei'], profile.spacing_z, rng)
    if appearance == 'realistic':
        look = looks[vi]
        a = profile.volumes[vi]['appearance']
        def colour(c, name):
            return np.asarray(c if c is not None else a[name]['mean'], np.float32) * np.exp(rng.normal(0, 0.03, look.C))
        cc = {k: colour(c, 'cytoplasm') for k, c in t['cell_colours'].items()}
        nc = {k: colour(c, 'nucleus') for k, c in t['nucleus_colours'].items()}
        img = render_realistic(t['cells'], t['nuclei'], nucleoli, cc, nc, look, rng)
    else:
        img = render_randomized(t['cells'], t['nuclei'], nucleoli, rng, noise)
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return dict(image=img, cells=t['cells'], nuclei=t['nuclei'], nucleoli=nucleoli, assignments=t['assignments'],
                nucleolus_assignments=links, volume_index=vi, target=t['target'])


def as_volume(v, profile, name='synthetic'):
    """A synthetic volume as measure_gt_statistics.Volume (everything is training data)."""
    return Volume(name, v['image'], v['cells'].astype(np.int32), v['nuclei'].astype(np.int32),
                  np.ones(v['cells'].shape[1:], bool), profile.spacing_z, profile.ndim, not profile.substrate,
                  {int(k): int(c) for k, c in v['assignments'].items()}, v['nucleoli'].astype(np.int32), v['nucleolus_assignments'])


def _pooled(appearances):
    """Mean of the appearance measurements that calibrate() compares, over several volumes."""
    C = len(appearances[0]['background']['mean'])
    nanmean = lambda xs: np.nanmean(np.asarray(xs, np.float64), 0).tolist() if xs else None
    out = {}
    for name in ('background', 'cytoplasm', 'nucleus'):
        out[name] = {k: nanmean([a[name][k] for a in appearances if a[name].get(k) is not None])
                     for k in ('mean', 'noise_sigma', 'texture_std', 'coherence')}
        if out[name]['coherence'] is not None:
            out[name]['coherence'] = float(out[name]['coherence'])
    out['sharpness'] = {}
    for axis in ('xy', 'z'):
        rows = [a['sharpness'][axis] for a in appearances if axis in a.get('sharpness', {})]
        if rows:
            n = min(len(r) for r in rows)
            out['sharpness'][axis] = nanmean([r[:n] for r in rows])
    out['blur'] = {'halo_weight': nanmean([a['blur']['halo_weight'] for a in appearances if 'blur' in a])}
    contact = [a['membranes']['contact'] for a in appearances if 'contact' in a.get('membranes', {})]
    out['membranes'] = {'contact': nanmean(contact)} if contact else {}
    far = [(a.get('haze') or [dict(mean=a['background']['mean'])])[-1]['mean'] for a in appearances]
    out['haze'] = [dict(d0=0.0, mean=nanmean(far) or [0.0] * C)]
    return out


def _update(look, real, synth, ndim):
    """One calibration step of `look` from the real and the synthetic appearance (see calibrate)."""
    clip = lambda x: float(np.clip(x, 0.5, 2.0)) if x is not None and np.isfinite(x) and x > 0 else 1.0
    div = lambda a, b: a / b if a is not None and b is not None and b != 0 else None
    mean = lambda f: (lambda xs: float(np.mean(xs)) if xs else None)(
        [x for x in map(f, synth) if x is not None and np.isfinite(x)])
    # sigma ~ ratio^-1/2 for a pure blur; texture and noise flatten the dependence, so the full ratio is used
    look.core_xy = float(np.clip(max(look.core_xy, 0.3) * clip(div(mean(lambda a: sharp_ratio(a, 'xy', 1, 4)),
                                                                   sharp_ratio(real, 'xy', 1, 4))), 0, 12))
    if ndim == 3:
        look.core_z = float(np.clip(max(look.core_z, 0.3) * clip(div(mean(lambda a: sharp_ratio(a, 'z', 0, 2)),
                                                                     sharp_ratio(real, 'z', 0, 2))), 0, 6))
    halo = div(real.get('blur', {}).get('halo_weight'), mean(lambda a: a.get('blur', {}).get('halo_weight')))
    look.halo_weight = float(np.clip(max(look.halo_weight, 0.05) * clip(halo), 0, 0.9))
    for name in ('cytoplasm', 'nucleus'):
        have = np.nanmean([a[name]['mean'] for a in synth], 0)
        want = np.asarray(real[name]['mean'])
        # channels with almost no signal (< 3 of 255) are left alone: their ratios are noise
        look.gain[name] *= np.where(np.isfinite(have) & (want >= 3), np.clip(want / np.maximum(have, 1), 0.5, 2),
                                    1).astype(np.float32)
    far_real = np.asarray((real.get('haze') or [dict(mean=real['background']['mean'])])[-1]['mean'])
    far_synth = np.mean([(a.get('haze') or [dict(mean=a['background']['mean'])])[-1]['mean'] for a in synth], 0)
    look.background = np.maximum(look.background + (far_real - far_synth), 0).astype(np.float32)
    # Noise by the background only: inside the cells the noise estimate also sees the fine texture.
    # A background without noise (clipped at zero, as in the Hollandi images) gets none.
    noise = lambda a: float(np.nanmean(a['background'].get('noise_sigma', [np.nan])))
    texture = lambda a: np.nanmean([a[c].get('texture_std', [np.nan]) for c in ('cytoplasm', 'nucleus')])
    if np.isfinite(noise(real)) and noise(real) < 0.05:
        look.scale['noise'] = 0.0
    else:
        look.scale['noise'] *= clip(div(noise(real), mean(noise)))
    look.scale['texture'] *= clip(div(texture(real), mean(texture)))
    want = np.asarray(real.get('membranes', {}).get('contact', [np.nan] * look.C), np.float64)
    have = [a['membranes']['contact'] for a in synth if 'contact' in a.get('membranes', {})]
    if have:
        # the contrast without membranes is not zero (neighbours differ): additive step
        step = np.nan_to_num(want - np.nanmean(np.asarray(have, np.float64), 0))
        look.membrane = np.clip(look.membrane + 2.5 * step, 0, 5).astype(np.float32)
    for name in ('cytoplasm', 'nucleus'):
        coh = mean(lambda a: a[name].get('coherence'))
        if coh is not None and real[name].get('coherence') is not None and np.isfinite(real[name]['coherence']):
            tx = look.texture[name]
            tx['fibre'] = float(np.clip(tx['fibre'] + 1.5 * (real[name]['coherence'] - coh), 0, 1))


def calibrate(profile, shape, seed, rounds=3, n=2, verbose=True, **packing):
    """Rendering parameters per appearance volume such that the measurement of synthetic volumes
    (same code as for the annotated data) matches: core blur by the structure function of the image
    (D(2) / D(16) in XY, D(1) / D(3) along Z: sigma x ratio synthetic / real), halo weight by the halo share of
    the edge fit against the labels, class gains by the class means, the share of streaky (fibre)
    texture by the coherence of the fine texture, far background, noise (by the background), texture
    amplitude (factors limited to 0.5..2 per round) and the membrane contrast (additive steps). (The
    background 8-32 px from the cells depends on how much tissue a volume holds, so it is compared,
    not fitted; so is the colour spread between objects: in the annotated volume it includes the
    change of colour over the whole depth of the spheroid, which one synthetic volume does not span.)
    Profiles with PROFILES[...]['pooled'] calibrate all their volumes together against the mean of
    their measurements (2 x n synthetic volumes per round): one image holds few cells."""
    looks = make_looks(profile)
    groups = [sorted(looks)] if PROFILES[profile.name].get('pooled') else [[vi] for vi in sorted(looks)]
    for r in range(rounds):
        for group in groups:
            count = n if len(group) == 1 else 2 * n
            synth = []
            for i in range(count):
                vi = group[i % len(group)]
                # calibration volumes hold the median cell share: nearly empty ones make the comparison noisy
                v = make_volume(profile, shape, np.random.default_rng([seed, 99, r, group[0], i]), 'realistic', looks=looks,
                                volume_index=vi, target=float(np.median(profile.window_shares)), **packing)
                synth.append(measure_volume(as_volume(v, profile), list(shape), geometry=False)['appearance'])
            if len(group) == 1:
                real = profile.volumes[group[0]]['appearance']
            else:
                real = _pooled([profile.volumes[vi]['appearance'] for vi in group])
                synth = [_pooled(synth)]
            for vi in group:
                _update(looks[vi], real, synth, profile.ndim)
            look = looks[group[0]]
            if verbose:
                print(f'calibration round {r + 1}, volume(s) {group}: core blur z {look.core_z:.2f} xy {look.core_xy:.2f}, '
                      f'halo weight {look.halo_weight:.2f}, gains {np.round(look.gain["cytoplasm"], 2).tolist()} '
                      f'{np.round(look.gain["nucleus"], 2).tolist()}, background {np.round(look.background, 1).tolist()}, '
                      f'membranes {np.round(look.membrane, 2).tolist()}, fibre {look.texture["cytoplasm"]["fibre"]:.2f} '
                      f'{look.texture["nucleus"]["fibre"]:.2f}, '
                      + ', '.join(f'{k} {v:.2f}' for k, v in look.scale.items()), flush=True)
    return looks


# ------------------------------------------------------------------------------- comparison

def comparison_table(real, synth):
    """Markdown rows: statistic, real (median [p5-p95] or share), synthetic."""
    def f(d, digits=2):
        if d is None:
            return '–'
        if isinstance(d, dict):
            return '–' if not d.get('n') else f"{d['median']:.{digits}f} [{d['p05']:.{digits}f}–{d['p95']:.{digits}f}]"
        if isinstance(d, (list, tuple)):
            return '(' + ', '.join(f'{x:.1f}' for x in d) + ')'
        return f'{d:.{digits}f}'
    rows = ['| статистика | реальные (train) | синтетика |', '|---|---|---|']
    for level, title in (('cells', 'клетки'), ('nuclei', 'ядра')):
        for k, name in (('volume', 'объём, воксели'), ('extent_z', 'экстент Z, срезы'), ('radius_xy', 'экв. радиус XY'),
                        ('elongation', 'вытянутость XY'), ('flatness', 'сплюснутость'), ('sphericity', 'сферичность / круглость'),
                        ('solidity', 'solidity'), ('mode2', 'мода 2 log r'), ('rough_low', 'шероховатость 3–8'),
                        ('rough_high', 'шероховатость 9–24'), ('nonstar', 'не-звёздные лучи')):
            a, b = real.get(level, {}).get(k), synth.get(level, {}).get(k)
            if a or b:
                rows.append(f'| {title}: {name} | {f(a, 0 if k == "volume" else 2)} | {f(b, 0 if k == "volume" else 2)} |')
    if 'nucleoli' in synth:
        for k, name in (('volume', 'объём, воксели'), ('radius_xy', 'экв. радиус XY')):
            rows.append(f"| ядрышки (приняты): {name} | – | {f(synth['nucleoli'].get(k), 1)} |")
    hr, hs = real['hierarchy'], synth['hierarchy']
    for k, name in (('multinucleated_share', 'доля многоядерных'), ('without_nucleus_share', 'доля без ядра'),
                    ('outside_share_voxels', 'вокселей ядер вне клетки')):
        rows.append(f'| {name} | {f(hr.get(k), 3)} | {f(hs.get(k), 3)} |')
    for k, name in (('volume_ratio', 'объём ядра / клетки'), ('offset_xy', 'смещение ядра XY / радиус')):
        rows.append(f'| {name} | {f(hr.get(k))} | {f(hs.get(k))} |')
    pr, ps = real['packing'], synth['packing']
    for k, name in (('touching_share', 'доля касающихся клеток'), ('clusters_per_cell', 'кластеров на клетку')):
        rows.append(f'| {name} | {f(pr.get(k))} | {f(ps.get(k))} |')
    for k, name in (('contact_fraction', 'доля границы на контакте'), ('nn_gap_ratio', 'зазор до ближайшего (отношение)'),
                    ('nn_distance', 'расстояние до ближайшего, px'), ('envelope_density', 'плотность в оболочке'),
                    ('window_cell_share', 'доля клеток в окне')):
        rows.append(f'| {name} | {f(pr.get(k))} | {f(ps.get(k))} |')
    if 'appearance' in real and 'appearance' in synth:
        ar, as_ = real['appearance'], synth['appearance']
        for cls in ('background', 'cytoplasm', 'nucleus'):
            for k, name in (('mean', 'среднее'), ('between_object_std', 'σ между объектами'), ('texture_std', 'текстура σ'),
                            ('noise_sigma', 'шум σ')):
                if k in ar[cls] or k in as_[cls]:
                    rows.append(f'| {cls}: {name} | {f(ar[cls].get(k))} | {f(as_[cls].get(k))} |')
            tl = lambda a: a[cls].get('texture_length', {}).get('x')
            if tl(ar) or tl(as_):
                rows.append(f'| {cls}: длина корреляции текстуры X, px | {f(tl(ar))} | {f(tl(as_))} |')
        for k, name in (('sigma_xy', 'края относительно GT: σ_xy ядра, px'), ('sigma_z', 'края относительно GT: σ_z ядра, срезы'),
                        ('halo_weight', 'края относительно GT: доля ореола')):
            rows.append(f"| {name} | {f(ar['blur'].get(k), 2)} | {f(as_['blur'].get(k), 2)} |")
        for cls in ('cytoplasm', 'nucleus'):
            if ar[cls].get('coherence') is not None or as_[cls].get('coherence') is not None:
                rows.append(f"| {cls}: когерентность тонкой текстуры | {f(ar[cls].get('coherence'))} | {f(as_[cls].get('coherence'))} |")
        for k, name in (('contact', 'мембраны: контраст на контактах клеток'), ('free', 'мембраны: контраст на свободной границе')):
            m = lambda a: a.get('membranes', {}).get(k)
            if m(ar) or m(as_):
                rows.append(f"| {name} | {f(m(ar))} | {f(m(as_))} |")
        if as_.get('nucleolus_contrast') is not None:
            rows.append(f"| ядрышки (приняты): яркость в ядрышке / в остальном ядре после размытия | – | {f(as_['nucleolus_contrast'])} |")
        for k, name in (('sharpness_xy', 'резкость: D(2)/D(16) по XY'), ('sharpness_z', 'резкость: D(1)/D(3) по Z'),
                        ('haze_excess', 'дымка: фон в 8–32 px от клеток минус дальний фон')):
            if ar.get(k) is not None or as_.get(k) is not None:
                rows.append(f"| {name} | {f(ar.get(k), 3)} | {f(as_.get(k), 3)} |")
    return '\n'.join(rows)


def preview(profile, volumes, path, reals=()):
    """Real and synthetic slices side by side: XY through the middle of the tissue (raw and with
    outlines: cells yellow, nuclei cyan, nucleoli red), XY 8 slices higher, XZ (raw and outlined)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    items = [(title, v) for title, v in reals] + [(f'синтетика {i + 1}', v) for i, v in enumerate(volumes)]
    cols = 5 if profile.ndim == 3 else 2
    fig, axes = plt.subplots(len(items), cols, figsize=(4.5 * cols, 4.5 * len(items)), squeeze=False)

    def overlay(img, labels):
        o = img.astype(np.float32) / 255
        for lab, colour in zip(labels, ((1, 1, 0), (0, 1, 1), (1, 0, 0))):
            if lab is not None:
                o[find_boundaries(lab, mode='inner')] = colour
        return o
    for row, (title, v) in zip(axes, items):
        img, cells, nuclei, nucleoli = v['image'], v['cells'], v['nuclei'], v.get('nucleoli')
        z = int(np.argmax((cells > 0).reshape(cells.shape[0], -1).sum(1)))
        row[0].imshow(img[z])
        row[0].set_title(f'{title}: XY, срез {z}')
        row[1].imshow(overlay(img[z], [cells[z], nuclei[z], None if nucleoli is None else nucleoli[z]]))
        row[1].set_title('клетки, ядра, ядрышки')
        if cols == 5:
            zu = max(z - 8, 0)
            row[2].imshow(img[zu])
            row[2].set_title(f'XY, срез {zu} (выше на {z - zu})')
            y = int(np.argmax((cells[z] > 0).sum(1)))
            rep = max(1, int(round(profile.spacing_z)))
            xz = lambda a: np.repeat(a[:, y], rep, axis=0)
            row[3].imshow(xz(img))
            row[3].set_title(f'XZ, y = {y} (Z ×{rep})')
            row[4].imshow(overlay(xz(img), [xz(cells), xz(nuclei), None if nucleoli is None else xz(nucleoli)]))
        for a in row:
            a.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=60)
    plt.close(fig)


def real_windows(profile_name, shape, quantiles=(0.75, 1.0)):
    """Windows of the real training data at the given quantiles of the cell share (1.0: the densest),
    for previews."""
    from measure_gt_statistics import SOURCES
    found = []
    shape3 = tuple(shape) if len(shape) == 3 else (1,) + tuple(shape)
    for v in SOURCES[profile_name]():
        fg = v.cells > 0
        wz = min(shape3[0], fg.shape[0])
        for z0 in range(0, fg.shape[0] - wz + 1, max(1, wz // 4)):
            for y0 in range(0, fg.shape[1] - shape3[1] + 1, shape3[1] // 4):
                for x0 in range(0, fg.shape[2] - shape3[2] + 1, shape3[2] // 4):
                    if v.train[y0:y0 + shape3[1], x0:x0 + shape3[2]].all():
                        sl = (slice(z0, z0 + wz), slice(y0, y0 + shape3[1]), slice(x0, x0 + shape3[2]))
                        found.append((float(fg[sl].mean()), v, sl))
    found.sort(key=lambda b: b[0])
    out = []
    for q in quantiles:
        share, v, sl = found[int(round(q * (len(found) - 1)))]
        out.append(dict(image=v.image[sl], cells=v.cells[sl], nuclei=v.nuclei[sl], nucleoli=None, name=v.name, share=share))
    return out


def compare_shape(shape):
    """Volumes for the comparison table: XY doubled, so that fewer objects are cut by the border
    (whole objects are compared; the annotated region is much larger than a tile)."""
    return tuple(shape[:-2]) + tuple(2 * s for s in shape[-2:])


def compared(profile, shape, seed, n, appearance, noise, looks, packing):
    """Summary of the measurement of n synthetic volumes of compare_shape(shape)."""
    big = compare_shape(shape)
    rows = []
    for i in range(n):
        v = make_volume(profile, big, np.random.default_rng([seed, 5, i]), appearance, noise, looks, **packing)
        rows.append(measure_volume(as_volume(v, profile), list(shape), appearance=True, donors=False))
    return summarize(rows), big


# ------------------------------------------------------------------------------------ output

def write_volume(root, split, stem, v, ndim):
    for sub in ('images', 'cytoplasm_masks', 'nucleus_masks', 'nucleolus_masks', 'assignments', 'assignments_nucleolus_nucleus'):
        (root / split / sub).mkdir(parents=True, exist_ok=True)
    squeeze = (lambda a: a[0]) if ndim == 2 else (lambda a: a)
    tifffile.imwrite(root / split / 'images' / f'{stem}.tiff', squeeze(v['image']), photometric='rgb')
    for sub, key in (('cytoplasm_masks', 'cells'), ('nucleus_masks', 'nuclei'), ('nucleolus_masks', 'nucleoli')):
        tifffile.imwrite(root / split / sub / f'{stem}.tiff', squeeze(v[key]).astype(np.uint16), photometric='minisblack')
    (root / split / 'assignments' / f'{stem}.json').write_text(json.dumps({str(k): int(c) for k, c in sorted(v['assignments'].items())}))
    (root / split / 'assignments_nucleolus_nucleus' / f'{stem}.json').write_text(
        json.dumps({str(k): int(c) for k, c in sorted(v['nucleolus_assignments'].items())}))
    extents = []
    for labels in (v['nuclei'], v['cells']):
        for box in ndi.find_objects(squeeze(labels)):
            if box is not None:
                extents.append([s.stop - s.start for s in box])
    return {'filename': f'{stem}.tiff', 'cells_median_extent_p': np.median(extents, 0).tolist() if extents else None,
            'rescale_factor': 1.0, 'n_nuclei': int(len(np.unique(v['nuclei'])) - 1), 'n_cells': int(len(np.unique(v['cells'])) - 1),
            'n_nucleoli': int(len(np.unique(v['nucleoli'])) - 1), 'target_cell_share': round(float(v['target']), 4)}


def generate(stats_path, profile_name, output, counts=(60, 8, 8), shape=None, seed=0, appearance='realistic', noise=0.02,
             max_overlap=0.3, compare=4, calibration_rounds=5, render_params=None):
    stats = json.loads(Path(stats_path).read_text())
    profile = Profile(stats, profile_name)
    shape = tuple(shape or PROFILES[profile_name]['shape'])
    if len(shape) != profile.ndim:
        raise ValueError(f'{profile_name} is {profile.ndim}D: --shape needs {profile.ndim} values')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty')
    output.mkdir(parents=True, exist_ok=True)
    packing = dict(max_overlap=max_overlap)
    looks = None
    if appearance == 'realistic':
        looks = make_looks(profile, render_params) if render_params else calibrate(profile, shape, seed, calibration_rounds, **packing)
    split_info = dict(generator='dataset_prep/synthetic/make_synthetic_from_stats.py', stats=str(Path(stats_path).resolve()),
                      profile=profile_name, shape=list(shape), seed=seed, appearance=appearance,
                      noise=noise if appearance != 'realistic' else None, packing=packing, nucleolus_assumed=NUCLEOLUS,
                      levels=['nucleolus', 'nucleus', 'cytoplasm'], rescale_factor=1.0, created=time.strftime('%Y-%m-%d %H:%M'),
                      render_params_from=str(render_params) if render_params else 'calibrate()')
    for split_id, (split, count) in enumerate(zip(('train', 'val', 'test'), counts)):
        meta, names = [], []
        for i in range(count):
            rng = np.random.default_rng([seed, split_id, i])
            v = make_volume(profile, shape, rng, appearance, noise, looks, **packing)
            stem = f'syn_{profile_name}_{split}_{i:04d}'
            meta.append(write_volume(output, split, stem, v, profile.ndim))
            names.append(stem + '.tiff')
            print(f'{split} {i + 1}/{count}: {meta[-1]["n_cells"]} cells, {meta[-1]["n_nuclei"]} nuclei, {meta[-1]["n_nucleoli"]} nucleoli',
                  flush=True)
        (output / split / 'meta.json').write_text(json.dumps(meta, indent=2))
        split_info[split] = names
    (output / 'split_info.json').write_text(json.dumps(split_info, indent=2))
    if looks:
        (output / 'render_params.json').write_text(json.dumps({profile.volumes[vi]['name']: l.params() for vi, l in looks.items()}, indent=2))
    report = [f'# Синтетика {profile_name}', '', usage_notes(profile, shape, counts, appearance, noise)]
    if compare:
        summary, big = compared(profile, shape, seed, compare, appearance, noise, looks, packing)
        report += ['', '## Сравнение с разметкой', '', f'Синтетика: {compare} отдельных объёмов {list(big)} с теми же параметрами '
                   '(XY вдвое больше, чтобы край реже обрезал целые объекты).', '', comparison_table(stats['datasets'][profile_name], summary)]
    (output / 'synthetic_report.md').write_text('\n'.join(report) + '\n')
    print('->', output)


def usage_notes(profile, shape, counts, appearance, noise):
    n = NUCLEOLUS
    return '\n'.join([
        f'Профиль {profile.name}: {profile.source["description"]} Объёмы {list(shape)}, train/val/test = {list(counts)}, '
        f'заливка: {"по измеренным статистикам с калибровкой" if appearance == "realistic" else f"случайная, шум {noise}"}.',
        '',
        'Использование: `data.dataset_dir` = эта папка; `training.train_region_crops` = false (папки train_regions нет). '
        'Два уровня — `model.levels` ["nucleus", "cytoplasm"] (nucleolus_masks не читаются); три уровня — '
        '["nucleolus", "nucleus", "cytoplasm"] (пары берут assignments и assignments_nucleolus_nucleus).',
        '',
        'Измерено по разметке (train): формы клеток и ядер (радиальная функция по срезам и её спектр), размеры, ядра на клетку, '
        'смещение ядер, упаковка (доля клеток, зазоры, направления к соседям, кластеры, плотность), цвета объектов, '
        'текстура, размытие, шум, мембраны, профиль по глубине.',
        '',
        f'Принято без измерений (разметки ядрышек нет): ядрышек на ядро 1 + Пуассон({n["count_mean"]}), не больше {n["count_max"]}; '
        f'экв. радиус {n["radius_ratio"][0]}–{n["radius_ratio"][1]} радиуса ядра; вытянутость {n["elongation"][0]}–{n["elongation"][1]}; '
        f'яркость {n["darkness"][0]}–{n["darkness"][1]} яркости ядра; отступ от края ядра {n["margin"]} px. '
        'Ядро целиком внутри своей клетки, ядрышко — внутри своего ядра (в разметке '
        f'{profile.summary["hierarchy"]["outside_share_voxels"]:.1%} вокселей ядер лежат вне своей клетки).'])


def run_preview(stats_path, profile_name, path, n=3, shape=None, seed=0, appearance='realistic', noise=0.02, max_overlap=0.3,
                calibration_rounds=5, compare=4, render_params=None):
    """Preview figure and comparison table of a few synthetic volumes, without writing a dataset."""
    stats = json.loads(Path(stats_path).read_text())
    profile = Profile(stats, profile_name)
    shape = tuple(shape or PROFILES[profile_name]['shape'])
    packing = dict(max_overlap=max_overlap)
    t0 = time.time()
    looks = None
    if appearance == 'realistic':
        looks = make_looks(profile, render_params) if render_params else calibrate(profile, shape, seed, calibration_rounds, **packing)
    volumes = [make_volume(profile, shape, np.random.default_rng([seed, 7, i]), appearance, noise, looks, **packing) for i in range(n)]
    print(f'{len(volumes)} volumes in {time.time() - t0:.0f} s', flush=True)
    reals = real_windows(profile_name, shape)
    preview(profile, volumes, path, [(f'реальные (train), доля клеток {r["share"]:.2f}', r) for r in reals])
    if looks:
        Path(path).with_suffix('.json').write_text(json.dumps({profile.volumes[vi]['name']: l.params() for vi, l in looks.items()}, indent=2))
    if compare:
        summary, big = compared(profile, shape, seed, compare, appearance, noise, looks, packing)
        table = comparison_table(stats['datasets'][profile_name], summary)
        Path(path).with_suffix('.md').write_text(f'# Превью синтетики {profile_name}\n\nСинтетика: {compare} объёмов {list(big)}.\n\n{table}\n')
        print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stats', required=True)
    parser.add_argument('--profile', default='hela', choices=list(PROFILES))
    parser.add_argument('--output')
    parser.add_argument('--counts', type=int, nargs=3, default=(60, 8, 8), metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--shape', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--appearance', default='realistic', choices=('realistic', 'randomized'))
    parser.add_argument('--noise', type=float, default=0.02, help='randomized mode: Gaussian noise sigma / 255')
    parser.add_argument('--max-overlap', type=float, default=0.3)
    parser.add_argument('--calibration-rounds', type=int, default=5)
    parser.add_argument('--render-params', help='JSON of calibrated rendering parameters (render_params.json of a dataset or '
                                                    'the .json of a preview) used instead of a new calibration')
    parser.add_argument('--preview', help='PNG: preview and comparison of a few volumes instead of a dataset')
    parser.add_argument('--compare', type=int, default=4, help='synthetic volumes (XY doubled) measured for the comparison table')
    args = parser.parse_args()
    common = dict(shape=args.shape, seed=args.seed, appearance=args.appearance, noise=args.noise, max_overlap=args.max_overlap,
                  calibration_rounds=args.calibration_rounds, compare=args.compare, render_params=args.render_params)
    if args.preview:
        run_preview(args.stats, args.profile, args.preview, **common)
    elif args.output:
        generate(args.stats, args.profile, args.output, counts=args.counts, **common)
    else:
        parser.error('--output or --preview is needed')


if __name__ == '__main__':
    main()
