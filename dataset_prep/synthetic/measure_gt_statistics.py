"""Statistics of the annotated cells and nuclei and of their images, for synthetic CytoNucs data.

Measured on training data only:
  hela      the HeLa spheroid at half XY resolution (patches_split_hela_xy2/train_regions); voxels
            outside the train blocks of regions.json are masked before anything is measured.
  ea        the three EA.hy926 volumes, downscaled x2 in XY like the HeLa set; voxels outside the
            blocks that patches_split/split_manifest.json gives to 'train' are masked. No current
            split (patches_split_hela, patches_split_hela_xy2) holds EA.hy926, so its masked val and
            test blocks stay clean for an independent evaluation.
  hollandi  the 2D set (RGB 2048 x 2048, two levels) without HOLLANDI_HELD_OUT, the validation
            images of the StarDist 2D runs in models/cytonucs_hollandi*_stardist_2d.

Per level: size, flatness, elongation, XY orientation, sphericity (circularity in 2D), solidity,
the outline of every slice as a radial function r(phi) around the slice centroid with the power of
the Fourier modes of log r (mode 2 = elongation, 3-8 = low-frequency roughness, 9-24 = high-
frequency roughness), and the share of rays along which the slice is not star-convex. Packing:
cell share of fixed-size windows, density inside the tissue envelope, nearest neighbours (distance,
direction, gap ratio = centre distance / sum of the two radii along it), contacts, clusters.
Hierarchy: nuclei per cell, nucleus / cell volume, nucleus offset, nucleus voxels outside the cell.
Appearance per class and channel: means, spread between objects and within them (texture) with its
correlation length, noise (Immerkaer estimator by intensity), blur (edge width of the image against
the blurred piecewise-constant label image), profiles at the cell border, around and inside the
nucleus, depth profile along Z, background against the distance to the nearest cell (haze).

Whole objects (no voxel on a masked block or on the volume border) are the donors from which
make_synthetic_from_stats.py draws new objects. measure_volume() is also used there to compare the
synthetic data with these numbers.

Writes <output>/gt_statistics.json and <output>/gt_statistics_report.md.
Usage:
  python dataset_prep/synthetic/measure_gt_statistics.py --output DIR [--datasets hela ea hollandi] [--update]
"""
import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

EMESE = Path('data/Cytonucs_Emese_Hela')
HOLLANDI = Path('data/Cytonucs_Hollandi')
HELA_REGIONS = EMESE / 'patches_split_hela_xy2' / 'train_regions'
EA_MANIFEST = EMESE / 'patches_split' / 'split_manifest.json'
HOLLANDI_HELD_OUT = ('Image0019.tiff', 'Image0036.tiff')
HELA_SPACING_Z = 2.5          # Z step / XY pixel at half XY resolution (model.anisotropy of v5)

K_RAYS = 64                   # rays of one slice outline
N_MODES = 24                  # Fourier modes of log r(phi) kept: k = 0..N_MODES
STEP = 0.5                    # ray sampling step, px
PHI = 2 * np.pi * np.arange(K_RAYS) / K_RAYS
LOW_MODES, HIGH_MODES = range(3, 9), range(9, N_MODES + 1)
PROFILE_BINS = 13             # distance profiles at 1..12 px from a border (and >= 13)
MIN_WINDOW_SHARE = 0.02      # windows with fewer cell voxels count as empty (the generator draws from the others)


@dataclass
class Volume:
    """One annotated image. 2D images have one slice (Z = 1). Labels are 0 outside training data."""
    name: str
    image: np.ndarray                       # (Z, Y, X, C) uint8
    cells: np.ndarray                       # (Z, Y, X) int32
    nuclei: np.ndarray
    train: np.ndarray                       # (Y, X) bool, training pixels
    spacing_z: float = 1.0                  # Z step in XY pixels
    ndim: int = 3
    z_cut: bool = True                      # an object on the first or last slice is cut
    assignments: dict = None                # nucleus -> cell; by overlap when None
    nucleoli: np.ndarray = None
    nucleolus_assignments: dict = None      # nucleolus -> nucleus
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------- sources

def _masked(volume):
    for labels in (volume.cells, volume.nuclei):
        labels[:, ~volume.train] = 0
    return volume


def load_hela(root=HELA_REGIONS):
    entries = json.loads((root / 'regions.json').read_text())
    for name, entry in entries.items():
        image = tifffile.imread(root / 'images' / name)
        cells = tifffile.imread(root / 'cytoplasm_masks' / name).astype(np.int32)
        nuclei = tifffile.imread(root / 'nucleus_masks' / name).astype(np.int32)
        train = np.zeros(cells.shape[1:], bool)
        for y0, y1, x0, x1 in entry['train_blocks']:
            train[y0:y1, x0:x1] = True
        pairs = root / 'assignments' / (Path(name).stem + '.json')
        assignments = {int(k): int(v) for k, v in json.loads(pairs.read_text()).items()} if pairs.is_file() else None
        yield _masked(Volume(name, image, cells, nuclei, train, HELA_SPACING_Z, 3, True, assignments,
                             info=dict(train_blocks=entry['train_blocks'])))


def load_ea(factor=2, manifest=EA_MANIFEST, spacing_z=HELA_SPACING_Z):
    """EA.hy926 volumes at 1/factor XY resolution. The voxel size is not in the files; the Z step is
    taken as in the HeLa volume (assumed). Cells lie on the coverslip and span the whole stack, so an
    object on the first or last slice is not counted as cut."""
    from downscale_split import downscale_image, downscale_labels
    blocks = {v['name']: v['blocks'] for v in json.loads(Path(manifest).read_text())['volumes']}
    for path in sorted((EMESE / 'images').glob('2D_EA*.tif')):
        key = path.stem.replace('.', '_').replace('(', '_').replace(')', '')
        image = tifffile.imread(path)
        cells = tifffile.imread(EMESE / 'cyto_masks' / path.name).astype(np.int32)
        nuclei = tifffile.imread(EMESE / 'nuc_masks' / path.name).astype(np.int32)
        train = np.zeros(cells.shape[1:], bool)
        used = [b for b in blocks[key] if b['split'] == 'train']
        for b in used:
            train[b['y0']:b['y1'], b['x0']:b['x1']] = True
        if factor > 1:
            image, cells, nuclei = downscale_image(image, factor), downscale_labels(cells, factor), downscale_labels(nuclei, factor)
            train = train[::factor, ::factor]
        yield _masked(Volume(key, image, cells, nuclei, train, spacing_z, 3, False,
                             info=dict(train_blocks=[[b['y0'] // factor, b['y1'] // factor, b['x0'] // factor, b['x1'] // factor]
                                                     for b in used], xy_downscale=factor)))


def load_hollandi(root=HOLLANDI, held_out=HOLLANDI_HELD_OUT, factor=1):
    from downscale_split import downscale_image, downscale_labels
    for path in sorted((root / 'images').glob('*.tif*')):
        if path.name in held_out:
            continue
        image = tifffile.imread(path)[None]
        cells = tifffile.imread(root / 'cyto_masks' / path.name).astype(np.int32)[None]
        nuclei = tifffile.imread(root / 'nuc_masks' / path.name).astype(np.int32)[None]
        if factor > 1:
            image, cells, nuclei = downscale_image(image, factor), downscale_labels(cells, factor), downscale_labels(nuclei, factor)
        yield Volume(path.stem, image[..., :3], cells, nuclei, np.ones(cells.shape[1:], bool), 1.0, 2, False,
                     info=dict(xy_downscale=factor))


SOURCES = dict(hela=load_hela, ea=load_ea, hollandi=load_hollandi)


# --------------------------------------------------------------------------------- outlines

def radial_function(mask2d, center):
    """r(phi) of one slice: distance from `center` to the outermost object pixel along each of K_RAYS
    rays (+ STEP / 2), and per ray whether it leaves the object before that pixel (not star-convex)."""
    h, w = mask2d.shape
    t = np.arange(0, np.hypot(h, w) + 1, STEP)
    y = np.rint(center[0] + np.sin(PHI)[:, None] * t).astype(int)
    x = np.rint(center[1] + np.cos(PHI)[:, None] * t).astype(int)
    ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    inside = np.zeros(y.shape, bool)
    inside[ok] = mask2d[y[ok], x[ok]]
    last = np.where(inside.any(1), inside.shape[1] - 1 - np.argmax(inside[:, ::-1], 1), 0)
    gap = (np.cumsum(inside[:, ::-1], 1)[:, ::-1] > 0) & ~inside
    return t[last] + STEP / 2, gap.any(1)


def log_radius_modes(r):
    """Complex Fourier modes c_0..c_N_MODES of log r(phi): log r = sum_k Re(c_k exp(i k phi))."""
    c = np.fft.rfft(np.log(np.maximum(r, STEP))) / K_RAYS
    c[1:] *= 2
    return c[:N_MODES + 1]


def radius_from_modes(modes, phi):
    k = np.arange(len(modes))
    return np.exp(np.real(np.exp(1j * np.multiply.outer(phi, k)) @ np.asarray(modes)))


def mode_power(modes):
    """Variance of log r carried by each mode k >= 1."""
    return np.abs(np.asarray(modes)[1:]) ** 2 / 2


def slice_centre(m):
    """Centre of the radial function of one slice: its centroid when that lies at least half the
    largest inner distance from the border, else the innermost pixel nearest to the centroid (a
    crescent seen from its centroid would give a radial function with spikes)."""
    sy, sx = np.nonzero(m)
    c = np.array([sy.mean(), sx.mean()])
    d = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
    iy, ix = int(round(c[0])), int(round(c[1]))
    if d[iy, ix] >= 0.5 * d.max():
        return c
    top = np.argwhere(d >= d.max() - 0.5)
    return top[np.argmin(((top - c) ** 2).sum(1))].astype(np.float64)


def outline_record(mask):
    """Slices of one object (mask: (nz, h, w) bool crop): the XY centroid of the object, and per slice
    with pixels its index, the offset of its centre (slice_centre) from the object centroid, area,
    the log-radius modes and the share of non-star-convex rays."""
    zz, yy, xx = np.nonzero(mask)
    centre = np.array([yy.mean(), xx.mean()])
    slices = []
    for z in range(mask.shape[0]):
        m = mask[z]
        area = int(m.sum())
        if not area:
            continue
        c = slice_centre(m)
        r, gap = radial_function(m, c)
        slices.append(dict(z=z, offset=c - centre, area=area, modes=log_radius_modes(r), gap=float(gap.mean())))
    return centre, slices


# --------------------------------------------------------------------------------- features

def _hull_volume(mask, spacing_z):
    """Volume (area in 2D) of the convex hull of the voxel corners, in XY-pixel units."""
    from scipy.spatial import ConvexHull, QhullError
    edge = mask & ~ndi.binary_erosion(mask)
    pts = np.argwhere(edge).astype(np.float64)
    planar = mask.shape[0] == 1
    corners = np.array([(dz, dy, dx) for dz in (-.5, .5) for dy in (-.5, .5) for dx in (-.5, .5)])
    if planar:
        pts, corners = pts[:, 1:], np.unique(corners[:, 1:], axis=0)
    pts = (pts[:, None, :] + corners[None]).reshape(-1, pts.shape[1])
    if not planar:
        pts[:, 0] *= spacing_z
    try:
        return ConvexHull(pts).volume
    except QhullError:
        return np.nan


def shape_features(mask, spacing_z, ndim):
    """Descriptors of one object (mask: (nz, h, w) bool crop) compared between real and synthetic data."""
    from skimage.measure import find_contours, marching_cubes, mesh_surface_area
    n = int(mask.sum())
    zz, yy, xx = np.nonzero(mask)
    areas = mask.reshape(mask.shape[0], -1).sum(1)
    cov = np.cov(np.stack([yy, xx])) if n > 2 else np.eye(2)
    lam, vec = np.linalg.eigh(cov)
    lam = np.maximum(lam, 1e-6)
    major = vec[:, 1]
    f = dict(volume=n, extent_z=int(np.ptp(zz) + 1), extent_y=int(np.ptp(yy) + 1), extent_x=int(np.ptp(xx) + 1),
             radius_xy=float(np.sqrt(areas.max() / np.pi)),
             semi_major_xy=float(2 * np.sqrt(lam[1])), semi_minor_xy=float(2 * np.sqrt(lam[0])),
             elongation=float(np.sqrt(lam[1] / lam[0])),
             orientation=float(np.degrees(np.arctan2(major[0], major[1])) % 180))
    hull = _hull_volume(mask, spacing_z)
    if ndim == 3:
        v = n * spacing_z
        f['radius_3d'] = float((3 * v / (4 * np.pi)) ** (1 / 3))
        f['semi_z'] = float(np.sqrt(5 * np.var(zz * spacing_z))) if n > 1 else spacing_z / 2
        f['flatness'] = f['semi_z'] / np.sqrt(f['semi_major_xy'] * f['semi_minor_xy'])
        verts, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5, spacing=(spacing_z, 1, 1))
        f['sphericity'] = float(np.pi ** (1 / 3) * (6 * v) ** (2 / 3) / mesh_surface_area(verts, faces))
        f['solidity'] = float(v / hull) if hull > 0 else np.nan
    else:
        p = sum(np.hypot(*np.diff(c, axis=0).T).sum() for c in find_contours(np.pad(mask[0], 1).astype(np.float32), 0.5))
        f['sphericity'] = float(4 * np.pi * n / p ** 2) if p > 0 else np.nan
        f['solidity'] = float(n / hull) if hull > 0 else np.nan
    _, slices = outline_record(mask)
    largest = max(slices, key=lambda s: s['area'])
    power = mode_power(largest['modes'])
    f.update(mode1=float(np.abs(largest['modes'][1])), mode2=float(np.abs(largest['modes'][2])),
             rough_low=float(np.sqrt(power[[k - 1 for k in LOW_MODES]].sum())),
             rough_high=float(np.sqrt(power[[k - 1 for k in HIGH_MODES]].sum())),
             nonstar=float(np.mean([s['gap'] for s in slices])))
    return f, slices


# ----------------------------------------------------------------------------- per volume

def whole_labels(labels, train, z_cut):
    """Labels with no voxel on the image border, on the border of the training area or (z_cut) on
    the first or last slice."""
    forbidden = ndi.binary_dilation(~train, np.ones((3, 3), bool), border_value=1)
    forbidden[[0, -1], :] = forbidden[:, [0, -1]] = True
    cut = set(np.unique(labels[:, forbidden]).tolist())
    if z_cut and labels.shape[0] > 1:
        cut |= set(np.unique(labels[[0, -1]]).tolist())
    present = set(np.unique(labels).tolist()) - {0}
    return sorted(present - cut)


def overlap_assignments(inner, outer):
    """inner label -> the outer label covering most of its voxels (absent when none does)."""
    both = (inner > 0) & (outer > 0)
    if not both.any():
        return {}
    pairs, counts = np.unique(np.stack([inner[both], outer[both]]), axis=1, return_counts=True)
    best = {}
    for (a, b), c in zip(pairs.T, counts):
        if c > best.get(int(a), (0, 0))[1]:
            best[int(a)] = (int(b), int(c))
    return {a: b for a, (b, c) in best.items()}


def face_contacts(labels):
    """Pairs (a, b), a < b, of labels sharing a voxel face, and per label its border voxel count
    and the count of those touching another label."""
    pairs = set()
    border = {}
    touch = {}
    nd = labels.ndim
    for axis in range(nd):
        if labels.shape[axis] < 2:
            continue
        a = np.moveaxis(labels, axis, 0)
        lo, hi = a[:-1], a[1:]
        diff = lo != hi
        for x, y in ((lo, hi), (hi, lo)):
            sel = diff & (x > 0)
            u, c = np.unique(x[sel], return_counts=True)
            for k, v in zip(u, c):
                border[int(k)] = border.get(int(k), 0) + int(v)
            sel &= y > 0
            u, c = np.unique(x[sel], return_counts=True)
            for k, v in zip(u, c):
                touch[int(k)] = touch.get(int(k), 0) + int(v)
        sel = diff & (lo > 0) & (hi > 0)
        if sel.any():
            p = np.unique(np.sort(np.stack([lo[sel], hi[sel]], 1), 1), axis=0)
            pairs.update(map(tuple, p.tolist()))
    return pairs, border, touch


def ray_extent(labels, label, centre, direction, spacing_z, limit):
    """Distance from `centre` (voxel coordinates) to the outermost voxel of `label` along a physical
    direction (unit vector, XY-pixel units)."""
    t = np.arange(0, limit, STEP)
    p = centre[None] + t[:, None] * (direction / np.array([spacing_z, 1, 1]))[None]
    idx = np.rint(p).astype(int)
    ok = np.all((idx >= 0) & (idx < labels.shape), 1)
    inside = np.zeros(len(t), bool)
    inside[ok] = labels[tuple(idx[ok].T)] == label
    return float(t[len(t) - 1 - np.argmax(inside[::-1])] + STEP / 2) if inside.any() else 0.0


def tissue_envelope(fg, radius, spacing_z):
    """Tissue around the cells: closing of the cell union with an ellipsoid of `radius` px in XY
    (radius / spacing_z slices in Z), holes filled per slice. Computed on a grid coarsened so that
    the radius is about 4 cells of it in XY (an exact closing with a large ball is very slow)."""
    if not fg.any():
        return fg.copy()
    f = max(1, int(radius // 4))
    Z, H, W = fg.shape
    hp, wp = -(-H // f) * f, -(-W // f) * f
    small = np.pad(fg, [(0, 0), (0, hp - H), (0, wp - W)]).reshape(Z, hp // f, f, wp // f, f).any((2, 4))
    r = max(1.0, radius / f)
    rz = int(round(radius / spacing_z)) if Z > 1 else 0
    ri = int(np.ceil(r))
    g = np.ogrid[-rz:rz + 1, -ri:ri + 1, -ri:ri + 1]
    ball = (g[0] * spacing_z / max(radius, 1)) ** 2 + (g[1] / r) ** 2 + (g[2] / r) ** 2 <= 1
    pad = [(rz, rz), (ri, ri), (ri, ri)]
    closed = ndi.binary_closing(np.pad(small, pad), ball)[rz:rz + Z, ri:ri + small.shape[1], ri:ri + small.shape[2]]
    closed = np.stack([ndi.binary_fill_holes(s) for s in closed])
    env = np.repeat(np.repeat(closed, f, 1), f, 2)[:, :H, :W]
    return env | fg


def packing_stats(v, whole_cells, centres, radii, window):
    """Contacts, clusters, nearest neighbours, envelope density and window cell shares."""
    cells, sz = v.cells, v.spacing_z
    pairs, border, touch = face_contacts(cells)
    labels = sorted(set(np.unique(cells).tolist()) - {0})
    neighbours = {k: set() for k in labels}
    for a, b in pairs:
        neighbours[a].add(b)
        neighbours[b].add(a)
    # clusters: connected components of the contact graph of all cells in the training area
    parent = {k: k for k in labels}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for a, b in pairs:
        parent[root(a)] = root(b)
    sizes = np.unique([root(k) for k in labels], return_counts=True)[1] if labels else np.array([])
    out = dict(n_cells=len(labels), n_whole=len(whole_cells), n_clusters=int(len(sizes)),
               cluster_sizes=sorted(sizes.tolist(), reverse=True),
               touching=[len(neighbours[k]) for k in whole_cells],
               contact_fraction=[touch.get(k, 0) / max(border.get(k, 1), 1) for k in whole_cells])
    # nearest neighbours of whole cells among all cells (physical distance between centroids)
    keys = list(centres)
    pts = np.array([centres[k] for k in keys]) * np.array([sz, 1, 1]) if keys else np.zeros((0, 3))
    nn = []
    if len(keys) > 1:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        for k in whole_cells:
            i = keys.index(k)
            d, j = tree.query(pts[i], 2)
            j, d = j[1], d[1]
            u = (pts[j] - pts[i]) / max(d, 1e-6)
            limit = 3 * d + 10
            ri = ray_extent(cells, k, np.array(centres[k]), u, sz, limit)
            rj = ray_extent(cells, keys[j], np.array(centres[keys[j]]), -u, sz, limit)
            nn.append(dict(distance=float(d), radius_sum=float(radii[k] + radii[keys[j]]),
                           gap_ratio=float(d / (ri + rj)) if ri + rj > 0 else np.nan,
                           elevation=float(np.degrees(np.arcsin(min(1.0, abs(u[0]))))),
                           touching=bool(keys[j] in neighbours[k])))
    out['nearest'] = nn
    # envelope: closing of the cell union with an ellipsoid of half the median cell radius
    fg = cells > 0
    env = tissue_envelope(fg, 0.5 * np.median(list(radii.values())) if radii else 1.0, sz)
    env[:, ~v.train] = False
    out['envelope_density'] = float(fg[env].mean()) if env.any() else np.nan
    out['envelope_fraction'] = float(env[:, v.train].mean())
    out['cell_fraction'] = float(fg[:, v.train].mean())
    out['cells_per_megavoxel_envelope'] = float(len(labels) / max(env.sum(), 1) * 1e6)
    # cell share of windows lying in the training area
    wz, wy, wx = (min(window[0], cells.shape[0]),) + tuple(window[1:]) if v.ndim == 3 else (1,) + tuple(window)
    shares = []
    for z0 in range(0, cells.shape[0] - wz + 1, max(1, wz // 4)):
        for y0 in range(0, cells.shape[1] - wy + 1, wy // 2):
            for x0 in range(0, cells.shape[2] - wx + 1, wx // 2):
                if v.train[y0:y0 + wy, x0:x0 + wx].all():
                    shares.append(float(fg[z0:z0 + wz, y0:y0 + wy, x0:x0 + wx].mean()))
    out['window'] = [wz, wy, wx]
    out['window_cell_share'] = shares
    return out


def _profile(dist, values, weights=None):
    """Mean of values (n, C) per integer distance bin 1..PROFILE_BINS (the last bin holds the rest)."""
    b = np.clip(np.rint(dist).astype(int), 1, PROFILE_BINS) - 1
    s = np.zeros((PROFILE_BINS, values.shape[1]))
    n = np.bincount(b, minlength=PROFILE_BINS).astype(np.float64)
    for c in range(values.shape[1]):
        s[:, c] = np.bincount(b, values[:, c], minlength=PROFILE_BINS)
    return s, n


def immerkaer_sigma(image2d, mean2d):
    """Per-pixel |response| of the Immerkaer noise kernel (sigma = median / (0.6745 * 6) for white
    Gaussian noise) and the local 3 x 3 mean, without the outermost pixels."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    return np.abs(ndi.convolve(image2d, k, mode='reflect'))[1:-1, 1:-1], mean2d[1:-1, 1:-1]


CORE_XY = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
CORE_Z = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
HALO = (6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)
BLUR_REACH = 32               # px around the cells where the blur is fitted (the halo reaches that far)


def edge_blur(image, model, region, spacing_z, ndim, rng=None, samples=2_000_000):
    """Blur of the image against the piecewise-constant image of the labels (`model`: object
    means): image ~ a + b * G_core * model + c * G_halo * model inside `region`, by the best R^2 of
    the linear fit (b, c >= 0) over a grid of the core sigmas (Z slices, XY px) and the halo sigma
    (physical, XY px; Z sigma = halo / spacing_z). The halo is the out-of-focus light around the
    objects (halo_weight = c / (b + c)); `region` reaches BLUR_REACH px beyond the objects so that
    the halo is seen. A single Gaussian is reported too."""
    rng = rng or np.random.default_rng(0)
    gray, flat = image.mean(-1), model.mean(-1).astype(np.float32)
    idx = np.flatnonzero(region)
    if len(idx) > samples:
        idx = np.sort(rng.choice(idx, samples, replace=False))
    y = gray.ravel()[idx].astype(np.float64)
    z_ok = ndim == 3 and image.shape[0] > 1
    halos = {h: ndi.gaussian_filter(flat, (h / spacing_z if z_ok else 0, h, h)).ravel()[idx].astype(np.float64) for h in HALO}
    best, single = None, None
    for cxy in CORE_XY:
        bxy = ndi.gaussian_filter(flat, (0, cxy, cxy)) if cxy > 0 else flat
        for cz in (CORE_Z if z_ok else (0.0,)):
            core = (ndi.gaussian_filter1d(bxy, cz, axis=0) if cz > 0 else bxy).ravel()[idx].astype(np.float64)
            r2 = np.corrcoef(core, y)[0, 1] ** 2
            if single is None or r2 > single[2]:
                single = (cz, cxy, r2)
            for h, halo in halos.items():
                X = np.stack([np.ones_like(core), core, halo], 1)
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                if coef[1] < 0 or coef[2] < 0:
                    continue                                # not a blur
                r2 = 1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
                if best is None or r2 > best['r2']:
                    best = dict(sigma_z=float(cz), sigma_xy=float(cxy), halo=float(h), r2=float(r2), offset=float(coef[0]),
                                core_gain=float(coef[1]), halo_gain=float(coef[2]),
                                halo_weight=float(coef[2] / (coef[1] + coef[2])) if coef[1] + coef[2] else 0.0)
    best = best or dict(sigma_z=single[0], sigma_xy=single[1], halo=HALO[0], r2=single[2], halo_weight=0.0)
    best['single'] = dict(sigma_z=float(single[0]), sigma_xy=float(single[1]), r2=float(single[2]))
    return best


SHARP_XY = (1, 2, 4, 8, 16)
SHARP_Z = (1, 2, 3)


def sharpness(image, inside):
    """Structure function D(l) = mean (I(x + l) - I(x))^2 of the gray image over pairs of voxels
    inside the cells, for lags SHARP_XY along Y and X and SHARP_Z along Z. Its fall towards small
    lags measures the blur from the image alone (no annotation involved)."""
    gray = image.mean(-1)
    out = {}
    for name, axes, lags in (('xy', (1, 2), SHARP_XY), ('z', (0,), SHARP_Z)):
        rows = []
        for lag in lags:
            num = den = 0.0
            for ax in axes:
                if gray.shape[ax] <= lag:
                    continue
                a = [slice(None)] * 3
                b = [slice(None)] * 3
                a[ax], b[ax] = slice(0, -lag), slice(lag, None)
                both = inside[tuple(a)] & inside[tuple(b)]
                d = (gray[tuple(b)] - gray[tuple(a)])[both]
                num += float((d.astype(np.float64) ** 2).sum())
                den += len(d)
            if den:
                rows.append(num / den)
        if rows:
            out[name] = rows
    return out


MEMBRANE_WINDOW = 5           # px: local maximum around an annotated border finds a membrane within +-2 px


def membrane_contrast(img, cells, cls, train3, d_cell, near_other, d_nuc_out, cell_ref):
    """Brightness of the membranes per channel: local maximum (MEMBRANE_WINDOW px in XY) at the
    border pixels of the cell bodies, over the cell's interior mean, minus the same at interior
    pixels (>= 6 px from the border, >= 4 px from a nucleus). Borders with another cell ('contact')
    and with the background ('free') apart. The annotated border may miss the membrane by a pixel
    or two, so a profile along the distance to the border smears it; the local maximum does not."""
    mx = ndi.maximum_filter(img, size=(1, MEMBRANE_WINDOW, MEMBRANE_WINDOW, 1))
    ref = cell_ref[cells]
    body = (cls == 1) & train3
    interior = body & (d_cell >= 6) & (d_nuc_out >= 4)
    out = {}
    C = img.shape[-1]
    sels = (('contact', body & (d_cell == 1) & near_other), ('free', body & (d_cell == 1) & ~near_other))
    for c in range(C):
        ok = ref[..., c] >= 1                                # cells with a reference brightness in this channel
        inner = interior & ok
        if inner.sum() < 100:
            continue
        ratio_in = mx[..., c][inner] / ref[..., c][inner]
        out.setdefault('interior_spread', [np.nan] * C)[c] = float(ratio_in.std())
        for name, sel in sels:
            sel = sel & ok
            if sel.sum() >= 100:
                values = mx[..., c][sel] / ref[..., c][sel]
                out.setdefault(name, [np.nan] * C)[c] = float(values.mean() - ratio_in.mean())
                # unevenness along the border: spread of the same ratio over the border pixels
                out.setdefault(name + '_spread', [np.nan] * C)[c] = float(values.std())
    return out


def coherence(image, inside, sigma_d=1.0, sigma_i=4.0):
    """Orientation of the fine texture: energy-weighted coherence (l1 - l2) / (l1 + l2) of the XY
    structure tensor (derivative scale sigma_d, integration scale sigma_i) of the gray image minus
    its Gaussian smoothing (sigma 4 px, so that slow intensity ramps do not count) over the pixels
    `inside`; 0 for isotropic texture, towards 1 for parallel fibres."""
    num = den = 0.0
    gray = image.mean(-1)
    for z in range(gray.shape[0]):
        if not inside[z].any():
            continue
        g = gray[z] - ndi.gaussian_filter(gray[z], 4.0)
        gy = ndi.gaussian_filter(g, sigma_d, order=(1, 0))
        gx = ndi.gaussian_filter(g, sigma_d, order=(0, 1))
        jyy, jxx, jxy = (ndi.gaussian_filter(a, sigma_i) for a in (gy * gy, gx * gx, gx * gy))
        trace = jyy + jxx
        diff = np.sqrt((jyy - jxx) ** 2 + 4 * jxy ** 2)
        num += float(diff[inside[z]].sum())
        den += float(trace[inside[z]].sum())
    return num / den if den > 0 else np.nan


def appearance_stats(v, whole_cells, cell_nuclei, blur_fit=True):
    """Intensities per class and channel, texture, noise, blur, border profiles, depth, haze."""
    img = v.image.astype(np.float32)
    C = img.shape[-1]
    cells, nuclei = v.cells, v.nuclei
    train3 = np.broadcast_to(v.train, cells.shape)
    cls = np.zeros(cells.shape, np.int8)           # 0 background, 1 cytoplasm, 2 nucleus
    cls[cells > 0] = 1
    cls[nuclei > 0] = 2
    out = {}
    for k, name in enumerate(('background', 'cytoplasm', 'nucleus')):
        sel = (cls == k) & train3
        vals = img[sel]
        if not len(vals):                                   # a class without voxels (small synthetic volumes)
            out[name] = dict(mean=[np.nan] * C, std=[np.nan] * C, p01=[np.nan] * C, p99=[np.nan] * C, voxels=0)
            continue
        out[name] = dict(mean=vals.mean(0).tolist(), std=vals.std(0).tolist(),
                         p01=np.percentile(vals, 1, 0).tolist(), p99=np.percentile(vals, 99, 0).tolist(),
                         voxels=int(sel.sum()))
    # per-object means: cytoplasm of each cell, each nucleus
    cyto = np.where(cls == 1, cells, 0)
    lab_c = np.unique(cyto)[1:]
    lab_n = np.unique(nuclei)[1:]
    means_c = np.stack([ndi.mean(img[..., c], cyto, lab_c) for c in range(C)], 1) if len(lab_c) else np.zeros((0, C))
    means_n = np.stack([ndi.mean(img[..., c], nuclei, lab_n) for c in range(C)], 1) if len(lab_n) else np.zeros((0, C))
    colour_c = dict(zip(lab_c.tolist(), means_c.tolist()))
    colour_n = dict(zip(lab_n.tolist(), means_n.tolist()))
    out['cytoplasm']['between_object_std'] = means_c.std(0).tolist()
    out['nucleus']['between_object_std'] = means_n.std(0).tolist()
    # piecewise-constant model image: object means, background = mean background of the slice
    model = np.zeros_like(img)
    for z in range(img.shape[0]):
        bsel = (cls[z] == 0) & v.train
        model[z] = img[z][bsel].mean(0) if bsel.any() else 0
    lut_c = np.zeros((int(cells.max()) + 1, C), np.float32)
    lut_c[lab_c] = means_c
    lut_n = np.zeros((int(nuclei.max()) + 1, C), np.float32)
    lut_n[lab_n] = means_n
    model[cls == 1] = lut_c[cyto[cls == 1]]
    model[cls == 2] = lut_n[nuclei[cls == 2]]
    # texture: residual from the object mean, away from borders; correlation along X, Y, Z
    d_cell = np.zeros(cells.shape, np.float32)
    d_nuc_in = np.zeros(cells.shape, np.float32)
    d_nuc_out = np.full(cells.shape, 99, np.float32)
    near_other = np.zeros(cells.shape, bool)
    for z in range(cells.shape[0]):
        if cells[z].any():
            # distance to the nearest voxel not of the same cell; type of that voxel
            for lab, box in enumerate(ndi.find_objects(cells[z]), 1):
                if box is None:
                    continue
                box = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in box)
                m = cells[z][box] == lab
                d, idx = ndi.distance_transform_edt(np.pad(m, 1), return_indices=True)
                d, idx = d[1:-1, 1:-1], idx[:, 1:-1, 1:-1] - 1
                ny = np.clip(idx[0] + box[0].start, 0, cells.shape[1] - 1)
                nx = np.clip(idx[1] + box[1].start, 0, cells.shape[2] - 1)
                other = cells[z][ny, nx] > 0
                d_cell[z][box][m] = d[m]
                near_other[z][box][m] = other[m]
        if nuclei[z].any():
            d_nuc_in[z] = ndi.distance_transform_edt(nuclei[z] > 0)
            d_nuc_out[z] = ndi.distance_transform_edt(nuclei[z] == 0)
    resid = img - model
    interior = {'cytoplasm': (cls == 1) & (d_cell >= 4) & (d_nuc_out >= 3) & train3,
                'nucleus': (cls == 2) & (d_nuc_in >= 3) & train3}
    for name, sel in interior.items():
        r = resid[sel]
        o = out[name]
        o['texture_std'] = r.std(0).tolist() if len(r) else [np.nan] * C
        o['texture_rel'] = (r.std(0) / np.maximum(img[sel].mean(0), 1)).tolist() if len(r) else [np.nan] * C
        acf = {}
        for axis, lags in (('x', range(1, 41)), ('y', range(1, 41)), ('z', range(1, 7))):
            ax = {'z': 0, 'y': 1, 'x': 2}[axis]
            if cells.shape[ax] < 2:
                continue
            vals = []
            for lag in lags:
                if lag >= cells.shape[ax]:
                    break
                a = [slice(None)] * 3
                b = [slice(None)] * 3
                a[ax], b[ax] = slice(0, -lag), slice(lag, None)
                both = sel[tuple(a)] & sel[tuple(b)]
                if both.sum() < 100:
                    break
                ra, rb = resid[tuple(a)][both], resid[tuple(b)][both]
                cc = [(np.mean(ra[:, c] * rb[:, c]) / max(np.sqrt(np.mean(ra[:, c] ** 2) * np.mean(rb[:, c] ** 2)), 1e-9))
                      for c in range(C)]
                vals.append([float(x) for x in cc])
            acf[axis] = vals
        o['texture_acf'] = acf
        o['texture_length'] = {axis: [float(_one_over_e([row[c] for row in rows])) for c in range(C)]
                               for axis, rows in acf.items() if rows}
        o['coherence'] = coherence(img, sel)
    # noise: Immerkaer kernel response against the local mean, per channel (inner pixels of the training area)
    keep = v.train[1:-1, 1:-1]
    resp_all = np.concatenate([np.stack([immerkaer_sigma(img[z, ..., c], ndi.uniform_filter(img[z, ..., c], 3))[0][keep]
                                         for c in range(C)], 1) for z in range(img.shape[0])])
    mean_all = np.concatenate([np.stack([ndi.uniform_filter(img[z, ..., c], 3)[1:-1, 1:-1][keep] for c in range(C)], 1)
                               for z in range(img.shape[0])])
    noise = {}
    for c in range(C):
        bins = np.clip((mean_all[:, c] // 16).astype(int), 0, 15)
        noise[c] = [[b * 16 + 8, float(np.median(resp_all[bins == b, c]) / (0.6745 * 6)), int((bins == b).sum())]
                    for b in range(16) if (bins == b).sum() >= 1000]
    out['noise_by_intensity'] = noise
    cls_inner = cls[:, 1:-1, 1:-1][:, keep].reshape(-1)
    for name, k in (('background', 0), ('cytoplasm', 1), ('nucleus', 2)):
        sel = cls_inner == k
        out[name]['noise_sigma'] = (np.median(resp_all[sel], 0) / (0.6745 * 6)).tolist() if sel.any() else [np.nan] * C
    del resp_all, mean_all
    # blur: edge width against the piecewise-constant label image, near the objects
    fg = cells > 0
    near = ndi.binary_dilation(fg, np.ones((1, 2 * BLUR_REACH + 1, 2 * BLUR_REACH + 1), bool)) & train3
    if near.any() and blur_fit:
        zz, yy, xx = np.nonzero(near)
        box = (slice(zz.min(), zz.max() + 1), slice(yy.min(), yy.max() + 1), slice(xx.min(), xx.max() + 1))
        out['blur'] = edge_blur(img[box], model[box], near[box], v.spacing_z, v.ndim)
    out['sharpness'] = sharpness(img, (cells > 0) & train3)
    if v.nucleoli is not None and v.nucleoli.any():
        # visibility of the nucleoli: gray level inside them over the rest of their nuclei
        gray = img.mean(-1)
        inner = (v.nucleoli > 0) & train3
        rest = (nuclei > 0) & (v.nucleoli == 0) & train3 & np.isin(nuclei, np.unique(nuclei[inner]))
        out['nucleolus_contrast'] = float(gray[inner].mean() / max(gray[rest].mean(), 1e-6)) if rest.any() else None
    # border profiles, relative to the cell's interior cytoplasm mean (absolute values too)
    cell_ref = np.zeros((int(cells.max()) + 1, C), np.float32)
    sel = interior['cytoplasm']
    if sel.any():
        labs = np.unique(cells[sel])
        ref = np.stack([ndi.mean(img[..., c], np.where(sel, cells, 0), labs) for c in range(C)], 1)
        cell_ref[labs] = ref
    for c_lab in whole_cells:
        if not cell_ref[c_lab].any() and c_lab in colour_c:
            cell_ref[c_lab] = colour_c[c_lab]
    out['membranes'] = membrane_contrast(img, cells, cls, train3, d_cell, near_other, d_nuc_out, cell_ref)
    wset = np.zeros(int(cells.max()) + 1, bool)
    wset[whole_cells] = True
    in_whole = wset[cells] & (cls == 1)
    refv = np.maximum(cell_ref[cells[in_whole]], 1)
    vals = img[in_whole]
    profiles = {}
    for name, sel2 in (('cell_border_contact', near_other[in_whole]), ('cell_border_free', ~near_other[in_whole])):
        s, n = _profile(d_cell[in_whole][sel2], vals[sel2] / refv[sel2])
        a, _ = _profile(d_cell[in_whole][sel2], vals[sel2])
        profiles[name] = dict(ratio=(s / np.maximum(n, 1)[:, None]).tolist(), absolute=(a / np.maximum(n, 1)[:, None]).tolist(),
                              count=n.tolist())
    s, n = _profile(d_nuc_out[in_whole], vals / refv)
    profiles['perinuclear'] = dict(ratio=(s / np.maximum(n, 1)[:, None]).tolist(), count=n.tolist())
    nsel = (cls == 2) & train3
    nref = np.maximum(lut_n[nuclei[nsel]], 1)
    s, n = _profile(d_nuc_in[nsel], img[nsel] / nref)
    profiles['nucleus_border'] = dict(ratio=(s / np.maximum(n, 1)[:, None]).tolist(), count=n.tolist())
    out['profiles'] = profiles
    # depth: mean of cell voxels and of background per slice, Z relative to the tissue centre
    if img.shape[0] > 1:
        zc = float(np.average(np.arange(img.shape[0]), weights=(fg & train3).sum((1, 2)) + 1e-9))
        rows = []
        for z in range(img.shape[0]):
            f_ = fg[z] & v.train
            b_ = ~fg[z] & v.train
            rows.append(dict(dz=float(z - zc), cells=img[z][f_].mean(0).tolist() if f_.sum() > 100 else None,
                             background=img[z][b_].mean(0).tolist() if b_.any() else None, cell_voxels=int(f_.sum())))
        out['depth'] = dict(tissue_centre_z=zc, slices=rows)
    # haze: background against the physical distance to the nearest cell voxel
    if fg.any() and (~fg & train3).any():
        dist = ndi.distance_transform_edt(~fg, sampling=(v.spacing_z, 1, 1)) if fg.shape[0] > 1 else \
            ndi.distance_transform_edt(~fg[0])[None]
        bsel = ~fg & train3
        d = dist[bsel]
        edges = np.array([0, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 1e9])
        bi = np.digitize(d, edges) - 1
        rows = []
        bv = img[bsel]
        for b in range(len(edges) - 1):
            s = bi == b
            if s.sum() >= 200:
                rows.append(dict(d0=float(edges[b]), d1=float(min(edges[b + 1], 9999)), mean=bv[s].mean(0).tolist(),
                                 std=bv[s].std(0).tolist(), voxels=int(s.sum())))
        out['haze'] = rows
    return out, colour_c, colour_n


def _one_over_e(acf):
    """Lag (interpolated) at which a correlation sequence for lags 1, 2, ... falls to 1/e."""
    prev_l, prev = 0.0, 1.0
    for lag, value in enumerate(acf, 1):
        if value <= 1 / np.e:
            return prev_l + (prev - 1 / np.e) / max(prev - value, 1e-9) * (lag - prev_l)
        prev_l, prev = lag, value
    return float(len(acf)) if acf else np.nan


def measure_volume(v, window=(32, 256, 256), appearance=True, donors=True, geometry=True, blur_fit=True):
    """Objects, hierarchy, packing and appearance of one volume. Returns a dict with lists
    'cells', 'nuclei' (feature rows of whole objects), 'donors', 'hierarchy', 'packing', 'appearance'.
    geometry=False measures the appearance only."""
    t0 = time.time()
    sz, nd = v.spacing_z, v.ndim
    whole_c = whole_labels(v.cells, v.train, v.z_cut)
    if not geometry:
        assign = v.assignments if v.assignments is not None else overlap_assignments(v.nuclei, v.cells)
        nuclei_of = {}
        for n, c in assign.items():
            nuclei_of.setdefault(c, []).append(n)
        return dict(name=v.name, appearance=appearance_stats(v, whole_c, nuclei_of, blur_fit)[0], seconds=round(time.time() - t0, 1))
    whole_n = set(whole_labels(v.nuclei, v.train, v.z_cut))
    assign = v.assignments if v.assignments is not None else overlap_assignments(v.nuclei, v.cells)
    present_n = set(np.unique(v.nuclei).tolist()) - {0}
    assign = {n: c for n, c in assign.items() if n in present_n}
    nuclei_of = {}
    for n, c in assign.items():
        nuclei_of.setdefault(c, []).append(n)
    boxes_c, boxes_n = ndi.find_objects(v.cells), ndi.find_objects(v.nuclei)
    centres, radii = {}, {}
    for lab, box in enumerate(boxes_c, 1):
        if box is None:
            continue
        m = v.cells[box] == lab
        c = np.array(ndi.center_of_mass(m)) + [s.start for s in box]
        centres[lab] = c.tolist()
        radii[lab] = float(np.sqrt(m.reshape(m.shape[0], -1).sum(1).max() / np.pi))
    rows = {'cells': [], 'nuclei': [], 'nucleoli': []}
    records = {}
    for level, labels, boxes, keep in (('cells', v.cells, boxes_c, whole_c), ('nuclei', v.nuclei, boxes_n, sorted(whole_n))):
        for lab in keep:
            box = boxes[lab - 1]
            m = labels[box] == lab
            f, slices = shape_features(m, sz, nd)
            f['label'] = lab
            rows[level].append(f)
            records[(level, lab)] = (box, m, slices)
    if v.nucleoli is not None and v.nucleoli.any():
        for lab, box in enumerate(ndi.find_objects(v.nucleoli), 1):
            if box is not None:
                f, _ = shape_features(v.nucleoli[box] == lab, sz, nd)
                f['label'] = lab
                rows['nucleoli'].append(f)
    # hierarchy of whole cells
    hier = dict(nuclei_per_cell=[], volume_ratio=[], total_ratio=[], offset_xy=[], offset_z=[], outside=[], outside_voxels=0,
                outside_none_voxels=0, nucleus_voxels=0)
    for c in whole_c:
        ns = [n for n in nuclei_of.get(c, [])]
        hier['nuclei_per_cell'].append(len(ns))
        cbox, cm, _ = records[('cells', c)]
        cvol = cm.sum()
        cc = np.array(centres[c])
        rz = (cbox[0].stop - cbox[0].start) * sz / 2
        tot = 0
        for n in ns:
            nbox = boxes_n[n - 1]
            nm = v.nuclei[nbox] == n
            nv = int(nm.sum())
            tot += nv
            out_vox = int((v.cells[nbox][nm] != c).sum())
            hier['outside'].append(out_vox / nv)
            hier['outside_voxels'] += out_vox
            hier['outside_none_voxels'] += int((v.cells[nbox][nm] == 0).sum())
            hier['nucleus_voxels'] += nv
            hier['volume_ratio'].append(nv / cvol)
            nc = np.array(ndi.center_of_mass(nm)) + [s.start for s in nbox]
            hier['offset_xy'].append(float(np.hypot(*(nc[1:] - cc[1:])) / max(radii[c], 1)))
            hier['offset_z'].append(float((nc[0] - cc[0]) * sz / max(rz, 1)) if nd == 3 else 0.0)
        hier['total_ratio'].append(tot / cvol)
    packing = packing_stats(v, whole_c, centres, radii, window)
    out = dict(name=v.name, info=v.info, shape=list(v.image.shape), spacing_z=sz, ndim=nd, rows=rows,
               hierarchy=hier, packing=packing)
    colour_c, colour_n = {}, {}
    if appearance:
        out['appearance'], colour_c, colour_n = appearance_stats(v, whole_c, nuclei_of)
    if donors:
        out['donors'] = [donor_record(v, c, records, nuclei_of, whole_n, boxes_n, colour_c, colour_n) for c in whole_c]
        zc = out.get('appearance', {}).get('depth', {}).get('tissue_centre_z')
        for d in out['donors']:
            d['dz_tissue'] = None if zc is None else float(centres[d['label']][0] - zc)
    out['seconds'] = round(time.time() - t0, 1)
    return out


def _slices_json(slices, z0):
    return [dict(z=int(s['z'] + z0), dy=round(float(s['offset'][0]), 3), dx=round(float(s['offset'][1]), 3), area=s['area'],
                 modes=[[round(float(c.real), 5), round(float(c.imag), 5)] for c in s['modes']]) for s in slices]


def donor_record(v, c, records, nuclei_of, whole_n, boxes_n, colour_c, colour_n):
    """A whole cell as a donor: its slices (z relative to the cell's first slice), XY centroid,
    position of its first slice in the volume, colours, and its nuclei with their offsets."""
    box, m, slices = records[('cells', c)]
    centre, _ = outline_record(m)
    z0 = box[0].start
    cy, cx = centre[0] + box[1].start, centre[1] + box[2].start
    d = dict(label=int(c), volume=int(m.sum()), nz=int(m.shape[0]), z0=int(z0), z_rel=float(z0 / max(v.cells.shape[0] - 1, 1)),
             slices=_slices_json(slices, 0), colour=colour_c.get(c), nuclei=[])
    for n in nuclei_of.get(c, []):
        nbox = boxes_n[n - 1]
        nm = v.nuclei[nbox] == n
        ncentre, nslices = outline_record(nm)
        d['nuclei'].append(dict(label=int(n), volume=int(nm.sum()), whole=bool(n in whole_n),
                                dz=int(nbox[0].start - z0), dy=round(float(ncentre[0] + nbox[1].start - cy), 3),
                                dx=round(float(ncentre[1] + nbox[2].start - cx), 3),
                                slices=_slices_json(nslices, 0), colour=colour_n.get(n)))
    return d


# --------------------------------------------------------------------------------- summary

def describe(values):
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)], np.float64)
    if not len(a):
        return dict(n=0)
    p = np.percentile(a, [5, 25, 50, 75, 95])
    return dict(n=int(len(a)), mean=float(a.mean()), std=float(a.std()), p05=float(p[0]), p25=float(p[1]),
                median=float(p[2]), p75=float(p[3]), p95=float(p[4]))


FEATURES = ('volume', 'extent_z', 'extent_y', 'extent_x', 'radius_xy', 'radius_3d', 'semi_major_xy', 'semi_minor_xy',
            'semi_z', 'flatness', 'elongation', 'sphericity', 'solidity', 'mode1', 'mode2', 'rough_low', 'rough_high',
            'nonstar')


def orientation_test(angles):
    """Mean resultant length of the doubled XY orientation angles and the Rayleigh p-value (small p:
    orientations are not uniform)."""
    a = np.radians(np.asarray(angles)) * 2
    if not len(a):
        return dict(n=0)
    R = float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))
    n = len(a)
    z = n * R ** 2
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n)))
    return dict(n=n, resultant=R, rayleigh_p=min(1.0, p), mean_deg=float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2 % 180),
                z=float(z))


def mode_spectrum(volumes, level):
    """Mean power of each Fourier mode of log r over the largest slice of every whole object."""
    powers = []
    for vol in volumes:
        for d in vol.get('donors', []):
            objs = [d] if level == 'cells' else d['nuclei']
            for o in objs:
                s = max(o['slices'], key=lambda s: s['area'])
                c = np.array([complex(*m) for m in s['modes']])
                powers.append(mode_power(c))
    return np.mean(powers, 0).tolist() if powers else []


def summarize(volumes):
    """Pooled summary of measure_volume() results of one dataset."""
    out = {}
    for level in ('cells', 'nuclei', 'nucleoli'):
        rows = [r for v in volumes for r in v['rows'][level]]
        if not rows:
            continue
        out[level] = {k: describe([r.get(k) for r in rows]) for k in FEATURES if any(k in r for r in rows)}
        out[level]['orientation'] = orientation_test([r['orientation'] for r in rows])
        out[level]['mode_power'] = mode_spectrum(volumes, level)
    h = {k: [x for v in volumes for x in v['hierarchy'][k]] for k in ('nuclei_per_cell', 'volume_ratio', 'total_ratio', 'offset_xy',
                                                                       'offset_z', 'outside')}
    counts = np.bincount(h['nuclei_per_cell']) if h['nuclei_per_cell'] else np.array([])
    out['hierarchy'] = dict(cells=len(h['nuclei_per_cell']), nuclei_per_cell_counts=counts.tolist(),
                            multinucleated_share=float((np.asarray(h['nuclei_per_cell']) > 1).mean()) if h['nuclei_per_cell'] else None,
                            without_nucleus_share=float((np.asarray(h['nuclei_per_cell']) == 0).mean()) if h['nuclei_per_cell'] else None,
                            volume_ratio=describe(h['volume_ratio']), total_ratio=describe(h['total_ratio']),
                            offset_xy=describe(h['offset_xy']), offset_z=describe(h['offset_z']),
                            outside_share_voxels=float(sum(v['hierarchy']['outside_voxels'] for v in volumes) /
                                                       max(sum(v['hierarchy']['nucleus_voxels'] for v in volumes), 1)),
                            outside_no_cell_share_voxels=float(sum(v['hierarchy'].get('outside_none_voxels', 0) for v in volumes) /
                                                               max(sum(v['hierarchy']['nucleus_voxels'] for v in volumes), 1)))
    p = [v['packing'] for v in volumes]
    nn = [x for q in p for x in q['nearest']]
    out['packing'] = dict(
        cells=sum(q['n_cells'] for q in p), whole_cells=sum(q['n_whole'] for q in p),
        touching_share=float(np.mean([t > 0 for q in p for t in q['touching']])) if any(q['touching'] for q in p) else None,
        touching_neighbours=describe([t for q in p for t in q['touching']]),
        contact_fraction=describe([t for q in p for t in q['contact_fraction']]),
        clusters_per_cell=float(sum(q['n_clusters'] for q in p) / max(sum(q['n_cells'] for q in p), 1)),
        singleton_share=float(sum(sum(1 for s in q['cluster_sizes'] if s == 1) for q in p) / max(sum(q['n_cells'] for q in p), 1)),
        nn_distance=describe([x['distance'] for x in nn]), nn_gap_ratio=describe([x['gap_ratio'] for x in nn]),
        nn_gap_ratio_touching=describe([x['gap_ratio'] for x in nn if x['touching']]),
        nn_gap_ratio_free=describe([x['gap_ratio'] for x in nn if not x['touching']]),
        nn_elevation=describe([x['elevation'] for x in nn]), nn_touching_share=float(np.mean([x['touching'] for x in nn])) if nn else None,
        envelope_density=describe([q['envelope_density'] for q in p]), envelope_fraction=describe([q['envelope_fraction'] for q in p]),
        cell_fraction=describe([q['cell_fraction'] for q in p]), window=p[0]['window'] if p else None,
        window_cell_share=describe([x for q in p for x in q['window_cell_share'] if x > MIN_WINDOW_SHARE]),
        empty_window_share=float(np.mean([x <= MIN_WINDOW_SHARE for q in p for x in q['window_cell_share']]))
        if any(q['window_cell_share'] for q in p) else None)
    if all('appearance' in v for v in volumes):
        out['appearance'] = {}
        for name in ('background', 'cytoplasm', 'nucleus'):
            rows = [v['appearance'][name] for v in volumes]
            out['appearance'][name] = {k: np.nanmean([r[k] for r in rows if k in r], 0).tolist()
                                       for k in ('mean', 'std', 'between_object_std', 'texture_std', 'texture_rel', 'noise_sigma',
                                                 'coherence')
                                       if any(k in r for r in rows)}
            lengths = [r.get('texture_length', {}) for r in rows]
            out['appearance'][name]['texture_length'] = {a: np.nanmean([l[a] for l in lengths if a in l], 0).tolist()
                                                         for a in ('x', 'y', 'z') if any(a in l for l in lengths)}
        out['appearance']['blur'] = {k: describe([v['appearance']['blur'][k] for v in volumes if 'blur' in v['appearance']])
                                     for k in ('sigma_z', 'sigma_xy', 'halo', 'halo_weight', 'r2')}
        out['appearance']['membranes'] = {k: np.mean([v['appearance']['membranes'][k] for v in volumes
                                                       if k in v['appearance'].get('membranes', {})], 0).tolist()
                                          for k in ('contact', 'free', 'contact_spread', 'interior_spread')
                                          if any(k in v['appearance'].get('membranes', {}) for v in volumes)}
        out['appearance']['nucleolus_contrast'] = _mean_or_none([v['appearance'].get('nucleolus_contrast') for v in volumes])
        out['appearance'].update(sharpness_xy=_mean_ratio(volumes, 'xy', 1, 4), sharpness_z=_mean_ratio(volumes, 'z', 0, 2),
                                 haze_excess=_mean_or_none([haze_excess(v['appearance']) for v in volumes]))
    return out


def haze_excess(a, near=(8, 32)):
    """Mean background brightness 8-32 px from the cells above the far background (the halo)."""
    rows = a.get('haze') or []
    if not rows:
        return None
    far = np.mean(rows[-1]['mean'])
    sel = [np.mean(r['mean']) - far for r in rows if near[0] <= r['d0'] < near[1]]
    return float(np.mean(sel)) if sel else None


def sharp_ratio(a, axis, lo, hi):
    """D(lag lo) / D(lag hi) of the image structure function (sharpness()); falls with blur."""
    rows = a.get('sharpness', {}).get(axis)
    return rows[lo] / rows[hi] if rows and len(rows) > hi and rows[hi] > 0 else None


def _mean_or_none(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def _mean_ratio(volumes, axis, lo, hi):
    return _mean_or_none([sharp_ratio(v['appearance'], axis, lo, hi) for v in volumes])


# ---------------------------------------------------------------------------------- report

def _fmt(d, scale=1.0, digits=2):
    if not d or not d.get('n'):
        return '–'
    return f"{d['median'] * scale:.{digits}f} [{d['p05'] * scale:.{digits}f}–{d['p95'] * scale:.{digits}f}]"


def report(stats):
    lines = ['# Annotation statistics for the CytoNucs synthetic data', '',
             'Median [5th–95th percentile] over the whole objects of the train part. Units: XY pixels '
             '(in 3D, volumes and radii in XY pixels with the Z step taken into account), Z extent in slices.', '']
    for ds, s in stats['datasets'].items():
        src = stats['sources'][ds]
        lines += [f'## {ds}', '', src['description'], '']
        lines += ['| feature | cells | nuclei |', '|---|---|---|']
        names = dict(volume='volume, voxels', extent_z='Z extent, slices', radius_xy='equivalent radius XY', radius_3d='equivalent radius 3D',
                     semi_major_xy='semi-major axis XY', semi_minor_xy='semi-minor axis XY', semi_z='semi-axis Z (physical)',
                     flatness='flatness (Z / XY)', elongation='elongation XY', sphericity='sphericity (2D: circularity)',
                     solidity='solidity', mode2='mode 2 of log r (ellipse)', rough_low='roughness, modes 3–8',
                     rough_high='roughness, modes 9–24', nonstar='share of non-star-convex rays')
        for k, title in names.items():
            if k in s.get('cells', {}) or k in s.get('nuclei', {}):
                lines.append(f"| {title} | {_fmt(s.get('cells', {}).get(k))} | {_fmt(s.get('nuclei', {}).get(k))} |")
        for lvl in ('cells', 'nuclei'):
            o = s.get(lvl, {}).get('orientation', {})
            if o.get('n'):
                lines.append(f"| orientation XY ({lvl}) | R = {o['resultant']:.2f}, p = {o['rayleigh_p']:.2g} | |")
        h = s['hierarchy']
        lines += ['', f"Hierarchy: {h['cells']} whole cells; nuclei per cell (0, 1, 2, …): {h['nuclei_per_cell_counts']}; "
                      f"multinucleated {h['multinucleated_share']:.1%}, without a nucleus {h['without_nucleus_share']:.1%}; "
                      f"nucleus / cell volume {_fmt(h['volume_ratio'])}; nucleus offset in XY / cell radius {_fmt(h['offset_xy'])}; "
                      f"in Z / cell half-height {_fmt(h['offset_z'])}; nucleus voxels outside their cell {h['outside_share_voxels']:.2%} "
                      f"(outside every cell {h['outside_no_cell_share_voxels']:.2%}, the rest in a neighbouring cell).", '']
        p = s['packing']
        lines += [f"Packing: cells in the train region {p['cells']}, whole {p['whole_cells']}; touching a neighbour {p['touching_share']:.0%}; "
                  f"touching neighbours {_fmt(p['touching_neighbours'], digits=0)}; boundary share in contact {_fmt(p['contact_fraction'])}; "
                  f"clusters per cell {p['clusters_per_cell']:.2f}, singletons {p['singleton_share']:.0%}; "
                  f"distance to the nearest {_fmt(p['nn_distance'], digits=1)} px, gap ratio {_fmt(p['nn_gap_ratio'])}, "
                  f"angle to the XY plane {_fmt(p['nn_elevation'], digits=0)}°; density in the tissue envelope {_fmt(p['envelope_density'])}; "
                  f"cell fraction of the volume {_fmt(p['cell_fraction'])}; in non-empty windows {p['window']}: {_fmt(p['window_cell_share'])} "
                  f"(empty windows {p['empty_window_share']:.0%}).", '']
        if 'appearance' in s:
            a = s['appearance']
            lines += ['| class | mean RGB | σ between objects | texture σ | correlation length X | noise σ |', '|---|---|---|---|---|---|']
            for name in ('background', 'cytoplasm', 'nucleus'):
                r = a[name]
                tl = r.get('texture_length', {}).get('x')
                lines.append(f"| {name} | {_v(r.get('mean'))} | {_v(r.get('between_object_std'))} | {_v(r.get('texture_std'))} | "
                             f"{_v(tl)} | {_v(r.get('noise_sigma'), 2)} |")
            b = a['blur']
            lines += ['', f"Blur relative to the GT boundaries (core + halo): core σ_xy {_fmt(b['sigma_xy'], digits=1)} px, "
                          f"σ_z {_fmt(b['sigma_z'], digits=1)} slices; halo σ {_fmt(b['halo'], digits=0)} px, its weight "
                          f"{_fmt(b['halo_weight'])}; R² {_fmt(b['r2'])}.", '']
    lines += ['## Measured and assumed', '', *stats['assumptions'], '']
    return '\n'.join(lines)


def _v(values, digits=1):
    return '–' if values is None else '(' + ', '.join(f'{x:.{digits}f}' for x in values) + ')'


ASSUMPTIONS = [
    '- The files carry no voxel size. For HeLa at half XY resolution the Z step is 2.5 XY pixels (model.anisotropy of v5); '
    'the same is assumed for EA.hy926 (after halving XY).',
    '- Nucleoli are annotated nowhere: their generator parameters are set by hand, not measured.',
    '- A nucleus belongs to a cell by the assignments file (HeLa) or by the largest overlap (EA.hy926, Hollandi).',
    '- Blur: the width of the image edges relative to the GT boundaries; it includes annotation inaccuracy.',
    '- Noise: Immerkær estimate (3x3 kernel) by median, as a function of the local mean; uint8 quantisation is part of it.',
]

DESCRIPTIONS = dict(
    hela='HeLa spheroid at half XY resolution (patches_split_hela_xy2/train_regions), train blocks of regions.json only.',
    ea='EA.hy926, three volumes halved in XY; train blocks of patches_split/split_manifest.json only '
       '(the val / test blocks are masked and can serve as an independent check).',
    hollandi='Hollandi 2D, RGB 2048², without Image0019 and Image0036.')
WINDOWS = dict(hela=(32, 256, 256), ea=(16, 256, 256), hollandi=(512, 512))


def collect(volumes_by_dataset, windows=None, descriptions=None):
    """Statistics of {dataset: [Volume]}: the 'sources', 'datasets' (pooled summary) and 'volumes'
    (per volume, with donors) sections that make_synthetic_from_stats.py reads."""
    windows = windows or WINDOWS
    stats = dict(created=time.strftime('%Y-%m-%d %H:%M'), k_rays=K_RAYS, n_modes=N_MODES, step=STEP, profile_bins=PROFILE_BINS,
                 assumptions=ASSUMPTIONS, sources={}, datasets={}, volumes={})
    for ds, volumes in volumes_by_dataset.items():
        results = []
        for v in volumes:
            r = measure_volume(v, windows[ds])
            print(f"{ds} {v.name}: {len(r['rows']['cells'])} whole cells, {len(r['rows']['nuclei'])} whole nuclei, {r['seconds']} s",
                  flush=True)
            results.append(r)
        stats['sources'][ds] = dict(description=(descriptions or DESCRIPTIONS).get(ds, ds), volumes=[r['name'] for r in results],
                                    ndim=results[0]['ndim'], spacing_z=results[0]['spacing_z'], window=list(windows[ds]),
                                    held_out=list(HOLLANDI_HELD_OUT) if ds == 'hollandi' else None,
                                    info=[r['info'] for r in results])
        stats['datasets'][ds] = summarize(results)
        stats['volumes'][ds] = results
    return json.loads(json.dumps(stats, default=_json_default))


def measure(datasets, output, update=False):
    """Measure `datasets` into <output>/gt_statistics.json; with update=True the other datasets of
    an existing file are kept."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stats = collect({ds: SOURCES[ds]() for ds in datasets})
    old = output / 'gt_statistics.json'
    if update and old.is_file():
        previous = json.loads(old.read_text())
        for key in ('sources', 'datasets', 'volumes'):
            stats[key] = {**{k: v for k, v in previous[key].items() if k not in datasets}, **stats[key]}
    (output / 'gt_statistics.json').write_text(json.dumps(stats))
    (output / 'gt_statistics_report.md').write_text(report(stats))
    print('->', output / 'gt_statistics.json')
    return stats


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True)
    parser.add_argument('--datasets', nargs='+', default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument('--update', action='store_true', help='keep the other datasets of an existing gt_statistics.json')
    args = parser.parse_args()
    measure(args.datasets, args.output, args.update)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # downscale_split.py of dataset_prep
    main()
