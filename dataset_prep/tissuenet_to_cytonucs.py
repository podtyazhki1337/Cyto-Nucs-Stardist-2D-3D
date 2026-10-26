"""TissueNet v1.1 (2D multiplexed tissue images: a nuclear and a membrane channel, whole-cell and nuclear labels)
as a two-level CytoNucs dataset: cells with their nuclei.

Read from the official npz files (tissuenet_v1.1_{train,val,test}.npz) one image at a time (the arrays are read
as a stream, never loaded whole):
  - images: an ordinary 2D RGB image (uint8), the two stains merged as in scripts/preprocess_cytonucs.ipynb:
    every stain scaled to 0..1 by its own percentiles 1 and 99.9 in the image and coloured, nuclear green,
    membrane blue (the colours of DeepCell's create_rgb_image in the TissueNet README), red empty;
  - labels: y[..., 0] whole cell -> cytoplasm_masks, y[..., 1] nucleus -> nucleus_masks (relabelled 1..n);
  - nucleus -> cell pairs: the cell with the largest overlap when at least half of the nucleus lies in cells
    (99 % of the test nuclei); other nuclei have no cell. Cells without a nucleus are kept as they are (the
    nucleus is outside the section: the nuclear annotation is complete);
  - splits as published: train = 2580 images of 512 x 512; val = --val-count of the 3118 val tiles of 256 x 256
    (random, --seed; checkpoint selection and StarDist thresholds); test = all 1324 test tiles of 256 x 256.
Writes <output>/{train,val,test}/{images,nucleus_masks,cytoplasm_masks,assignments}/<split>_<index>.tif(.json),
split_info.json and tissuenet.json (source index, specimen, experiment, pixel size and object counts per image).
The object arrays 'meta' of the npz files are read by an unpickler that only rebuilds numpy arrays.

With --val-all only the val split is converted, all 3118 tiles, into <output>/val_all (same names; the tiles of
<output>/val must come out identical, which is checked), listed in split_info.json as 'val_all' and described in
tissuenet_val_all.json; nothing else is touched.

Usage:
  python dataset_prep/tissuenet_to_cytonucs.py --raw /storage02/miroslavm/data/TissueNet/raw \\
      --output /storage02/miroslavm/data/TissueNet/cytonucs [--val-count 256] [--seed 0] [--val-all]
Licence: modified Apache 2.0, non-commercial (see raw/LICENSE); cite Greenwald, Miller et al., Nat. Biotechnol. 2022.
"""
import argparse
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import tifffile

ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'), ('numpy', 'ndarray'),
           ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar')}
COLUMNS = ('filename', 'experiment', 'pixel_size', 'screening_passed', 'time_step', 'specimen')


class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'{module}.{name} is not allowed in TissueNet metadata')


def header(stream):
    version = np.lib.format.read_magic(stream)
    return np.lib.format._read_array_header(stream, version)              # shape, fortran order, dtype


def metadata(archive):
    """One dict per image (the header row, repeated once per concatenated part, is dropped)."""
    with archive.open('meta.npy') as stream:
        header(stream)                                      # a pickled object array: its order is in the pickle
        rows = NumpyOnly(io.BytesIO(stream.read())).load()
    rows = [list(r) for r in rows if str(r[0]) != 'filename']
    return [dict(zip(COLUMNS, r)) for r in rows]


def images(archive, member):
    """(index, array) of every image of an npz member, read as a stream."""
    with archive.open(member) as stream:
        shape, fortran, dtype = header(stream)
        if fortran:
            raise ValueError(f'{member}: Fortran-ordered images cannot be read one at a time')
        size = int(np.prod(shape[1:])) * dtype.itemsize
        for index in range(shape[0]):
            yield index, np.frombuffer(stream.read(size), dtype).reshape(shape[1:])


def relabel(labels):
    """Labels 1..n in order of the old ids, and old id -> new id."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lookup = np.zeros(int(labels.max()) + 1, np.int64)
    lookup[ids] = np.arange(1, len(ids) + 1)
    return lookup[labels].astype(np.uint16), len(ids)


def nucleus_cells(nuclei, cells):
    """nucleus -> cell with the largest overlap, for nuclei lying at least half inside cells."""
    sizes = np.bincount(nuclei.ravel())
    both = (nuclei > 0) & (cells > 0)
    width = int(cells.max()) + 1
    keys, counts = np.unique(nuclei[both].astype(np.int64) * width + cells[both], return_counts=True)
    best = {}
    for key, count in zip(keys.tolist(), counts.tolist()):
        nucleus, cell = divmod(key, width)
        if count > best.get(nucleus, (0, 0))[0]:
            best[nucleus] = (count, cell)
    inside = np.bincount(nuclei[both].ravel(), minlength=len(sizes))
    return {int(n): int(cell) for n, (_, cell) in best.items() if inside[n] >= 0.5 * sizes[n]}


def rgb(x):
    """(Y, X, 2) stains (nuclear, membrane) -> (Y, X, 3) uint8 RGB: nuclear green, membrane blue."""
    x = np.asarray(x, np.float32)
    out = np.zeros(x.shape[:2] + (3,), np.uint8)
    for stain, colour in ((0, 1), (1, 2)):
        low, high = np.percentile(x[..., stain], (1, 99.9))
        scaled = (x[..., stain] - low) / (high - low) if high > low else np.zeros(x.shape[:2], np.float32)
        out[..., colour] = np.round(np.clip(scaled, 0, 1) * 255).astype(np.uint8)
    return out


def write_image(path, x):
    tifffile.imwrite(path, rgb(x), photometric='rgb', compression='zlib')


def convert(raw, output, val_count=256, seed=0, val_all=False):
    raw, output = Path(raw), Path(output)
    summary = dict(source=str(raw), val_count=val_count, seed=seed, images={})
    for split in (('val',) if val_all else ('train', 'val', 'test')):
        archive = zipfile.ZipFile(raw / f'tissuenet_v1.1_{split}.npz')
        meta = metadata(archive)
        with archive.open('X.npy') as stream:
            count = header(stream)[0][0]
        if len(meta) != count:
            raise ValueError(f'{split}: {len(meta)} metadata rows for {count} images')
        keep = (set(np.random.default_rng(seed).choice(count, val_count, replace=False).tolist())
                if split == 'val' and not val_all else set(range(count)))
        folder = output / ('val_all' if val_all else split)
        for sub in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments'):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        for (index, x), (_, y) in zip(images(archive, 'X.npy'), images(archive, 'y.npy')):
            if index not in keep:
                continue
            name = f'{split}_{index:05d}.tif'
            cells, n_cells = relabel(y[..., 0])
            nuclei, n_nuclei = relabel(y[..., 1])
            pairs = nucleus_cells(nuclei, cells)
            tifffile.imwrite(folder / 'nucleus_masks' / name, nuclei, photometric='minisblack', compression='zlib')
            tifffile.imwrite(folder / 'cytoplasm_masks' / name, cells, photometric='minisblack', compression='zlib')
            (folder / 'assignments' / f'{split}_{index:05d}.json').write_text(json.dumps(pairs))
            write_image(folder / 'images' / name, x)
            info = meta[index]
            summary['images'][name] = dict(split=split, index=index, specimen=str(info['specimen']),
                                           experiment=str(info['experiment']).split('/')[-1],
                                           pixel_size=float(info['pixel_size']), shape=list(x.shape[:2]),
                                           cells=n_cells, nuclei=n_nuclei, pairs=len(pairs))
        print(f'{split}: {len(keep)} of {count} images', flush=True)
    if val_all:
        return finish_val_all(output, summary)
    names = {s: sorted(n for n, v in summary['images'].items() if v['split'] == s) for s in ('train', 'val', 'test')}
    (output / 'split_info.json').write_text(json.dumps(dict(names, seed=seed, split='tissuenet_v1.1', rescale_factor=1.0,
                                                            levels=['nucleus', 'cytoplasm']), indent=1))
    (output / 'tissuenet.json').write_text(json.dumps(summary, indent=1))
    return summary


def finish_val_all(output, summary):
    """Check that the tiles of val come out identical in val_all, then list val_all in split_info.json."""
    names = sorted(summary['images'])
    for name in sorted(p.name for p in (output / 'val' / 'images').iterdir()):
        for sub in ('images', 'nucleus_masks', 'cytoplasm_masks'):
            if not np.array_equal(tifffile.imread(output / 'val' / sub / name), tifffile.imread(output / 'val_all' / sub / name)):
                raise ValueError(f'{name}: {sub} differs between val and val_all')
        pairs = Path(name).stem + '.json'
        if json.loads((output / 'val' / 'assignments' / pairs).read_text()) != json.loads((output / 'val_all' / 'assignments' / pairs).read_text()):
            raise ValueError(f'{name}: assignments differ between val and val_all')
    info = json.loads((output / 'split_info.json').read_text())
    info['val_all'] = names
    tmp = output / 'split_info.json.tmp'
    tmp.write_text(json.dumps(info, indent=1))
    tmp.replace(output / 'split_info.json')
    (output / 'tissuenet_val_all.json').write_text(json.dumps(summary, indent=1))
    print(f'val_all: {len(names)} images; the {len(list((output / "val" / "images").iterdir()))} val tiles are identical')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--val-count', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val-all', action='store_true', help='convert all val tiles into <output>/val_all only')
    args = parser.parse_args()
    convert(args.raw, args.output, args.val_count, args.seed, args.val_all)


if __name__ == '__main__':
    main()
