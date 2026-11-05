"""Masks whose labels ran past their integer type (uint8: one id for object k and object k + 256) relabelled in a
dataset view: in a mask whose largest label is the type's maximum, every part of one label of at least --min-part
pixels gets its own id (int32); smaller parts keep the id of the label's largest part. The nucleus -> cell pairs of
such an image are found again (helacytonuc_to_cytonucs.nucleus_cells).

HeLaCytoNuc's hand-drawn test masks are uint8; three images hold more than 255 nuclei (2495, 2497, 2608.tif), so
76-136 of their ids each name two nuclei far apart, which no single prediction can match.

The view's files are hard links to the source dataset: a changed mask replaces the link with a new file, the source
stays as it is. Writes label_overflow.json (per split: image, mask folder, labels split) at the top of the view.

Usage: python dataset_prep/fix_label_overflow.py --dataset DIR_view --splits test [--min-part 20]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

MASKS = ('nucleus_masks', 'cytoplasm_masks')


def relabel_parts(mask, min_part):
    """(int32 mask with every part >= min_part of a label under its own id, number of labels split)."""
    out = np.asarray(mask).astype(np.int32)
    next_id, split = int(out.max()) + 1, 0
    for label, sl in enumerate(ndi.find_objects(out), 1):
        if sl is None:
            continue
        parts, n = ndi.label(out[sl] == label)
        sizes = np.bincount(parts.ravel())[1:]
        big = [p + 1 for p in np.argsort(-sizes) if sizes[p] >= min_part]
        if len(big) < 2:
            continue
        split += 1
        view = out[sl]
        for p in big[1:]:
            view[parts == p] = next_id
            next_id += 1
    return out, split


def replace(path, array):
    """A new file at path (a hard link is removed first, so its source is not written through)."""
    path.unlink()
    tifffile.imwrite(path, array)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from helacytonuc_to_cytonucs import nucleus_cells
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--splits', nargs='+', required=True)
    parser.add_argument('--min-part', type=int, default=20)
    args = parser.parse_args()
    dataset, report = Path(args.dataset), {}
    for split in args.splits:
        rows = []
        for image in sorted((dataset / split / 'images').iterdir()):
            masks, changed = {}, False
            for folder in MASKS:
                path = dataset / split / folder / image.name
                mask = tifffile.imread(path)
                masks[folder] = mask
                if not np.issubdtype(mask.dtype, np.integer) or mask.max() < np.iinfo(mask.dtype).max:
                    continue
                fixed, split_labels = relabel_parts(mask, args.min_part)
                if split_labels:
                    replace(path, fixed)
                    masks[folder], changed = fixed, True
                    rows.append(dict(image=image.name, masks=folder, labels_split=split_labels,
                                     objects_before=int(len(np.unique(mask)) - 1), objects_after=int(len(np.unique(fixed)) - 1)))
            pairs = dataset / split / 'assignments' / (image.stem + '.json')
            if changed and pairs.exists():
                pairs.unlink()
                pairs.write_text(json.dumps(nucleus_cells(masks['nucleus_masks'], masks['cytoplasm_masks'])))
        report[split] = rows
        print(split, rows)
    (dataset / 'label_overflow.json').write_text(json.dumps(dict(min_part=args.min_part, splits=report), indent=1))


if __name__ == '__main__':
    main()
