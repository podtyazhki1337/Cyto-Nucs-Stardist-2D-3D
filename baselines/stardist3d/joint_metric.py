"""JTPR of the two separate StarDist models (nuclei, cells) from their saved predictions: a nucleus-cell pair of
the annotation counts if its nucleus and its cell are both matched at IoU 0.5 and the predicted nucleus is
assigned to that predicted cell (cytonucs_inference.compute_jtpr, assignment of predictions by overlap)."""
import os, sys, json, contextlib, io
os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from pathlib import Path
import numpy as np, tifffile
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from cytonucs_inference import compute_jtpr
DATA = Path('/storage02/miroslavm/data/Cytonucs_Emese_Hela/patches_split_hela')
out = {}
for split in ('val', 'test'):
    joint = total = 0
    for path in sorted((DATA / split / 'images').iterdir()):
        gt_n = tifffile.imread(DATA / split / 'nucleus_masks' / path.name); gt_c = tifffile.imread(DATA / split / 'cytoplasm_masks' / path.name)
        pairs = {int(k): int(v) for k, v in json.loads((DATA / split / 'assignments' / (path.stem + '.json')).read_text()).items()}
        n = tifffile.imread(HERE / 'models/stardist3d_hela_nucleus/predictions' / split / path.name)
        c = tifffile.imread(HERE / 'models/stardist3d_hela_cytoplasm/predictions' / split / path.name)
        with contextlib.redirect_stdout(io.StringIO()):
            _, details = compute_jtpr(n, c, gt_n, gt_c, iou_thresh=0.5, assignments_gt=pairs)
        joint += int(details['joint_tp']); total += int(details['total_pairs'])
    out[split] = dict(jtpr=joint / max(total, 1), joint_tp=joint, total_pairs=total)
    print(f"{split}: JTPR {out[split]['jtpr']:.3f} ({joint} of {total} pairs)")
(HERE / 'models' / 'joint_metric.json').write_text(json.dumps(out, indent=2))
