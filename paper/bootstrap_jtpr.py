"""Bootstrap confidence intervals of linked JTPR (IoU 0.5) over the annotated pairs of a 2D test split, for every method,
and of the paired difference CytoNucs minus each baseline. Per-pair joint TP comes
from compute_jtpr(with_pairs=True) on the predictions as score_predictions.py scores them (ignored regions dropped,
saved links when present, else pairing by overlap), so the JTPR equals the scorer's. 10 000 resamples of the pairs
(the same resample for all methods: paired); a second bootstrap over images (pairs of an image together) as a check.
Usage: python paper/bootstrap_jtpr.py DATA SPLIT OUT.json NAME=PREDICTIONS_DIR ... (the first NAME is CytoNucs)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the repository root: the CytoNucs modules
import numpy as np, tifffile
from score_hierarchy import ground_truth, score_image
from cytonucs_inference import compute_jtpr

if len(sys.argv) < 5 or sys.argv[1] in ('-h', '--help'):
    print(__doc__)
    sys.exit(0 if sys.argv[1:2] in (['-h'], ['--help']) else 2)
data, split, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
methods = [a.split('=', 1) for a in sys.argv[4:]]
levels = ['nucleus', 'cytoplasm']; pairs = [('nucleus', 'cytoplasm')]; key = 'nucleus_cytoplasm'
names = sorted(p.name for p in (data / split / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
hits = {m: [] for m, _ in methods}; image_of = []
for i, name in enumerate(names):
    gt, truths, ignore = ground_truth(data / split, name, levels, pairs)
    order = sorted(truths[key])
    image_of += [i] * len(order)
    for m, folder in methods:
        root = Path(folder) / split if (Path(folder) / split).is_dir() else Path(folder)
        pred = {level: tifffile.imread(root / level / name) for level in levels}
        pf = root / 'parents' / (Path(name).stem + '.json')
        parents = {levels[0]: {int(k): int(v) for k, v in json.loads(pf.read_text()).items()}} if pf.is_file() else None
        _, scored = score_image(gt, pred, pairs, truths, pred_parents=parents, ignore=ignore)
        _, det = compute_jtpr(scored['nucleus'], scored['cytoplasm'], gt['nucleus'], gt['cytoplasm'], iou_thresh=.5,
                              assignments_pred=(parents or {}).get('nucleus'), assignments_gt=truths[key], with_pairs=True)
        joint = {n for n, _ in det['joint_pairs']}
        hits[m] += [int(n in joint) for n in order]
H = {m: np.array(v) for m, v in hits.items()}; image_of = np.array(image_of); n = len(image_of)
rng = np.random.default_rng(0); B = 10000
idx = rng.integers(0, n, (B, n))
imgs = np.unique(image_of); members = [np.flatnonzero(image_of == k) for k in imgs]
img_idx = rng.integers(0, len(imgs), (B, len(imgs)))
def by_images(v):
    return np.array([np.concatenate([v[members[j]] for j in row]).mean() for row in img_idx])
ci = lambda s: [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
first = methods[0][0]; result = dict(data=str(data), split=split, pairs=int(n), images=int(len(imgs)), resamples=B, methods={}, differences={})
for m, _ in methods:
    v = H[m]; boot = v[idx].mean(1)
    result['methods'][m] = dict(jtpr=float(v.mean()), joint_tp=int(v.sum()), ci95_pairs=ci(boot), ci95_images=ci(by_images(v)))
for m, _ in methods[1:]:
    d = H[first] - H[m]; boot = d[idx].mean(1)
    result['differences'][f'{first} - {m}'] = dict(diff=float(d.mean()), ci95_pairs=ci(boot), ci95_images=ci(by_images(d)),
                                                  share_above_0=float((boot > 0).mean()), pairs_only_first=int(((H[first] == 1) & (H[m] == 0)).sum()),
                                                  pairs_only_other=int(((H[first] == 0) & (H[m] == 1)).sum()))
out.write_text(json.dumps(result, indent=1))
for m, r in result['methods'].items():
    print(f"{m}: {r['jtpr']:.3f} ({r['joint_tp']}/{n}) CI pairs [{r['ci95_pairs'][0]:.3f}, {r['ci95_pairs'][1]:.3f}] images [{r['ci95_images'][0]:.3f}, {r['ci95_images'][1]:.3f}]")
for k, r in result['differences'].items():
    print(f"{k}: {r['diff']:+.3f} CI pairs [{r['ci95_pairs'][0]:+.3f}, {r['ci95_pairs'][1]:+.3f}] images [{r['ci95_images'][0]:+.3f}, {r['ci95_images'][1]:+.3f}] P(>0) {r['share_above_0']:.3f} discordant {r['pairs_only_first']}/{r['pairs_only_other']}")
