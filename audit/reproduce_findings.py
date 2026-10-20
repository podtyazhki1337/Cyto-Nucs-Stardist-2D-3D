"""Isolated NumPy checks for the review; no model/data/training imports.

Runs selected functions extracted from the supplied source with AST.
The matching check supplies a minimal bounding-box/area implementation, not
scikit-image; it verifies matching logic, not scikit-image integration.
"""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / 'audit/history/20260915T134843518714Z_baseline/original'
results = {}


def extracted(name, namespace, strip_imports=False):
    tree = ast.parse((ROOT / 'cytonucs_inference.py').read_text(encoding='utf-8-sig'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    if strip_imports:
        node.body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(ROOT / 'cytonucs_inference.py'), 'exec'), namespace)
    return namespace[name]


# Even a perfect refinement cannot remove false-positive polyhedron voxels.
place = extracted('_place', {'np': np, 'nd': 3, 'nms_thresh': 0.3})
labels = np.ones((2, 3, 4), np.int32)
gt = np.zeros_like(labels, dtype=bool)
gt[:, 1:2, 1:3] = True
sl = tuple(slice(0, s) for s in labels.shape)
out = np.zeros_like(labels)
place(labels, 1, sl, sl, gt, out, {})
score = np.logical_and(out > 0, gt).sum() / np.logical_or(out > 0, gt).sum()
assert np.all(out == labels)
results['perfect_refinement_for_oversized_polyhedron'] = {
    'gt_voxels': int(gt.sum()), 'output_voxels': int((out > 0).sum()),
    'iou': float(score), 'perfect_mask_iou_before_place': 1.0,
}


# A neighbour's polyhedron forbids recovering territory even with perfect masks.
labels = np.zeros((1, 1, 8), np.int32)
labels[0, 0, :2] = 1
labels[0, 0, 2:6] = 2
gt = np.zeros_like(labels, dtype=bool)
gt[0, 0, :4] = True
sl = tuple(slice(0, s) for s in labels.shape)
out = np.zeros_like(labels)
place(labels, 1, sl, sl, gt, out, {})
assert not out[0, 0, 2:4].any()
results['perfect_refinement_cannot_reclaim_neighbour'] = {
    'gt_voxels': int(gt.sum()), 'output_voxels': int((out > 0).sum()),
}


def regionprops_minimal(arr):
    props = []
    for lab in np.unique(arr):
        if lab == 0:
            continue
        pts = np.argwhere(arr == lab)
        props.append(SimpleNamespace(label=int(lab), area=len(pts),
                     bbox=tuple(pts.min(0)) + tuple(pts.max(0) + 1)))
    return props


match = extracted('match_instances', {'np': np, 'regionprops': regionprops_minimal}, True)
gt = np.array([[[1, 1, 2, 2]]], np.int32)
pred = np.ones_like(gt)
matches = match(pred, gt, 0.5)
assert matches == {1: 1, 2: 1}
results['non_bijective_matching_at_iou_half'] = {'matches': matches, 'unique_predictions': len(set(matches.values()))}


# Formula in the supplied model is softplus(Wx + b), not softplus(Wx) + b.
z = math.log(math.expm1(1.0))
results['softplus_for_distance_one'] = {
    'total_preactivation': z, 'derivative': 1 / (1 + math.exp(-z)),
    'linear_contribution_if_bias_99': z - 99,
}
assert math.isclose(1 / (1 + math.exp(-z)), 1 - math.exp(-1))


# Reproduce the current model pooling algorithm vs inference's legacy formula.
cfg = json.loads((ROOT / 'config_cytonucs.json').read_text())
patch = cfg['training']['train_patch_size']
grid = cfg['model']['nucleus_grid']
size = [s // g for s, g in zip(patch, grid)]
pools = []
for _ in range(cfg['model']['unet_n_depth']):
    p = [2 if s % 2 == 0 and s // 2 >= 2 else 1 for s in size]
    pools.append(p)
    size = [s // v for s, v in zip(size, p)]
actual = np.array(grid) * np.prod(pools, axis=0)
aniso = cfg['model']['anisotropy']
legacy_pool = [2 if a < 2 * min(aniso) else 1 for a in aniso]
legacy = np.array(grid) * np.array(legacy_pool) ** cfg['model']['unet_n_depth']
assert actual.tolist() == [8, 32, 32]
assert legacy.tolist() == [1, 32, 32]
results['input_divisibility'] = {'model': actual.tolist(), 'inference': legacy.tolist(), 'pools': pools}


# Directions follow the official Rays_GoldenSpiral algorithm (unit normalized).
n = 96
z = np.linspace(-1, 1, n)
phi = (3 - math.sqrt(5)) * math.pi * np.arange(n)
rho = np.sqrt(1 - z * z)
base = np.stack([z, rho * np.sin(phi), rho * np.cos(phi)], axis=1)
directions = []
for a in (19, 28):
    v = base / [a, 1, 1]
    directions.append(v / np.linalg.norm(v, axis=1, keepdims=True))
angles = np.degrees(np.arccos(np.clip(np.sum(directions[0] * directions[1], axis=1), -1, 1)))
# Identical nested shapes at the same centre, but indexwise distances differ.
axes_n = np.array([2.0, 38.0, 38.0])
axes_c = axes_n * 1.001
dn = 1 / np.sqrt(np.sum((directions[0] / axes_n) ** 2, axis=1))
dc = 1 / np.sqrt(np.sum((directions[1] / axes_c) ** 2, axis=1))
# This particular geometry need not violate; report directional discrepancy.
results['different_ray_directions'] = {'max_angle_degrees': float(angles.max()),
                                       'different_indices': int((angles > 1e-5).sum())}
# An elongated shape along Z is a concrete counterexample to indexwise containment.
axes_n = np.array([8.0, 3.0, 3.0])
axes_c = axes_n * 1.001
dn = 1 / np.sqrt(np.sum((directions[0] / axes_n) ** 2, axis=1))
dc = 1 / np.sqrt(np.sum((directions[1] / axes_c) ** 2, axis=1))
assert (dn > dc).any()
results['nested_ellipsoids_false_containment_penalty'] = {
    'note': 'Analytic counterexample, not a measured HeLa sample.',
    'violating_rays': int((dn > dc).sum()), 'max_false_violation_px': float(np.maximum(dn-dc, 0).max()),
}


history = json.loads((ROOT / 'history.json').read_text())
results['history'] = {'training_epochs': len(history['train_loss']),
                      'last_fullval_epoch': history['fullval_epoch'][-1],
                      'best_jtpr': max(history['fullval_jtpr']),
                      'last_jtpr': history['fullval_jtpr'][-1]}
parsed = []
for p in sorted(ROOT.glob('*.py')):
    ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    parsed.append(p.name)
results['syntax_checked_files'] = parsed
path = WORKSPACE / 'audit' / 'baseline_recheck.json'
path.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
