"""Boundary-error split of the matched masks kept by the gates, by position and size (read-only).

Usage: python audit/analyze_boundary_split.py [epoch ...] -> audit/boundary_split_epoch<last>.json
Input: fixes_v3.2 history.json and epoch_NNNN_refinement.jsonl written with the boundary split
(cytonucs_refinement_audit.boundary_split). Masks are those written and kept by the calibrated
gates of that epoch; coverage/precision come from the written IoU, mask area and GT volume.
missing_share: fraction of the error voxels that are GT voxels the mask does not cover.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'fixes_v3.2'
SIZE_BINS = {'nucleus': (0, 5000, 15000, 40000), 'cytoplasm': (0, 5000, 15000, 40000)}


def rows(epoch, head, gate, min_area):
    out = []
    for line in (PACKAGE / f'epoch_{epoch:04d}_refinement.jsonl').read_text().splitlines():
        a = json.loads(line)['stage_audit'][head]
        for d in a['assembly_decisions']:
            b = d.get('boundary')
            area = d.get('connected_voxels', d['area'])
            if d['reason'] != 'written' or b is None or d['quality'] < gate or area < min_area:
                continue
            g = a['gt_geometry'][b['gt']]
            iou = d['written_gt_iou'][b['gt']]
            inter = iou * (area + g['volume']) / (1 + iou)
            out.append(dict(volume=g['volume'], iou=iou, coverage=inter / g['volume'], precision=inter / area,
                            missing=(g['volume'] - inter), xy_face=any(f[0] in 'yx' for f in g['faces'].split(',') if f),
                            **{k: b[k] for k in ('z_error_voxels', 'xy_error_voxels', 'mixed_error_voxels',
                                                 'error_voxels', 'extra_slices', 'missing_slices')}))
    return out


def summary(items):
    errors = max(sum(r['error_voxels'] for r in items), 1)
    return dict(n=len(items), iou_median=float(np.median([r['iou'] for r in items])),
                coverage_median=float(np.median([r['coverage'] for r in items])),
                precision_median=float(np.median([r['precision'] for r in items])),
                missing_share=float(sum(r['missing'] for r in items) / errors),
                **{p + '_share': float(sum(r[p + '_error_voxels'] for r in items) / errors) for p in ('z', 'xy', 'mixed')},
                slice_set_exact=float(np.mean([r['extra_slices'] + r['missing_slices'] == 0 for r in items])))


def main(epochs):
    history = json.loads((PACKAGE / 'history.json').read_text(encoding='utf-8'))
    report = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results={})
    for epoch in epochs:
        i = history['fullval_epoch'].index(epoch)
        for head in ('nucleus', 'cytoplasm'):
            items = rows(epoch, head, history[f'fullval_{head}_quality_gate'][i], history[f'fullval_{head}_area_gate'][i] or 0)
            edges = SIZE_BINS[head] + (np.inf,)
            result = dict(
                history={k: history[f'fullval_{k}'][i] for k in
                         [f'ap_{head}_quality_gated'] + [f'ap_{head}_quality_gated_{p}_exact' for p in ('z', 'xy', 'mixed')]},
                all=summary(items),
                xy_face={str(flag): summary([r for r in items if r['xy_face'] == flag]) for flag in (False, True)},
                volume={f'{lo}-{hi}': summary(sel) for lo, hi in zip(edges, edges[1:])
                        for sel in [[r for r in items if lo <= r['volume'] < hi]] if len(sel) >= 20})
            report['results'][f'{epoch}/{head}'] = result
            s = result['all']
            print(epoch, head, {k: round(v, 4) for k, v in result['history'].items()},
                  'missing share %.2f, Z/XY/rims %.3f/%.3f/%.3f' % (s['missing_share'], s['z_share'], s['xy_share'], s['mixed_share']))
    out = ROOT / 'audit' / f'boundary_split_epoch{epochs[-1]}.json'
    out.write_text(json.dumps(report, indent=2, default=float), encoding='utf-8')


if __name__ == '__main__':
    main([int(e) for e in sys.argv[1:]] or [815, 820])
