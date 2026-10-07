"""Retrospective AP after the gates with patch-edge fragments ignored (cytonucs_runtime.edge_fragment_ignored_ap)
on user validation JSONL, with the gates of each epoch from history. Reporting only.

Usage: python audit/edge_fragment_metric_retro.py -> audit/edge_fragment_metric_retro.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cytonucs_runtime import edge_fragment_ignored_ap

TAUS = np.round(np.arange(0.5, 0.951, 0.05), 2)
RUNS = (('fixes_v3.1', 635), ('fixes_v3.2', 725), ('fixes_v3.2', 730), ('fixes_v3.2', 735))


def evaluate(package, epoch):
    h = json.loads((ROOT / package / 'history.json').read_text(encoding='utf-8'))
    i = h['fullval_epoch'].index(epoch)
    out = {}
    for head in ('nucleus', 'cytoplasm'):
        gate, min_area = h[f'fullval_{head}_quality_gate'][i], h[f'fullval_{head}_area_gate'][i]
        fragment = lambda info: bool(info and info[0] and info[1] < min_area)
        masks, gt = [], []
        for line in (ROOT / package / f'epoch_{epoch:04d}_refinement.jsonl').read_text().splitlines():
            a = json.loads(line)['stage_audit'][head]
            edge = {g: ('y' in v['faces'] or 'x' in v['faces'], v['volume']) for g, v in a['gt_geometry'].items()}
            gt += [fragment(v) for v in edge.values()]
            for d in a['assembly_decisions']:
                if d['reason'] == 'written':
                    g, iou = max(d['written_gt_iou'].items(), key=lambda kv: kv[1], default=(None, 0.))
                    masks.append((d['quality'], iou, d.get('connected_voxels', d['area']), fragment(edge.get(g))))
        out[head] = dict(ap_quality_gated=h[f'fullval_ap_{head}_quality_gated'][i],
                         **edge_fragment_ignored_ap(masks, gt, gate, min_area, TAUS))
    return out


def main():
    report = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Reporting metric on user JSONL; not a checkpoint selection metric.',
                  results={f'{p}/{e}': evaluate(p, e) for p, e in RUNS})
    (ROOT / 'audit' / 'edge_fragment_metric_retro.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for k, v in report['results'].items():
        print(k, {h: (round(x['ap_quality_gated'], 3), round(x['ap'], 3), x['ignored_gt']) for h, x in v.items()})


if __name__ == '__main__':
    main()
