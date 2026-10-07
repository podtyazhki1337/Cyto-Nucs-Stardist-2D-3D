"""Read-only detection review of supplied v2.8/v2.11 results; never executes model code.

Usage: python audit/analyze_v2_11_detection.py  ->  audit/v2_11_detection_review.json

Edge distances use the volume shape inferred from refinement ROIs (max origin+shape).
Candidate locations are bounding-box lower corners (the JSONL stores no upper corner),
so "near edge" is a proxy: lower corner <=4 px from the low XY faces or within 40 px
of the high XY faces. All numbers are diagnostics, not JTPR gains.
"""
import json
import re
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
V28, V211 = ROOT / 'fixes_v2.8', ROOT / 'fixes_v2.11'
HEADS = ('nucleus', 'cytoplasm')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def volume_shape(audit):
    return np.max([np.add(r['origin'], r['shape']) for r in audit['rois']], axis=0)


def near_edge(lo, shape):
    return bool(lo[1] <= 4 or lo[2] <= 4 or lo[1] >= shape[1] - 40 or lo[2] >= shape[2] - 40)


def fullval_table(history_path):
    h = json.loads(history_path.read_text())
    keys = ('jtpr', 'nucleus_tp', 'nucleus_fp', 'nucleus_fn', 'cytoplasm_tp', 'cytoplasm_fp', 'cytoplasm_fn')
    table = []
    for i, epoch in enumerate(h['fullval_epoch']):
        if epoch is not None and epoch >= 400:
            table.append(dict(epoch=int(epoch), **{k: h['fullval_' + k][i] for k in keys}))
    return table


def proposal_counts(console_path):
    """Per-validation sums of preselected peaks and the peak windows actually used."""
    text = console_path.read_text(encoding='utf-8', errors='replace')
    result = {}
    for head in HEADS:
        found = re.findall(r'\[%s\] \d+ threshold candidates -> (\d+) preselected \(win=\(([\d, ]+)\)' % head, text)
        counts = [int(n) for n, _ in found]
        result[head] = dict(windows=sorted(set(w for _, w in found)), volumes_logged=len(counts),
                            per_block_of_61=[sum(counts[i:i + 61]) for i in range(0, len(counts), 61)])
    return result


def summarize(path):
    data = rows(path)
    out = dict(volumes=len(data), pair_failures=dict(sum((Counter(r['pair_failures']) for r in data), Counter())))
    for head in HEADS:
        monitoring, refinement, sweeps = Counter(), Counter(), defaultdict(lambda: np.zeros(3))
        missed, written = Counter(), []
        edge = defaultdict(Counter)
        prompts_near_edge = prompts = 0
        ranking = Counter()
        quality, best_iou = [], []
        for r in data:
            a = r['stage_audit'][head]
            shape = volume_shape(a)
            monitoring.update({k: v for k, v in a['monitoring'].items() if isinstance(v, (int, float))})
            refinement.update({k: v for k, v in r['refinement'][head].items() if isinstance(v, int)})
            for s in a['proposals']['sweeps']:
                sweeps[s['threshold']] += [s['gt_above_threshold'], s['gt_with_local_max'], s['background_local_maxima']]
            for roi in a['rois']:
                c = roi['center_global']
                prompts += 1
                prompts_near_edge += min(c[1], c[2], shape[1] - 1 - c[1], shape[2] - 1 - c[2]) <= 4
            before = a['stages']['before_nms']['best_iou_per_gt']
            final = a['stages']['final']['best_iou_per_gt']
            parts = a['fragmentation']['before_nms']['parts_per_gt']
            centers = a['prompt_centers_per_gt']
            for g in final:
                if final[g] >= .5:
                    continue
                if before[g] >= .5:
                    missed['good_candidate_lost_in_assembly'] += 1
                elif g not in centers:
                    missed['no_prompt_center'] += 1
                elif not parts.get(g):
                    missed['center_but_no_dominant_part'] += 1
                elif max(p['gt_coverage'] for p in parts[g]) < .5:
                    missed['center_dominant_part_coverage_below_0_5'] += 1
                else:
                    missed['center_dominant_part_coverage_at_least_0_5'] += 1
            decisions = a.get('assembly_decisions', [])
            location = {}
            by_gt = defaultdict(list)
            for d in decisions:
                for g, v in d['gt_iou'].items():
                    if v > location.get(g, (0, None))[0]:
                        location[g] = (v, d['origin'])
                if d['best_gt'] is not None:
                    by_gt[d['best_gt']].append(d)
                    quality.append(d['quality']); best_iou.append(d['best_iou'])
                if d['reason'] == 'written':
                    iou = max(d['written_gt_iou'].values(), default=0.)
                    written.append(dict(iou=iou, quality=d['quality'], area=d['connected_voxels'],
                                        edge=near_edge(d['origin'], shape)))
            if decisions:
                for g in final:
                    status = 'matched' if final[g] >= .5 else 'lost_in_assembly' if before[g] >= .5 else 'no_good_candidate'
                    if g in location:
                        edge[status]['n'] += 1
                        edge[status]['near_xy_edge'] += near_edge(location[g][1], shape)
                for g, candidates in by_gt.items():
                    if len(candidates) < 2:
                        continue
                    top = max(candidates, key=lambda d: d['quality'])
                    if max(d['best_iou'] for d in candidates) >= .5 and top['best_iou'] < .5:
                        ranking['top_quality_bad_while_good_exists'] += 1
                        ranking['of_which_final_missed'] += final[g] < .5
        head_result = dict(monitoring=dict(monitoring), refinement=dict(refinement),
                           sweeps_gt_above_gt_localmax_background_localmax={
                               str(k): v.astype(int).tolist() for k, v in sorted(sweeps.items())},
                           prompts=prompts, prompts_within_4px_xy_edge=int(prompts_near_edge),
                           missed_final=dict(missed))
        if written:
            iou = np.array([w['iou'] for w in written]); q = np.array([w['quality'] for w in written])
            area = np.array([w['area'] for w in written]); tp = iou >= .5
            classes = dict(tp=tp, fp_partial=(iou < .5) & (iou > .1), fp_touch=(iou > 0) & (iou <= .1), fp_no_gt=iou == 0)
            head_result['written'] = {k: dict(count=int(m.sum()),
                                              quality_q10_50_90=np.round(np.quantile(q[m], [.1, .5, .9]), 3).tolist() if m.any() else None,
                                              area_q10_50=np.quantile(area[m], [.1, .5]).astype(int).tolist() if m.any() else None)
                                      for k, m in classes.items()}
            head_result['written']['fp_no_gt_near_xy_edge'] = int(sum(w['edge'] for w in written if w['iou'] == 0))
            head_result['quality_gate_counterfactual'] = {
                str(t): dict(fp_removed=int((~tp & (q < t)).sum()), tp_removed=int((tp & (q < t)).sum()))
                for t in (.2, .3, .35, .4, .5)}
            head_result['area_gate_counterfactual'] = {
                str(t): dict(fp_removed=int((~tp & (area < t)).sum()), tp_removed=int((tp & (area < t)).sum()))
                for t in (200, 500, 1000)}
            head_result['gt_location_by_status'] = {k: dict(v) for k, v in edge.items()}
            head_result['quality_ranking'] = dict(ranking, spearman_quality_vs_candidate_best_iou=float(
                spearmanr(quality, best_iou).correlation))
        out[head] = head_result
    return out


def history_means(path, keys, last=25):
    h = json.loads(path.read_text())
    return {k: float(np.mean([v for v in h[k][-last:] if v is not None])) for k in keys if k in h}


def main():
    inputs = sorted(p for d in (V28, V211) for p in d.iterdir() if p.is_file())
    result = dict(
        scope='Read-only aggregation of user result packages; no weights/TIFF were available. '
              'Counterfactual gates assume removal of written masks only (low-quality masks are committed last), '
              'they are not rerun inference and are not JTPR gains.',
        script_sha256=digest(Path(__file__)),
        inputs={str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in inputs},
        v2_11_run_resume=json.loads((V211 / 'run.json').read_text())['requested_config']['training']['resume_from_checkpoint'],
        fullval_v2_11_history=fullval_table(V211 / 'history.json'),
        proposals={'v2.8': proposal_counts(V28 / 'console.log'), 'v2.11': proposal_counts(V211 / 'console.log')},
        epochs={'v2.8_420': summarize(V28 / 'epoch_0420_refinement.jsonl'),
                'v2.11_445': summarize(V211 / 'epoch_0445_refinement.jsonl')},
        quality_supervision_last25=history_means(V211 / 'history.json', [
            'train_nucleus_quality_supervised_fraction', 'train_cytoplasm_quality_supervised_fraction',
            'val_nucleus_quality_supervised_fraction', 'val_cytoplasm_quality_supervised_fraction']),
        nuclear_bce_background_by_epoch={
            k: {str(e): json.loads((V211 / 'history.json').read_text())[k][e - 1] for e in (410, 420, 430, 440, 449)}
            for k in ('train_nucleus_bce_background', 'val_nucleus_bce_background')},
    )
    partial = rows(V211 / 'epoch_0450_refinement.jsonl')
    result['v2.11_450_partial'] = dict(volumes=len(partial), pair_failures=dict(
        sum((Counter(r['pair_failures']) for r in partial), Counter())))
    target = ROOT / 'audit' / 'v2_11_detection_review.json'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
