"""Which annotated nucleus-cell pairs change their cell outcome when the cells predicted from nucleus prompts are removed
(inference.outer_from_inner_prompts false: cells only from the cell head's own proposals), and which candidates of the
joint network lie near those cells (the data behind Figure 6C).

  cache  (GPU) one ungated pass of the joint network over a split (as paper/operating_points.py cache-joint does for
         val): every candidate of both levels with its mask, offset, prompt centre and sigma, detection score, prompt
         head ('nucleus' = a cell candidate from a nucleus prompt, 'cytoplasm' = the cell head's own proposal), predicted
         quality (the estimated IoU) and mask threshold, plus the nucleus-cell evidence pairs -> OUT/cache/<split>/
  pairs  (CPU) every annotated pair of the split, the model as deployed (F) vs the ablation: the annotated cell's best
         predicted cell (IoU), cell matched (one-to-one at 0.5), joint TP (nucleus and cell matched, predicted link
         right). For the pairs whose cell or joint outcome differs, F's cell candidates near the annotated cell from the
         ungated cache, at F's own settings: prompt head, prompt centre / sigma, detection score, predicted quality,
         IoU with the annotated cell and status: 'prob' (prompt below the prob threshold of its head), 'gate'
         (predicted quality below the cell gate), 'kept' (the best candidate, IoU >= 0.8, of a final F cell label) or
         'assembly' (passed both, not in the final labels: NMS / competition / size gate / nesting filter)
         -> OUT/pairs_<split>/ (summary.json, changed_pairs.json / .md)
Layout of OUT for pairs: OUT/F/predictions_<split> and OUT/ownparents/predictions_<split>, the --predictions folders of
score_hierarchy.py (with parents/) for the deployed configuration and the one with outer_from_inner_prompts false;
OUT/cache/<split> from the cache step (--out OUT/cache). --settings: a file with the deployed settings under "own" (the
chosen_joint_*.json of paper/operating_points.py search, or the network.json of its cache-joint).

Usage (from the repository root):
  python paper/prompt_source_pairs.py cache --config CFG --weights W --out OUT/cache --split test [--gpu 0]
  python paper/prompt_source_pairs.py pairs --out OUT --data DATA --settings S.json --split test
"""
import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROLES = ('nucleus', 'cytoplasm')


def cache(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import tempfile
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    from cytonucs_config import CytoNucsConfig, read_config
    from cytonucs_inference import CytoNucsPredictor
    from cytonucs_model import build_cytonucs_model
    from cytonucs_runtime import normalize_percentiles, read_model_image
    from cytonucs_trainer import DiceCallback
    raw = read_config(args.config, args.weights)
    raw['mode'] = 'inference'
    raw['inference']['checkpoint_path'] = str(args.weights)
    inference = raw['inference']
    prob = {f'prob_thresh_{r}': float(inference[f'prob_thresh_{r}']) for r in ROLES}
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    folder = Path(raw['data']['dataset_dir']) / args.split
    names = sorted(p.name for p in (folder / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))[:args.limit or None]
    with tempfile.TemporaryDirectory() as tmp:
        raw['output']['basedir'] = tmp
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = CytoNucsConfig.from_json(str(path))
            model = build_cytonucs_model(cfg, name='op', basedir=tmp)
            model.load_weights(str(args.weights))
        predictor = CytoNucsPredictor(model=model, config=cfg, nms_thresh=inference['nms_thresh'], **prob)
        predictor.calibrating = True
        nd = int(cfg.ndim)
        started = time.time()
        for name in names:
            target = out / (Path(name).stem + '.pkl')
            if target.is_file():
                continue
            image = np.asarray(read_model_image(folder / 'images' / name, nd, int(cfg.n_image_channels)), np.float32)
            if image.ndim == nd:
                image = image[..., None]
            image = normalize_percentiles(image, 1, 99.8, axis=tuple(range(nd)))
            sink = {}
            predictor._refinement_candidate_sink = sink
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    DiceCallback._predict_full_volume(SimpleNamespace(model=model), predictor, image)
            finally:
                predictor._refinement_candidate_sink = None
            cands = {r: [c for c in sink.get(r, []) if c is not None] for r in ROLES}
            index = {r: {id(c): i for i, c in enumerate(cands[r])} for r in ROLES}
            evidence = [(index['nucleus'][id(n)], index['cytoplasm'].get(id(c)) if c is not None else None)
                        for n, c in sink.get('evidence', []) if n is not None and id(n) in index['nucleus']]
            strip = lambda c: {k: v for k, v in c.items() if k != 'roi'}                     # noqa: E731
            with open(target, 'wb') as f:
                pickle.dump(dict(name=name, shape=tuple(int(s) for s in image.shape[:nd]),
                                 candidates={r: [strip(c) for c in cands[r]] for r in ROLES}, evidence=evidence), f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            print(f'cache {name}: {len(cands["nucleus"])} / {len(cands["cytoplasm"])} candidates, {len(evidence)} evidence '
                  f'({time.time() - started:.0f} s)', flush=True)
    (out.parent / f'network_{args.split}.json').write_text(json.dumps(dict(
        config=str(args.config), weights=str(args.weights), split=args.split, prob=prob,
        mask_default=float(inference.get('mask_thresh', 0.45)), images=names), indent=1))
    print('->', out)


def iou_in(mask, lo, target):
    """IoU of a candidate mask (window at lo) with a full-image boolean mask."""
    sl = tuple(slice(int(a), int(a) + n) for a, n in zip(lo, mask.shape))
    inter = int(np.logical_and(mask, target[sl]).sum())
    union = int(mask.sum()) + int(target.sum()) - inter
    return inter / union if union else 0.0


def best_iou(pred, target):
    """(pred id, IoU) of the predicted label with the highest IoU with the boolean mask target (0, 0.0 if none)."""
    ids, inter = np.unique(pred[target], return_counts=True)
    keep = ids > 0
    if not keep.any():
        return 0, 0.0
    ids, inter = ids[keep], inter[keep]
    areas = np.bincount(pred.ravel())[ids]
    iou = inter / (areas + int(target.sum()) - inter)
    k = int(np.argmax(iou))
    return int(ids[k]), float(iou[k])


def pairs(args):
    """(CPU) Every annotated nucleus-cell pair of the split, F vs ownparents: the GT cell's best predicted cell (IoU), cell
    matched (one-to-one at 0.5), joint TP (nucleus and cell matched, predicted link right). For the pairs whose cell or
    joint outcome differs, F's cell candidates near the GT cell from the ungated cache, at F's own settings:
    prompt head, prompt centre / sigma, detection score, predicted quality (estimated IoU), IoU with the GT cell and
    status: 'prob' (prompt below the prob threshold of its head), 'gate' (predicted quality below the cell gate),
    'kept' (the best candidate, IoU >= 0.8, of a final F cell label) or 'assembly' (passed both, not in the final
    labels: NMS / competition / size gate / nesting filter)."""
    import tifffile
    from cytonucs_inference import match_instances
    from cytonucs_postprocess import rethreshold_candidate
    from score_hierarchy import ground_truth, without_ignored
    O = Path(args.out)
    own = json.loads(Path(args.settings).read_text())['own']
    prob = {r: own[r]['prob'] for r in ROLES}
    data = Path(args.data) / args.split
    names = sorted(p.name for p in (data / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))[:args.limit or None]
    rows, changed = [], []
    for name in names:
        stem = Path(name).stem
        gt, truths, ignore = ground_truth(data, name, list(ROLES), [('nucleus', 'cytoplasm')])
        pred = {}
        for v in ('F', 'ownparents'):
            root = O / v / f'predictions_{args.split}'
            labs = {}
            for r in ROLES:
                lab = np.asarray(tifffile.imread(root / r / name))
                if ignore and ignore.get(r) is not None:
                    lab = without_ignored(lab, ignore[r])
                labs[r] = lab
            links = {int(k): int(val) for k, val in json.loads((root / 'parents' / f'{stem}.json').read_text()).items()}
            pred[v] = dict(labs=labs, links=links, m={r: match_instances(labs[r], gt[r]) for r in ROLES})
        image_rows = []
        for n, c in truths['nucleus_cytoplasm'].items():
            gcell = gt['cytoplasm'] == c
            row = dict(image=name, gt_nucleus=int(n), gt_cell=int(c), cell_area=int(gcell.sum()))
            for v, p in pred.items():
                pn, pc = p['m']['nucleus'].get(n), p['m']['cytoplasm'].get(c)
                bid, biou = best_iou(p['labs']['cytoplasm'], gcell)
                row[v] = dict(cell_iou=round(biou, 3), best_cell=bid, cell_matched=pc is not None, nucleus_matched=pn is not None,
                              joint=bool(pn is not None and pc is not None and p['links'].get(pn) == pc))
            image_rows.append(row)
        rows += image_rows
        todo = [r for r in image_rows if r['F']['cell_matched'] != r['ownparents']['cell_matched'] or r['F']['joint'] != r['ownparents']['joint']]
        if todo:
            item = pickle.loads((O / 'cache' / args.split / f'{stem}.pkl').read_bytes())
            fcells = pred['F']['labs']['cytoplasm']
            cands = []
            for k, cand in enumerate(item['candidates']['cytoplasm']):
                status = None
                if not cand.get('residual', False) and cand.get('detection_score', 1.0) < prob[cand.get('prompt_head', 'nucleus')]:
                    status = 'prob'
                if abs(own['cytoplasm']['mask'] - float(cand.get('threshold', 0.45))) > 1e-9:
                    orig = cand
                    cand = rethreshold_candidate(orig, own['cytoplasm']['mask']) or dict(orig, mask=np.zeros((0, 0), bool))
                    cand.update({key: orig[key] for key in ('detection_score', 'prompt_head', 'residual', 'predicted_quality',
                                                            'prompt_center', 'prompt_sigma') if key in orig})
                if status is None and cand.get('predicted_quality', 1.0) < own['cytoplasm']['gate']:
                    status = 'gate'
                cands.append((k, cand, status))
            for r in todo:
                gcell = gt['cytoplasm'] == r['gt_cell']
                gnuc = gt['nucleus'] == r['gt_nucleus']
                near = []
                for k, cand, status in cands:
                    if cand['mask'].size == 0:
                        continue
                    iou = iou_in(cand['mask'], cand['lo'], gcell)
                    ctr = np.clip(np.round(np.asarray(cand['prompt_center'])).astype(int), 0, np.array(gcell.shape) - 1)
                    if iou < 0.1 and not gcell[tuple(ctr)]:
                        continue
                    near.append(dict(index=k, head=cand.get('prompt_head'), residual=bool(cand.get('residual', False)),
                                     prompt_center=[round(float(x), 1) for x in cand['prompt_center']],
                                     prompt_sigma=[round(float(x), 2) for x in np.ravel(cand['prompt_sigma'])],
                                     prompt_in_gt_nucleus=bool(gnuc[tuple(ctr)]),
                                     detection_score=round(float(cand.get('detection_score', 1.0)), 3),
                                     predicted_quality=round(float(cand.get('predicted_quality', float('nan'))), 3),
                                     iou_gt_cell=round(iou, 3), status=status, lo=[int(x) for x in cand['lo']],
                                     shape=[int(x) for x in cand['mask'].shape], _mask=cand['mask']))
                labels = {int(v) for v in np.unique(fcells[gcell]) if v}
                for lab in labels:
                    target = fcells == lab
                    scored = [(iou_in(d['_mask'], d['lo'], target), i) for i, d in enumerate(near) if d['status'] is None]
                    if scored:
                        best, i = max(scored)
                        if best >= 0.8:
                            near[i]['status'] = 'kept'
                            near[i]['final_label'] = lab
                            near[i]['iou_final_label'] = round(best, 3)
                for d in near:
                    d['status'] = d['status'] or 'assembly'
                    d.pop('_mask')
                r['candidates'] = sorted(near, key=lambda d: -d['iou_gt_cell'])
                changed.append(r)
        print(name, len(image_rows), len(todo), flush=True)
    summary = dict(pairs=len(rows))
    for key in ('cell_matched', 'joint'):
        summary[key] = dict(F=sum(r['F'][key] for r in rows), ownparents=sum(r['ownparents'][key] for r in rows),
                            lost=sum(r['F'][key] and not r['ownparents'][key] for r in rows),
                            gained=sum(r['ownparents'][key] and not r['F'][key] for r in rows))
    lost = [r for r in changed if r['F']['cell_matched'] and not r['ownparents']['cell_matched']]
    for r in lost:
        kept = [d for d in r['candidates'] if d['status'] == 'kept' and d.get('final_label') == r['F']['best_cell']]
        r['F_kept_from'] = kept[0]['head'] if kept else None
        r['own_proposals'] = sum(d['head'] == 'cytoplasm' for d in r['candidates'])
    summary['cell_lost_F_kept_from'] = {h: sum(r.get('F_kept_from') == h for r in lost) for h in ('nucleus', 'cytoplasm', None)}
    out = O / f'pairs_{args.split}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    (out / 'changed_pairs.json').write_text(json.dumps(changed, indent=1, default=str))
    lines = [f"# {args.data} {args.split}: F vs cells only from their own proposals", '',
             '```', json.dumps(summary, indent=1, default=str), '```', '',
             '## Cell matched in F, lost without the cells from nucleus prompts (F IoU high first)', '',
             '| image | GT nucleus | GT cell | cell area | F cell IoU | own-proposals cell IoU | F joint / own joint | F cell kept from | own proposals near | candidates (head: quality / IoU GT / status) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(lost, key=lambda r: (r['ownparents']['cell_iou'] - r['F']['cell_iou'])):
        cs = '; '.join(f"{d['head'][0]}: {d['predicted_quality']:.2f} / {d['iou_gt_cell']:.2f} / {d['status']}" for d in r['candidates'][:6])
        lines.append(f"| {r['image']} | {r['gt_nucleus']} | {r['gt_cell']} | {r['cell_area']} | {r['F']['cell_iou']:.3f} | "
                     f"{r['ownparents']['cell_iou']:.3f} | {int(r['F']['joint'])} / {int(r['ownparents']['joint'])} | {r['F_kept_from']} | "
                     f"{r['own_proposals']} | {cs} |")
    (out / 'changed_pairs.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=1, default=str))
    print('->', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('cache', 'pairs'))
    parser.add_argument('--config')
    parser.add_argument('--weights')
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--data', help='pairs: the dataset folder')
    parser.add_argument('--settings', help='pairs: a file with the deployed settings under "own"')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root: the CytoNucs modules
    if args.step == 'cache':
        cache(args)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        pairs(args)


if __name__ == '__main__':
    main()
