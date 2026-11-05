"""AP and JTPR on whole held-out blocks of the spatial split instead of thin tiles.

The val and test tiles are 16-slice slabs of the held-out 512 px blocks: almost every cell touches a
tile face (138 of 141 val cells for HeLa), so the tile metrics score fragments that inference on a
real volume never produces. Here every held-out block is predicted over the full Z range of the
volume, with --margin px of the surrounding volume as image context (its labels are not used), and an
object belongs to the block whose XY area holds its centroid: the GT objects of the block (complete,
from the whole-volume annotation) are matched with the predicted objects whose centroid lies in it.

Usage:
  python evaluate_blocks.py --config CONFIG --weights WEIGHTS [--splits test val] [--margin 192] [--gpu 0]
Needs <dataset>/train_regions (dataset_prep/add_train_regions.py: the whole source volumes) and
split_manifest.json. A model trained on a downscaled copy (dataset_prep/downscale_split.py) reads
the image at its own resolution; its labels are enlarged and scored against the native annotation.
Writes <weights>.blocks.json (or --output).
"""
import argparse
import contextlib
import io
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
from scipy import ndimage as ndi

from score_native_resolution import TAUS, enlarge_labels

LEVEL_FILES = {'nucleus': 'nucleus_masks', 'cytoplasm': 'cytoplasm_masks'}


def block_windows(manifest, split, margin, factor=1):
    """(volume name, block (y0, y1, x0, x1), window (y0, y1, x0, x1)) of the held-out blocks of a split,
    native coordinates; window edges are multiples of `factor`."""
    for volume in manifest['volumes']:
        height, width = volume['shape'][1:]
        for block in volume['blocks']:
            if block['split'] != split:
                continue
            y0, x0 = (max(block[k] - margin, 0) // factor * factor for k in ('y0', 'x0'))
            y1 = min(-(-(block['y1'] + margin) // factor) * factor, height)
            x1 = min(-(-(block['x1'] + margin) // factor) * factor, width)
            yield volume['name'], (block['y0'], block['y1'], block['x0'], block['x1']), (y0, y1, x0, x1)


def objects_in_block(labels, block, window):
    """`labels` (a window) with only the objects whose centroid lies in the block's XY area."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if not len(ids):
        return np.zeros_like(labels)
    centres = ndi.center_of_mass(np.ones_like(labels, np.uint8), labels, ids)
    keep = [int(i) for i, c in zip(ids, centres)
            if block[0] <= c[-2] + window[0] < block[1] and block[2] <= c[-1] + window[2] < block[3]]
    return np.where(np.isin(labels, keep), labels, 0).astype(labels.dtype)


def evaluate(config_path, weights, splits=('test', 'val'), margin=192, save=None, gates=None, mask_thresholds=None,
             thresholds=None, tta=False):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        if int(os.environ.get('GPU_MEMORY_LIMIT_MB', 0) or 0):          # a cap when the queue shares the GPU
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.environ['GPU_MEMORY_LIMIT_MB']))])
        else:
            tf.config.experimental.set_memory_growth(gpu, True)
    from cytonucs_runtime import normalize_percentiles as normalize
    from stardist.matching import matching
    from cytonucs_config import CytoNucsConfig, read_config
    from cytonucs_model import build_cytonucs_model
    from cytonucs_inference import CytoNucsPredictor, compute_jtpr
    from cytonucs_trainer import DiceCallback
    from cytonucs_runtime import accuracy_profile, read_model_image

    raw = read_config(config_path, weights)
    dataset = Path(raw['data']['dataset_dir'])
    manifest = json.loads((dataset / 'split_manifest.json').read_text())
    factor = int(manifest.get('xy_downscale', 1))
    native = Path(manifest['downscaled_from']) if factor > 1 else dataset
    native_manifest = json.loads((native / 'split_manifest.json').read_text())
    raw['mode'] = 'inference'
    raw['inference']['checkpoint_path'] = str(weights)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        raw['output']['basedir'] = tmp
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = CytoNucsConfig.from_json(str(path))
            model = build_cytonucs_model(cfg, name='blocks', basedir=tmp)
            model.load_weights(str(weights))
        if gates:
            # Replace calibrated gates ('<head>' quality, '<head>_min_area' size); null removes one.
            merged = dict(getattr(model, 'loaded_quality_gates', None) or {}, **gates)
            model.loaded_quality_gates = {k: v for k, v in merged.items() if v is not None}
        if mask_thresholds:
            model.loaded_mask_thresholds = dict(getattr(model, 'loaded_mask_thresholds', None) or {}, **mask_thresholds)
        print('gates', getattr(model, 'loaded_quality_gates', None), 'mask thresholds', getattr(model, 'loaded_mask_thresholds', None), flush=True)
        inference = raw['inference']
        predictor = CytoNucsPredictor(model=model, config=cfg, prob_thresh_nucleus=inference['prob_thresh_nucleus'],
                                      prob_thresh_cytoplasm=inference['prob_thresh_cytoplasm'], nms_thresh=inference['nms_thresh'])
        if thresholds:
            # Thresholds chosen after training (evaluation.optimize_thresholds): 'auto' reads them next to the weights.
            path = Path(weights).parent / 'optimized_thresholds.json' if thresholds == 'auto' else Path(thresholds)
            chosen = json.loads(path.read_text())
            predictor.prob_thresh_nucleus = float(chosen['prob_thresh_nucleus'])
            predictor.prob_thresh_cytoplasm = float(chosen['prob_thresh_cytoplasm'])
            predictor.nms_thresh = float(chosen['nms_thresh'])
            for head in ('nucleus', 'cytoplasm'):
                if isinstance(chosen.get('mask_thresh_' + head), (int, float)):
                    setattr(cfg, 'mask_thresh_' + head, float(chosen['mask_thresh_' + head]))
            print('thresholds', chosen, flush=True)
        cfg.tta_flips = bool(tta) or bool(getattr(cfg, 'tta_flips', False))
        runner = SimpleNamespace(model=predictor.model)
        channels = int(getattr(cfg, 'n_image_channels', cfg.n_channel_in))
        cache = {}
        for split in splits:
            counts = {(level, ''): np.zeros((3, len(TAUS)), np.int64) for level in LEVEL_FILES}
            joint, objects, blocks = [0, 0], {level: 0 for level in LEVEL_FILES}, []
            for name, block, window in block_windows(native_manifest, split, margin, factor):
                if name not in cache:
                    file = name + '.tiff'
                    cache = {name: dict(
                        image=read_model_image(dataset / 'train_regions' / 'images' / file, 3, channels),
                        **{level: tifffile.imread(native / 'train_regions' / folder / file) for level, folder in LEVEL_FILES.items()},
                        pairs={int(k): int(v) for k, v in json.loads(
                            (native / 'train_regions' / 'assignments' / (name + '.json')).read_text()).items()})}
                volume = cache[name]
                y0, y1, x0, x1 = window
                image = volume['image'][:, y0 // factor:y1 // factor, x0 // factor:x1 // factor]
                image = normalize(image.astype(np.float32), 1, 99.8, axis=(0, 1, 2)).astype(np.float32)
                started = time.time()
                with contextlib.redirect_stdout(io.StringIO()):
                    nuclei, cells, _ = DiceCallback._predict_full_volume(runner, predictor, image)
                seconds = time.time() - started
                gt = {level: volume[level][:, y0:y1, x0:x1] for level in LEVEL_FILES}
                pred = {'nucleus': enlarge_labels(nuclei, factor, gt['nucleus'].shape),
                        'cytoplasm': enlarge_labels(cells, factor, gt['cytoplasm'].shape)}
                gt_in = {level: objects_in_block(labels, block, window) for level, labels in gt.items()}
                pred_in = {level: objects_in_block(labels, block, window) for level, labels in pred.items()}
                if save:
                    stem = f'{split}_{name}_y{block[0]:04d}_x{block[2]:04d}'
                    Path(save).mkdir(parents=True, exist_ok=True)
                    for level in LEVEL_FILES:
                        tifffile.imwrite(Path(save) / f'{stem}_{level}_gt.tiff', gt_in[level], photometric='minisblack')
                        tifffile.imwrite(Path(save) / f'{stem}_{level}_pred.tiff', pred_in[level], photometric='minisblack')
                for level in LEVEL_FILES:
                    objects[level] += int(len(np.unique(gt_in[level])) - 1)
                    for k, r in enumerate(matching(gt_in[level], pred_in[level], thresh=TAUS, criterion='iou')):
                        counts[level, ''][:, k] += (int(r.tp), int(r.fp), int(r.fn))
                _, details = compute_jtpr(pred_in['nucleus'], pred_in['cytoplasm'], gt_in['nucleus'], gt_in['cytoplasm'],
                                          iou_thresh=0.5, assignments_pred=getattr(predictor, '_prompt_assignments', None),
                                          assignments_gt=volume['pairs'])
                joint[0] += int(details['joint_tp'])
                joint[1] += int(details['total_pairs'])
                blocks.append(dict(volume=name, block=list(block), window=list(window), inference_s=round(seconds, 1),
                                   window_voxels=int(np.prod(image.shape[:3])), jtpr=details['joint_tp'] / max(details['total_pairs'], 1),
                                   pairs=int(details['total_pairs']), fail_nucleus=int(details['joint_fail_nucleus']),
                                   fail_cell=int(details['joint_fail_cell']), fail_link=int(details['joint_fail_link'])))
            result = dict(blocks=blocks, jtpr=joint[0] / joint[1] if joint[1] else None, pairs=joint[1],
                          objects=objects, margin=margin)
            for level in LEVEL_FILES:
                tp, fp, fn = counts[level, '']
                ap = accuracy_profile(tp, fp, fn)
                result[f'ap_{level}'] = float(ap.mean())
                result[f'ap50_{level}'] = float(ap[0])
                result[f'recall50_{level}'] = float(tp[0] / max(tp[0] + fn[0], 1))
                result[f'precision50_{level}'] = float(tp[0] / max(tp[0] + fp[0], 1))
            results[split] = result
            print(f"{split:5s} {len(blocks)} block(s) | nuclei {objects['nucleus']}: AP {result['ap_nucleus']:.3f} "
                  f"(AP50 {result['ap50_nucleus']:.3f}, recall {result['recall50_nucleus']:.2f}, precision {result['precision50_nucleus']:.2f}) | "
                  f"cells {objects['cytoplasm']}: AP {result['ap_cytoplasm']:.3f} (AP50 {result['ap50_cytoplasm']:.3f}, "
                  f"recall {result['recall50_cytoplasm']:.2f}, precision {result['precision50_cytoplasm']:.2f}) | "
                  f"JTPR {result['jtpr']:.3f} of {joint[1]} pairs", flush=True)
    return dict(results, weights=str(weights), factor=factor, native=str(native))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--splits', nargs='+', default=['test', 'val'])
    parser.add_argument('--margin', type=int, default=192, help='native px of image context around a block')
    parser.add_argument('--gpu', help='CUDA_VISIBLE_DEVICES; "" runs on the CPU')
    parser.add_argument('--output', help='default: <weights>.blocks.json')
    parser.add_argument('--save', help='folder for the GT and predicted labels of every block')
    parser.add_argument('--gates', type=json.loads, help='JSON overriding the calibrated gates, e.g. {"cytoplasm_min_area": null}')
    parser.add_argument('--mask-thresholds', type=json.loads, help='JSON overriding the calibrated mask thresholds, e.g. {"cytoplasm": 0.5}')
    parser.add_argument('--thresholds', help="'auto' (optimized_thresholds.json next to the weights) or a JSON file")
    parser.add_argument('--tta', action='store_true', help='average the probability maps over flips (inference.tta_flips)')
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    results = evaluate(args.config, args.weights, args.splits, args.margin, args.save, args.gates, args.mask_thresholds,
                       args.thresholds, args.tta)
    output = Path(args.output or str(args.weights) + '.blocks.json')
    output.write_text(json.dumps(results, indent=1))
    print('->', output)


if __name__ == '__main__':
    main()
