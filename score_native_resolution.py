"""AP and JTPR of a checkpoint trained on a copy of the dataset at lower XY resolution
(dataset_prep/downscale_split.py, preset v5), against the ORIGINAL annotation: the val / test tiles
of the downscaled copy are predicted, the predicted labels are enlarged by the factor in Y and X and
matched with the native masks. These numbers compare with runs trained at native resolution
(v4, StarDist); the full validation during training scores against the downscaled masks.

Usage:
  python score_native_resolution.py --config config_cytonucs_v5.json \
      --weights models/cytonucs_emese_HK_3_training_v5/weights_best.weights.h5 [--splits val test] [--gpu 0]
The downscaled dataset is data.dataset_dir of the config; the native dataset and the factor are read
from its split_manifest.json ('downscaled_from', 'xy_downscale'), unless --native is given.
Writes <weights>.native_scores.json. Quality gates and mask thresholds are those saved with the weights,
probability and NMS thresholds those of the config's inference section.
"""
import argparse
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def downscaling(dataset):
    """(native dataset, factor) recorded in split_manifest.json by dataset_prep/downscale_split.py."""
    manifest = json.loads((Path(dataset) / 'split_manifest.json').read_text())
    factor = int(manifest.get('xy_downscale', 1))
    if factor < 2 or not manifest.get('downscaled_from'):
        raise ValueError(f'{dataset} is not a downscaled copy (no xy_downscale / downscaled_from in split_manifest.json)')
    return Path(manifest['downscaled_from']), factor


def enlarge_labels(labels, factor, shape):
    """Every pixel repeated factor x factor times in Y and X, then cut or padded with background to `shape`."""
    big = np.repeat(np.repeat(np.asarray(labels), factor, axis=-2), factor, axis=-1)
    out = np.zeros(shape, big.dtype)
    z, h, w = (min(a, b) for a, b in zip(shape, big.shape))
    out[:z, :h, :w] = big[:z, :h, :w]
    return out


def score(config_path, weights, splits=('val', 'test'), native=None):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    from stardist.matching import matching
    from cytonucs_config import CytoNucsConfig, read_config
    from cytonucs_model import build_cytonucs_model
    from cytonucs_inference import CytoNucsPredictor, compute_jtpr
    from cytonucs_data_generator import CytoNucsDataGenerator
    from cytonucs_trainer import DiceCallback, without_xy_face_objects
    from cytonucs_runtime import accuracy_profile

    raw = read_config(config_path, weights)
    dataset = Path(raw['data']['dataset_dir'])
    source, factor = downscaling(dataset)
    native = Path(native) if native else source
    raw['mode'] = 'inference'
    raw['inference']['checkpoint_path'] = str(weights)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        raw['output']['basedir'] = tmp
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = CytoNucsConfig.from_json(str(path))
            model = build_cytonucs_model(cfg, name='score_native', basedir=tmp)
            model.load_weights(str(weights))
        if cfg.n_pair_channels:
            raise NotImplementedError('hierarchies of three or more levels are not scored here')
        print(f'{dataset} = {native} / {factor} in XY | gates {model.loaded_quality_gates} | '
              f'mask thresholds {model.loaded_mask_thresholds}', flush=True)
        inference = raw['inference']
        for split in splits:
            with contextlib.redirect_stdout(io.StringIO()):
                gen = CytoNucsDataGenerator(dataset / split, cfg, cfg.rays, augmenter=None, shuffle=False, subset='val')
                predictor = CytoNucsPredictor(model=model, config=cfg, prob_thresh_nucleus=inference['prob_thresh_nucleus'],
                                              prob_thresh_cytoplasm=inference['prob_thresh_cytoplasm'],
                                              nms_thresh=inference['nms_thresh'])
            roles = list(gen.role_levels)
            counts = {(role, part): np.zeros((3, len(TAUS)), np.int64) for role in roles for part in ('', '_whole')}
            joint = [0, 0]
            runner = SimpleNamespace(model=predictor.model)
            for sample in sorted(gen.data_pool, key=lambda s: s['filename']):
                with contextlib.redirect_stdout(io.StringIO()):
                    nuclei, cells, _ = DiceCallback._predict_full_volume(runner, predictor, np.asarray(sample['image'], np.float32))
                gt = {role: tifffile.imread(native / split / f'{level}_masks' / sample['filename'])
                      for role, level in gen.role_levels.items()}
                pred = {role: enlarge_labels(labels, factor, gt[role].shape)
                        for role, labels in (('nucleus', nuclei), ('cytoplasm', cells)) if role in gt}
                pairs = native / split / gen.assignments_name / (Path(sample['filename']).stem + '.json')
                if gen.paired and pairs.is_file():
                    pairs = {int(k): int(v) for k, v in json.loads(pairs.read_text()).items()}
                    _, j = compute_jtpr(pred['nucleus'], pred['cytoplasm'], gt['nucleus'], gt['cytoplasm'], iou_thresh=0.5,
                                        assignments_pred=getattr(predictor, '_prompt_assignments', None), assignments_gt=pairs)
                    joint[0] += int(j['joint_tp'])
                    joint[1] += int(j['total_pairs'])
                for role in roles:
                    for part, (g, p) in (('', (gt[role], pred[role])), ('_whole', without_xy_face_objects(gt[role], pred[role]))):
                        for k, r in enumerate(matching(g, p, thresh=TAUS, criterion='iou')):
                            counts[role, part][:, k] += (int(r.tp), int(r.fp), int(r.fn))
            result = dict(volumes=len(gen.data_pool), jtpr=joint[0] / joint[1] if joint[1] else None)
            for (role, part), c in counts.items():
                ap = accuracy_profile(*c)
                result[f'ap_{role}{part}'] = float(ap.mean())
                result[f'ap50_{role}{part}'] = float(ap[0])
            results[split] = result
            line = ' | '.join(f"AP {role} {result[f'ap_{role}']:.3f} (whole {result[f'ap_{role}_whole']:.3f}, "
                              f"AP50 {result[f'ap50_{role}']:.3f})" for role in roles)
            jtpr = '' if result['jtpr'] is None else f" | JTPR {result['jtpr']:.3f}"
            print(f"{split:5s} {result['volumes']:2d} tiles | {line}{jtpr}", flush=True)
    return dict(results, native=str(native), downscaled=str(dataset), factor=factor, weights=str(weights))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--splits', nargs='+', default=['val', 'test'])
    parser.add_argument('--native', help='native dataset (default: downscaled_from of the split manifest)')
    parser.add_argument('--gpu', help='CUDA_VISIBLE_DEVICES; "" runs on the CPU')
    parser.add_argument('--output', help='default: <weights>.native_scores.json')
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    results = score(args.config, args.weights, args.splits, args.native)
    output = Path(args.output or str(args.weights) + '.native_scores.json')
    output.write_text(json.dumps(results, indent=1))
    print('->', output)


if __name__ == '__main__':
    main()
