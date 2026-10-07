"""Bounded probability audit: no NMS/refinement, one forward per validation patch."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def probability_model(model):
    import tensorflow as tf
    return tf.keras.Model(model.keras_model.inputs, {
        head: model.keras_model.output[head + '_prob']
        for head in ('nucleus', 'cytoplasm')})


def summarize(prob, labels, grid, thresholds, border=2):
    """GT coverage means a candidate voxel exists; it is NOT instance recall."""
    from scipy.ndimage import maximum
    gt = labels[tuple(slice(None, None, int(g)) for g in grid)]
    prob = np.asarray(prob)[tuple(slice(0, s) for s in gt.shape)]
    if prob.shape != gt.shape or not np.isfinite(prob).all():
        raise ValueError('Probability shape mismatch or nonfinite prediction')
    keep = np.ones(gt.shape, bool)
    if border > 0:
        for axis in (-2, -1):
            sl = [slice(None)] * gt.ndim
            sl[axis] = slice(0, border); keep[tuple(sl)] = False
            sl[axis] = slice(max(0, gt.shape[axis] - border), None); keep[tuple(sl)] = False
    ids = np.unique(labels); ids = ids[ids > 0]
    sampled = np.unique(gt[keep]); sampled = sampled[sampled > 0]
    maxima = np.asarray(maximum(np.where(keep, prob, -1.), labels=gt, index=sampled)) if sampled.size else np.array([])
    return {
        'gt_count': int(ids.size), 'gt_with_grid_voxels_inside_border': int(sampled.size),
        'prob_max': float(prob.max()), 'prob_max_inside_border': float(prob[keep].max()) if keep.any() else None,
        'gt_instance_max_quantiles': np.quantile(maxima, [0, .1, .5, .9, 1]).tolist() if maxima.size else [],
        'thresholds': [{
            'threshold': float(t),
            'gt_with_candidate': int(np.count_nonzero(maxima > t)),
            'foreground_candidate_voxels': int(np.count_nonzero(keep & (gt > 0) & (prob > t))),
            'background_candidate_voxels': int(np.count_nonzero(keep & (gt == 0) & (prob > t))),
        } for t in thresholds],
    }


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, help='Fixed epoch weights; geometry.json must be alongside')
    parser.add_argument('--output', required=True, help='New JSON path; existing file is never overwritten')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose a new path')
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error('Checkpoint does not exist')
    import tensorflow as tf
    import tifffile
    from csbdeep.utils import normalize
    from cytonucs_config import CytoNucsConfig
    from cytonucs_model import build_cytonucs_model
    from cytonucs_geometry import architecture_plan, pad_to_multiple
    from cytonucs_gpu_setup import setup_gpu
    from cytonucs_runtime import sanitized
    import importlib.metadata

    requested = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if requested.get('model', {}).get('use_gpu', False):
        setup_gpu(**requested.get('gpu_config', {}))
    cfg = CytoNucsConfig.from_json(args.config, checkpoint_path=checkpoint)
    tf.keras.utils.set_random_seed(cfg.seed)
    folder = Path(requested['data']['dataset_dir']) / 'val'
    files = sorted(p for p in (folder / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    if not files:
        raise ValueError('No validation TIFF files')
    selected = [files[i] for i in np.linspace(0, len(files)-1, min(3, len(files)), dtype=int)]
    # No model artifacts are written by this script into the training directory.
    model = build_cytonucs_model(cfg, name='probability_audit', basedir=str(output.parent))
    model.load_weights(checkpoint)
    net = probability_model(model)
    forward = tf.function(lambda x: net(x, training=False), reduce_retracing=True)
    thresholds = sorted(set([.2, .3, .4, .5, .6, float(cfg.prob_thresh_nucleus), float(cfg.prob_thresh_cytoplasm)]))
    report = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_hash(checkpoint),
                  config=sanitized(requested), seed=cfg.seed, script_sha256=file_hash(__file__),
                  geometry=json.loads((checkpoint.parent/'geometry.json').read_text()),
                  packages={p: importlib.metadata.version(p) for p in ('tensorflow','keras','stardist','numpy','csbdeep')},
                  split='val', selected_files=[str(p) for p in selected],
                  scope='GT candidate coverage before NMS; not TP/FP, recall or JTPR', results=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        def save():
            stream.seek(0); json.dump(report, stream, indent=2); stream.truncate(); stream.flush()
        save()
        for path in selected:
            img = tifffile.imread(path)
            if cfg.ndim == 3 and img.ndim == 4 and img.shape[1] in (3, 4) and img.shape[-1] != cfg.n_channel_in:
                img = np.moveaxis(img, 1, -1)
            if img.ndim == cfg.ndim:
                img = img[..., None]
            if img.shape[-1] != cfg.n_channel_in:
                raise ValueError(f'Unexpected channels: {path}: {img.shape}')
            if any(s > limit for s, limit in zip(img.shape[:cfg.ndim], (16, 512, 512) if cfg.ndim == 3 else (512, 512))):
                raise ValueError('This bounded audit accepts patches up to 16x512x512; no full-volume inference')
            source_shape = img.shape[:cfg.ndim]
            img = normalize(img.astype(np.float32), 1, 99.8, axis=tuple(range(cfg.ndim)))
            img = pad_to_multiple(img, architecture_plan(cfg)['input_multiple'])
            x = np.concatenate([img, np.zeros(img.shape[:cfg.ndim]+(2*cfg.prompts_per_crop,), np.float32)], axis=-1)
            pred = forward(tf.convert_to_tensor(x[None], tf.float32))
            row = dict(file=str(path), image_sha256=file_hash(path))
            for head in ('nucleus','cytoplasm'):
                mask_path = folder / (head+'_masks') / path.name
                labels = tifffile.imread(mask_path)
                if labels.shape != source_shape:
                    raise ValueError(f'GT/image shape mismatch: {mask_path}')
                row[head] = summarize(pred[head].numpy()[0,...,0], labels, getattr(cfg, head+'_grid'), thresholds,
                                      int(getattr(cfg,'nms_border_exclude_xy',2)))
                row[head]['gt_sha256'] = file_hash(mask_path)
            report['results'].append(row); save()
            print(f'Probability audit: {len(report["results"])}/{len(selected)} — {path.name}')
        report['status'] = 'complete'; save()
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
