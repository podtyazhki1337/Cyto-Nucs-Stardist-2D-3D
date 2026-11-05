"""Per-object semantic scores (as inference.semantic_scores computes them) for predictions already saved by
score_hierarchy.py --predictions, from the detection pass alone: the semantic foreground maps of a two-level model
are read, the proposals and the prompted refinement are skipped, and every saved object gets the mean probability
of its level's head over its voxels. Writes <predictions>/semantic/<stem>.json for semantic_gate.py. Much lighter
than predicting again: the refinement windows of large objects need most of a 24 GB card.

Usage:
  python semantic_scores.py --config CONFIG --weights WEIGHTS --split test --predictions DIR [--gpu 0] [--memory-limit MB]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--predictions', required=True, help='folder of saved predictions (<level>/<name>)')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--memory-limit', type=int, help='GPU memory cap in MB')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import tensorflow as tf
    import tifffile
    for gpu in tf.config.list_physical_devices('GPU'):
        if args.memory_limit:
            tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=args.memory_limit)])
        else:
            tf.config.experimental.set_memory_growth(gpu, True)
    from cytonucs_config import CytoNucsConfig, read_config
    from cytonucs_inference import CytoNucsPredictor, semantic_object_scores
    from cytonucs_model import build_cytonucs_model
    from cytonucs_runtime import normalize_percentiles, read_model_image
    from cytonucs_trainer import DiceCallback

    class SemanticOnly(CytoNucsPredictor):
        """The detection pass of the predictor, keeping the semantic maps; no proposals, no refinement."""
        def _stardist_postprocess(self, prob, dist, prob_thresh, nms_thresh):
            self._last_coords, self._last_probs, self._last_dists = np.zeros((0, prob.ndim), np.int32), np.zeros(0), None
            self._last_residual = None
            return np.zeros(self._last_img_shape, np.int32)

        def _apply_semantic(self, nucleus_instances, cytoplasm_instances, semantic, grids):
            self.semantic_maps, self.semantic_grids = semantic, grids
            return nucleus_instances, cytoplasm_instances

    raw = read_config(args.config)
    raw['mode'] = 'inference'
    raw['inference']['checkpoint_path'] = str(args.weights)
    raw['inference']['semantic_scores'] = True
    folder, predictions = Path(raw['data']['dataset_dir']) / args.split, Path(args.predictions)
    with tempfile.TemporaryDirectory() as tmp:
        raw['output']['basedir'] = tmp
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = CytoNucsConfig.from_json(str(path))
            model = build_cytonucs_model(cfg, name='semantic', basedir=tmp)
            model.load_weights(str(args.weights))
    if cfg.n_pair_channels:
        raise SystemExit('semantic_scores.py handles models of one or two levels')
    inference = raw['inference']
    thresholds = dict(prob_thresh_nucleus=inference['prob_thresh_nucleus'],
                      prob_thresh_cytoplasm=inference['prob_thresh_cytoplasm'], nms_thresh=inference['nms_thresh'])
    (predictions / 'semantic').mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in (predictions / cfg.levels[0]).iterdir())
    for k, name in enumerate(names, 1):
        started = time.time()
        image = read_model_image(folder / 'images' / name, int(cfg.ndim), int(cfg.n_image_channels))
        image = np.asarray(image, np.float32)
        image = image[..., None] if image.ndim == int(cfg.ndim) else image
        image = normalize_percentiles(image, 1, 99.8, axis=tuple(range(int(cfg.ndim))))      # as predict_hierarchy
        predictor = SemanticOnly(model=model, config=cfg, **thresholds)
        with contextlib.redirect_stdout(io.StringIO()):
            DiceCallback._predict_full_volume(SimpleNamespace(model=model), predictor, image)
        scores = {level: semantic_object_scores(tifffile.imread(predictions / level / name), predictor.semantic_maps[role],
                                                predictor.semantic_grids[role])
                  for role, level in cfg.level_roles.items() if (predictions / level / name).is_file()}
        (predictions / 'semantic' / (Path(name).stem + '.json')).write_text(json.dumps(scores))
        print(f'{k}/{len(names)} {name}: {time.time() - started:.0f} s', flush=True)


if __name__ == '__main__':
    main()
