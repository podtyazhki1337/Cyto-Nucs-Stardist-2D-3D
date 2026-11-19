"""The 'auto' inference settings of a config calibrated on validation for trained weights, as at the end of training
(cytonucs_calibration.calibrate_after_training), written to a file of their own. For a variant of the inference (e.g.
an ablation switch such as inference.outer_from_inner_prompts) whose own best settings are wanted next to the ones
the weights were calibrated with: the search starts from AUTO_INFERENCE_DEFAULTS, as in training, whatever the
weights' <weights>.inference.json holds, and that file is not touched.

Usage: python calibrate_inference.py --config CONFIG --weights WEIGHTS --output OUT.json [--gpu 0] [--budget 300]
OUT.json: {values, report}; put the values into the inference section of a scoring config (score_hierarchy.py).
"""
import argparse
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--budget', type=float, default=300.0, help='seconds, as at the end of training')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    from cytonucs_calibration import calibrate_after_training
    from cytonucs_config import AUTO_INFERENCE_DEFAULTS, CytoNucsConfig, auto_inference_keys, read_config
    from cytonucs_data_generator import CytoNucsDataGenerator
    from cytonucs_inference import CytoNucsPredictor
    from cytonucs_model import build_cytonucs_model

    raw = read_config(args.config)
    keys = auto_inference_keys(raw['inference'])
    if not keys:
        raise SystemExit(f'{args.config}: no inference setting is "auto"')
    raw['inference'].update({k: AUTO_INFERENCE_DEFAULTS[k] for k in keys})          # the start of training's search
    raw['mode'] = 'inference'
    raw['inference']['checkpoint_path'] = str(args.weights)
    with tempfile.TemporaryDirectory() as tmp:
        raw['output']['basedir'] = tmp
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = CytoNucsConfig.from_json(str(path))
            model = build_cytonucs_model(cfg, name='calibrate', basedir=tmp)
            model.load_weights(str(args.weights))
        val = CytoNucsDataGenerator(dataset_dir=Path(raw['data']['dataset_dir']) / 'val', config=cfg, rays=cfg.rays,
                                    augmenter=None, shuffle=False, subset='val')
        inference = raw['inference']
        predictor = CytoNucsPredictor(model=model, config=cfg, prob_thresh_nucleus=inference['prob_thresh_nucleus'],
                                      prob_thresh_cytoplasm=inference['prob_thresh_cytoplasm'],
                                      nms_thresh=inference['nms_thresh'])
        values, report = calibrate_after_training(predictor, val.data_pool, keys, budget=args.budget)
    Path(args.output).write_text(json.dumps(dict(values=values, report=report, config=str(args.config),
                                                 weights=str(args.weights)), indent=2))
    print(f'{keys} -> {values} -> {args.output}')


if __name__ == '__main__':
    main()
