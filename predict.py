"""Nested instance segmentation of a folder of images with a trained CytoNucs model (no annotation needed).

The run folder of the model holds the weights (weights_best.weights.h5, with the inference settings calibrated for
them in weights_best.weights.h5.inference.json) and geometry.json (ray sets, grids and data settings measured at
the start of training); the configuration is the one the model was trained with (e.g. configs/<run>.json).

Usage:
  python predict.py --config CONFIG --weights RUN/weights_best.weights.h5 --input IMAGES_DIR --output OUT_DIR [--gpu 0]

Input: 2D images (YX or YXC) or 3D volumes (ZYX or ZYXC) as TIFF (also PNG / JPEG in 2D), with the channels the model
was trained on. Each image is normalised to its 1st / 99.8th percentiles, as in training, and tiled as needed.
Output, for every image <name>:
  OUT_DIR/<level>/<name>          instance labels of every level of the model (TIFF, 0 = background)
  OUT_DIR/parents/<stem>.json     {inner id: outer id}: the parent of every object of the innermost level
  OUT_DIR/parents_<level>/<stem>.json   the same for the next levels of a hierarchy of three or more
"""
import argparse
import contextlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

SUFFIXES = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='configuration of the trained model')
    parser.add_argument('--weights', required=True, help='weights file in the run folder (geometry.json next to it)')
    parser.add_argument('--input', required=True, help='folder of images')
    parser.add_argument('--output', required=True, help='folder for the labels')
    parser.add_argument('--gpu', default='0', help="GPU index ('' for the CPU)")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    from cytonucs_config import CytoNucsConfig, read_config
    from cytonucs_hierarchy import predict_hierarchy, read_calibration
    from cytonucs_model import build_cytonucs_model
    from cytonucs_runtime import read_model_image
    from score_hierarchy import save_prediction

    raw = read_config(args.config, args.weights)               # 'auto' settings: those calibrated for these weights
    raw['mode'] = 'inference'
    raw['inference']['checkpoint_path'] = str(args.weights)
    images = sorted(p for p in Path(args.input).iterdir() if p.suffix.lower() in SUFFIXES)
    if not images:
        raise SystemExit(f'no images in {args.input}')
    output = Path(args.output)
    with tempfile.TemporaryDirectory() as tmp:
        raw['output']['basedir'] = tmp
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = CytoNucsConfig.from_json(str(path))
            model = build_cytonucs_model(cfg, name='predict', basedir=tmp)
            model.load_weights(str(args.weights))
        calibration = read_calibration(args.weights)
        inference = raw['inference']
        thresholds = dict(prob_thresh_nucleus=inference['prob_thresh_nucleus'],
                          prob_thresh_cytoplasm=inference['prob_thresh_cytoplasm'], nms_thresh=inference['nms_thresh'])
        levels = list(cfg.levels)
        print(f'{len(images)} images, levels {levels} (inner first)')
        for image_path in images:
            started = time.time()
            image = read_model_image(image_path, int(cfg.ndim), int(cfg.n_image_channels))
            with contextlib.redirect_stdout(io.StringIO()):
                pred = predict_hierarchy(model, cfg, image, calibration=calibration, thresholds=thresholds)
            name = image_path.stem + '.tif' if image_path.suffix.lower() not in ('.tif', '.tiff') else image_path.name
            save_prediction(output, name, pred['levels'])
            for index, (child, links) in enumerate((pred.get('parents') or {}).items()):
                folder = output / ('parents' if child == levels[0] else f'parents_{child}')
                folder.mkdir(parents=True, exist_ok=True)
                (folder / (image_path.stem + '.json')).write_text(json.dumps({int(k): int(v) for k, v in links.items()}))
            counts = ', '.join(f'{level} {len(set(pred["levels"][level].ravel().tolist()) - {0})}' for level in levels)
            print(f'{image_path.name}: {counts} ({time.time() - started:.1f} s)')
    print('->', output)


if __name__ == '__main__':
    main()
