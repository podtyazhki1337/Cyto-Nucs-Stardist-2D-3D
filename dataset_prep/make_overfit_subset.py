"""A few training volumes as BOTH train and val: can the network reproduce data it trains on?

Copies `count` original (not augmented, non-empty) volumes of <dataset>/train into <output>/train and
<output>/val with their masks and assignments. Training on this folder without augmentation is a
memorisation test of the network on real data: validation is done on the very volumes it trains on.
With --config and --checkpoint it also writes <output>/overfit_config.json: the given training config
continued from that checkpoint on the subset, augmentation off, learning rate held at its initial value.

Usage:
  python dataset_prep/make_overfit_subset.py --dataset PREPROCESSED_DIR --output OUT_DIR \\
      --config config_cytonucs_v3_2.json --checkpoint models/RUN/weights_epoch_830.weights.h5
  python train_cytonucs.py OUT_DIR/overfit_config.json
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import tifffile

FOLDERS = ('images', 'nucleus_masks', 'cytoplasm_masks')


def build(dataset, output, count=4):
    dataset, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty: choose a new output folder')
    train = dataset / 'train'
    names = sorted(p.name for p in (train / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    originals = [n for n in names if Path(n).stem.endswith('_a00')] or names     # '_aNN' marks offline copies
    objects = {n: int(tifffile.imread(train / 'cytoplasm_masks' / n).max()) for n in originals}
    filled = [n for n in originals if objects[n] > 0]
    if len(filled) < count:
        raise ValueError(f'Only {len(filled)} non-empty original volumes in {train}')
    chosen = [filled[i] for i in np.linspace(0, len(filled) - 1, count).round().astype(int)]   # spread over the list
    for split in ('train', 'val'):
        for sub in FOLDERS + ('assignments',):
            (output / split / sub).mkdir(parents=True, exist_ok=True)
        for name in chosen:
            for sub in FOLDERS:
                shutil.copy2(train / sub / name, output / split / sub / name)
            pairs = train / 'assignments' / (Path(name).stem + '.json')
            if pairs.exists():
                shutil.copy2(pairs, output / split / 'assignments' / pairs.name)
    info = dict(train=chosen, val=chosen, split='overfit_subset', rescale_factor=1.0, source=str(dataset),
                cells_per_volume={n: objects[n] for n in chosen})
    (output / 'split_info.json').write_text(json.dumps(info, indent=2))
    return info


def write_config(base_config, output, checkpoint, epochs=40):
    """The base config continued from `checkpoint` on the subset for `epochs` epochs as a separate experiment."""
    raw = json.loads(Path(base_config).read_text(encoding='utf-8'))
    saved = re.search(r'epoch_(\d+)', Path(checkpoint).name)
    if saved is None:
        raise ValueError('--checkpoint must be a numbered checkpoint (weights_epoch_NNN.weights.h5)')
    name = raw['experiment']['name'] + '_overfit_check'
    raw['mode'] = 'train'
    raw['experiment']['name'] = name
    raw['data']['dataset_dir'] = str(Path(output))
    raw['preprocessing']['run_preprocessing'] = False
    raw['augmentation']['enable'] = False
    raw['training'].update(resume_from_checkpoint=str(checkpoint), reset_best_on_resume=True,
                           train_epochs=int(saved.group(1)) + epochs, lr_decay_rate=1.0)
    # Numeric mask thresholds: nothing is calibrated for this experiment.
    raw['inference'].update(checkpoint_path=f"{raw['output']['basedir']}/{name}/weights_best.weights.h5",
                            mask_thresh_nucleus=raw['inference']['mask_thresh'],
                            mask_thresh_cytoplasm=raw['inference']['mask_thresh'])
    path = Path(output) / 'overfit_config.json'
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, help='folder with train/images, train/nucleus_masks, ...')
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--config', help='training config to continue on the subset')
    parser.add_argument('--checkpoint', help='numbered checkpoint of that training')
    parser.add_argument('--epochs', type=int, default=40, help='epochs to train on the subset')
    args = parser.parse_args()
    if bool(args.config) != bool(args.checkpoint):
        parser.error('--config and --checkpoint are given together')
    info = build(args.dataset, args.output, args.count)
    for name in info['train']:
        print(f"{name}: {info['cells_per_volume'][name]} cells")
    print('Written to', args.output, '(train and val hold the same volumes)')
    if args.config:
        print('Config:', write_config(args.config, args.output, args.checkpoint, args.epochs))


if __name__ == '__main__':
    main()
