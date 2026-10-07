"""The real entry point on a dataset written by dataset_prep/make_spatial_split_patches.py (synthetic, CPU).

Checks that the layout is read without the old preprocessing step, that validation tiles of another size than
the training patch pass full validation, and that inference runs on the test split. Not a quality measurement.
Usage: python audit/check_spatial_split_pipeline.py -> audit/spatial_split_pipeline.json
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'dataset_prep')); sys.path.insert(0, str(ROOT / 'tests'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import numpy as np

PRESET = 'config_cytonucs_v3_2.json'
PATCH, STRIDE = (16, 64, 64), (8, 56, 56)


def config(root, mode, **inference):
    raw = json.loads((ROOT / PRESET).read_text(encoding='utf-8'))
    raw['mode'] = mode
    raw['experiment'].update(name='split_check', use_comet=False)
    raw['data']['dataset_dir'] = str(root / 'dataset')
    raw['preprocessing']['run_preprocessing'] = False
    raw['output']['basedir'] = str(root / 'models')
    raw['model'].update(n_rays=12, use_gpu=False, unet_n_filter_base=2, net_conv_after_unet=4, mask_refinement_filters=4)
    raw['training'].update(train_patch_size=list(PATCH), steps_per_epoch=3, train_epochs=2, resume_from_checkpoint=None,
                           reset_best_on_resume=False, prefetch_workers=1, prefetch_depth=1, predicted_prompt_start_epoch=1)
    raw['inference'].update(dict(refinement_patch_size=[0, PATCH[1], PATCH[2]], eval_splits=['val'], mask_thresh_nucleus=0.45,
                                 mask_thresh_cytoplasm=0.45,
                                 checkpoint_path=str(root / 'models' / 'split_check' / 'weights_best.weights.h5')), **inference)
    raw['evaluation'].update(eval_dice_every=2, save_every_n_epochs=2, min_eval_epoch=0)
    raw['gpu_config']['visible_devices'] = None
    path = root / f'config_{mode}.json'
    path.write_text(json.dumps(raw, indent=2), encoding='utf-8')
    return path


def main():
    from test_spatial_split import source
    import make_spatial_split_patches as prep
    from cytonucs_runtime import run_logged
    import train_cytonucs
    (ROOT / '.cache').mkdir(exist_ok=True)
    report = dict(scope='Synthetic CPU run of the real entry point on the spatially split layout; contracts only.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checks={})
    checks = report['checks']
    with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
        root = Path(tmp)
        volumes = [dict(zip(('image', 'nucleus', 'cytoplasm'), source(seed, shape)), name=name)
                   for name, seed, shape in (('first', 0, (40, 200, 270)), ('second', 1, (17, 140, 140)))]
        manifest = prep.build(volumes, root / 'dataset', PATCH, STRIDE, augmentations=1, seed=1)
        count = {s: sum(f['split'] == s for f in manifest['files']) for s in prep.SPLITS}
        shapes = {s: sorted({tuple(f['shape']) for f in manifest['files'] if f['split'] == s}) for s in prep.SPLITS}
        report.update(files=count, tile_shapes={s: [list(x) for x in v] for s, v in shapes.items()},
                      summary=manifest['summary'], volumes={v['name']: v['summary'] for v in manifest['volumes']})
        checks['every_volume_in_every_split'] = all(v['summary'][s]['files'] > 0 for v in manifest['volumes'] for s in prep.SPLITS)
        checks['train_patches_cross_block_borders'] = any(len(f['blocks']) > 1 for f in manifest['files'] if f['split'] == 'train')
        checks['validation_tiles_differ_from_the_training_patch'] = any(x != PATCH for x in shapes['val'])
        run_logged(str(config(root, 'train')), train_cytonucs.main)
        run_dir = root / 'models' / 'split_check'
        history = json.loads((run_dir / 'history.json').read_text())
        checks['full_validation_ran'] = history.get('fullval_epoch') == [2]
        rows = (run_dir / 'visualizations' / 'epoch_0002_refinement.jsonl').read_text().splitlines()
        checks['every_validation_tile_evaluated'] = len(rows) == count['val']
        checks['validation_loss_finite'] = bool(np.isfinite(history['val_loss'][-1]))
        run_logged(str(config(root, 'inference', eval_splits=['test'])), train_cytonucs.main)
        results = json.loads((run_dir / 'inference_test' / 'results.json').read_text())
        checks['inference_on_test_split'] = results.get('n_images') == count['test'] and results.get('dataset') == 'test'
    report['status'] = 'passed' if all(checks.values()) else 'failed'
    (ROOT / 'audit' / 'spatial_split_pipeline.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESULT', report['status'], checks, count, shapes)


if __name__ == '__main__':
    main()
