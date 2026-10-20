"""End-to-end CPU run of the real entry point on synthetic TIFFs (not a JTPR measurement).

Phases through train_cytonucs.run_logged/main with the v2.12 preset scaled down:
  A train from scratch with full validation, B resume with reset_best,
  C inference with mask_quality_min='auto', D optimize_thresholds,
  E 'auto' inference on a checkpoint without calibration must fail.
Writes audit/e2e_<preset>_run.json. Usage: python audit/e2e_v2_12_run.py [config_cytonucs_v2_13.json]
"""
import json
import os
import shutil
import sys
import tempfile
import time
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESET = sys.argv[1] if len(sys.argv) > 1 else 'config_cytonucs_v2_12.json'
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import numpy as np
import tifffile
import h5py


def make_split(folder, count, seed):
    """3x3 cells per 8x48x48 volume, one nucleus each; border cells touch XY faces."""
    for sub in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments'):
        (folder / sub).mkdir(parents=True)
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[:8, :48, :48]
    for i in range(count):
        cells = np.zeros((8, 48, 48), np.uint16); nuclei = np.zeros_like(cells); pairs = {}
        label = 0
        for cy in range(3):
            for cx in range(3):
                label += 1
                y0, y1 = (0 if cy == 0 else cy*16+1), (48 if cy == 2 else cy*16+15)
                x0, x1 = (0 if cx == 0 else cx*16+1), (48 if cx == 2 else cx*16+15)
                cells[1:7, y0:y1, x0:x1] = label  # only the centre cell avoids XY faces
                c = (3.5 + rng.uniform(-.5, .5), cy*16 + 8 + rng.uniform(-1, 1), cx*16 + 8 + rng.uniform(-1, 1))
                inside = ((zz-c[0])/2.2)**2 + ((yy-c[1])/4.)**2 + ((xx-c[2])/4.)**2 <= 1
                nuclei[inside] = 100 + label
                pairs[100 + label] = label
        image = .1 + .4*(cells > 0) + .5*(nuclei > 0) + rng.normal(0, .03, cells.shape)
        for sub, arr in (('images', image.astype(np.float32)), ('nucleus_masks', nuclei), ('cytoplasm_masks', cells)):
            tifffile.imwrite(folder / sub / f'vol{i}.tif', arr, photometric='minisblack', metadata={'axes': 'ZYX'})
        (folder / 'assignments' / f'vol{i}.json').write_text(json.dumps(pairs))


def config(root, **changes):
    raw = json.loads((ROOT / PRESET).read_text(encoding='utf-8'))
    raw['experiment']['name'] = 'e2e'
    raw['data']['dataset_dir'] = str(root / 'data')
    raw['output']['basedir'] = str(root / 'models')
    raw['model'].update(n_rays=12, n_channel_in=1, use_gpu=False, unet_n_depth=2, unet_n_filter_base=2,
                        net_conv_after_unet=4, mask_refinement_filters=4,
                        unet_pool_schedule=[[1, 2, 2], [2, 2, 2]],
                        unet_kernel_schedule=[[1, 3, 3], [3, 3, 3], [3, 3, 3]])
    raw['training'].update(train_patch_size=[8, 32, 32], steps_per_epoch=2, train_epochs=4,
                           resume_from_checkpoint=None, reset_best_on_resume=False,
                           predicted_prompt_start_epoch=1, prefetch_workers=1, prefetch_depth=1)
    raw['inference'].update(refinement_patch_size=[0, 32, 32], eval_splits=['val'],
                            checkpoint_path=str(root / 'models' / 'e2e' / 'weights_best.weights.h5'))
    raw['evaluation'].update(eval_dice_every=2, save_every_n_epochs=2, min_eval_epoch=0,
                             prob_thresh_range=[.4, .5], mask_thresh_range=[.45, .6])
    raw['gpu_config']['visible_devices'] = None
    for section, values in changes.items():
        if isinstance(values, dict):
            raw[section].update(values)
        else:
            raw[section] = values
    path = root / f'config_{len(list(root.glob("config_*.json")))}.json'
    path.write_text(json.dumps(raw, indent=2), encoding='utf-8')
    return path


def run(path):
    from cytonucs_runtime import run_logged
    import train_cytonucs
    start = time.time()
    result = run_logged(str(path), train_cytonucs.main)
    return result, round(time.time() - start, 1)


def main():
    (ROOT / '.cache').mkdir(exist_ok=True)
    report = dict(scope='Synthetic CPU entry-point run; checks control flow/contracts, not segmentation quality.',
                  preset=PRESET,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), phases={}, checks={})
    with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
        root = Path(tmp)
        make_split(root / 'data' / 'train', 4, 1)
        make_split(root / 'data' / 'val', 2, 2)
        run_dir = root / 'models' / 'e2e'
        checks = report['checks']

        # A: fresh training with full validation at epochs 2 and 4.
        _, seconds = run(config(root))
        history = json.loads((run_dir / 'history.json').read_text())
        best = json.loads((run_dir / 'best_fullval.json').read_text())
        epochs = history['fullval_epoch']
        report['phases']['A_train'] = dict(seconds=seconds, fullval_epoch=epochs, best=best,
            fullval_jtpr=history['fullval_jtpr'], interior_pairs=history.get('fullval_interior_total_pairs'),
            quality_gates=[history.get(f'fullval_{h}_quality_gate') for h in ('nucleus', 'cytoplasm')],
            train_nucleus_quality_supervised_fraction=history.get('train_nucleus_quality_supervised_fraction'))
        checks['A_fullval_epochs'] = epochs == [2, 4]
        checks['A_all_fullval_keys_aligned'] = all(len(v) == len(epochs) for k, v in history.items() if k.startswith('fullval_'))
        checks['A_interior_pairs_logged'] = history.get('fullval_interior_total_pairs') == [2, 2]
        from cytonucs_runtime import read_quality_gates
        gates = read_quality_gates(run_dir / 'weights_best.weights.h5')
        at_best = epochs.index(best['epoch'])
        gate_heads = [h for h in ('nucleus', 'cytoplasm') if gates is not None and h in gates]
        checks['A_best_has_gates_of_its_epoch'] = bool(gates is not None and gate_heads and
            all(gates[h] == history[f'fullval_{h}_quality_gate'][at_best] for h in gate_heads)
            and all(gates.get(h + '_min_area', 0) == (history.get(f'fullval_{h}_area_gate') or [0] * len(epochs))[at_best]
                    for h in gate_heads)
            and best.get('quality_gates') == gates)
        preset_raw = json.loads((ROOT / PRESET).read_text(encoding='utf-8'))
        lp = preset_raw['loss_params']
        if lp.get('prob_object_peak_weight') or lp.get('cytoplasm_prob_object_weight') or lp.get('mask_quality_rank_weight'):
            wanted = [f'{s}_{h}_bce_object_peak_weighted' for s in ('train', 'val') for h in ('nucleus', 'cytoplasm')
                      if lp.get('prob_object_peak_weight')]
            wanted += [f'{s}_cytoplasm_bce_object_weighted' for s in ('train', 'val') if lp.get('cytoplasm_prob_object_weight')]
            wanted += [f'{s}_{h}_quality_{m}' for s in ('train', 'val') for h in ('nucleus', 'cytoplasm')
                       for m in ('pair_accuracy', 'rank_pairs') if lp.get('mask_quality_rank_weight')]
            report['phases']['A_train']['new_loss_history'] = {k: history.get(k) for k in wanted}
            checks['A_v2_16_losses_logged'] = all(
                history.get(k) and all(v is not None and np.isfinite(v) for v in history[k]) for k in wanted)
        if preset_raw['inference'].get('residual_proposals'):
            rows = [json.loads(line) for line in
                    (run_dir / 'visualizations' / 'epoch_0004_refinement.jsonl').read_text().splitlines()]
            report['phases']['A_train']['residual'] = [{h: {k: r['refinement'][h].get(k) for k in
                ('residual_peaks', 'residual_prompts', 'written_residual')} for h in ('nucleus', 'cytoplasm')} for r in rows]
            checks['A_residual_pass_and_geometry_logged'] = all(
                'residual_prompts' in r['refinement'][h] and 'written_residual' in r['refinement'][h]
                and 'gt_geometry' in r['stage_audit'][h] for r in rows for h in ('nucleus', 'cytoplasm'))
        mask_auto = 'auto' in (preset_raw['inference'].get('mask_thresh_nucleus'), preset_raw['inference'].get('mask_thresh_cytoplasm'))
        if mask_auto:
            from cytonucs_runtime import read_mask_thresholds
            # Calibrated once after training, on the reloaded best weights.
            thresholds = read_mask_thresholds(run_dir / 'weights_best.weights.h5')
            calibration = json.loads((run_dir / 'mask_calibration.json').read_text())
            report['phases']['A_train']['mask_calibration'] = calibration
            checks['A_mask_thresholds_after_training_only'] = (
                thresholds is not None and calibration['thresholds'] == thresholds
                and calibration['checkpoint_epoch'] == best['epoch']
                and not any('mask_thresh' in k for k in history)
                and json.loads((run_dir / 'final_results.json').read_text())['mask_thresholds'] == thresholds)
        if preset_raw['inference'].get('mask_quality_min') == 'auto':
            checks['A_edge_fragment_metric_logged'] = all(
                len(history.get(f'fullval_ap_{h}_quality_gated_edge_ignored') or []) == len(epochs)
                for h in ('nucleus', 'cytoplasm'))
            # Z-slice / XY-contour split of the boundary error: one value per validation and head,
            # AP with an error removed never below the AP after gates.
            checks['A_boundary_split_logged'] = all(
                len(history.get(f'fullval_ap_{h}_quality_gated_{kind}_exact') or []) == len(epochs)
                and all(v is None or v >= g - 1e-12 for v, g in zip(history[f'fullval_ap_{h}_quality_gated_{kind}_exact'],
                                                                      history[f'fullval_ap_{h}_quality_gated']))
                for h in ('nucleus', 'cytoplasm') for kind in ('z', 'xy', 'mixed'))
        checks['A_epoch_checkpoints_have_gates'] = all(read_quality_gates(run_dir / f'weights_epoch_{e:03d}.weights.h5') is not None for e in (2, 4))
        checks['A_jsonl_written'] = all((run_dir / 'visualizations' / f'epoch_{e:04d}_refinement.jsonl').exists() for e in (2, 4))
        checks['A_final_results'] = (run_dir / 'final_results.json').exists()

        # B: resume epoch 4 -> 6 with reset_best (v2.12 transition semantics).
        _, seconds = run(config(root, training=dict(resume_from_checkpoint=str(run_dir / 'weights_epoch_004.weights.h5'),
                                                    reset_best_on_resume=True, train_epochs=6)))
        history = json.loads((run_dir / 'history.json').read_text())
        report['phases']['B_resume'] = dict(seconds=seconds, fullval_epoch=history['fullval_epoch'],
                                            train_loss_len=len(history['train_loss']))
        checks['B_resume_continues'] = history['fullval_epoch'] == [2, 4, 6] and len(history['train_loss']) == 6
        checks['B_history_aligned'] = all(len(v) == 3 for k, v in history.items() if k.startswith('fullval_'))

        # C: standalone inference with calibrated gates.
        _, seconds = run(config(root, mode='inference'))
        report['phases']['C_inference_auto'] = dict(seconds=seconds)
        if mask_auto:
            from cytonucs_runtime import read_mask_thresholds
            checks['C_inference_uses_best_mask_thresholds'] = json.loads(
                (run_dir / 'inference_val' / 'results.json').read_text()).get('mask_thresholds') == read_mask_thresholds(run_dir / 'weights_best.weights.h5')
        checks['C_inference_outputs'] = any((run_dir / 'inference_val' / 'pred_masks').glob('*.tiff')) and json.loads((run_dir / 'inference_val' / 'results.json').read_text()).get('quality_gates') == read_quality_gates(run_dir / 'weights_best.weights.h5')

        # D: threshold optimisation entry point (ungated search).
        result, seconds = run(config(root, mode='optimize_thresholds'))
        report['phases']['D_optimize_thresholds'] = dict(seconds=seconds, result=result)
        checks['D_threshold_search_returns'] = isinstance(result, dict) and 'jtpr' in result

        # E: 'auto' on a checkpoint without calibration fails loudly.
        stripped = run_dir / 'weights_nogate.weights.h5'
        shutil.copy(run_dir / 'weights_epoch_004.weights.h5', stripped)
        shutil.copy(str(run_dir / 'weights_epoch_004.weights.h5') + '.architecture.json', str(stripped) + '.architecture.json')
        with h5py.File(stripped, 'a') as f:
            del f.attrs['cytonucs_quality_gates']
        try:
            run(config(root, mode='inference', inference=dict(checkpoint_path=str(stripped))))
            checks['E_auto_without_gates_fails'] = False
        except ValueError as error:
            checks['E_auto_without_gates_fails'] = 'no calibrated quality gates' in str(error)

        # F: resume across refiner pooling modes (weights compatible, fine-tuning continues).
        preset_pool = json.loads((ROOT / PRESET).read_text(encoding='utf-8'))['model'].get('refinement_pool', 'isotropic')
        other_pool = 'isotropic' if preset_pool == 'auto' else 'auto'
        _, seconds = run(config(root, model=dict(refinement_pool=other_pool),
                                training=dict(resume_from_checkpoint=str(run_dir / 'weights_epoch_006.weights.h5'),
                                              reset_best_on_resume=True, train_epochs=8)))
        history = json.loads((run_dir / 'history.json').read_text())
        console = sorted((run_dir / 'runs').glob('*/console.log'))[-1].read_text(encoding='utf-8')
        report['phases']['F_refiner_pool_transition'] = dict(seconds=seconds, pools=[preset_pool, other_pool],
                                                              fullval_epoch=history['fullval_epoch'])
        checks['F_pool_transition_resumes'] = (f'Refiner pooling: {preset_pool} -> {other_pool}' in console
                                               and history['fullval_epoch'] == [2, 4, 6, 8])
        # G: stand-alone calibration of a checkpoint without thresholds (interrupted run).
        if mask_auto:
            from cytonucs_runtime import read_mask_thresholds
            target = run_dir / 'weights_epoch_004.weights.h5'
            _, seconds = run(config(root, mode='calibrate', inference=dict(checkpoint_path=str(target))))
            report['phases']['G_calibrate_mode'] = dict(seconds=seconds, thresholds=read_mask_thresholds(target))
            checks['G_calibrate_mode_writes_checkpoint'] = (read_mask_thresholds(target) is not None
                                                            and (run_dir / 'mask_calibration.json').exists())
        report['status'] = 'complete' if all(checks.values()) else 'failed_checks'
    target = ROOT / 'audit' / ('e2e_' + Path(PRESET).stem.replace('config_cytonucs_', '') + '_run.json')
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print('E2E', report['status'], json.dumps(checks))


if __name__ == '__main__':
    main()
