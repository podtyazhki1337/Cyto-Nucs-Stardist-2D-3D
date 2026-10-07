"""Does training drop objects cut only by the real volume XY face from object-level losses?

Real CytoNucsDataGenerator + real without_truncated_objects / quality_supervision_weights
on synthetic TIFFs. Each 8x48x48 volume has one nucleus/cell cut by the volume face x=0
and one interior nucleus/cell; crops are 8x32x32 (smaller than the volume in XY, as
448 of 512 on the server), no augmentation. A crop that contains the face-cut object
starts at x=0, so that crop face IS the volume face: at inference the object looks the same.

Since v2.17 the generator also emits per-object <head>_crop_cut masks (fewer voxels than in
the source volume); the loss uses them instead of the per-axis flags. The summary reports both.

Usage: python audit/check_truncation_flags.py -> audit/truncation_flags_check.json
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1'); os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
import tifffile

FULL_101 = None


def make_split(folder, count):
    for sub in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments'):
        (folder / sub).mkdir(parents=True)
    zz, yy, xx = np.mgrid[:8, :48, :48]
    global FULL_101
    for i in range(count):
        cells = np.zeros((8, 48, 48), np.uint16); nuclei = np.zeros_like(cells)
        cells[1:7, 14:34, 0:12] = 1                       # cut by the volume face x=0
        nuclei[((zz - 3.5) / 2.2) ** 2 + ((yy - 24) / 4.) ** 2 + (xx / 4.) ** 2 <= 1] = 101
        cells[1:7, 14:34, 26:46] = 2                      # interior
        nuclei[((zz - 3.5) / 2.2) ** 2 + ((yy - 24) / 4.) ** 2 + ((xx - 36) / 4.) ** 2 <= 1] = 102
        image = .1 + .4 * (cells > 0) + .5 * (nuclei > 0)
        for sub, arr in (('images', image.astype(np.float32)), ('nucleus_masks', nuclei), ('cytoplasm_masks', cells)):
            tifffile.imwrite(folder / sub / f'vol{i}.tif', arr, photometric='minisblack', metadata={'axes': 'ZYX'})
        (folder / 'assignments' / f'vol{i}.json').write_text(json.dumps({101: 1, 102: 2}))
        FULL_101 = int(np.count_nonzero(nuclei == 101))


def main():
    from test_fixes import config
    from cytonucs_data_generator import CytoNucsDataGenerator
    from cytonucs_losses import without_truncated_objects
    rows = []
    with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp, contextlib.redirect_stdout(io.StringIO()):
        make_split(Path(tmp) / 'train', 4)
        cfg = config(train_patch_size=(8, 32, 32), nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1),
                     fg_crop_prob=0.0, prompts_per_crop=2)
        gen = CytoNucsDataGenerator(dataset_dir=Path(tmp) / 'train', config=cfg, rays=cfg.rays, augmenter=None,
                                    shuffle=True, subset='train')
        for step in range(200):
            gen.epoch = step
            _, y = gen[step % len(gen)]
            nuclei = np.asarray(y['nucleus_instances']); flags = np.asarray(y['crop_truncation_axes'])
            kept = without_truncated_objects(tf.constant(nuclei), tf.constant(nuclei), tf.constant(flags), 3).numpy()
            for b in range(len(nuclei)):
                present = set(np.unique(nuclei[b])) - {0}
                if 101 not in present:
                    continue
                # Touching the crop face x=0 alone does not prove the crop starts at the volume
                # face: a crop starting at x=1..3 cuts this object artificially. Whole = same
                # voxel count as in the source volume.
                at_volume_face = bool(np.any(nuclei[b][..., 0] == 101))
                whole = int(np.count_nonzero(nuclei[b] == 101)) == FULL_101
                mask = np.asarray(y['nucleus_crop_cut'])[b] if 'nucleus_crop_cut' in y else None
                rows.append(dict(flags=flags[b].tolist(), face_object_touches_crop_x0=at_volume_face,
                                 face_object_whole=whole,
                                 face_object_kept=bool(101 in set(np.unique(kept[b]))),
                                 face_object_marked_by_crop_cut_mask=None if mask is None else
                                 bool(mask[nuclei[b] == 101].max()),
                                 interior_present=bool(102 in present),
                                 interior_kept=bool(102 in set(np.unique(kept[b])))))
    touching = [r for r in rows if r['face_object_touches_crop_x0']]
    whole = [r for r in touching if r['face_object_whole']]
    summary = dict(samples_with_face_object=len(rows), touching_crop_face_x0=len(touching),
                   of_them_whole_crop_face_is_volume_face=len(whole),
                   of_them_cut_by_crop=len(touching) - len(whole),
                   whole_dropped_by_axis_flags=sum(not r['face_object_kept'] for r in whole),
                   whole_marked_by_crop_cut_mask=sum(bool(r['face_object_marked_by_crop_cut_mask']) for r in whole),
                   cut_marked_by_crop_cut_mask=sum(bool(r['face_object_marked_by_crop_cut_mask'])
                                                   for r in touching if not r['face_object_whole']),
                   xy_flag_set=sum(r['flags'][2] == 1 for r in touching))
    report = dict(scope='Real generator and loss helper on synthetic TIFFs; mechanism check, not a metric.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                                 for f in ('cytonucs_data_generator.py', 'cytonucs_losses.py')},
                  summary=summary, rows=rows)
    (ROOT / 'audit' / 'truncation_flags_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
