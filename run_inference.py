"""Прогон обученной модели по исходникам + GT ядер. Маски пишутся рядом со скриптом."""
import json
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import tifffile
from tqdm import tqdm
from cytonucs_runtime import normalize_percentiles as normalize
from stardist.matching import matching_dataset

from cytonucs_config import CytoNucsConfig
from cytonucs_model import build_cytonucs_model
from cytonucs_inference import CytoNucsPredictor, compute_jtpr
from cytonucs_trainer import compute_dice
from cytonucs_gpu_setup import setup_gpu

# ======================= ЕДИНСТВЕННОЕ МЕСТО ДЛЯ ПРАВКИ =======================
CONFIG     = 'config_cytonucs.json'
CHECKPOINT = '/storage02/miroslavm/Multi-Nuclear-Stardist-2D-3D/models/cytonucs_emese_3_stardist_3d_2dculture/weights_best.weights.h5'
# CHECKPOINT = 'models/cytonucs_emese_3_stardist_3d_2dculture/weights_best.weights.h5'

IMAGES   = '/FREYJA/Akos/Annotation/Cropped_1040_2048_20_186.tif'
NUC_GT   = '/FREYJA/Akos/Annotation/Cropped_1040_2048_20_186_mask.tif'
CYTO_GT  = None      # None -> GT клеток = GT ядер, маппинг тождественный {L: L}

PROB_NUC = PROB_CYTO = NMS = MASK_THRESH = None   # None -> берём из конфига
# =============================================================================

HERE = Path(__file__).resolve().parent
OUT = HERE / 'pred'
OUT.mkdir(exist_ok=True)

EXT = ('*.tif', '*.tiff', '*.TIF', '*.TIFF')


def find(p):
    """Путь может быть как каталогом, так и одним файлом."""
    p = Path(p)
    if p.is_file():
        return [p]
    fs = []
    for e in EXT:
        fs.extend(p.glob(e))
    return sorted(fs)


# ---------- конфиг с геометрией ИЗ ПАПКИ ЧЕКПОИНТА ----------
# from_json ищет geometry.json как output.basedir/experiment.name/geometry.json.
# При чужом чекпоинте подтянулись бы лучи и гриды другого прогона: формы
# тензоров совпадут, load_weights пройдёт молча, а полиэдры соберутся по
# неверной анизотропии. Поэтому подменяем эти два поля на путь к весам.
ckpt = Path(CHECKPOINT).resolve()
assert ckpt.exists(), f"нет чекпоинта: {ckpt}"

cfg_dict = json.load(open(CONFIG))
cfg_dict['mode'] = 'inference'
cfg_dict.setdefault('experiment', {})['name'] = ckpt.parent.name
cfg_dict.setdefault('output', {})['basedir'] = str(ckpt.parent.parent)
tmp_cfg = ckpt.parent / '_check_config.json'
json.dump(cfg_dict, open(tmp_cfg, 'w'), indent=2)

model_config = CytoNucsConfig.from_json(str(tmp_cfg))
ndim = int(model_config.ndim)
patch = tuple(model_config.train_patch_size)

if cfg_dict['model'].get('use_gpu', False):
    g = cfg_dict.get('gpu_config', {})
    setup_gpu(memory_limit=g.get('memory_limit'),
              allow_growth=g.get('allow_growth', True),
              visible_devices=g.get('visible_devices'))

model = build_cytonucs_model(config=model_config, name=ckpt.parent.name,
                             basedir=str(ckpt.parent.parent))
model.load_weights(str(ckpt))
print(f"✅ {ckpt}")

inf = cfg_dict.get('inference', {})
predictor = CytoNucsPredictor(
    model=model, config=model_config,
    prob_thresh_nucleus=PROB_NUC if PROB_NUC is not None else inf.get('prob_thresh_nucleus', 0.5),
    prob_thresh_cytoplasm=PROB_CYTO if PROB_CYTO is not None else inf.get('prob_thresh_cytoplasm', 0.3),
    nms_thresh=NMS if NMS is not None else inf.get('nms_thresh', 0.3))
if MASK_THRESH is not None:
    predictor.mask_thresh = float(MASK_THRESH)
print(f"   prob_nuc={predictor.prob_thresh_nucleus} prob_cyto={predictor.prob_thresh_cytoplasm} "
      f"nms={predictor.nms_thresh} mask={getattr(predictor, 'mask_thresh', None)}")

# ---------- данные ----------
images = find(IMAGES)
nucs = find(NUC_GT)
cytos = find(CYTO_GT) if CYTO_GT else list(nucs)
same_mask = CYTO_GT is None
assert images, f"нет изображений в {IMAGES}"
assert len(images) == len(nucs) == len(cytos), \
    f"разное число файлов: img={len(images)} nuc={len(nucs)} cyto={len(cytos)}"
if same_mask:
    print("ℹ️  GT клеток = GT ядер, маппинг ядро->клетка тождественный")
print(f"📂 {len(images)} шт -> {OUT}")


def load_img(p):
    a = np.asarray(imageio.volread(str(p)) if ndim == 3 else imageio.imread(str(p)))
    print(f"  прочитано: {a.shape} {a.dtype} range[{a.min()},{a.max()}]")

    # RGB-стек может лежать как (Z,C,Y,X) — приводим к (Z,Y,X,C)
    if ndim == 3 and a.ndim == 4 and a.shape[-1] not in (3, 4) and a.shape[1] in (3, 4):
        a = np.moveaxis(a, 1, -1)
    if a.ndim == ndim:
        a = a[..., None]

    n_need = int(getattr(model_config, 'n_channel_in', 1))
    if a.shape[-1] != n_need:
        if a.shape[-1] == 1:
            # Сеть ждёт n_channel_in каналов, а файл одноканальный.
            # Размножаем — это единственный способ прогнать, но каналы
            # станут идентичными, тогда как при обучении они несли разные
            # окраски. Предсказание будет вне обучающего распределения.
            print(f"  ⚠️  файл одноканальный, модель ждёт {n_need} — "
                  f"канал размножен, результат вне обучающего распределения")
            a = np.repeat(a, n_need, axis=-1)
        elif a.shape[-1] > n_need:
            print(f"  ⚠️  каналов {a.shape[-1]}, беру первые {n_need}")
            a = a[..., :n_need]
        else:
            raise ValueError(f"каналов {a.shape[-1]}, модель ждёт {n_need}")

    return normalize(a.astype(np.float32), 1, 99.8, axis=tuple(range(ndim)))


def load_mask(p):
    return np.asarray(imageio.volread(str(p)) if ndim == 3 else imageio.imread(str(p)))


# ---------- прогон ----------
gt_n, pr_n, gt_c, pr_c, rows = [], [], [], [], []
tp = tot = 0

for i in tqdm(range(len(images)), desc='inference'):
    img = load_img(images[i])
    sp = img.shape[:ndim]

    if any(s > p for s, p in zip(sp, patch)):
        nt = tuple(max(1, int(np.ceil(s / p))) for s, p in zip(sp, patch))
        if ndim == 3:
            nt = (1, nt[1], nt[2])      # Z управляется fixed-Z окнами внутри
        npred, cpred, _ = predictor._predict_with_tiling(img, n_tiles=nt, normalize_img=False)
    else:
        npred, cpred, _ = predictor.predict_instances(img, normalize_img=False)

    stem = images[i].stem
    tifffile.imwrite(str(OUT / f'{stem}_nucleus.tif'), npred.astype(np.uint16))
    tifffile.imwrite(str(OUT / f'{stem}_cytoplasm.tif'), cpred.astype(np.uint16))

    ngt, cgt = load_mask(nucs[i]), load_mask(cytos[i])
    if npred.shape != ngt.shape:
        sl = tuple(slice(0, min(a, b)) for a, b in zip(npred.shape, ngt.shape))
        npred, cpred, ngt, cgt = npred[sl], cpred[sl], ngt[sl], cgt[sl]

    # При общей маске объект L — это одновременно ядро L и клетка L.
    asg_gt = {int(v): int(v) for v in np.unique(ngt) if v > 0} if same_mask else None

    j, d = compute_jtpr(nucleus_pred=npred, cytoplasm_pred=cpred,
                        nucleus_gt=ngt, cytoplasm_gt=cgt, iou_thresh=0.5,
                        assignments_pred=getattr(predictor, '_prompt_assignments', None),
                        assignments_gt=asg_gt)
    tp += int(d['joint_tp'])
    tot += int(d['total_pairs'])

    gt_n.append(ngt); pr_n.append(npred)
    gt_c.append(cgt); pr_c.append(cpred)
    rows.append((images[i].name, len(np.unique(npred)) - 1, len(np.unique(cpred)) - 1,
                 compute_dice(ngt, npred), compute_dice(cgt, cpred), j))

# ---------- отчёт ----------
print("\n" + "=" * 78)
print(f"{'файл':40s} {'nuc':>5s} {'cyto':>5s} {'Dice_n':>7s} {'Dice_c':>7s} {'JTPR':>6s}")
for nm, a, b, dn, dc, jj in rows:
    print(f"{nm[:40]:40s} {a:5d} {b:5d} {dn:7.3f} {dc:7.3f} {jj:6.3f}")

s_n = matching_dataset(gt_n, pr_n, thresh=0.5, show_progress=False)
s_c = matching_dataset(gt_c, pr_c, thresh=0.5, show_progress=False)
jtpr = tp / tot if tot else 0.0
print("=" * 78)
print(f"nucleus   AP@50={s_n.accuracy:.4f} prec={s_n.precision:.4f} rec={s_n.recall:.4f} "
      f"TP={s_n.tp} FP={s_n.fp} FN={s_n.fn}")
print(f"cytoplasm AP@50={s_c.accuracy:.4f} prec={s_c.precision:.4f} rec={s_c.recall:.4f} "
      f"TP={s_c.tp} FP={s_c.fp} FN={s_c.fn}")
print(f"Dice: nuc={np.mean([r[3] for r in rows]):.4f} cyto={np.mean([r[4] for r in rows]):.4f}")
print(f"JTPR = {tp}/{tot} = {jtpr:.4f}")

json.dump({'checkpoint': str(ckpt), 'same_mask': same_mask,
           'nucleus_ap50': float(s_n.accuracy), 'cytoplasm_ap50': float(s_c.accuracy),
           'dice_nucleus': float(np.mean([r[3] for r in rows])),
           'dice_cytoplasm': float(np.mean([r[4] for r in rows])),
           'jtpr': float(jtpr), 'jtpr_tp': tp, 'jtpr_pairs': tot},
          open(OUT / 'metrics.json', 'w'), indent=2)
print(f"💾 {OUT}")