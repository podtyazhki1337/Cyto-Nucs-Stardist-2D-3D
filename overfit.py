import json
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
from skimage.measure import regionprops
from cytonucs_config import CytoNucsConfig
from cytonucs_losses import compute_stardist_gt

CFG = 'config_cytonucs.json'
cd  = json.load(open(CFG))
cfg = CytoNucsConfig.from_json(CFG)

base = Path(cd['data']['dataset_dir'])
root = base / 'train' if (base / 'train').exists() else base
N_FILES = int(cd.get('_replace_me', 0)) or 120   # сколько тайлов брать

def find(d):
    fs = []
    for e in ('*.tif', '*.tiff', '*.TIF', '*.TIFF'):
        fs.extend(d.glob(e))
    return sorted(fs)

for name, sub in (('nucleus', 'nucleus_masks'), ('cytoplasm', 'cytoplasm_masks')):
    files = find(root / sub)[:N_FILES]
    grid  = tuple(int(g) for g in getattr(cfg, f'{name}_grid'))
    rays  = getattr(cfg, f'rays_{name}')
    aniso = getattr(cfg, f'rays_anisotropy_{name}')

    B, S, M, MN, NZ = [], [], [], [], []
    n_obj = n_used = 0

    for f in files:
        lbl = np.asarray(imageio.volread(str(f)))
        if lbl.max() == 0:
            continue
        prob, dist = compute_stardist_gt(lbl, cfg, grid, rays=rays, aniso=aniso)
        prob = np.asarray(prob, np.float32)[..., 0]
        dist = np.asarray(dist, np.float32)
        lsub = lbl[tuple(slice(None, None, g) for g in grid)]
        lsub = lsub[tuple(slice(0, s) for s in prob.shape)]

        for r in regionprops(lbl):
            n_obj += 1
            m = (lsub == r.label)
            if not m.any():
                continue
            # Центр = самый ГЛУБОКИЙ воксель объекта. Обрезка краем тайла
            # укорачивает лучи только в сторону среза, поэтому объекты не
            # выбрасываем: при 288 тайлах 448x448 почти все касаются грани,
            # и фильтр по bbox оставлял 3 объекта из 25.
            c = np.unravel_index(int(np.argmax(np.where(m, prob, -1.0))), prob.shape)
            v = dist[c]
            if v.max() <= 0:
                continue
            B.append(np.median(v)); S.append(np.std(v))
            M.append(v.max());      MN.append(v.min())
            # доля лучей короче 10% максимума — индикатор сильной обрезки
            NZ.append(float(np.mean(v < 0.1 * v.max())))
            n_used += 1

    B, S, M, MN, NZ = map(np.asarray, (B, S, M, MN, NZ))
    print(f"\n=== {name}: файлов={len(files)} объектов={n_obj} центров={n_used}")
    if n_used == 0:
        continue

    # Отбраковка сильно срезанных: у них аномально много коротких лучей
    ok = NZ <= np.percentile(NZ, 60)
    for tag, sel in (('ВСЕ', np.ones_like(ok)), ('лучшие 60% по целостности', ok)):
        b, s, mx, mn = B[sel], S[sel], M[sel], MN[sel]
        print(f"  [{tag}] n={sel.sum():4d} | "
              f"bias={np.median(b):6.1f} (IQR {np.percentile(b,25):.0f}-{np.percentile(b,75):.0f}) | "
              f"spread={np.median(s):6.1f} | "
              f"max_ray={np.median(mx):6.1f} (IQR {np.percentile(mx,25):.0f}-{np.percentile(mx,75):.0f}) | "
              f"min_ray={np.median(mn):5.1f}")
    print(f"  в geometry.json: bias={getattr(cfg, f'dist_bias_init_{name}'):.1f}  "
          f"spread={getattr(cfg, f'dist_spread_{name}', float('nan')):.1f}  "
          f"max={getattr(cfg, f'median_max_dist_{name}'):.1f}")