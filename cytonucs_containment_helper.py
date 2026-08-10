"""
Helper functions for containment loss computation in numpy.
Can be used in custom training loop or post-processing.
"""
import numpy as np
from skimage.measure import regionprops


def _to_shape(arr, shape):
    """Приводит карту к заданной пространственной форме ближайшим соседом.
    Нужно потому, что nucleus_grid и cytoplasm_grid могут различаться, и тогда
    карты голов приходят разного размера."""
    arr = np.asarray(arr)
    if tuple(arr.shape) == tuple(shape):
        return arr
    idx = np.ix_(*[np.clip((np.arange(s) * a // max(s, 1)), 0, a - 1)
                   for s, a in zip(shape, arr.shape)])
    return arr[idx]


def compute_containment_penalty(
        nucleus_instances,
        cytoplasm_instances,
        cytoplasm_dist_pred,
        assignments,
        containment_margin=2.0,
        rays=None,
        grid=None
):
    """
    Compute containment penalty in numpy (for monitoring or post-processing).

    Для многоядерных клеток штраф считается по КАЖДОМУ ядру и усредняется
    по клетке — цитоплазма должна вмещать все свои ядра.

    Радиус клетки берётся ВДОЛЬ НАПРАВЛЕНИЯ центр клетки -> центр ядра, а не
    как среднее по лучам. Среднее занижено кратно: при анизотропном наборе
    лучей большинство направлений упирается в верх/низ плоской клетки и даёт
    короткие значения, из-за чего штраф срабатывал бы на нормальных клетках.
    Без rays поведение падает обратно на среднее (обратная совместимость).

    grid — прореживание, с которым посчитана cytoplasm_dist_pred. Центроиды из
    regionprops живут в full-res координатах, а карта дистанций — в grid-
    разрешении; прежняя версия индексировала карту full-res координатами, и
    проверка границ отбрасывала ВСЕ клетки, то есть функция тихо возвращала 0
    на любом изображении крупнее патча.

    Args:
        nucleus_instances: (Z, Y, X) nucleus instance labels
        cytoplasm_instances: (Z, Y, X) cytoplasm instance labels
        cytoplasm_dist_pred: (z, y, x, K) predicted cytoplasm distances (grid-res)
        assignments: dict {nucleus_id: cytoplasm_id}
        containment_margin: tolerance in voxels
        rays: объект лучей цитоплазмы (config.rays_cytoplasm), опционально
        grid: прореживание карты дистанций, опционально

    Returns:
        penalty: scalar, average containment violation
        details: dict with per-cell penalties
    """
    if len(assignments) == 0:
        return 0.0, {}

    cytoplasm_dist_pred = np.asarray(cytoplasm_dist_pred)
    nd = int(np.asarray(cytoplasm_instances).ndim)

    if grid is None:
        # Выводим grid из соотношения форм: карта дистанций прорежена ровно так.
        grid = tuple(max(1, int(round(s / max(d, 1))))
                     for s, d in zip(np.asarray(cytoplasm_instances).shape,
                                     cytoplasm_dist_pred.shape[:nd]))
    grid = tuple(int(g) for g in grid)

    verts = None
    if rays is not None:
        try:
            verts = np.asarray(rays.vertices, np.float32)          # (K, nd)
            vnorm = np.linalg.norm(verts, axis=1)
            vunit = verts / np.maximum(vnorm[:, None], 1e-9)
        except Exception:
            verts = None

    nucleus_props = {prop.label: prop for prop in regionprops(nucleus_instances)}
    cytoplasm_props = {prop.label: prop for prop in regionprops(cytoplasm_instances)}

    # Group nuclei by cell
    cell_to_nuclei = {}
    for nuc_id, cell_id in assignments.items():
        cell_to_nuclei.setdefault(cell_id, []).append(nuc_id)

    dist_shape = cytoplasm_dist_pred.shape[:nd]
    cell_penalties = {}

    for cell_id, nucleus_ids in cell_to_nuclei.items():
        if cell_id not in cytoplasm_props:
            continue
        cell_center = np.array(cytoplasm_props[cell_id].centroid, np.float64)

        # Центроид вогнутой клетки может лежать вне её самой — тогда лучи там
        # фоновые. Подтягиваем к ближайшему вокселю СВОЕЙ клетки.
        cc_int = np.clip(np.round(cell_center).astype(int),
                         0, np.array(cytoplasm_instances.shape) - 1)
        if cytoplasm_instances[tuple(cc_int)] != cell_id:
            pts = np.argwhere(cytoplasm_instances == cell_id)
            if len(pts) == 0:
                continue
            cc_int = pts[np.argmin(np.abs(pts - cell_center).sum(axis=1))]
            cell_center = cc_int.astype(np.float64)

        # full-res -> grid-координаты карты дистанций
        cc_grid = tuple(int(np.clip(cc_int[a] // grid[a], 0, dist_shape[a] - 1))
                        for a in range(nd))
        predicted_radii = np.asarray(cytoplasm_dist_pred[cc_grid], np.float64)

        penalties = []
        for nuc_id in nucleus_ids:
            if nuc_id not in nucleus_props:
                continue
            nucleus_center = np.array(nucleus_props[nuc_id].centroid, np.float64)
            vec = nucleus_center - cell_center
            distance = float(np.linalg.norm(vec))
            if distance < 1e-6:
                continue

            if verts is not None and len(predicted_radii) == len(verts):
                # Радиус вдоль направления на ядро: берём луч с максимальным
                # косинусом и переводим его длину в воксели (|d * v|).
                u = vec / distance
                k = int(np.argmax(vunit @ u))
                radius = float(predicted_radii[k] * vnorm[k])
            else:
                radius = float(np.mean(predicted_radii))

            if distance > radius + containment_margin:
                penalties.append((distance - radius) / (radius + 1e-6))

        if len(penalties) > 0:
            cell_penalties[cell_id] = {
                'penalty': float(np.mean(penalties)),
                'num_nuclei': len(nucleus_ids),
                'violations': len(penalties)
            }

    if len(cell_penalties) > 0:
        total_penalty = float(np.mean([v['penalty'] for v in cell_penalties.values()]))
    else:
        total_penalty = 0.0

    return total_penalty, cell_penalties


def monitor_wbr_metric(nucleus_prob, cytoplasm_prob, cytoplasm_instances=None,
                       prob_thresh=0.5):
    """
    Monitor WBR metric in numpy (for logging/visualization):
    доля МАССЫ ЯДРА, оказавшаяся вне клетки.

    Три отличия от прежней версии.

    1. Нормировка на площадь ядра, а не на площадь фона. Прежнее деление на
       число вокселей с cytoplasm_prob < 0.5 давало почти весь том в
       знаменателе, и метрика была порядка 1e-5 при любом состоянии модели.
    2. Контейнер предпочтительно БИНАРНЫЙ (cytoplasm_instances). cytoplasm_prob
       это edt_prob: выше 0.5 лежит лишь 14-25% площади клетки, поэтому порог
       по нему относит бо́льшую часть самой клетки к «снаружи».
    3. Карты голов приводятся к общей форме: nucleus_grid и cytoplasm_grid
       могут различаться, и тогда logical_and падал по broadcast.

    Args:
        nucleus_prob: (Z, Y, X) nucleus probability
        cytoplasm_prob: (Z, Y, X) cytoplasm probability
        cytoplasm_instances: (Z, Y, X) бинарный/instance контейнер, опционально
        prob_thresh: порог бинаризации вероятностей

    Returns:
        ratio: fraction of nucleus outside cytoplasm, [0, 1]
    """
    nucleus_prob = np.asarray(nucleus_prob)
    shape = nucleus_prob.shape

    if cytoplasm_instances is not None:
        inside = _to_shape(np.asarray(cytoplasm_instances) > 0, shape)
    else:
        inside = _to_shape(np.asarray(cytoplasm_prob), shape) > prob_thresh

    nucleus_mask = nucleus_prob > prob_thresh
    n_nuc = int(nucleus_mask.sum())
    if n_nuc == 0:
        return 0.0
    return float(np.logical_and(nucleus_mask, ~inside).sum()) / n_nuc