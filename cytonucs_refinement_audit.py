"""Read-only GT diagnostics. Never use these metrics to select predictions."""
import numpy as np
from scipy.ndimage import find_objects
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


class RefinementStageAudit:
    stages = ('threshold', 'component', 'before_nms', 'after_nms', 'competition', 'final')

    def __init__(self, gt, threshold=.5):
        self.gt = np.asarray(gt)
        if self.gt.ndim not in (2, 3) or not np.issubdtype(self.gt.dtype, np.integer) or np.any(self.gt < 0):
            raise ValueError('GT must be nonnegative 2D/3D integer labels')
        ids, areas = np.unique(self.gt, return_counts=True)
        positive = ids > 0
        self.ids = ids[positive]
        self.areas = areas[positive]
        self.index = {int(v): i for i, v in enumerate(self.ids)}
        self.threshold = float(threshold)
        self.rows = {s: [] for s in self.stages}
        self.rois = []
        self.parts = {s: {} for s in ('before_nms', 'final')}
        self.proposals = None
        self.assembly_decisions = []
        self._gt_slices = None

    def boundary_split(self, mask, lo, lab):
        """Where the error voxels of a written mask against GT `lab` lie (3D only).

        Each voxel of the symmetric difference falls in exactly one part:
        z  - on a Z slice that only one of the two masks occupies (whole extra/missing slices);
        xy - otherwise, in a (y, x) column that only one of the two masks occupies (the contour
             of the XY projection);
        mixed - on shared slices inside the shared projection (rims of objects a few slices
             thick, where Z and XY cannot be told apart).
        iou_<part>_exact is the IoU with that part of the error removed: its extra voxels
        dropped, its missing voxels counted as found.
        """
        if self._gt_slices is None:
            self._gt_slices = find_objects(self.gt)
        gt_sl = self._gt_slices[lab - 1]
        p_lo = np.asarray(lo, int); p_hi = p_lo + mask.shape
        g_lo = np.array([s.start for s in gt_sl]); g_hi = np.array([s.stop for s in gt_sl])
        u_lo, u_hi = np.minimum(p_lo, g_lo), np.maximum(p_hi, g_hi)
        g = self.gt[tuple(slice(a, b) for a, b in zip(u_lo, u_hi))] == lab
        p = np.zeros(g.shape, bool)
        p[tuple(slice(a, a + s) for a, s in zip(p_lo - u_lo, mask.shape))] = mask
        extra, missing = p & ~g, g & ~p
        z = np.logical_xor(p.any(axis=(1, 2)), g.any(axis=(1, 2)))[:, None, None]
        xy = np.logical_xor(p.any(axis=0), g.any(axis=0))[None] & ~z
        inter, union = int(np.count_nonzero(p & g)), int(np.count_nonzero(p | g))
        result = dict(gt=str(lab), extra_slices=int(np.count_nonzero(z[:, 0, 0] & p.any(axis=(1, 2)))),
                      missing_slices=int(np.count_nonzero(z[:, 0, 0] & g.any(axis=(1, 2)))),
                      error_voxels=union - inter)
        for part, where in (('z', z), ('xy', xy), ('mixed', ~z & ~xy)):
            e, m = int(np.count_nonzero(extra & where)), int(np.count_nonzero(missing & where))
            result[part + '_error_voxels'] = e + m
            result['iou_' + part + '_exact'] = float((inter + m) / max(union - e, 1))
        return result

    def record_assembly_decision(self, rank, reason, candidate, remainder=None, **details):
        """Observe the decision after it was made; never return a selection signal."""
        row = self.rows['after_nms'][rank]
        overlaps = {str(int(self.ids[j])): float(v) for j, v in row.items()}
        best = max(overlaps, key=overlaps.get) if overlaps else None
        after = {}
        boundary = None
        if remainder is not None:
            mask = remainder['mask']; lo = remainder['lo']
            sl = tuple(slice(int(x), int(x+s)) for x, s in zip(lo, mask.shape))
            ids, counts = np.unique(self.gt[sl][mask], return_counts=True)
            area = int(np.count_nonzero(mask))
            after = {str(int(lab)): float(count/(area+self.areas[self.index[int(lab)]]-count))
                     for lab, count in zip(ids, counts) if lab > 0}
            matched = max(after, key=after.get) if after else None
            # Error split for masks matched to a GT (IoU >= 0.5 is the unique match).
            if self.gt.ndim == 3 and matched is not None and after[matched] >= self.threshold:
                boundary = self.boundary_split(mask, lo, int(matched))
        if boundary is not None:
            details['boundary'] = boundary
        self.assembly_decisions.append(dict(rank=int(rank), reason=reason,
            area=int(candidate['area']), origin=list(map(int,candidate['lo'])),
            best_gt=best, best_iou=overlaps.get(best,0.), gt_iou=overlaps,
            written_gt_iou=after, **details))

    def record_proposals(self, prob, grid, border, window, selected, threshold, aniso, residual=None):
        """Explain raw detection proposals without changing selection or using GT in inference.

        residual: grid coordinates of window-suppressed peaks offered to the second pass.
        """
        from scipy.ndimage import maximum, maximum_filter
        from stardist.utils import edt_prob
        ss = tuple(slice(None,None,int(g)) for g in grid)
        sampled = self.gt[ss]
        if sampled.shape != prob.shape:
            raise ValueError('Proposal audit probability/GT grid shape mismatch')
        eligible = np.ones(prob.shape, bool)
        if border > 0:
            for axis in (-2,-1):
                sl=[slice(None)]*prob.ndim;sl[axis]=slice(0,border);eligible[tuple(sl)]=False
                sl[axis]=slice(max(0,prob.shape[axis]-border),None);eligible[tuple(sl)]=False
        masked = np.where(eligible,prob,-np.inf)
        local = eligible & (masked >= maximum_filter(masked,size=tuple(window),mode='constant',cval=-np.inf))
        chosen = np.zeros_like(eligible)
        if len(selected):chosen[tuple(np.asarray(selected).T)] = True
        def maxima(values, allowed):
            return np.asarray(maximum(np.where(allowed,values,-1.),labels=sampled,index=self.ids)) if len(self.ids) else np.array([])
        raw = maxima(prob,np.ones_like(eligible));inside=maxima(prob,eligible);peaks=maxima(prob,local)
        retained=maxima(prob,chosen)
        residual_ids = set()
        if residual is not None and len(residual):
            pts = np.clip(np.round(np.asarray(residual)).astype(int), 0, np.array(prob.shape)-1)
            residual_ids = set(map(int, sampled[tuple(pts.T)])) - {0}
        present=set(map(int,np.unique(sampled)));border_present=set(map(int,np.unique(sampled[eligible])))
        ideal = edt_prob(self.gt,anisotropy=aniso)[ss]                 # as the training targets: no hole filling
        ideal_max=maxima(ideal,np.ones_like(eligible))
        by_gt={};counts={k:0 for k in ('no_grid_voxel','border_only','border_suppressed','below_threshold','local_max_suppressed','plateau_or_cap_removed','selected')}
        for j,lab in enumerate(self.ids):
            reason=('no_grid_voxel' if lab not in present else 'border_only' if lab not in border_present
                    else 'border_suppressed' if raw[j] > threshold and inside[j] <= threshold
                    else 'below_threshold' if inside[j] <= threshold else 'local_max_suppressed' if peaks[j] <= threshold
                    else 'plateau_or_cap_removed' if retained[j] <= threshold else 'selected')
            counts[reason]+=1
            by_gt[str(int(lab))]=dict(reason=reason,max_prob_grid=float(raw[j]),max_prob_after_border=float(inside[j]),
                max_legacy_edt_target_on_grid=float(ideal_max[j]),residual_peak=int(lab) in residual_ids)
        self.proposals=dict(threshold=float(threshold),grid=list(grid),border_grid_voxels=int(border),window=list(window),
            reason_counts=counts,per_gt=by_gt,
            local_max_suppressed_with_residual_peak=sum(v['reason']=='local_max_suppressed' and v['residual_peak'] for v in by_gt.values()),
            legacy_gt_target_max_at_or_below_threshold=sum(lab in present and ideal_max[j] <= threshold for j,lab in enumerate(self.ids)),
            sweeps=[dict(threshold=float(t),gt_above_threshold=int(np.count_nonzero(inside>t)),
                         gt_with_local_max=int(np.count_nonzero(peaks>t)),
                         background_local_maxima=int(np.count_nonzero(local & (sampled==0) & (prob>t))))
                    for t in sorted(set([.3,.4,.5,float(threshold)]))],
            scope='Detection-grid coverage, not instance recall/JTPR. Sweeps precede plateau collapse and candidate cap; legacy EDT maxima are diagnostic targets, not model predictions.')

    def add(self, stage, candidate):
        if candidate is None:
            return
        mask = candidate['mask']
        lo = np.asarray(candidate['lo'], int)
        hi = lo + mask.shape
        if len(lo) != self.gt.ndim or np.any(lo < 0) or np.any(hi > self.gt.shape):
            raise ValueError('Candidate outside GT coordinates')
        sl = tuple(slice(int(a), int(b)) for a,b in zip(lo,hi))
        ids, counts = np.unique(self.gt[sl][mask], return_counts=True)
        area = int(np.count_nonzero(mask))
        row = {}
        for lab, intersection in zip(ids, counts):
            if lab > 0:
                index = self.index[int(lab)]
                row[index] = float(intersection / (area + self.areas[index] - intersection))
        self.rows[stage].append(row)
        if stage in self.parts and area and np.any(ids > 0):
            positive = np.flatnonzero(ids > 0)
            k = positive[np.argmax(counts[positive])]
            lab, intersection = int(ids[k]), int(counts[k])
            purity = intersection / area
            coverage = intersection / self.areas[self.index[lab]]
            # Ignore tiny specks and predictions mostly outside the assigned GT.
            # These diagnostic cutoffs never affect masks or production decisions.
            if purity >= .5 and coverage >= .1:
                overlap = mask & (self.gt[sl] == lab)
                bounds = []
                for axis in range(self.gt.ndim):
                    axes = tuple(a for a in range(self.gt.ndim) if a != axis)
                    occupied = np.flatnonzero(np.any(overlap, axis=axes))
                    bounds.append([int(lo[axis]+occupied[0]), int(lo[axis]+occupied[-1]+1)])
                self.parts[stage].setdefault(str(lab), []).append(dict(
                    gt_coverage=float(coverage), prediction_purity=float(purity), intersection_bounds=bounds))

    def observe(self, stage, candidates=None, labels=None):
        if labels is not None:
            if labels.shape != self.gt.shape:
                raise ValueError('Prediction/GT shape mismatch')
            for lab, sl in enumerate(find_objects(labels), 1):
                if sl is not None:
                    self.add(stage, dict(lo=[s.start for s in sl], mask=labels[sl] == lab))
        else:
            for candidate in candidates:
                self.add(stage, candidate)

    def record_roi(self, origin, shape, padded_shape, center, grid):
        # Compact geometry evidence; masks and feature arrays are not retained.
        self.rois.append(dict(origin=list(map(int,origin)), shape=list(map(int,shape)),
            padded_shape=list(map(int,padded_shape)), center_global=list(map(float,center)),
            grid=list(map(int,grid))))

    def report(self):
        result = {}
        for stage, rows in self.rows.items():
            best = np.zeros(len(self.ids), np.float64)
            rr, cc = [], []
            candidate_best = []
            for i, row in enumerate(rows):
                candidate_best.append(max(row.values(), default=0.))
                for j, value in row.items():
                    best[j] = max(best[j], value)
                    if value >= self.threshold:
                        rr.append(i);cc.append(j)
            graph = csr_matrix((np.ones(len(rr)), (rr,cc)), shape=(len(rows),len(self.ids)))
            matched = int(np.count_nonzero(maximum_bipartite_matching(graph, perm_type='column') >= 0)) if rr else 0
            result[stage] = dict(candidates=len(rows), gt_count=len(self.ids),
                gt_covered_at_iou=int(np.count_nonzero(best >= self.threshold)),
                oracle_one_to_one_tp=matched,
                best_iou_per_gt={str(int(k)):float(v) for k,v in zip(self.ids,best)},
                candidate_best_iou_quantiles=np.quantile(candidate_best,[0,.25,.5,.75,1]).tolist() if rows else [])
        transitions = {}
        for before,after in zip(self.stages,self.stages[1:]):
            a=result[before]['best_iou_per_gt'];b=result[after]['best_iou_per_gt']
            transitions[before+'->'+after] = dict(
                lost_gt_ids=[k for k in a if a[k]>=self.threshold and b[k]<self.threshold],
                gained_gt_ids=[k for k in a if a[k]<self.threshold and b[k]>=self.threshold])
        fragmentation = {}
        for stage, parts in self.parts.items():
            axial_pairs = 0
            for group in parts.values():
                if self.gt.ndim == 3:
                    for i, a in enumerate(group):
                        aa = a['intersection_bounds']
                        for b in group[i+1:]:
                            bb = b['intersection_bounds']
                            z_disjoint = aa[0][1] <= bb[0][0] or bb[0][1] <= aa[0][0]
                            xy_overlap = all(max(aa[d][0],bb[d][0]) < min(aa[d][1],bb[d][1]) for d in (1,2))
                            axial_pairs += int(z_disjoint and xy_overlap)
            fragmentation[stage] = dict(gt_with_multiple_parts=sum(len(p)>1 for p in parts.values()),
                z_disjoint_xy_bbox_pairs=axial_pairs, parts_per_gt=parts)
        prompt_counts = {}
        outside = 0
        for roi in self.rois:
            center = np.round(roi['center_global']).astype(int)
            if np.any(center < 0) or np.any(center >= self.gt.shape):
                outside += 1;continue
            lab = int(self.gt[tuple(center)])
            if not lab:
                outside += 1;continue
            prompt_counts[str(lab)] = prompt_counts.get(str(lab),0) + 1
        duplicates = np.zeros(len(self.ids),int)
        for row in self.rows['before_nms']:
            for j,value in row.items():duplicates[j] += int(value >= self.threshold)
        monitoring = dict(gt_count=len(self.ids), gt_without_prompt_center=len(self.ids)-len(prompt_counts),
            gt_multiple_prompt_centers=sum(n>1 for n in prompt_counts.values()), prompt_centers_outside_gt=outside,
            gt_multiple_good_candidates=int(np.count_nonzero(duplicates > 1)),
            gt_without_good_candidate=len(self.ids)-result['before_nms']['gt_covered_at_iou'],
            final_gt_multiple_parts=fragmentation['final']['gt_with_multiple_parts'],
            final_z_disjoint_xy_bbox_pairs=fragmentation['final']['z_disjoint_xy_bbox_pairs'])
        if self.proposals is not None:
            monitoring.update({'proposal_'+k:v for k,v in self.proposals['reason_counts'].items()})
            monitoring['legacy_gt_target_below_threshold']=int(self.proposals['legacy_gt_target_max_at_or_below_threshold'])
            monitoring['proposal_local_max_suppressed_with_residual_peak']=int(self.proposals['local_max_suppressed_with_residual_peak'])
        # Attribution is per GT coverage, not a causal A/B or one-to-one matching.
        lost_assembly = []
        before = result['before_nms']['best_iou_per_gt']
        final = result['final']['best_iou_per_gt']
        decisions = {d['rank']: d for d in self.assembly_decisions}
        for gt_id, value in before.items():
            if value < self.threshold or final[gt_id] >= self.threshold:
                continue
            good = []
            for d in self.assembly_decisions:
                iou = d['gt_iou'].get(gt_id, 0.)
                if iou < self.threshold:
                    continue
                blockers = []
                for rank in d['blockers']:
                    b = decisions[rank]
                    blockers.append(dict(rank=rank, quality=b['quality'], best_gt=b['best_gt'],
                        candidate_iou_with_lost_gt=b['gt_iou'].get(gt_id,0.),
                        written_iou_with_lost_gt=b['written_gt_iou'].get(gt_id,0.)))
                good.append(dict(rank=d['rank'], quality=d['quality'], before_iou=iou,
                    written_iou=d['written_gt_iou'].get(gt_id,0.), reason=d['reason'], blockers=blockers))
            lost_assembly.append(dict(gt_id=gt_id, best_before=value, best_final=final[gt_id],
                                      good_candidates=good))
        # Size and volume-face contact of every GT: separates truncated slivers from whole objects.
        geometry = {}
        for lab, sl in enumerate(find_objects(self.gt), 1):
            if sl is None or lab not in self.index:
                continue
            faces = [f'{"zyx"[a - self.gt.ndim + 3]}{side}' for a in range(self.gt.ndim)
                     for side, hit in ((0, sl[a].start == 0), (1, sl[a].stop == self.gt.shape[a])) if hit]
            geometry[str(lab)] = dict(volume=int(self.areas[self.index[lab]]), faces=','.join(faces))
        monitoring['gt_touching_volume_face'] = sum(bool(g['faces']) for g in geometry.values())
        return dict(iou_threshold=self.threshold, gt_geometry=geometry,
            scope='Candidate coverage and maximum one-to-one matching; overlapping stages are not final detection metrics or JTPR',
            stages=result, transitions=transitions, rois=self.rois,
            assembly_decisions=self.assembly_decisions, assembly_lost_gt=lost_assembly,
            monitoring=monitoring, proposals=self.proposals, prompt_centers_per_gt=prompt_counts, fragmentation=fragmentation,
            fragmentation_definition='Dominant GT overlap: prediction purity >=0.5 and GT coverage >=0.1; Z disjoint intersection bounds with overlapping XY bounding boxes are a diagnostic proxy, not proof of a Z split. ROI centers count refinement jobs, not independent objects.')
