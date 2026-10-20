"""Diagnostic controls; GT prompts are never part of production inference."""
import numpy as np
from cytonucs_postprocess import assemble_masks, assign_final_masks
from cytonucs_refinement_audit import RefinementStageAudit
from cytonucs_inference import compute_jtpr, predict_instances_prompted


def compare_nested_assembly(config, shape, candidates, gt, assignments):
    """Both-head A/B on identical cached candidates; no extra network forwards."""
    results = {}
    for name, modes in (
        ('baseline', ('voxel', 'object_quality')),
        ('nested', ('object_quality_containment', 'object_quality_containment'))):
        labels, stats, reports = {}, {}, {}
        for head, mode in zip(('nucleus', 'cytoplasm'), modes):
            observer = RefinementStageAudit(gt[head])
            labels[head], stats[head] = assemble_masks(
                candidates[head], shape, getattr(config, 'mask_nms_' + head),
                config.mask_containment_threshold if head == 'cytoplasm' or name == 'nested' else 0.,
                getattr(config, 'mask_overlap_' + head), observer=observer,
                assembly_mode=mode, min_remaining_fraction=config.mask_min_remaining_fraction)
            reports[head] = observer.report()
        pairs = assign_final_masks(labels['nucleus'], labels['cytoplasm'], config.pair_min_overlap,
                                   candidates['evidence'], config.pair_min_mask_iou)
        jtpr, counts = compute_jtpr(labels['nucleus'], labels['cytoplasm'], gt['nucleus'], gt['cytoplasm'],
                                   assignments_pred=pairs, assignments_gt=assignments)
        results[name] = dict(jtpr=jtpr, pairs=counts, assembly=stats, stage_audit=reports,
                             modes=dict(zip(('nucleus', 'cytoplasm'), modes)))
    return results


def compare_assembly(config, shape, candidates, nucleus_labels, gt, assignments):
    results = {}
    for mode in ('voxel', 'object', 'object_quality'):
        audit = RefinementStageAudit(gt['cytoplasm'])
        cells, stats = assemble_masks(candidates['cytoplasm'], shape,
            config.mask_nms_cytoplasm, config.mask_containment_threshold,
            config.mask_overlap_cytoplasm, observer=audit, assembly_mode=mode,
            min_remaining_fraction=config.mask_min_remaining_fraction)
        pairs = assign_final_masks(nucleus_labels, cells, config.pair_min_overlap,
                                   candidates['evidence'], config.pair_min_mask_iou)
        jtpr, counts = compute_jtpr(nucleus_labels,cells,gt['nucleus'],gt['cytoplasm'],
                                   assignments_pred=pairs,assignments_gt=assignments)
        report = audit.report()
        stages = ('before_nms','after_nms','competition','final')
        report['stages'] = {k:v for k,v in report['stages'].items() if k in stages}
        report['transitions'] = {k:v for k,v in report['transitions'].items() if k.split('->')[0] in stages}
        results[mode] = dict(jtpr=jtpr, pairs=counts, assembly=stats, stage_audit=report)
    return results


def gt_prompt_inputs(labels):
    """One interior point per GT object, bbox/4 Gaussian width as in training.

    No GT mask is sent to the network. A sparse anchor label map only identifies
    the points for the existing runtime, and cannot prescribe an output shape.
    """
    anchors = np.zeros(labels.shape, np.int32)
    centers, sigmas = [], []
    for i, lab in enumerate(np.unique(labels[labels>0]),1):
        points=np.argwhere(labels==lab)
        extent=points.max(0)-points.min(0)+1
        mean=points.mean(0)
        center=points[np.argmin(np.sum(((points-mean)/np.maximum(extent,1))**2,axis=1))]
        anchors[tuple(center)] = i
        centers.append(center);sigmas.append(np.maximum(extent/4.,1))
    return anchors, np.asarray(centers,dtype=int).reshape((-1,labels.ndim)), sigmas


def evaluate_gt_prompts(model, config, image, gt, assignments):
    args={h:gt_prompt_inputs(g) for h,g in gt.items()}
    sigmas={(h,tuple(center)):sigma for h,(_,centers,widths) in args.items()
            for center,sigma in zip(centers,widths)}
    audits={h:RefinementStageAudit(g) for h,g in gt.items()}
    n,c,pairs=predict_instances_prompted(model,config,image,args['nucleus'][1],
        np.ones(len(args['nucleus'][1])),args['cytoplasm'][1],np.ones(len(args['cytoplasm'][1])),
        mask_thresh=config.mask_thresh,nuc_labels=args['nucleus'][0],cyto_labels=args['cytoplasm'][0],
        stage_audit=audits,prompt_sigmas=sigmas)
    jtpr,counts=compute_jtpr(n,c,gt['nucleus'],gt['cytoplasm'],assignments_pred=pairs,assignments_gt=assignments)
    return dict(scope='GT-center/bbox prompt oracle, not deployable inference or validation JTPR; base ROI without detector-ray expansion',
                stage_audit={h:a.report() for h,a in audits.items()},jtpr_oracle=jtpr,pairs=counts,
                refinement=model.last_refinement_diagnostics)
