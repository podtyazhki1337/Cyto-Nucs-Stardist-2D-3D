"""Evaluate two named validation volumes on one checkpoint, without training."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit.diagnose_probability import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--compare-assembly',action='store_true',help='Compare cell modes and both-head nested assembly on identical cached candidates')
    parser.add_argument('--gt-prompts',action='store_true',help='Extra forward passes with GT center/bbox prompts; diagnostic oracle only')
    parser.add_argument('--pad-mode',choices=('reflect','constant'),default='reflect',
                        help='Detection input padding for volumes not divisible by the network multiple (A/B)')
    parser.add_argument('--files',nargs='+',default=[
        'hela_spheroid_z029_y1344_x0896_a01.tiff',
        'hela_spheroid_z016_y0896_x0896_a00.tiff'])
    args=parser.parse_args()
    output=Path(args.output);checkpoint=Path(args.checkpoint)
    if output.exists():parser.error('Output exists; choose another path')
    if not checkpoint.is_file():parser.error('Checkpoint missing')
    if any(Path(n).name!=n for n in args.files):parser.error('--files accepts validation basenames only')
    requested=json.loads(Path(args.config).read_text(encoding='utf-8'))
    from cytonucs_gpu_setup import setup_gpu
    if requested.get('model',{}).get('use_gpu'):setup_gpu(**requested.get('gpu_config',{}))
    import tensorflow as tf
    import tifffile
    from csbdeep.utils import normalize
    from cytonucs_config import CytoNucsConfig
    from cytonucs_model import build_cytonucs_model
    from cytonucs_inference import CytoNucsPredictor, compute_jtpr
    from cytonucs_trainer import DiceCallback
    from cytonucs_runtime import read_model_image, sanitized
    from cytonucs_refinement_audit import RefinementStageAudit
    from types import SimpleNamespace
    cfg=CytoNucsConfig.from_json(args.config,checkpoint_path=checkpoint)
    if cfg.refinement_mode!='mask':parser.error('Requires mask refinement')
    tf.keras.utils.set_random_seed(cfg.seed)
    folder=Path(requested['data']['dataset_dir'])/'val'
    for name in args.files:
        for subdir in ('images','nucleus_masks','cytoplasm_masks'):
            if not (folder/subdir/name).is_file():raise FileNotFoundError(folder/subdir/name)
    model=build_cytonucs_model(cfg,name='stage_probe',basedir=str(output.parent))
    model.load_weights(checkpoint)
    predictor=CytoNucsPredictor(model,cfg,prob_thresh_nucleus=cfg.prob_thresh_nucleus,
                              prob_thresh_cytoplasm=cfg.prob_thresh_cytoplasm,nms_thresh=cfg.nms_thresh)
    predictor.input_pad_mode=args.pad_mode
    report=dict(status='running',checkpoint=str(checkpoint.resolve()),checkpoint_sha256=file_hash(checkpoint),
        config=sanitized(requested),effective_config=sanitized(vars(cfg)),seed=cfg.seed,
        geometry=json.loads((checkpoint.parent/'geometry.json').read_text()),
        source_sha256={p.name:file_hash(p) for p in ROOT.glob('cytonucs_*.py')},script_sha256=file_hash(__file__),
        ablation_source_sha256=file_hash(ROOT/'audit/refinement_ablation.py'),
        packages={p:importlib.metadata.version(p) for p in ('tensorflow','keras','numpy','stardist')},
        split='val',selected_files=args.files,input_pad_mode=args.pad_mode,results=[])
    # Effective config includes non-JSON runtime objects (rays); record only serializable fields.
    report['effective_config']={k:v for k,v in report['effective_config'].items()
                               if isinstance(v,(str,int,float,bool,type(None),list,tuple,dict))}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        def save():
            stream.seek(0);json.dump(report,stream,ensure_ascii=False,indent=2,default=str);stream.truncate();stream.flush()
        save()
        try:
            for name in args.files:
                image_path=folder/'images'/name
                image=read_model_image(image_path,cfg.ndim,cfg.n_channel_in).astype(np.float32)
                image=normalize(image,1,99.8,axis=tuple(range(cfg.ndim)))
                gt={h:tifffile.imread(folder/(h+'_masks')/name) for h in ('nucleus','cytoplasm')}
                if any(g.shape!=image.shape[:cfg.ndim] for g in gt.values()):raise ValueError('GT/image mismatch')
                audits={h:RefinementStageAudit(g) for h,g in gt.items()}
                predictor._refinement_stage_audit=audits
                candidate_sink={} if args.compare_assembly else None
                predictor._refinement_candidate_sink=candidate_sink
                try:
                    n,c,_=DiceCallback._predict_full_volume(SimpleNamespace(model=model),predictor,image)
                finally:
                    predictor._refinement_stage_audit=None
                    predictor._refinement_candidate_sink=None
                assignment_path=folder/'assignments'/(Path(name).stem+'.json')
                assignments=({int(k):int(v) for k,v in json.loads(assignment_path.read_text()).items()}
                             if assignment_path.exists() else None)
                jtpr,pairs=compute_jtpr(n,c,gt['nucleus'],gt['cytoplasm'],assignments_pred=predictor._prompt_assignments,
                                      assignments_gt=assignments)
                report['results'].append(dict(filename=name,image_sha256=file_hash(image_path),
                    gt_sha256={h:file_hash(folder/(h+'_masks')/name) for h in gt},
                    stage_audit={h:a.report() for h,a in audits.items()},
                    refinement=model.last_refinement_diagnostics,jtpr=jtpr,pairs=pairs,
                    gt_assignment_source=str(assignment_path) if assignments is not None else 'geometric_fallback',
                    gt_assignment_sha256=file_hash(assignment_path) if assignments is not None else None))
                save()  # Preserve baseline even if an optional ablation fails later.
                from audit.refinement_ablation import compare_assembly, compare_nested_assembly, evaluate_gt_prompts
                if args.compare_assembly:
                    report['results'][-1]['assembly_comparison']=compare_assembly(cfg,gt['nucleus'].shape,
                        candidate_sink,n,gt,assignments)
                    if cfg.mask_containment_threshold > 0:
                        report['results'][-1]['nested_assembly_comparison']=compare_nested_assembly(
                            cfg,gt['nucleus'].shape,candidate_sink,gt,assignments)
                    save()
                if args.gt_prompts:
                    report['results'][-1]['gt_prompt_oracle']=evaluate_gt_prompts(model,cfg,image,gt,assignments)
                    save()
                del candidate_sink
                save();print(f'Stage probe: {len(report["results"])}/{len(args.files)} {name}')
            report['status']='complete';save()
        except Exception as error:
            report['status']='failed';report['error']=str(error);save();raise
    print(f'Saved {output}')


if __name__=='__main__':main()
