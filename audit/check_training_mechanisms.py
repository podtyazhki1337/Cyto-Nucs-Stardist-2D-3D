"""CPU synthetic mechanism audit, not an estimate of checkpoint gradients/JTPR."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import hashlib
import platform
import numpy as np
import tensorflow as tf
import stardist
from cytonucs_config import CytoNucsConfig
from cytonucs_geometry import architecture_plan, detection_receptive_field
from cytonucs_model import build_cytonucs_model
from cytonucs_losses import CytoNucsLoss, compute_stardist_gt


def main(output):
    if output.exists():
        raise FileExistsError(output)
    cfg = CytoNucsConfig(ndim=3, n_rays=96, n_channel_in=1,
        train_patch_size=(16,64,64), nucleus_grid=(1,2,2), cytoplasm_grid=(1,2,2),
        anisotropy=(5,1,1), unet_n_depth=4, unet_n_filter_base=2,
        net_conv_after_unet=4, mask_refinement_filters=4, prompts_per_crop=1,
        train_batch_size=1, median_max_dist_nucleus=54., median_max_dist_cytoplasm=99.)
    zz,yy,xx = np.indices(cfg.train_patch_size)
    n = (((zz-8)/2.5)**2+((yy-32)/14)**2+((xx-32)/14)**2 <= 1).astype('int32')
    c = (((zz-8)/5.)**2+((yy-32)/25)**2+((xx-32)/25)**2 <= 1).astype('int32')
    from stardist.rays3d import Rays_GoldenSpiral
    targets = {}
    for key,lab,aniso in [('nucleus',n,(19,1,1)),('cytoplasm',c,(28,1,1))]:
        p,d = compute_stardist_gt(lab,cfg,(1,2,2),
            rays=Rays_GoldenSpiral(96,anisotropy=aniso),aniso=aniso)
        targets[key] = (tf.constant(p[None]),tf.constant(d[None],tf.float32))
    prompt = np.exp(-.5*(((zz-8)/2.5)**2+((yy-32)/7)**2+((xx-32)/7)**2))
    img = .1+ .4*c + .4*n + np.random.default_rng(42).normal(0,.03,n.shape)
    x = tf.constant(np.stack([img,prompt,prompt],axis=-1)[None],tf.float32)
    runs=[]
    for seed in (42,43,44):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model=build_cytonucs_model(cfg, name='gradient_audit').keras_model
        loss=CytoNucsLoss(cfg)
        with tf.GradientTape(persistent=True) as tape:
            pred=model(x,training=True)
            terms={}
            for key,lab,w in [('nucleus',n,.2),('cytoplasm',c,.5)]:
                p,d=targets[key]
                ls=loss._stardist_loss(p,pred[key+'_prob'],d,pred[key+'_dist'],
                    name=key,prob_logit=pred[key+'_prob_logit'])
                terms[key+'_prob']=ls['bce']
                terms[key+'_dist']=w*ls['distance']
                target=tf.constant(lab[None,::1,::2,::2,None],tf.float32)
                # Both production refinement terms; a single object gives target=foreground.
                z=pred[key+'_mask_logit']
                terms[key+'_mask_ce']=.2*loss._refinement_ce_loss(z,target,target)
                # Use the model's mask output for a differentiable Dice probe.
                pm=pred[key+'_mask']
                terms[key+'_mask_dice_probe']=.2*(1-(2*tf.reduce_sum(pm*target)+1e-6)/
                    (tf.reduce_sum(pm)+tf.reduce_sum(target)+1e-6))
        variables=model.trainable_variables
        groups={
            'backbone':[i for i,v in enumerate(variables) if 'enc_' in v.path or 'pre_pool' in v.path],
            'nuc_decoder':[i for i,v in enumerate(variables) if 'nuc_dec' in v.path],
            'cyto_decoder':[i for i,v in enumerate(variables) if 'cyto_dec' in v.path],
            'prob_heads':[i for i,v in enumerate(variables) if 'prob_logit' in v.path],
            'mask_heads':[i for i,v in enumerate(variables) if 'refine_conv' in v.path or 'mask_out' in v.path],
        }
        assert all(groups.values()),groups
        measures={}; gradients={}
        for name,value in terms.items():
            gs=tape.gradient(value,variables); gradients[name]=gs
            measures[name]={'loss':float(value),'groups':{group:{
                'connected_variables':sum(gs[i] is not None for i in indices),
                'norm':float(tf.linalg.global_norm([gs[i] for i in indices if gs[i] is not None]))}
                for group,indices in groups.items()}}
            if '_mask_' in name:
                for group in ('backbone','nuc_decoder','cyto_decoder','prob_heads'):
                    assert measures[name]['groups'][group]['connected_variables']==0,(name,group)
                assert measures[name]['groups']['mask_heads']['norm']>0
        cosines={}
        names=['nucleus_prob','nucleus_dist','cytoplasm_prob','cytoplasm_dist']
        for i,a in enumerate(names):
            for b in names[i+1:]:
                pairs=[(gradients[a][j],gradients[b][j]) for j in groups['backbone']
                       if gradients[a][j] is not None and gradients[b][j] is not None]
                numerator=sum(tf.reduce_sum(ga*gb) for ga,gb in pairs)
                denom=tf.linalg.global_norm([ga for ga,gb in pairs])*tf.linalg.global_norm([gb for ga,gb in pairs])
                cosines[a+' vs '+b]=float(numerator/(denom+1e-12))
        runs.append({'seed':seed,'terms':measures,'backbone_cosines':cosines})
        del tape
    # Probability BCE pressure at a constant initial logit: exact dL/dz=(.5-target)/N.
    pressure={}
    for key,(p,d) in targets.items():
        a=p.numpy(); fg=a>.01; g=.5-a
        pressure[key]={'foreground_fraction':float(fg.mean()),
            'target_mean':float(a.mean()),
            'initial_background_abs_logit_gradient_sum':float(np.abs(g[~fg]).sum()/g.size),
            'initial_foreground_abs_logit_gradient_sum':float(np.abs(g[fg]).sum()/g.size)}
    fullcfg=CytoNucsConfig(ndim=3,n_rays=96,train_patch_size=(16,448,448),
        nucleus_grid=(1,2,2),cytoplasm_grid=(1,2,2),unet_n_depth=4)
    plan=architecture_plan(fullcfg)
    result={'scope':'synthetic random initialization; not epoch30; no training or JTPR measurement',
        'versions':{'python':platform.python_version(),'tensorflow':tf.__version__,'numpy':np.__version__,
                    'stardist':stardist.__version__},
        'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
            ['cytonucs_model.py','cytonucs_losses.py','cytonucs_geometry.py','audit/check_training_mechanisms.py']},
        'synthetic_patch':list(cfg.train_patch_size),'n_rays':96,'ray_anisotropy':{'nucleus':[19,1,1],'cytoplasm':[28,1,1]},
        'production_architecture_plan':plan,'production_receptive_field':detection_receptive_field(plan),
        'probability_pressure':pressure,'runs':runs,'assertions':'passed'}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Saved',output)


if __name__=='__main__':
    main(Path(sys.argv[1]))
