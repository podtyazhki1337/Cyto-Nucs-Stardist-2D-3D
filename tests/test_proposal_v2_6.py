import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
from test_fixes import config,ROOT
from cytonucs_config import CytoNucsConfig
from cytonucs_losses import compute_stardist_gt,CytoNucsLoss
from cytonucs_inference import CytoNucsPredictor
from cytonucs_refinement_audit import RefinementStageAudit


class ProposalTests(unittest.TestCase):
    def test_perfect_legacy_target_can_fail_threshold(self):
        cfg=config(nucleus_prob_grid_normalize=True)
        labels=np.zeros((8,32,32),np.int32);labels[2:5,2:5,2:5]=17
        args=dict(rays=cfg.rays_nucleus,aniso=(19,1,1))
        old,d=compute_stardist_gt(labels,cfg,cfg.nucleus_grid,**args)
        new,d2=compute_stardist_gt(labels,cfg,cfg.nucleus_grid,normalize_prob=True,**args)
        self.assertEqual(old.max(),.5);self.assertEqual(np.count_nonzero(old>.5),0)
        self.assertEqual(new.max(),1);self.assertGreater(np.count_nonzero(new>.5),0)
        np.testing.assert_array_equal(d,d2)
        np.testing.assert_array_equal(new[labels[:,::2,::2]==0],0)

    def test_fallback_gt_matches_direct_normalized(self):
        cfg=config(nucleus_prob_grid_normalize=True)
        lab=np.zeros(cfg.train_patch_size,np.int32);lab[2:5,2:5,2:5]=9
        loss=CytoNucsLoss(cfg)
        p,d=loss._star_dist_wrapper_nucleus(tf.constant(lab[None]))
        expected,_=compute_stardist_gt(lab,cfg,cfg.nucleus_grid,rays=cfg.rays_nucleus,
                    aniso=cfg.rays_anisotropy_nucleus,normalize_prob=True)
        np.testing.assert_array_equal(p[0],expected)

    def test_audit_reason_partition_and_threshold_sweep(self):
        gt=np.zeros((6,14,14),np.int32)
        gt[2,1,1]=1  # not sampled at grid2
        gt[2,0,0]=2  # removed by border
        gt[2,4,4]=3  # low probability
        gt[2,8,8]=4  # suppressed by stronger nearby background peak
        gt[4,8,4]=5  # survives
        p=np.zeros((6,7,7),np.float32);p[2,0,0]=.9;p[2,2,2]=.35;p[2,4,4]=.6;p[2,4,5]=.9;p[4,4,2]=.9
        a=RefinementStageAudit(gt)
        a.record_proposals(p,(1,2,2),1,(3,3,3),np.array([[2,4,5],[4,4,2]]),.5,(19,1,1))
        r=a.report()['proposals']['reason_counts']
        self.assertEqual(r,dict(no_grid_voxel=1,border_only=1,border_suppressed=0,below_threshold=1,local_max_suppressed=1,plateau_or_cap_removed=0,selected=1))
        self.assertEqual(sum(r.values()),5)
        self.assertGreater(a.proposals['sweeps'][0]['gt_above_threshold'],a.proposals['sweeps'][-1]['gt_above_threshold'])

    def test_actual_inference_audit_does_not_change_anchors(self):
        cfg=config(proposal_mode='peaks');gt=np.zeros((8,32,32),np.int32);gt[2:5,10:18,10:18]=1
        p=np.zeros((8,16,16),np.float32);p[3,7,7]=.9
        stub=SimpleNamespace(config=cfg,ndim=3,rays_nucleus=cfg.rays_nucleus,rays_cytoplasm=cfg.rays_cytoplasm,_current_type='nucleus')
        d=np.ones(p.shape+(12,),np.float32)
        baseline=CytoNucsPredictor._stardist_postprocess(stub,p,d,.5,.3)
        a=RefinementStageAudit(gt);stub._refinement_stage_audit={'nucleus':a}
        result=CytoNucsPredictor._stardist_postprocess(stub,p,d,.5,.3)
        np.testing.assert_array_equal(result,baseline)
        self.assertEqual(a.proposals['reason_counts']['selected'],1)
        CytoNucsPredictor._stardist_postprocess(stub,p*.1,d,.5,.3)
        self.assertEqual(a.proposals['reason_counts']['below_threshold'],1)

    def test_json_contract(self):
        raw=json.loads((ROOT/'config_cytonucs_v2_6.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            raw['training']['resume_from_checkpoint']=None;raw['output']['basedir']=tmp
            p=Path(tmp)/'c.json';p.write_text(json.dumps(raw));cfg=CytoNucsConfig.from_json(p)
            for sec,settings in cfg.FIXES_SETTINGS.items():
                for key in settings:self.assertEqual(getattr(cfg,key),raw[sec][key])
            self.assertTrue(cfg.nucleus_prob_grid_normalize)
        with self.assertRaises(ValueError):config(nucleus_prob_grid_normalize='yes')


if __name__=='__main__':unittest.main()
