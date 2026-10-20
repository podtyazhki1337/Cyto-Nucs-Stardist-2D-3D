import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from cytonucs_config import CytoNucsConfig
from cytonucs_postprocess import mask_candidate, assemble_masks
from cytonucs_refinement_audit import RefinementStageAudit
from audit.refinement_ablation import compare_assembly,gt_prompt_inputs,evaluate_gt_prompts
import cytonucs_inference as inference


class ObjectAssemblyTests(unittest.TestCase):
    def test_new_preset_loads_checkpoint_geometry_and_all_settings(self):
        raw=json.loads((ROOT/'config_cytonucs_v2_2.json').read_text())
        self.assertEqual(raw['mode'],'inference')
        self.assertTrue(raw['inference']['checkpoint_path'].endswith('weights_epoch_080.weights.h5'))
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            raw['output']['basedir']=tmp
            raw['inference']['checkpoint_path']=str(Path(tmp)/'weights_epoch_080.weights.h5')
            (Path(tmp)/'geometry.json').write_bytes((ROOT/'fixes_v2/Epoch70/geometry.json').read_bytes())
            p=Path(tmp)/'config.json';p.write_text(json.dumps(raw))
            cfg=CytoNucsConfig.from_json(p)
            for section,keys in cfg.FIXES_SETTINGS.items():
                for key in keys:self.assertEqual(getattr(cfg,key),raw[section][key],key)
            self.assertEqual(cfg.mask_assembly_cytoplasm,'object_quality')
            self.assertEqual(cfg.mask_assembly_nucleus,'voxel')
        for bad in [dict(mask_assembly_cytoplasm='unknown'),dict(mask_min_remaining_fraction=1.1)]:
            with self.assertRaises(ValueError):config(**bad)

    def test_selected_whole_object_cannot_be_cut_by_later_voxel_scores(self):
        shape=(4,10,10);full=np.ones(shape,bool);stripe=np.zeros(shape,bool);stripe[:,:,4:6]=1
        main=mask_candidate(np.full(shape,.9),full,(0,0,0))
        fragment=mask_candidate(np.full(shape,.8),stripe,(0,0,0))
        fragment['score'][:]=.99 # local logits can beat a globally higher-ranked object
        baseline,_=assemble_masks([main,fragment],shape,overlap_threshold=0.)
        fixed,stats=assemble_masks([main,fragment],shape,assembly_mode='object')
        self.assertLess(np.count_nonzero(baseline==1),np.count_nonzero(full))
        np.testing.assert_array_equal(fixed,np.ones(shape))
        self.assertEqual(stats['rejected_remainder'],1)

    def test_learned_quality_is_rank_only_no_absolute_gate(self):
        shape=(3,10,10);whole=np.ones(shape,bool);stripe=np.zeros(shape,bool);stripe[:,:,4:6]=1
        main=mask_candidate(np.full(shape,.8),whole,(0,0,0));main['predicted_quality']=.20
        fragment=mask_candidate(np.full(shape,.99),stripe,(0,0,0));fragment['predicted_quality']=.10
        out,stats=assemble_masks([fragment,main],shape,assembly_mode='object_quality')
        self.assertEqual(stats['written'],1);self.assertTrue(np.all(out==1))
        # Even a quality far below .25 does not erase an otherwise accepted object.
        fragment['predicted_quality']=float('nan')
        self.assertTrue(np.isfinite(assemble_masks([fragment],shape,assembly_mode='object_quality')[0]).all())

    def test_neighbours_remain_distinct_and_low_remainder_is_not_new_fragment(self):
        shape=(5,10,20);left=np.zeros(shape,bool);left[:,:,:10]=1;right=np.zeros(shape,bool);right[:,:,9:]=1
        a=mask_candidate(np.full(shape,.9),left,(0,0,0));b=mask_candidate(np.full(shape,.8),right,(0,0,0))
        out,s=assemble_masks([a,b],shape,assembly_mode='object')
        self.assertEqual(s['written'],2);self.assertTrue(np.all(out[:,:,:10]==1));self.assertTrue(np.all(out[:,:,10:]==2))
        reversed_out,_=assemble_masks([b,a],shape,assembly_mode='object')
        np.testing.assert_array_equal(out,reversed_out)

    def test_pre_nms_no_longer_discards_candidate_before_ownership(self):
        shape=(2,6,10);a=np.zeros(shape,bool);b=np.zeros(shape,bool);a[:,:,:6]=1;b[:,:,4:]=1
        ca=mask_candidate(np.full(shape,.9),a,(0,0,0));cb=mask_candidate(np.full(shape,.8),b,(0,0,0))
        out,s=assemble_masks([ca,cb],shape,dedup_iou=.1,assembly_mode='object',min_remaining_fraction=.5)
        self.assertEqual(s['after_mask_nms'],2);self.assertEqual(s['written'],2)
        legacy,ls=assemble_masks([ca,cb],shape,dedup_iou=.1)
        self.assertEqual(ls['written'],1)

    def test_gt_inputs_are_interior_points_and_bbox_sigmas(self):
        gt=np.zeros((9,17,19),np.int32);gt[1:8,2:14,2:15]=99;gt[2:7,4:12,4:13]=0
        labels,centers,sigmas=gt_prompt_inputs(gt)
        self.assertEqual(np.count_nonzero(labels),1)
        self.assertEqual(gt[tuple(centers[0])],99)
        np.testing.assert_allclose(sigmas[0],[7/4,12/4,13/4])
        z,c,s=gt_prompt_inputs(np.zeros_like(gt));self.assertEqual(c.shape,(0,3));self.assertEqual(s,[])

    def test_ablation_reuses_candidates_without_mutation(self):
        gt=np.ones((2,8,8),np.int32);c=mask_candidate(np.full(gt.shape,.8),gt>0,(0,0,0));c['predicted_quality']=.2
        source=dict(nucleus=[c],cytoplasm=[c],evidence=[(c,c)])
        before=c['score'].copy()
        result=compare_assembly(config(),gt.shape,source,gt,{'nucleus':gt,'cytoplasm':gt},{1:1})
        self.assertEqual(set(result),{'voxel','object','object_quality'})
        self.assertTrue(all(v['jtpr']==1 for v in result.values()))
        np.testing.assert_array_equal(c['score'],before)

    def test_gt_prompt_width_reaches_runtime_and_gt_shape_is_not_output(self):
        cfg=config(nucleus_grid=(1,1,1),cytoplasm_grid=(1,1,1),mask_assembly_cytoplasm='object')
        gt=np.zeros((8,32,32),np.int32);gt[1:7,4:28,6:26]=1
        seen=[]
        def refine(inputs):
            prompt=inputs[-1];seen.append(prompt.numpy())
            # Deliberately predict nothing: oracle cannot copy GT into output.
            z=tf.zeros(tf.concat([tf.shape(prompt)[:-1],[3]],0));return z,z
        with patch.object(inference,'_refinement_runners',return_value=(lambda x:[x]*7,refine)), \
             patch.object(inference,'_cytoplasm_refinement_runner',return_value=lambda x:refine(x)[1]):
            report=evaluate_gt_prompts(SimpleNamespace(),cfg,np.zeros(gt.shape+(1,),np.float32),
                                       {'nucleus':gt,'cytoplasm':gt},{1:1})
        self.assertEqual(report['jtpr_oracle'],0)
        self.assertEqual(report['stage_audit']['cytoplasm']['stages']['final']['candidates'],0)
        _,coords,sigmas=gt_prompt_inputs(gt);center=coords[0];sigma=sigmas[0]
        ax=np.indices(gt.shape);expected=np.exp(-.5*sum(((ax[a]-center[a])/sigma[a])**2 for a in range(3)))
        np.testing.assert_allclose(seen[0][0,...,0],expected,atol=1e-6)


if __name__=='__main__':unittest.main()
