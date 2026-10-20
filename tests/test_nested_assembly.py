import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from cytonucs_config import CytoNucsConfig
from cytonucs_postprocess import mask_candidate, assemble_masks
from audit.refinement_ablation import compare_nested_assembly, evaluate_gt_prompts
import cytonucs_inference as inference


class NestedAssemblyTests(unittest.TestCase):
    def candidate(self, mask, quality, probability=.9):
        c = mask_candidate(np.full(mask.shape, probability), mask, np.zeros(mask.ndim, int))
        c['predicted_quality'] = quality
        return c

    def assemble(self, candidates, shape):
        return assemble_masks(candidates, shape, assembly_mode='object_quality_containment',
                              containment_threshold=.95)

    def test_nested_both_rank_directions_and_input_order(self):
        shape = (5, 12, 12)
        full = np.ones(shape, bool)
        small = np.zeros(shape, bool); small[1:4, 4:8, 4:8] = True
        for big_quality, small_quality in ((.9, .6), (.6, .9)):
            a, b = self.candidate(full, big_quality), self.candidate(small, small_quality, .99)
            out, stats = self.assemble([a, b], shape)
            reverse, _ = self.assemble([b, a], shape)
            np.testing.assert_array_equal(out, reverse)
            self.assertEqual(stats['written'], 1)
            self.assertEqual(stats['rejected_nested'], 1)
            np.testing.assert_array_equal(out > 0, full if big_quality > small_quality else small)
            # A bad rank remains a bad mask choice, but cannot create an outer shell ID.
            if small_quality > big_quality:
                old, old_stats = assemble_masks([a, b], shape, assembly_mode='object_quality')
                self.assertEqual(old_stats['written'], 2)

    def test_neighbours_and_separate_z_fragments_not_merged(self):
        shape=(6, 10, 20)
        a=np.zeros(shape,bool); a[:,:,:11]=1
        b=np.zeros(shape,bool); b[:,:,9:]=1
        out, stats=self.assemble([self.candidate(a,.9),self.candidate(b,.8)],shape)
        self.assertEqual(stats['written'],2)
        self.assertEqual(stats['rejected_nested'],0)
        a[:]=0; b[:]=0; a[:2]=1; b[4:]=1
        self.assertEqual(self.assemble([self.candidate(a,.9),self.candidate(b,.8)],shape)[1]['written'],2)

    def test_rejected_candidate_cannot_suppress_other_objects(self):
        shape=(3,10,20)
        a=np.zeros(shape,bool); a[:,:,:10]=1
        b=np.ones(shape,bool)
        c=np.zeros(shape,bool); c[:,:,12:]=1
        out, stats=self.assemble([self.candidate(a,.9), self.candidate(b,.8), self.candidate(c,.7)],shape)
        self.assertEqual(stats['written'],2)
        self.assertTrue(np.all(out[:,:,12:]>0))
        self.assertEqual(stats['rejected_nested'],1)

    def test_empty_invalid_and_equal_quality_prefer_whole(self):
        shape=(3,8,8)
        self.assertEqual(self.assemble([],shape)[1]['rejected_nested'],0)
        a=np.ones(shape,bool); b=np.zeros(shape,bool); b[:,2:4,2:4]=1
        out, stats=self.assemble([self.candidate(b,.8),self.candidate(a,.8)],shape)
        self.assertTrue(np.all(out==1))
        with self.assertRaises(ValueError):
            config(mask_assembly_nucleus='object_quality_containment',mask_containment_threshold=0)

    def test_preset_json_contract(self):
        raw=json.loads((ROOT/'config_cytonucs_v2_8.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            raw['training']['resume_from_checkpoint']=None
            raw['output']['basedir']=tmp
            path=Path(tmp)/'config.json';path.write_text(json.dumps(raw))
            cfg=CytoNucsConfig.from_json(path)
        for section, settings in cfg.FIXES_SETTINGS.items():
            for key in settings:
                value=raw[section][key]
                effective=getattr(cfg,key)
                if isinstance(effective,tuple): effective=list(effective)
                self.assertEqual(effective,value,key)
        self.assertEqual(cfg.mask_assembly_nucleus,'object_quality_containment')
        self.assertEqual(cfg.mask_assembly_cytoplasm,'object_quality_containment')

    def test_both_head_ablation_uses_same_candidates_without_mutation(self):
        shape=(3,10,10); full=np.ones(shape,bool); small=np.zeros(shape,bool);small[:,4:6,4:6]=1
        a=self.candidate(full,.9,.8);b=self.candidate(small,.6,.99)
        source=dict(nucleus=[a,b],cytoplasm=[a,b],evidence=[(a,a),(b,b)])
        before=[c['score'].copy() for c in source['nucleus']]
        result=compare_nested_assembly(config(mask_containment_threshold=.95),shape,source,
                                      {'nucleus':full.astype(int),'cytoplasm':full.astype(int)},{1:1})
        for head in ('nucleus','cytoplasm'):
            self.assertEqual(result['nested']['assembly'][head]['written'],1)
            self.assertEqual(result['nested']['assembly'][head]['rejected_nested'],1)
        for c,old in zip(source['nucleus'],before):np.testing.assert_array_equal(c['score'],old)

    def test_inference_passes_containment_to_both_heads(self):
        cfg=config(nucleus_grid=(1,1,1),cytoplasm_grid=(1,1,1),
                   mask_assembly_nucleus='object_quality_containment',
                   mask_assembly_cytoplasm='object_quality_containment',mask_containment_threshold=.95)
        gt=np.zeros((8,32,32),np.int32);gt[1:7,4:28,6:26]=1
        def refine(inputs):
            z=tf.zeros(tf.concat([tf.shape(inputs[-1])[:-1],[3]],0))
            return z,z
        with patch.object(inference,'_refinement_runners',return_value=(lambda x:[x]*7,refine)), \
             patch.object(inference,'_cytoplasm_refinement_runner',return_value=lambda x:refine(x)[1]):
            report=evaluate_gt_prompts(SimpleNamespace(),cfg,np.zeros(gt.shape+(1,),np.float32),
                                       {'nucleus':gt,'cytoplasm':gt},{1:1})
        for head in ('nucleus','cytoplasm'):
            self.assertEqual(report['refinement'][head]['assembly_mode'],'object_quality_containment')
            self.assertEqual(report['refinement'][head]['rejected_nested'],0)


if __name__ == '__main__':
    unittest.main()
