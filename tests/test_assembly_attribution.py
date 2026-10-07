import unittest
import numpy as np
from cytonucs_postprocess import mask_candidate, assemble_masks
from cytonucs_refinement_audit import RefinementStageAudit


class AssemblyAttributionTests(unittest.TestCase):
    def c(self, shape, start, stop, quality):
        mask=np.zeros(shape,bool); mask[...,start:stop]=1
        c=mask_candidate(np.full(shape,.9),mask,np.zeros(len(shape),int))
        c['predicted_quality']=quality
        return c

    def test_clipped_neighbour_does_not_become_nested_duplicate(self):
        # A takes 4 columns of B; remaining B=[8,12) is inside C=[8,16).
        # Original B=[4,12) and C overlap by only 50%, so C is a neighbour.
        shape=(3,5,16)
        candidates=[self.c(shape,0,8,.9),self.c(shape,4,12,.8),self.c(shape,8,16,.7)]
        out,stats=assemble_masks(candidates,shape,assembly_mode='object_quality_containment',
                                 containment_threshold=.95)
        self.assertEqual(stats['written'],3)
        self.assertEqual(stats['rejected_nested'],0)
        self.assertTrue(np.all(out[...,12:]==3))

    def test_observer_records_wrong_nested_winner_without_changing_masks(self):
        shape=(3,5,12);gt=np.ones(shape,np.int32)
        candidates=[self.c(shape,0,12,.7),self.c(shape,4,7,.9)]
        audit=RefinementStageAudit(gt)
        kwargs=dict(assembly_mode='object_quality_containment',containment_threshold=.95)
        expected,stats=assemble_masks(candidates,shape,**kwargs)
        actual,observed=assemble_masks(candidates,shape,observer=audit,**kwargs)
        np.testing.assert_array_equal(actual,expected);self.assertEqual(stats,observed)
        report=audit.report();lost=report['assembly_lost_gt'][0]
        self.assertEqual(lost['gt_id'],'1')
        d=lost['good_candidates'][0]
        self.assertEqual(d['before_iou'],1.)
        self.assertEqual(d['reason'],'nested')
        self.assertEqual(d['blockers'][0]['written_iou_with_lost_gt'],.25)
        self.assertGreater(d['blockers'][0]['quality'],d['quality'])

    def test_connected_remainder_and_written_damage_are_distinguished(self):
        shape=(3,5,12);gt=np.ones(shape,np.int32)
        stripe=self.c(shape,4,8,.9);whole=self.c(shape,0,12,.7)
        audit=RefinementStageAudit(gt)
        assemble_masks([whole,stripe],shape,assembly_mode='object_quality',observer=audit)
        d=audit.report()['assembly_lost_gt'][0]['good_candidates'][0]
        self.assertEqual(d['reason'],'insufficient_connected_remainder')
        trace=audit.assembly_decisions[1]
        self.assertEqual(trace['free_voxels'],120)
        self.assertEqual(trace['connected_voxels'],60)
        # Accept the damaged remainder under the existing lower fraction option.
        audit=RefinementStageAudit(gt)
        assemble_masks([whole,stripe],shape,assembly_mode='object_quality',observer=audit,
                       min_remaining_fraction=.3)
        d=audit.report()['assembly_lost_gt'][0]['good_candidates'][0]
        self.assertEqual(d['reason'],'written')
        self.assertAlmostEqual(d['written_iou'],1/3)

    def test_empty_and_sparse_gt_and_fully_occupied(self):
        shape=(2,4,8);gt=np.full(shape,9001,np.int32)
        a=self.c(shape,0,8,.9);b=self.c(shape,0,8,.8)
        audit=RefinementStageAudit(gt)
        assemble_masks([a,b],shape,assembly_mode='object_quality',observer=audit)
        self.assertEqual(audit.assembly_decisions[1]['reason'],'occupied')
        self.assertEqual(audit.assembly_decisions[0]['best_gt'],'9001')
        self.assertEqual(audit.report()['assembly_lost_gt'],[])
        empty=RefinementStageAudit(np.zeros(shape,np.int32))
        assemble_masks([],shape,assembly_mode='object_quality',observer=empty)
        self.assertEqual(empty.report()['assembly_decisions'],[])


if __name__=='__main__':unittest.main()
