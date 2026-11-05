import unittest
import torch
from rltrain.batch import PolicyVersions,tensorize,response_hidden


def records():
    return [
        {'prompt_token_ids':[1,2,3],'response_token_ids':[4,5],'reward':0.,'group_id':0},
        {'prompt_token_ids':[1,2,3],'response_token_ids':[6],'reward':1.,'group_id':0},
        {'prompt_token_ids':[7],'response_token_ids':[8,9,10,11],'reward':2.,'group_id':1},
        {'prompt_token_ids':[7],'response_token_ids':[8,12],'reward':0.,'group_id':1},
    ]


class BatchTests(unittest.TestCase):
    def test_causal_shift_predicts_first_response_from_last_prompt(self):
        b=tensorize(records(),0)
        h=torch.arange(4*5).view(4,5,1).float()
        selected=response_hidden(h,b['positions'])
        self.assertEqual(float(selected[0,0,0]),2.)
        self.assertEqual(float(selected[2,0,0]),10.)
        self.assertEqual(float(selected[0,1,0]),3.)

    def test_padded_out_of_range_positions_clamped_and_zero_weighted(self):
        b=tensorize(records(),0)
        self.assertLess(int(b['positions'].max()),b['input_ids'].shape[1])
        self.assertEqual(float(b['weights'][1,1:].sum()),0.)
        self.assertEqual(int(b['attention_mask'][1].sum()),4)

    def test_full_batch_sequence_mean_preserved(self):
        b=tensorize(records(),0)
        torch.testing.assert_close(b['weights'].sum(-1),torch.full((4,),.25))

    def test_matching_old_rollout_version(self):
        PolicyVersions(0,0,0).validate(3,3)
        with self.assertRaises(ValueError):PolicyVersions(0,1,0).validate(1,3)

    def test_future_and_over_stale_rejected(self):
        with self.assertRaises(ValueError):PolicyVersions(2,2,0).validate(1,3)
        with self.assertRaises(ValueError):PolicyVersions(0,0,0).validate(4,3)

    def test_empty_response_rejected(self):
        r=records();r[0]['response_token_ids']=[]
        with self.assertRaises(ValueError):tensorize(r,0)

    def test_frozen_epoch_validity_not_reference_current_equality(self):
        PolicyVersions(4,4,0).validate(5,2)
        with self.assertRaises(ValueError):PolicyVersions(-1,-1,0).validate(1,2)


if __name__=='__main__':unittest.main()
