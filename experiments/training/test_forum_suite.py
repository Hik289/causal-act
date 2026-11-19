import unittest
import torch
from torch import nn
from build_forum_quads import split_forums, set_subscription
from forum_expert_iteration import RoleLoRA, causal_report, role_parameters


class ProtocolTests(unittest.TestCase):
    def test_subscription_checks_persisted_state_when_ajax_dom_is_stale(self):
        from types import SimpleNamespace
        class Page:
            backend=False
            visible=False
            reloads=0
            first=None
            def __init__(self):self.first=self
            def evaluate(self, script):return {"subscribed":self.visible}
            def locator(self, selector):return self
            def click(self):self.backend=not self.backend
            def reload(self, **kwargs):self.visible=self.backend;self.reloads+=1
            def expect_response(self, predicate, **kwargs):return self
            def __enter__(self):return SimpleNamespace(value=SimpleNamespace(status=200))
            def __exit__(self, *args):pass
        page=Page()
        self.assertTrue(set_subscription(page,True)["subscribed"])
        self.assertEqual(page.reloads,1)
        self.assertFalse(set_subscription(page,False)["subscribed"])

    def test_splits_preserve_groups_and_test_size(self):
        p=split_forums([f"forum{i}" for i in range(75)],20261119)
        self.assertEqual(len(p["test"])*4,100)
        self.assertFalse(set(p["train"])&set(p["test"]))
        self.assertFalse(set(p["val"])&set(p["test"]))
        self.assertEqual(p,split_forums([f"forum{i}" for i in reversed(range(75))],20261119))

    def test_insufficient_groups_cannot_silently_shrink_test(self):
        with self.assertRaises(ValueError):split_forums(["one","two"],42)

    def test_lora_initializes_as_exact_frozen_backbone(self):
        base=nn.Linear(5,3);adapt=RoleLoRA(base,rank=2);x=torch.randn(4,5)
        self.assertTrue(torch.equal(adapt(x),base(x)))
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
        adapt(x).sum().backward()
        self.assertIsNotNone(adapt.up["A"].grad)
        self.assertIsNone(adapt.up["V"].grad)
        self.assertIsNone(adapt.up["I"].grad)

    def test_negative_pair_is_kept_and_missing_evidence_is_not_dropped(self):
        row={"id":"x","size":[100,100],"variants":{
            "original":{"action":0,"bbox":[1,1,3,3]},"target":{"action":1},"unrelated":{"action":0}}}
        pred={"x":{"original":{"action":0},"target":{"action":0},"unrelated":{"action":0}}}
        r=causal_report([row],pred)
        self.assertEqual(r["metrics"]["PairCausalAcc"]["denominator"],1)
        self.assertEqual(r["metrics"]["PairCausalAcc"]["numerator"],0)
        self.assertEqual(r["metrics"]["EvidenceRecall"]["denominator"],1)
        self.assertEqual(r["metrics"]["EvidenceRecall"]["numerator"],0)
        self.assertEqual(r["metrics"]["UnsafeAction"]["numerator"],1)
        self.assertIsNone(r["area_buckets"][">5%"]["PairCausalAcc"]["value"])

    def test_pair_causal_accuracy_requires_unrelated_correctness(self):
        row={"id":"x","size":[100,100],"variants":{
            "original":{"action":0,"bbox":[1,1,3,3]},"target":{"action":1},"unrelated":{"action":0}}}
        pred={"x":{"original":{"action":0},"target":{"action":1},"unrelated":{"action":1}}}
        metrics=causal_report([row],pred)["metrics"]
        self.assertEqual(metrics["TargetPairAcc"]["numerator"],1)
        self.assertEqual(metrics["PairCausalAcc"]["numerator"],0)


if __name__=="__main__":unittest.main()
