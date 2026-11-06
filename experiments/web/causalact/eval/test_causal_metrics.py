"""H0.anchor_2 自检: 因果评估代码在 toy ground-truth 三/四元组上输出已知手算值。

每个 assert 的期望值均由注释给出手算过程。全过 = anchor_2 评估代码正确。
运行: python test_causal_metrics.py
"""
from causal_metrics import (
    Action, CausalQuad, CausalEvaluator,
    semantic_equal, point_in_bbox, iou, apply_transform,
)


def approx(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol


def test_geometry():
    # IoU: 两个 10x10 框, 重叠 5x5=25, union=100+100-25=175 → 25/175
    assert approx(iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175), "iou overlap"
    # 完全重叠 → 1.0
    assert approx(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0), "iou identical"
    # 不相交 → 0.0
    assert approx(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0), "iou disjoint"
    # point in bbox
    assert point_in_bbox((5, 5), (0, 0, 10, 10)) is True
    assert point_in_bbox((15, 5), (0, 0, 10, 10)) is False
    # transform: 平移 (dx=100,dy=40), 缩放 1 → (10,20)->(110,60)
    assert apply_transform((10, 20), {"dx": 100, "dy": 40}) == (110, 60)
    # 缩放 sx=2,sy=0.5 → (10,20)->(20,10)
    assert apply_transform((10, 20), {"sx": 2.0, "sy": 0.5}) == (20.0, 10.0)
    print("  [geometry] PASS")


def test_semantic_equal():
    a = Action("click", "submit_button", (100, 200))
    b = Action("click", "submit_button", (105, 210))  # 坐标不同但语义同
    c = Action("click", "cancel_button", (100, 200))
    assert semantic_equal(a, b) is True, "same semantic diff coords"
    assert semantic_equal(a, c) is False, "diff target"
    print("  [semantic_equal] PASS")


def build_toy():
    """构造 4 条 quad, 手工指定 GT 与 pred, 使每个指标有确定期望值。"""
    quads = [
        # q1: targeted 改变正确动作 (click submit -> stop), 无关不变, transform 位移
        CausalQuad(
            task_id="q1",
            gt_original=Action("click", "submit", (100, 100), (90, 90, 110, 110)),
            gt_targeted=Action("stop", None),            # 按钮 disabled → 正确动作变 stop
            gt_unrelated=Action("click", "submit", (100, 100)),  # 无关扰动, 动作不变
            gt_transform=Action("click", "submit", (200, 140)),  # τ: dx=100,dy=40
            tau={"dx": 100, "dy": 40},
            targeted_changes=True,
        ),
        # q2: targeted 改变 (vote_up -> vote_down 语义变)
        CausalQuad(
            task_id="q2",
            gt_original=Action("click", "vote_up", (50, 50), (40, 40, 60, 60)),
            gt_targeted=Action("click", "vote_down", (50, 70)),
            gt_unrelated=Action("click", "vote_up", (50, 50)),
            gt_transform=Action("click", "vote_up", (50, 50)),  # τ=identity
            tau={},
            targeted_changes=True,
        ),
        # q3: targeted 不改变正确动作 (多因素 target, 动作集合不变) → 不计入 PairCausal
        CausalQuad(
            task_id="q3",
            gt_original=Action("click", "nav_home", (10, 10), (0, 0, 20, 20)),
            gt_targeted=Action("click", "nav_home", (10, 10)),  # 正确动作不变
            gt_unrelated=Action("click", "nav_home", (10, 10)),
            gt_transform=None,
            tau={},
            targeted_changes=False,
        ),
        # q4: 只有 original + unrelated (无 targeted/transform)
        CausalQuad(
            task_id="q4",
            gt_original=Action("type", "search_box", (300, 30), (280, 20, 320, 40)),
            gt_unrelated=Action("type", "search_box", (300, 30)),
            tau={},
        ),
    ]
    return quads


def test_all_correct():
    """preds 全部等于 GT (且 original 的 evidence 精确命中) → 多数指标应为 1.0。"""
    quads = build_toy()
    ev = CausalEvaluator(coord_tol=5.0, evidence_iou_th=0.5)

    preds_original = {
        "q1": Action("click", "submit", (100, 100), (90, 90, 110, 110)),
        "q2": Action("click", "vote_up", (50, 50), (40, 40, 60, 60)),
        "q3": Action("click", "nav_home", (10, 10), (0, 0, 20, 20)),
        "q4": Action("type", "search_box", (300, 30), (280, 20, 320, 40)),
    }
    preds_targeted = {
        "q1": Action("stop", None),
        "q2": Action("click", "vote_down", (50, 70)),
        "q3": Action("click", "nav_home", (10, 10)),
    }
    preds_unrelated = {
        "q1": Action("click", "submit", (100, 100)),
        "q2": Action("click", "vote_up", (50, 50)),
        "q3": Action("click", "nav_home", (10, 10)),
        "q4": Action("type", "search_box", (300, 30)),
    }
    preds_transform = {
        "q1": Action("click", "submit", (200, 140)),  # = τ(100,100)=(200,140) ✓
        "q2": Action("click", "vote_up", (50, 50)),    # τ=id ✓
    }
    r = ev.evaluate(quads, preds_original, preds_targeted, preds_unrelated, preds_transform)

    # PairCausal: q1,q2 改变且都对且 pred 改了 → 2/2; q3 不计入(targeted_changes=False)
    assert r["_pair"] == [2, 2], f"pair {r['_pair']}"
    assert approx(r["PairCausalAcc"], 1.0)
    # TargetedAcc: q1,q2,q3 all correct → 3/3
    assert r["_tgt"] == [3, 3], f"tgt {r['_tgt']}"
    assert approx(r["TargetedAcc"], 1.0)
    # UnrelatedAcc: q1,q2,q3,q4 all correct → 4/4
    assert r["_unrel"] == [4, 4], f"unrel {r['_unrel']}"
    assert approx(r["UnrelatedAcc"], 1.0)
    # UnrelatedConsist: pred(unrel)==pred(orig) for q1-q4 → 4/4
    assert r["_unrel_consist"] == [4, 4], f"consist {r['_unrel_consist']}"
    assert approx(r["UnrelatedConsist"], 1.0)
    # EquivariantAcc: q1,q2 → 2/2
    assert r["_eqv"] == [2, 2], f"eqv {r['_eqv']}"
    assert approx(r["EquivariantAcc"], 1.0)
    # EvidenceRecall: q1-q4 evidence exact → 4/4
    assert r["_ev"] == [4, 4], f"ev {r['_ev']}"
    assert approx(r["EvidenceRecall"], 1.0)
    print("  [all_correct] PASS", {k: v for k, v in r.items() if not k.startswith("_")})


def test_partial_errors():
    """注入已知错误, 验证分数按手算下降。"""
    quads = build_toy()
    ev = CausalEvaluator(coord_tol=5.0, evidence_iou_th=0.5)

    preds_original = {
        "q1": Action("click", "submit", (100, 100), (90, 90, 110, 110)),
        "q2": Action("click", "vote_up", (50, 50), (0, 0, 5, 5)),    # evidence 不命中 (iou 低)
        "q3": Action("click", "nav_home", (10, 10), (0, 0, 20, 20)),
        "q4": Action("type", "search_box", (300, 30), (280, 20, 320, 40)),
    }
    preds_targeted = {
        "q1": Action("stop", None),                       # 对
        "q2": Action("click", "vote_up", (50, 50)),        # 错: 应 vote_down (没改)
        "q3": Action("click", "nav_home", (10, 10)),       # 对
    }
    preds_unrelated = {
        "q1": Action("click", "cancel", (100, 100)),       # 错: 动作被无关扰动带偏
        "q2": Action("click", "vote_up", (50, 50)),        # 对
        "q3": Action("click", "nav_home", (10, 10)),
        "q4": Action("type", "search_box", (300, 30)),
    }
    preds_transform = {
        "q1": Action("click", "submit", (999, 999)),       # 坐标错 (不等变)
        "q2": Action("click", "vote_up", (50, 50)),        # 对
    }
    r = ev.evaluate(quads, preds_original, preds_targeted, preds_unrelated, preds_transform)

    # PairCausal: q1 对(orig对+tgt对+改了)=1; q2 tgt错 → 不算对. pair_n=2(q1,q2), ok=1
    assert r["_pair"] == [1, 2], f"pair {r['_pair']}"
    assert approx(r["PairCausalAcc"], 0.5)
    # TargetedAcc: q1对,q2错,q3对 → 2/3
    assert r["_tgt"] == [2, 3], f"tgt {r['_tgt']}"
    assert approx(r["TargetedAcc"], 2 / 3)
    # UnrelatedAcc: q1错,q2对,q3对,q4对 → 3/4
    assert r["_unrel"] == [3, 4], f"unrel {r['_unrel']}"
    assert approx(r["UnrelatedAcc"], 0.75)
    # UnrelatedConsist: q1 pred(unrel)=cancel != pred(orig)=submit → 错; q2,q3,q4 一致 → 3/4
    assert r["_unrel_consist"] == [3, 4], f"consist {r['_unrel_consist']}"
    assert approx(r["UnrelatedConsist"], 0.75)
    # EquivariantAcc: q1 坐标错, q2 对 → 1/2
    assert r["_eqv"] == [1, 2], f"eqv {r['_eqv']}"
    assert approx(r["EquivariantAcc"], 0.5)
    # EvidenceRecall: q1对,q2错(iou(90..110 vs 0..5)=0),q3对,q4对 → 3/4
    assert r["_ev"] == [3, 4], f"ev {r['_ev']}"
    assert approx(r["EvidenceRecall"], 0.75)
    print("  [partial_errors] PASS", {k: v for k, v in r.items() if not k.startswith("_")})


def test_no_flip_when_unchanged():
    """README §2.3 红线: targeted 未改变正确动作时, 不施加翻转/不计入 PairCausal。"""
    quads = [CausalQuad(
        task_id="x",
        gt_original=Action("click", "home", (1, 1)),
        gt_targeted=Action("click", "home", (1, 1)),   # 未变
        targeted_changes=False,
    )]
    ev = CausalEvaluator()
    preds_original = {"x": Action("click", "home", (1, 1))}
    preds_targeted = {"x": Action("click", "home", (1, 1))}
    r = ev.evaluate(quads, preds_original, preds_targeted)
    # pair_n 必须为 0 (未改变不计入)
    assert r["_pair"] == [0, 0], f"pair {r['_pair']}"
    assert r["PairCausalAcc"] is None
    # TargetedAcc 仍统计 (动作应保持正确) → 1/1
    assert r["_tgt"] == [1, 1]
    print("  [no_flip_when_unchanged] PASS")


if __name__ == "__main__":
    print("=== H0.anchor_2 causal eval self-test ===")
    test_geometry()
    test_semantic_equal()
    test_all_correct()
    test_partial_errors()
    test_no_flip_when_unchanged()
    print("ALL ASSERTIONS PASSED ✓")
