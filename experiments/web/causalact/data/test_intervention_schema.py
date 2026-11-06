"""H0.anchor_3 自检: 干预数据 loader 的 shape/range/统计量/缺失率验证。

验证两方面:
  1. 合法样本通过 validate 且统计量=手算值
  2. 各类非法样本 (坐标越界/bbox乱序/confidence越界/targeted矛盾/unrelated偏移)
     被 validate 正确 raise ValidationError
运行: python test_intervention_schema.py
"""
from causalact_runtime import location
import json, tempfile, os
from intervention_schema import (
    InterventionSample, EditSpec, ActionRec,
    sample_from_dict, load_jsonl, dump_jsonl,
    validate_sample, validate_dataset, ValidationError,
    VIEWPORT_W, VIEWPORT_H,
)


def _base_valid_dict(tid="t1"):
    return {
        "task_id": tid, "site": "reddit", "intent": "subscribe to the food forum",
        "screenshot_path": location('CAUSALACT_RUN_DIR', f'scratch/{tid}.png'), "image_size": [1280, 2048],
        "base_action": {"action_type": "click", "target_id": "subscribe",
                        "coordinates": [916, 129], "evidence_bbox": [860, 110, 972, 150], "confidence": 0.8},
        "original_action": {"action_type": "click", "target_id": "subscribe",
                            "coordinates": [916, 129], "evidence_bbox": [860, 110, 972, 150], "confidence": 1.0},
        "target_edit": {"edit_type": "vote_state", "category": "targeted",
                        "object_id": "subscribe", "attribute": "subscribed", "value": True,
                        "region_bbox": [860, 110, 972, 150], "backend_mutated": True,
                        "reversible_action": "unsubscribe"},
        "target_action": {"action_type": "click", "target_id": "unsubscribe", "coordinates": [916, 129]},
        "target_screenshot_path": location('CAUSALACT_RUN_DIR', f'scratch/{tid}_tgt.png'),
        "targeted_changes": True,
        "unrelated_edit": {"edit_type": "recolor", "category": "unrelated",
                           "attribute": "backgroundColor", "value": "#ffcc00"},
        "unrelated_action": {"action_type": "click", "target_id": "subscribe", "coordinates": [916, 129]},
        "unrelated_screenshot_path": location('CAUSALACT_RUN_DIR', f'scratch/{tid}_unrel.png'),
        "layout_edit": {"edit_type": "move", "category": "transform",
                        "region_bbox": [860, 110, 972, 150], "tau": {"dx": 120, "dy": 40}},
        "mapped_action": {"action_type": "click", "target_id": "subscribe", "coordinates": [1036, 169]},
        "layout_screenshot_path": location('CAUSALACT_RUN_DIR', f'scratch/{tid}_tf.png'),
        "oracle_metadata": {"render_valid": True},
        "data_split": "train",
    }


def test_valid_sample():
    s = sample_from_dict(_base_valid_dict())
    warns = validate_sample(s)
    assert warns == [], f"unexpected warns: {warns}"
    print("  [valid_sample] PASS")


def test_roundtrip_jsonl():
    s = sample_from_dict(_base_valid_dict("rt1"))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.jsonl")
        dump_jsonl([s], p)
        back = load_jsonl(p)
    assert len(back) == 1
    assert back[0].task_id == "rt1"
    assert back[0].target_edit.backend_mutated is True
    assert back[0].mapped_action.coordinates == [1036, 169]
    print("  [roundtrip_jsonl] PASS")


def _expect_raise(d, msg):
    s = sample_from_dict(d)
    try:
        validate_sample(s)
    except ValidationError:
        return
    raise AssertionError(f"expected ValidationError for: {msg}")


def test_invalid_cases():
    # 坐标越界 (x 远超宽)
    d = _base_valid_dict("c1"); d["original_action"]["coordinates"] = [99999, 100]
    _expect_raise(d, "coord out of range")
    # bbox 乱序 (x1<x0)
    d = _base_valid_dict("c2"); d["original_action"]["evidence_bbox"] = [500, 100, 100, 150]
    _expect_raise(d, "bbox not ordered")
    # confidence 越界
    d = _base_valid_dict("c3"); d["base_action"]["confidence"] = 1.5
    _expect_raise(d, "confidence > 1")
    # targeted_changes=True 但 target_action 语义==original (矛盾)
    d = _base_valid_dict("c4")
    d["target_action"] = {"action_type": "click", "target_id": "subscribe", "coordinates": [916, 129]}
    _expect_raise(d, "targeted_changes contradiction")
    # unrelated_action 语义 != original (unrelated 应不变)
    d = _base_valid_dict("c5")
    d["unrelated_action"] = {"action_type": "click", "target_id": "cancel", "coordinates": [10, 10]}
    _expect_raise(d, "unrelated drifted")
    # image_size 非法
    d = _base_valid_dict("c6"); d["image_size"] = [0, 100]
    _expect_raise(d, "bad image_size")
    print("  [invalid_cases] PASS (6/6 raised)")


def test_dom_only_warning():
    # targeted edit backend_mutated=False → 应产生 warning (非 raise)
    d = _base_valid_dict("w1"); d["target_edit"]["backend_mutated"] = False
    s = sample_from_dict(d)
    warns = validate_sample(s)
    assert any("DOM-only" in w for w in warns), f"expected DOM-only warn, got {warns}"
    print("  [dom_only_warning] PASS")


def test_dataset_stats():
    # 构造 4 样本: 全 valid; 2 有 target(changes), 1 无 target, 统计手算
    ds = []
    for i in range(4):
        d = _base_valid_dict(f"d{i}")
        if i == 3:
            # 去掉 target/unrelated/layout → missing
            d["target_edit"] = None; d["target_action"] = None; d["targeted_changes"] = False
            d["unrelated_edit"] = None; d["unrelated_action"] = None
            d["layout_edit"] = None; d["mapped_action"] = None
        ds.append(sample_from_dict(d))
    r = validate_dataset(ds)
    assert r["n"] == 4
    # missing target: 1/4
    assert abs(r["missing_rate"]["target"] - 0.25) < 1e-9, r["missing_rate"]
    assert abs(r["missing_rate"]["unrelated"] - 0.25) < 1e-9
    assert abs(r["missing_rate"]["layout"] - 0.25) < 1e-9
    # targeted_changes: d0,d1,d2 True → 3/4
    assert abs(r["targeted_changes_rate"] - 0.75) < 1e-9, r["targeted_changes_rate"]
    # backend_mutated: d0,d1,d2 → 3
    assert r["backend_mutated_count"] == 3, r["backend_mutated_count"]
    assert r["splits"] == {"train": 4}
    assert r["n_warnings"] == 0, r["warnings_sample"]
    print("  [dataset_stats] PASS", {k: r[k] for k in ("missing_rate", "targeted_changes_rate", "backend_mutated_count")})


if __name__ == "__main__":
    print("=== H0.anchor_3 intervention loader self-test ===")
    test_valid_sample()
    test_roundtrip_jsonl()
    test_invalid_cases()
    test_dom_only_warning()
    test_dataset_stats()
    print("ALL ASSERTIONS PASSED ✓")
