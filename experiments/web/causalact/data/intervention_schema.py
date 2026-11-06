"""causalact (CausalAct) — 干预样本数据 schema + loader + 验证 (H0.anchor_3)

README §12.1 训练样本:
  (task, state, observation, base_action, original_action, target_edit,
   target_action, unrelated_edit, unrelated_action, layout_edit, mapped_action,
   oracle_metadata)

本模块定义:
  - InterventionSample 的 JSON schema (dataclass)
  - 从 JSONL 加载
  - validate(): shape / range / 统计量 / 缺失率 检查 (对齐 §12 spec)

编辑 mask: 归一化到 [0,1] 的 (H, W) 浮点数组 (或 bbox 列表)。
坐标: 像素坐标, 落在 viewport (1280×720 / 或截图尺寸) 内。
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import json


# viewport / 截图尺寸 (VWA run 用 1280×720 viewport; 截图可能 full_page 更高)
VIEWPORT_W = 1280
VIEWPORT_H = 720
# SoM 截图常见高度上限 (full page), 坐标校验用宽松上界
MAX_IMG_H = 8192
MAX_IMG_W = 2048


@dataclass
class EditSpec:
    """一个干预编辑 (target / unrelated / layout)。"""
    edit_type: str                  # disable / recolor / move / text_change / vote_state ...
    category: str                   # 'targeted' | 'unrelated' | 'transform'
    object_id: Optional[str] = None
    attribute: Optional[str] = None
    value: Optional[Any] = None
    region_bbox: Optional[list] = None      # [x0,y0,x1,y1] 被编辑区域
    tau: dict = field(default_factory=dict)  # transform 的变换参数 (仅 transform)
    backend_mutated: bool = False            # target: 是否真实改了后端状态
    reversible_action: Optional[str] = None  # 还原用的逆动作 (如 unsubscribe)


@dataclass
class ActionRec:
    action_type: str
    target_id: Optional[str] = None
    coordinates: Optional[list] = None        # [x, y]
    evidence_bbox: Optional[list] = None       # [x0,y0,x1,y1]
    confidence: float = 1.0


@dataclass
class InterventionSample:
    task_id: str
    site: str                                  # 'reddit'
    intent: str
    screenshot_path: str                       # original 截图路径
    image_size: list                           # [W, H] of original screenshot
    base_action: ActionRec                     # gpt-5.6-sol Base 候选动作
    original_action: ActionRec                 # 环境 oracle 正确动作

    target_edit: Optional[EditSpec] = None
    target_action: Optional[ActionRec] = None
    target_screenshot_path: Optional[str] = None
    targeted_changes: bool = False             # 环境核验: 正确动作是否真改变

    unrelated_edit: Optional[EditSpec] = None
    unrelated_action: Optional[ActionRec] = None
    unrelated_screenshot_path: Optional[str] = None

    layout_edit: Optional[EditSpec] = None     # transform/equivariant
    mapped_action: Optional[ActionRec] = None
    layout_screenshot_path: Optional[str] = None

    oracle_metadata: dict = field(default_factory=dict)
    data_split: str = "train"                   # train/val/test; 原场景与编辑版同划分


# ──────────────────────────────────────────────────────────────────────
# 构造 / 序列化
# ──────────────────────────────────────────────────────────────────────
def _mk_action(d: dict | None) -> Optional[ActionRec]:
    if d is None:
        return None
    return ActionRec(**d)


def _mk_edit(d: dict | None) -> Optional[EditSpec]:
    if d is None:
        return None
    return EditSpec(**d)


def sample_from_dict(d: dict) -> InterventionSample:
    return InterventionSample(
        task_id=d["task_id"], site=d["site"], intent=d["intent"],
        screenshot_path=d["screenshot_path"], image_size=d["image_size"],
        base_action=_mk_action(d["base_action"]),
        original_action=_mk_action(d["original_action"]),
        target_edit=_mk_edit(d.get("target_edit")),
        target_action=_mk_action(d.get("target_action")),
        target_screenshot_path=d.get("target_screenshot_path"),
        targeted_changes=d.get("targeted_changes", False),
        unrelated_edit=_mk_edit(d.get("unrelated_edit")),
        unrelated_action=_mk_action(d.get("unrelated_action")),
        unrelated_screenshot_path=d.get("unrelated_screenshot_path"),
        layout_edit=_mk_edit(d.get("layout_edit")),
        mapped_action=_mk_action(d.get("mapped_action")),
        layout_screenshot_path=d.get("layout_screenshot_path"),
        oracle_metadata=d.get("oracle_metadata", {}),
        data_split=d.get("data_split", "train"),
    )


def load_jsonl(path: str) -> list[InterventionSample]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(sample_from_dict(json.loads(line)))
    return out


def dump_jsonl(samples: list[InterventionSample], path: str) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


# ──────────────────────────────────────────────────────────────────────
# 验证器 (H0.anchor_3): shape / range / 统计量 / 缺失率
# ──────────────────────────────────────────────────────────────────────
class ValidationError(Exception):
    pass


def _check_coords(coords, W, H, ctx):
    if coords is None:
        return
    if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
        raise ValidationError(f"{ctx}: coordinates must be [x,y], got {coords}")
    x, y = coords
    if not (0 <= x <= max(W, MAX_IMG_W)) or not (0 <= y <= max(H, MAX_IMG_H)):
        raise ValidationError(f"{ctx}: coords {coords} out of range (W={W},H={H})")


def _check_bbox(bbox, W, H, ctx):
    if bbox is None:
        return
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        raise ValidationError(f"{ctx}: bbox must be [x0,y0,x1,y1], got {bbox}")
    x0, y0, x1, y1 = bbox
    if not (x0 <= x1 and y0 <= y1):
        raise ValidationError(f"{ctx}: bbox not well-ordered {bbox}")
    if x0 < 0 or y0 < 0 or x1 > max(W, MAX_IMG_W) or y1 > max(H, MAX_IMG_H):
        raise ValidationError(f"{ctx}: bbox {bbox} out of range (W={W},H={H})")


def validate_sample(s: InterventionSample, strict_paths: bool = False) -> list[str]:
    """返回 warning 列表; 严重问题直接 raise ValidationError。"""
    warns = []
    if not (isinstance(s.image_size, (list, tuple)) and len(s.image_size) == 2):
        raise ValidationError(f"{s.task_id}: image_size must be [W,H]")
    W, H = s.image_size
    if W <= 0 or H <= 0 or W > MAX_IMG_W or H > MAX_IMG_H:
        raise ValidationError(f"{s.task_id}: image_size {s.image_size} invalid")

    # 动作坐标/证据框范围
    for name, act in [("base", s.base_action), ("original", s.original_action),
                      ("target", s.target_action), ("unrelated", s.unrelated_action),
                      ("mapped", s.mapped_action)]:
        if act is None:
            continue
        _check_coords(act.coordinates, W, H, f"{s.task_id}.{name}")
        _check_bbox(act.evidence_bbox, W, H, f"{s.task_id}.{name}")
        if not (0.0 <= act.confidence <= 1.0):
            raise ValidationError(f"{s.task_id}.{name}: confidence {act.confidence} not in [0,1]")

    # target 一致性: 若声称 targeted_changes 则必须有 target_action 且与 original 语义不同
    if s.targeted_changes:
        if s.target_action is None:
            raise ValidationError(f"{s.task_id}: targeted_changes=True 但无 target_action")
        same = (s.target_action.action_type == s.original_action.action_type and
                s.target_action.target_id == s.original_action.target_id)
        if same:
            raise ValidationError(
                f"{s.task_id}: targeted_changes=True 但 target_action 语义==original (矛盾)")
    # target red-line: 若有 target_edit 是 targeted 类, backend 应 mutated (不用 DOM 伪装)
    if s.target_edit is not None and s.target_edit.category == "targeted":
        if not s.target_edit.backend_mutated:
            warns.append(f"{s.task_id}: targeted edit backend_mutated=False (DOM-only, 可能过不了 executable oracle)")

    # unrelated 一致性: unrelated_action 应 == original (specificity)
    if s.unrelated_action is not None:
        same = (s.unrelated_action.action_type == s.original_action.action_type and
                s.unrelated_action.target_id == s.original_action.target_id)
        if not same:
            raise ValidationError(
                f"{s.task_id}: unrelated_action 语义 != original (unrelated 应保持正确动作)")

    # transform: 需 tau + bbox
    if s.layout_edit is not None:
        if s.layout_edit.category != "transform":
            warns.append(f"{s.task_id}: layout_edit.category!='transform'")
        _check_bbox(s.layout_edit.region_bbox, W, H, f"{s.task_id}.layout.region")

    # data split 一致性 (原场景与编辑版同划分 — schema 级单样本内保证)
    if s.data_split not in ("train", "val", "test"):
        raise ValidationError(f"{s.task_id}: data_split {s.data_split} invalid")

    return warns


def validate_dataset(samples: list[InterventionSample]) -> dict:
    """数据集级: 统计量 + 缺失率 + 逐样本验证。"""
    n = len(samples)
    if n == 0:
        raise ValidationError("empty dataset")
    all_warns = []
    miss = {"target": 0, "unrelated": 0, "layout": 0}
    tgt_changed = 0
    backend_mut = 0
    splits = {}
    for s in samples:
        all_warns += validate_sample(s)
        if s.target_edit is None:
            miss["target"] += 1
        if s.unrelated_edit is None:
            miss["unrelated"] += 1
        if s.layout_edit is None:
            miss["layout"] += 1
        if s.targeted_changes:
            tgt_changed += 1
        if s.target_edit is not None and s.target_edit.backend_mutated:
            backend_mut += 1
        splits[s.data_split] = splits.get(s.data_split, 0) + 1
    return {
        "n": n,
        "missing_rate": {k: v / n for k, v in miss.items()},
        "targeted_changes_rate": tgt_changed / n,
        "backend_mutated_count": backend_mut,
        "splits": splits,
        "n_warnings": len(all_warns),
        "warnings_sample": all_warns[:5],
    }
