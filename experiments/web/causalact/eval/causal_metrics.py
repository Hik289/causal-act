"""causalact (CausalAct) — 因果评估指标 (H0.anchor_2)

实现 README §2.2/§8 的因果敏感性指标, 在 forum 可完成子集口径下评估 Action Refiner:
  - PairCausalAcc   : 因果敏感性 (targeted pair 上原动作+改后动作都对, 且二者不同)
  - TargetedAcc     : 改后场景动作正确率 (sensitivity)
  - UnrelatedAcc    : 无关扰动下动作正确率 (specificity, 动作应不变)
  - UnrelatedConsist: 无关扰动下动作一致率 (pred(unrelated)==pred(original))
  - EquivariantAcc  : 变换场景动作等变正确率 (语义不变 + 坐标按 τ 映射)
  - EvidenceRecall  : 证据定位召回 (pred evidence_bbox 命中 GT 区域)

动作表示 (README §4.2):
  {"action_type":"click","target_id":"submit_button",
   "coordinates":[x,y],"evidence_bbox":[x0,y0,x1,y1],"confidence":0.82}

语义动作 (semantic action) 判等: 比较 (action_type, target_id)。
坐标等变: transform 场景下 pred.coordinates 应 ≈ τ(original.coordinates)。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# 动作表示与判等
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Action:
    action_type: str                       # click / type / scroll / stop / ...
    target_id: Optional[str] = None        # 语义目标 (按钮/链接 id)
    coordinates: Optional[tuple] = None     # (x, y) 执行坐标
    evidence_bbox: Optional[tuple] = None   # (x0, y0, x1, y1) 关键证据框
    confidence: float = 1.0

    def semantic_key(self) -> tuple:
        """语义动作 = (type, target)。坐标不参与语义判等。"""
        return (self.action_type, self.target_id)


def semantic_equal(a: Action, b: Action) -> bool:
    return a.semantic_key() == b.semantic_key()


# ──────────────────────────────────────────────────────────────────────
# 几何工具
# ──────────────────────────────────────────────────────────────────────
def point_in_bbox(pt: tuple, bbox: tuple) -> bool:
    x, y = pt
    x0, y0, x1, y1 = bbox
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def iou(b1: tuple, b2: tuple) -> float:
    ax0, ay0, ax1, ay1 = b1
    bx0, by0, bx1, by1 = b2
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    a1 = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    a2 = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def apply_transform(coords: tuple, tau: dict) -> tuple:
    """τ: {'dx':.., 'dy':.., 'sx':.., 'sy':..} 平移+缩放 (围绕原点)。"""
    x, y = coords
    sx = tau.get("sx", 1.0)
    sy = tau.get("sy", 1.0)
    dx = tau.get("dx", 0.0)
    dy = tau.get("dy", 0.0)
    return (x * sx + dx, y * sy + dy)


# ──────────────────────────────────────────────────────────────────────
# 样本容器: 一个因果四元组
# ──────────────────────────────────────────────────────────────────────
@dataclass
class CausalQuad:
    """一条因果样本: original + targeted + unrelated + transform 的 GT 正确动作。

    gt_*  : 环境 oracle 给出的正确动作 (ground truth)
    tau   : transform 的变换参数 (用于等变坐标核验)
    targeted_changes: 环境核验 targeted 是否真的改变了正确动作集合
                      (README §2.3 红线: 只有改变才施加翻转约束/计入 PairCausal)
    """
    task_id: str
    gt_original: Action
    gt_targeted: Optional[Action] = None
    gt_unrelated: Optional[Action] = None
    gt_transform: Optional[Action] = None
    tau: dict = field(default_factory=dict)
    targeted_changes: bool = True


# ──────────────────────────────────────────────────────────────────────
# 指标计算器
# ──────────────────────────────────────────────────────────────────────
class CausalEvaluator:
    def __init__(self, coord_tol: float = 5.0, evidence_iou_th: float = 0.5):
        self.coord_tol = coord_tol
        self.evidence_iou_th = evidence_iou_th

    # --- 单项判定 (供逐样本调试) ---
    def _coords_match(self, pred: Action, gt: Action) -> bool:
        if pred.coordinates is None or gt.coordinates is None:
            return True  # 非坐标动作 (stop/scroll) 不校坐标
        px, py = pred.coordinates
        gx, gy = gt.coordinates
        return abs(px - gx) <= self.coord_tol and abs(py - gy) <= self.coord_tol

    def evaluate(
        self,
        quads: list[CausalQuad],
        preds_original: dict[str, Action],
        preds_targeted: dict[str, Action] | None = None,
        preds_unrelated: dict[str, Action] | None = None,
        preds_transform: dict[str, Action] | None = None,
    ) -> dict:
        preds_targeted = preds_targeted or {}
        preds_unrelated = preds_unrelated or {}
        preds_transform = preds_transform or {}

        # 计数器
        pair_n = pair_ok = 0
        tgt_n = tgt_ok = 0
        unrel_n = unrel_ok = 0
        unrel_consist_n = unrel_consist_ok = 0
        eqv_n = eqv_ok = 0
        ev_n = ev_ok = 0

        for q in quads:
            po = preds_original.get(q.task_id)

            # --- Targeted Acc (sensitivity) ---
            if q.gt_targeted is not None and q.task_id in preds_targeted:
                pt = preds_targeted[q.task_id]
                tgt_n += 1
                if semantic_equal(pt, q.gt_targeted) and self._coords_match(pt, q.gt_targeted):
                    tgt_ok += 1

                # --- PairCausalAcc: 只在 targeted 真的改变正确动作时计入 ---
                if q.targeted_changes and po is not None:
                    pair_n += 1
                    orig_right = semantic_equal(po, q.gt_original) and self._coords_match(po, q.gt_original)
                    tgt_right = semantic_equal(pt, q.gt_targeted) and self._coords_match(pt, q.gt_targeted)
                    changed = not semantic_equal(po, pt)   # 模型确实改了决策
                    if orig_right and tgt_right and changed:
                        pair_ok += 1

            # --- Unrelated Acc (specificity): 正确动作 == 原动作 ---
            if q.gt_unrelated is not None and q.task_id in preds_unrelated:
                pu = preds_unrelated[q.task_id]
                unrel_n += 1
                if semantic_equal(pu, q.gt_unrelated) and self._coords_match(pu, q.gt_unrelated):
                    unrel_ok += 1
                # 一致性: pred(unrelated) == pred(original)
                if po is not None:
                    unrel_consist_n += 1
                    if semantic_equal(pu, po):
                        unrel_consist_ok += 1

            # --- Equivariant Acc: 语义不变 + 坐标按 τ 映射 ---
            if q.gt_transform is not None and q.task_id in preds_transform:
                px = preds_transform[q.task_id]
                eqv_n += 1
                sem_ok = semantic_equal(px, q.gt_transform)
                if px.coordinates is not None and po is not None and po.coordinates is not None:
                    mapped = apply_transform(po.coordinates, q.tau)
                    coord_ok = (abs(px.coordinates[0] - mapped[0]) <= self.coord_tol and
                                abs(px.coordinates[1] - mapped[1]) <= self.coord_tol)
                else:
                    coord_ok = True
                if sem_ok and coord_ok:
                    eqv_ok += 1

            # --- Evidence Recall: pred evidence 命中 GT 区域 (用 original) ---
            if po is not None and po.evidence_bbox is not None and q.gt_original.evidence_bbox is not None:
                ev_n += 1
                if iou(po.evidence_bbox, q.gt_original.evidence_bbox) >= self.evidence_iou_th:
                    ev_ok += 1

        def rate(ok, n):
            return (ok / n) if n > 0 else None

        return {
            "PairCausalAcc": rate(pair_ok, pair_n), "_pair": [pair_ok, pair_n],
            "TargetedAcc": rate(tgt_ok, tgt_n), "_tgt": [tgt_ok, tgt_n],
            "UnrelatedAcc": rate(unrel_ok, unrel_n), "_unrel": [unrel_ok, unrel_n],
            "UnrelatedConsist": rate(unrel_consist_ok, unrel_consist_n), "_unrel_consist": [unrel_consist_ok, unrel_consist_n],
            "EquivariantAcc": rate(eqv_ok, eqv_n), "_eqv": [eqv_ok, eqv_n],
            "EvidenceRecall": rate(ev_ok, ev_n), "_ev": [ev_ok, ev_n],
        }
