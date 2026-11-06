#!/usr/bin/env python3
"""Recalculate headline numbers from the released aggregate counts."""
import json
from pathlib import Path

def pct(n, d):
    return 100 * n / d

def main():
    data = json.loads((Path(__file__).resolve().parents[1] / "results" / "summary.json").read_text())
    d = data["forum_targeted"]
    n = d["heldout_pairs"]
    print(f'Forum diagnostic: {d["base_both_correct"]}/{n} -> {d["paired_supervision_both_correct"]}/{n}')
    d = data["forum_baseline"]
    print(f'Forum Base: {d["successful"]}/{d["scored"]} = {pct(d["successful"], d["scored"]):.1f}%')
    d = data["desktop_independent_matched"]
    n, a, b = d["tasks"], d["base_successful"], d["refiner_successful"]
    assert b-a == d["rescues"]-d["harms"]
    print(f'OSWorld independent: {a}/{n} -> {b}/{n} ({pct(b-a,n):+.1f} pp)')
    d = data["desktop_per_domain_passthrough_reconstruction"]
    n, a = d["baseline_pool"], d["base_successful"]
    assert d["rescues"] == sum(x["rescues"] for x in d["per_domain"].values())
    assert d["harms"] == sum(x["harms"] for x in d["per_domain"].values())
    b = a + d["rescues"] - d["harms"]
    assert b == d["reconstructed_refiner_successful"]
    print(f'OSWorld passthrough reconstruction: {a}/{n} -> {b}/{n} ({pct(b-a,n):+.1f} pp)')
    d = data["driving_base_smoke"]
    print(f'Bench2Drive Base smoke: {d["logged_routes"]} logged route, score {d["driving_score"]:.5f}')

if __name__ == "__main__":
    main()
