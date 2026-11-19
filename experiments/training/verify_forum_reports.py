"""Reconcile completed reports to the written metric definitions, without inference.

The frozen runner's initial PairCausalAcc counted original/target correctness.
Preserve that raw report and expose TargetPairAcc separately; the documented
PairCausalAcc also requires unrelated correctness. No predictions are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from forum_expert_iteration import atomic, causal_report


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run, variant):
    out=run/variant; raw=out/"report.json"; predictions=out/"heldout_predictions.jsonl"
    if not raw.exists():return False
    data=run/"data/dataset.jsonl"
    rows=[json.loads(x) for x in data.read_text().splitlines() if x.strip()]
    rows=[r for r in rows if r["split"]=="test"]
    records=[json.loads(x) for x in predictions.read_text().splitlines() if x.strip()]
    ids=[r["id"] for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!={r["id"] for r in rows}:
        raise RuntimeError("Incomplete or duplicate frozen predictions: do not shrink denominator")
    manifest={"raw_report_sha256":sha(raw),"predictions_sha256":sha(predictions),"data_sha256":sha(data)}
    destination=out/"verified_report.json"
    if destination.exists():
        if json.loads(destination.read_text())["verification"]["sources"]!=manifest:
            raise RuntimeError("A verified source changed; retain the previous report for audit")
        return True
    report=json.loads(raw.read_text())
    report.update(causal_report(rows,{r["id"]:r["predictions"] for r in records}))
    report.update(independent_forums=len({r["forum"] for r in rows}),
                  case_structure="two goals and two paraphrases per forum; cases are clustered, not IID tasks",
                  action_metric_scope="semantic click/stop correctness; pointer execution is not measured",
                  safety_metric_scope="oracle disagreement on an already-satisfied subscription goal; not real-world safety")
    report["verification"]={"sources":manifest,"metric_source_sha256":sha(Path(__file__).with_name("forum_expert_iteration.py")),
                            "prediction_changes":False,"additional_inference":False,
                            "PairCausalAcc":"original, targeted, and unrelated decisions are all correct",
                            "TargetPairAcc":"original and targeted decisions are both correct",
                            "note":"Raw report retained; metric definition reconciled independently of scores"}
    atomic(destination,report)
    return True


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);ap.add_argument("--wait",action="store_true")
    args=ap.parse_args();run=Path(args.run);protocol=json.loads((run/"protocol.json").read_text())
    while True:
        done=[v for v in protocol["variants"] if verify(run,v)]
        atomic(run/"verified_reports_status.json",{"completed":done,"planned":len(protocol["variants"]),
               "additional_inference":False,"metric_definition":"original/targeted/unrelated all correct"})
        if len(done)==len(protocol["variants"]) or not args.wait:break
        state=json.loads((run/"status.json").read_text())
        if state["phase"] in ["needs repair","finished"]:break
        time.sleep(15)


if __name__=="__main__":main()
