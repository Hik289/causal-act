"""Resume-safe launcher for one immutable forum component-study run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from build_forum_quads import write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);args=ap.parse_args()
    run=Path(args.run);p=json.loads((run/"protocol.json").read_text())
    lock=(run/"controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={"phase":"collecting","pid":os.getpid(),"jobs":{v:{"status":"queued"} for v in p["variants"]}}
    def save():write_json(run/"status.json",state)
    children=[]
    try:
        save();data=run/"data"
        if p.get("external_dataset"):
            while not (data/"dataset_complete.json").exists() or not (run/"model_ready.json").exists():
                state["phase"]="waiting for verified dataset/model handoff";save();time.sleep(15)
        if not (data/"dataset_complete.json").exists():
            if (data/"dataset.jsonl").exists():raise RuntimeError("Partial collection exists: repair only missing frozen forums before resuming")
            with (run/"collection.log").open("x") as log:
                child=subprocess.Popen([p["python"],"-u",str(run/"code/build_forum_quads.py"),"--out",str(data),"--auth",p["auth"],"--browser",p["browser"],"--seed",str(p["split_seed"]),"--forums",str(p["forums"]),"--test-forums","25","--val-forums","10"],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state["collection_pid"]=child.pid;save()
                if child.wait()!=0:raise RuntimeError("Dataset collection failed; preserve collection.log and any captures")
        meta=json.loads((data/"dataset_complete.json").read_text())
        if not meta["eligible_for_training"]:raise RuntimeError("Frozen held-out coverage incomplete; do not shrink test denominator")
        state.update(phase="training and held-out evaluation",collection_pid=None,data=meta);save()
        pending=[v for v in p["variants"] if not (run/v/"report.json").exists()]
        active={}
        while pending or active:
            for gpu in p["gpus"]:
                if gpu in active or not pending:continue
                if shutil.disk_usage(run).free<10*2**30:state["waiting_for_disk"]=True;save();continue
                free=subprocess.run(["nvidia-smi",f"--id={gpu}","--query-gpu=memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
                if int(free.stdout.strip())<p.get("memory_free_floor_mib",36000):continue
                variant=pending.pop(0);out=run/variant;out.mkdir(exist_ok=True)
                if (out/"heldout_predictions.jsonl").exists():raise RuntimeError("Partial held-out evaluation needs an explicit missing-only continuation")
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM="false",HF_HUB_OFFLINE="1")
                stream=(run/(variant+".log")).open("x")
                cmd=[p["python"],"-u",str(run/"code/forum_expert_iteration.py"),"--data",str(data/"dataset.jsonl"),"--model",p["model"],"--out",str(out),"--variant",variant,"--rounds","2","--epochs","3","--seed","42"]
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env)
                children.append(child);active[gpu]=(variant,child,stream);state["jobs"][variant]={"status":"running","pid":child.pid,"gpu":gpu};save()
            for gpu,(variant,child,stream) in list(active.items()):
                code=child.poll()
                if code is None:continue
                stream.close();state["jobs"][variant].update(status="finished" if code==0 else "invalid; needs repair",exit_code=code,pid=None);del active[gpu];save()
            if pending or active:time.sleep(10)
        state["phase"]="finished" if all((run/v/"report.json").exists() for v in p["variants"]) else "needs repair"
    except Exception:
        state.update(phase="needs repair",error=traceback.format_exc());raise
    finally:
        # Children already running retain their logs and valid output; do not
        # cancel them merely because another job requires repair.
        save()


if __name__=="__main__":main()
