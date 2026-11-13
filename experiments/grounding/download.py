import sys
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
repo='likaixin/ScreenSpot-Pro'
sha=Path(__file__).with_name("dataset.revision").read_text().strip()
(root/'revision.txt').write_text(sha+'\n')
snapshot_download(repo_id=repo,repo_type='dataset',revision=sha,local_dir=root/'dataset',allow_patterns=['annotations/*','images/*'],max_workers=6)
print('Data downloaded at revision',sha,flush=True)
