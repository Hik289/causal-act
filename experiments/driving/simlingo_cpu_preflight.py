"""Validate a pinned SimLingo architecture/checkpoint without driving or CUDA.

The native Hydra architecture and tokenizer settings are retained. Published
base-model assets must already be downloaded, pinned and hashed. This performs
no agent setup, forward pass, simulator launch or benchmark evaluation.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_json(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise RuntimeError('Bound JSON changed: ' + str(path))
    return json.loads(path.read_text())


def run(protocol_path, protocol_sha, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(scope='CPU architecture/checkpoint preparation only', valid=False,
        score=None, benchmark_scores=0, simulator_launches=0,
        native_agent_setup_executed=False, model_forward_executed=False,
        GPU_used=False, external_model_requests=0,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        spec = checked_json(protocol_path, protocol_sha)
        if spec['scope'] != 'cpu_checkpoint_preparation' or spec['method'] != 'SimLingo':
            raise RuntimeError('Only the pinned SimLingo CPU preparation is implemented')
        if sha(__file__) != spec['preflight_source_sha256']:
            raise RuntimeError('The executing CPU preflight source changed')
        root = Path(spec['source_root']).resolve()
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != spec['source_commit']:
            raise RuntimeError('Native source revision changed')
        for name, digest in spec['source_files_sha256'].items():
            if sha(root/name) != digest:
                raise RuntimeError('Native model source changed: ' + name)
        if sha(spec['checkpoint']) != spec['checkpoint_sha256']:
            raise RuntimeError('Published checkpoint changed')
        if sha(spec['config']) != spec['config_sha256']:
            raise RuntimeError('Published Hydra configuration changed')
        base = checked_json(spec['base_asset_manifest'], spec['base_asset_manifest_sha256'])
        if base['repo'] != spec['base_repo'] or base['revision'] != spec['base_revision']:
            raise RuntimeError('Base-model revision does not match the frozen assets')
        base_path = Path(base['path']).resolve()
        for name, digest in base['files_sha256'].items():
            if sha(base_path/name) != digest:
                raise RuntimeError('Base-model asset changed: ' + name)

        os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        sys.path.insert(0, str(root))
        import torch
        import importlib.metadata
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        import requests
        from transformers import AutoModel, AutoProcessor

        packages = {name: importlib.metadata.version(name) for name in spec['packages']}
        if packages != spec['packages']:
            raise RuntimeError('The frozen model preparation dependencies changed')
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA was initialized before the CPU-only check')
        if torch.__version__ != spec['reused_torch_version']:
            raise RuntimeError('The registered reused Torch version changed')
        if Path(torch.__file__).resolve() != Path(spec['reused_torch_file']).resolve():
            raise RuntimeError('The preparation did not retain the existing Torch installation')
        torch.set_num_threads(2)
        calls = collections.Counter()

        def no_cuda(*args, **kwargs):
            calls['cuda_attempts'] += 1
            raise RuntimeError('CUDA is forbidden during CPU checkpoint preparation')

        def no_network(*args, **kwargs):
            calls['network_attempts'] += 1
            raise RuntimeError('Download and bind assets before the CPU-only check')

        native_loader = AutoModel.from_pretrained

        def local_model(name, *args, **kwargs):
            if name != spec['base_repo']:
                raise RuntimeError('Unexpected native pretrained model: ' + str(name))
            if kwargs.get('trust_remote_code') is not True:
                raise RuntimeError('Preserve the native model loader contract')
            calls['native_base_model_loads'] += 1
            return native_loader(str(base_path), *args, local_files_only=True, **kwargs)

        with patch.object(torch.cuda, '_lazy_init', side_effect=no_cuda), \
             patch.object(requests.sessions.Session, 'request', side_effect=no_network), \
             patch.object(AutoModel, 'from_pretrained', side_effect=local_model):
            cfg = OmegaConf.load(spec['config'])
            if cfg.model.vision_model.variant != spec['base_repo'] or cfg.model.language_model.variant != spec['base_repo']:
                raise RuntimeError('Native vision/language model identifier changed')
            cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
            processor = AutoProcessor.from_pretrained(str(base_path),
                trust_remote_code=True, local_files_only=True)
            tokenizer = processor.tokenizer if 'tokenizer' in processor.__dict__ else processor
            tokenizer.add_special_tokens({'additional_special_tokens': [
                '<WAYPOINTS>', '<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>',
                '<ORG_WAYPOINTS>', '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>', '<TARGET_POINT>']})
            tokenizer.padding_side = 'left'
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                model = instantiate(cfg.model, cfg_data_module=cfg.data_module,
                    processor=processor, cache_dir=str(base_path), _recursive_=False).to('cpu')
            finally:
                torch.set_default_dtype(previous_dtype)
            state = torch.load(spec['checkpoint'], map_location='cpu', weights_only=True)
            if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
                raise RuntimeError('The native loader expects the published flat tensor state dictionary')
            loaded = model.load_state_dict(state, strict=True)
            if loaded.missing_keys or loaded.unexpected_keys:
                raise RuntimeError('Native strict checkpoint loading was incomplete')
            actual = model.state_dict()
            if set(actual) != set(state):
                raise RuntimeError('Strict state-key identity failed')
            for name, value in actual.items():
                if value.device.type != 'cpu' or not torch.equal(value, state[name].to(dtype=value.dtype)):
                    raise RuntimeError('Loaded tensor differs from the checkpoint: ' + name)
            devices = collections.Counter(str(p.device) for p in model.parameters())
            if set(devices) != {'cpu'} or torch.cuda.is_initialized() or calls['cuda_attempts'] or calls['network_attempts']:
                raise RuntimeError('CPU-only preparation boundary was crossed')
            result.update(valid=True, native_architecture_and_strict_checkpoint_passed=True,
                source_commit=commit, protocol_sha256=protocol_sha,
                checkpoint_sha256=spec['checkpoint_sha256'], config_sha256=spec['config_sha256'],
                base_revision=base['revision'], base_asset_manifest_sha256=spec['base_asset_manifest_sha256'],
                parameter_count=sum(p.numel() for p in model.parameters()),
                parameter_devices=dict(devices), tensor_keys=len(state), all_tensor_values_match=True,
                native_base_model_loads=calls['native_base_model_loads'],
                CUDA_attempts=calls['cuda_attempts'], network_attempts=calls['network_attempts'],
                torch_version=torch.__version__, reused_torch_file=str(Path(torch.__file__).resolve()),
                packages=packages, python_version=sys.version,
                published_torch_version=spec['published_torch_version'],
                runtime_difference='CPU validation reuses existing Torch; native GPU/kernel readiness remains pending',
                native_GPU_setup_passed=False, formal_evaluation_ready=False)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.protocol_sha256, args.output)))
