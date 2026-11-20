"""Prepare UI-TARS geometry and source identities without running inference.

This preparation does not change the deployed service, execute a benchmark,
or establish a complete UI-TARS Forum agent.
"""
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def native_parser(source, expected_sha256=None):
    path = Path(source) / 'codes/ui_tars/action_parser.py'
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise RuntimeError('Pinned native parser changed')
    spec = importlib.util.spec_from_file_location('pinned_uitars_action_parser', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def thumbnail_cap(service_source):
    caps = []
    for node in ast.walk(ast.parse(Path(service_source).read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'thumbnail':
            if len(node.args) != 1:
                raise RuntimeError('Unrecognized service thumbnail arguments')
            value = ast.literal_eval(node.args[0])
            if not isinstance(value, (tuple, list)) or len(value) != 2 or value[0] != value[1] or type(value[0]) is not int or value[0] <= 0:
                raise RuntimeError('Unrecognized service thumbnail dimensions')
            caps.append(value[0])
    if not caps or len(set(caps)) != 1:
        raise RuntimeError('Cannot establish one thumbnail cap from deployed service source')
    return caps[0]


def screen_box(value, screen_size):
    """Convert the native parser's normalized box, retaining out-of-frame errors."""
    box = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
        raise ValueError('Expected four finite native box coordinates')
    if len(screen_size) != 2 or any(not isinstance(x, int) or x <= 0 for x in screen_size):
        raise ValueError('Expected two positive screen dimensions')
    return [round((box[0] + box[2]) / 2 * screen_size[0], 3),
            round((box[1] + box[3]) / 2 * screen_size[1], 3)]


def parsed_action(text, contract):
    native = native_parser(contract['native_source'], contract['native_files_sha256']['codes/ui_tars/action_parser.py'])
    height, width = native.smart_resize(contract['service_thumbnail_size'][1], contract['service_thumbnail_size'][0],
        factor=native.IMAGE_FACTOR, min_pixels=contract['min_pixels'], max_pixels=contract['max_pixels'])
    if [width, height] != contract['model_frame_size']:
        raise RuntimeError('Native parser and deployed processor disagree on geometry')
    return native.parse_action_to_structure_output(text, factor=1000,
        origin_resized_height=contract['service_thumbnail_size'][1],
        origin_resized_width=contract['service_thumbnail_size'][0], model_type='qwen25vl',
        min_pixels=contract['min_pixels'], max_pixels=contract['max_pixels'])


def record(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Preparation must run with CUDA_VISIBLE_DEVICES empty')
    out = Path(args.output).resolve()
    if shutil.disk_usage(out.parent).free < 10 * 1024**3:
        raise RuntimeError('Preparation disk is below the existing10GiB floor')
    out.mkdir(parents=True, exist_ok=False)
    source, model_path, service = [Path(x).resolve() for x in [args.source, args.model_path, args.service_source]]
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD'], check=True)
    native = native_parser(source)
    if thumbnail_cap(service) != args.service_max_side:
        raise RuntimeError('Requested geometry differs from deployed service thumbnail code')
    from PIL import Image
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True).image_processor
    image = Image.new('RGB', (args.width, args.height))
    image.thumbnail((args.service_max_side, args.service_max_side))
    grid = processor(images=[image], return_tensors='np')['image_grid_thw'][0]
    if processor.patch_size * processor.merge_size != native.IMAGE_FACTOR:
        raise RuntimeError('Native image factor differs from the actual processor')
    contract = {'kind': 'offline UI-TARS preparation; no inference or benchmark score',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'native_source': str(source), 'native_commit': commit, 'native_url': 'https://github.com/bytedance/UI-TARS',
        'native_files_sha256': {n: file_sha256(source / n) for n in ['codes/ui_tars/action_parser.py', 'codes/ui_tars/prompt.py', 'README_coordinates.md']},
        'model_path': str(model_path), 'model': 'ui-tars-1.5-7b', 'dtype': 'deployed service fp16, device_map=auto',
        'service_source': str(service), 'service_sha256': file_sha256(service),
        'coordinate_space': 'qwen25vl_processed_pixels', 'screenshot_size': [args.width, args.height],
        'service_thumbnail_max_side': args.service_max_side, 'service_thumbnail_size': list(image.size),
        'image_grid_thw': [int(x) for x in grid], 'patch_size': processor.patch_size,
        'merge_size': processor.merge_size, 'min_pixels': processor.min_pixels, 'max_pixels': processor.max_pixels,
        'model_frame_size': [int(grid[2]) * processor.patch_size, int(grid[1]) * processor.patch_size],
        'processor_config_sha256': file_sha256(model_path / 'preprocessor_config.json'),
        'configuration_sha256': {p.name: file_sha256(p) for p in model_path.glob('*.json')},
        'selection': 'deployment metadata and synthetic geometry only; no task targets or scores',
        'actual_model_requests': 0, 'benchmark_scores': 0, 'full_agent_ready': False}
    mw, mh = contract['model_frame_size']
    text = "Thought: synthetic coordinate check.\nAction: click(point='<point>" + str(mw // 2) + ' ' + str(mh // 2) + "</point>')"
    parsed = parsed_action(text, contract)
    point = screen_box(parsed[0]['action_inputs']['start_box'], contract['screenshot_size'])
    if point != [args.width / 2, args.height / 2]:
        raise RuntimeError('Native coordinate center did not map to the screen center')
    contract['synthetic_center_check'] = {'response': text, 'native_parsed': parsed, 'screen_point': point}
    (out / 'frame_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    weights = sorted(model_path.glob('*.safetensors')) + sorted(model_path.glob('pytorch_model*.bin'))
    if not weights:
        raise RuntimeError('No actual model weights found')
    files = {}
    for path in weights:
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Model weight changed during identity preparation')
        files[path.name] = {'bytes': after.st_size, 'sha256': digest}
        (out / 'weight_hash_progress.json').write_text(json.dumps({'files_hashed': len(files), 'planned_files': len(weights)}, indent=2) + '\n')
    manifest = {'model_path': str(model_path), 'weights': files, 'configuration_sha256': contract['configuration_sha256'],
        'use': 'identity preparation only; does not establish the loaded service weights or an evaluation result'}
    (out / 'model_files.json').write_text(json.dumps(manifest, indent=2) + '\n')
    status = {'status': 'offline geometry and weight identity prepared', 'actual_model_requests': 0,
        'benchmark_scores': 0, 'frame_contract_sha256': file_sha256(out / 'frame_contract.json'),
        'model_files_sha256': file_sha256(out / 'model_files.json'), 'native_commit': commit,
        'real_inference_fixture': 'pending a free evaluation slot', 'full_Forum_adapter': 'not yet implemented',
        'browser_fixture': 'pending', 'formal_queue_started': False,
        'existing_model_service_modified': False}
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--service-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--service-max-side', type=int, default=640)
    args = parser.parse_args()
    if min(args.width, args.height, args.service_max_side) <= 0:
        parser.error('Image dimensions must be positive')
    record(args)
