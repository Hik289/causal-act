"""Freeze original Forum tasks and shared still-image inputs, without scoring."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
from bedrock_image_transport import normalize_image


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(tasks, checkout, output, expected_count):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ['tasks', 'original_images', 'canonical_images']:
        (output / name).mkdir()
    records = []
    for config in sorted(Path(tasks).glob('*.json'), key=lambda p: int(p.stem)):
        original = config.read_bytes()
        task = json.loads(original)
        if task.get('sites') != ['reddit']:
            raise ValueError('Forum-only input manifest required')
        frozen = output / 'tasks' / config.name
        frozen.write_bytes(original)
        inputs = task.get('image') or []
        inputs = [inputs] if isinstance(inputs, str) else inputs
        references = []
        for declaration in inputs:
            source = Path(declaration)
            if not source.is_absolute():
                source = Path(checkout) / source
            raw = source.read_bytes()
            original_hash = sha(raw)
            archived = output / 'original_images' / (original_hash + '.source')
            if not archived.exists():
                archived.write_bytes(raw)
                archived.chmod(0o444)
            with Image.open(io.BytesIO(raw)) as image:
                image.seek(0)
                image.load()
                original_format, original_mode = image.format, image.mode
                source_frames = getattr(image, 'n_frames', 1)
                pixel_hash = sha(image.convert('RGBA').tobytes())
                mode = 'RGBA' if image.mode in ('RGBA', 'LA') or 'transparency' in image.info else 'RGB'
                decoded = image.convert(mode)
                # Recreate from pixels to remove EXIF/text/profile metadata,
                # retaining the Base's first-frame orientation and dimensions.
                canonical = Image.frombytes(mode, decoded.size, decoded.tobytes())
                if sha(canonical.convert('RGBA').tobytes()) != pixel_hash:
                    raise ValueError('Canonicalization changed decoded first-frame pixels')
                stream = io.BytesIO()
                canonical.save(stream, format='PNG')
                encoded = stream.getvalue()
                dimensions = canonical.size
            cache = output / 'canonical_images' / (original_hash + '.png')
            if not cache.exists():
                cache.write_bytes(encoded)
                cache.chmod(0o444)
            if sha(cache.read_bytes()) != sha(encoded) or sha(archived.read_bytes()) != original_hash:
                raise ValueError('Frozen input hash mismatch')
            mime, transport = normalize_image('image/png', base64.b64encode(encoded).decode())
            references.append(dict(source_declaration=declaration, resolved_original=str(source.resolve()),
                original_cache_file=str(archived), original_sha256=original_hash, original_bytes=len(raw),
                original_format=original_format, original_mode=original_mode, source_frames=source_frames,
                selected_frame=0, pixel_rgba_sha256=pixel_hash, cache_file=str(cache), sha256=sha(encoded),
                width=dimensions[0], height=dimensions[1], canonical_mode=mode,
                transport_mime=mime, transport_sha256=sha(base64.b64decode(transport))))
        records.append(dict(manifest_index=int(config.stem), task_id=task['task_id'],
            config_file=str(frozen), source_config=str(config.resolve()), sha256=sha(original),
            reference_inputs=references, original_require_reset=task.get('require_reset', False)))
    if len(records) != expected_count or len({x['task_id'] for x in records}) != expected_count:
        raise ValueError('Unexpected task count or duplicate task identity')
    manifest = dict(kind='frozen input preparation; no benchmark scores', valid_scores=0, tasks=records,
                    total_tasks=len(records), image_tasks=sum(bool(x['reference_inputs']) for x in records),
                    image_inputs=sum(len(x['reference_inputs']) for x in records),
                    canonicalization='decoded first frame; no EXIF transpose; retain alpha; no resize; metadata-free PNG',
                    transport='existing shared relay JPEG policy above3700000bytes; dimensions unchanged')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--checkout', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-count', type=int, default=170)
    args = parser.parse_args()
    manifest = prepare(args.tasks, args.checkout, args.output, args.expected_count)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'tasks'}))
