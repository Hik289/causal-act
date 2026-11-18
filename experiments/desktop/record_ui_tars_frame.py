"""Record deployment geometry without using benchmark annotations or scores."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--service-max-side", required=True, type=int)
    parser.add_argument("--model", default="ui-tars-1.5-7b")
    parser.add_argument("--service-source", type=Path)
    args = parser.parse_args()
    if min(args.width, args.height, args.service_max_side) <= 0:
        parser.error("Image dimensions and service cap must be positive")
    from PIL import Image
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(args.model_path)).image_processor
    image = Image.new("RGB", (args.width, args.height))
    image.thumbnail((args.service_max_side, args.service_max_side))
    grid = processor(images=[image], return_tensors="np")["image_grid_thw"][0]
    contract = {
        "model": args.model,
        "coordinate_space": "qwen25vl_processed_pixels",
        "screenshot_size": [args.width, args.height],
        "service_thumbnail_size": list(image.size),
        "model_frame_size": [int(grid[2]) * processor.patch_size,
                             int(grid[1]) * processor.patch_size],
        "image_grid_thw": [int(v) for v in grid],
        "patch_size": processor.patch_size,
        "source": "https://github.com/bytedance/UI-TARS/blob/main/README_coordinates.md",
        "selection": "Processor metadata only; no benchmark target or score used",
        "processor_config_sha256": hashlib.sha256(
            (args.model_path / "preprocessor_config.json").read_bytes()).hexdigest(),
    }
    if args.service_source:
        contract["service_sha256"] = hashlib.sha256(args.service_source.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(contract, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
