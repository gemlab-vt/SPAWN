#!/usr/bin/env python
"""
Qwen-Image-Edit: insert a concept into a background scene, from a text prompt and/or a
reference image.

Text prompt (describe the concept):
    python edit.py \
        --image data/panorama_crops/0_right.png \
        --prompt "Add a wooden rowboat on the lake." \
        --out concept.png

Reference image (insert a given object, harmonized into the scene):
    python edit.py \
        --image data/panorama_crops/0_right.png \
        --reference my_boat.png \
        --prompt "Place this boat on the lake." \
        --out concept.png

--image is a direction-specific background (data/panorama_crops/{id}_{dir}.png). --reference
may be repeated for multiple objects. The output is the concept reference you then inject with
run_inject.sh (whose START_IMAGE is the separate first-frame reference data/reference/{id}.png).

Reference-guided editing uses Qwen-Image-Edit-2509's multi-image input via QwenImageEditPlusPipeline,
which requires diffusers >= 0.36. Without it, passing --reference is an error (the text-prompt-only
path still works on older diffusers).
"""
import argparse, os
from PIL import Image
import torch

# Prefer the multi-image "Plus" pipeline (Qwen-Image-Edit-2509) for reference-guided edits;
# fall back to the single-image pipeline when it isn't available.
try:
    from diffusers import QwenImageEditPlusPipeline as EditPipeline
    _HAS_PLUS = True
except ImportError:
    from diffusers import QwenImageEditPipeline as EditPipeline
    _HAS_PLUS = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="background scene image")
    ap.add_argument("--prompt", required=True, help="edit prompt (what to insert / how to place it)")
    ap.add_argument("--reference", action="append", default=[],
                    help="reference object image to insert (repeatable); omit for text-only editing")
    ap.add_argument("--out", default="concept.png", help="output edited image")
    ap.add_argument("--model", default="Qwen/Qwen-Image-Edit-2509",
                    help="HF repo id or a local path to the Qwen-Image-Edit model")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--true_cfg_scale", type=float, default=4.0)
    ap.add_argument("--negative_prompt", default=" ")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.reference and not _HAS_PLUS:
        raise SystemExit(
            "ERROR: --reference requires QwenImageEditPlusPipeline (diffusers >= 0.36), which this "
            f"build lacks (diffusers {__import__('diffusers').__version__}). Upgrade diffusers, or "
            "omit --reference to edit from the text prompt only."
        )

    pipeline = EditPipeline.from_pretrained(args.model)
    pipeline.to(torch.bfloat16).to("cuda")
    pipeline.set_progress_bar_config(disable=None)

    background = Image.open(args.image).convert("RGB")
    # background first, then any reference objects (multi-image input when supported)
    if args.reference and _HAS_PLUS:
        image_input = [background] + [Image.open(r).convert("RGB") for r in args.reference]
    else:
        image_input = background

    inputs = {
        "image": image_input,
        "prompt": args.prompt,
        "generator": torch.manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.steps,
    }
    with torch.inference_mode():
        out_image = pipeline(**inputs).images[0]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_image.save(args.out)
    print("saved:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
