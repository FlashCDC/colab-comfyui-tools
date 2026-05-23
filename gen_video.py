#!/usr/bin/env python3
"""
gen_video.py — Local client for Colab ComfyUI server

Usage:
    # T2V: pure text-to-video (no image needed)
    python gen_video.py --prompt "a cat walking in the rain, cinematic lighting"

    # I2V: image + text → video
    python gen_video.py --image input.png --prompt "cute cartoon, gentle motion"

    # Explicit mode
    python gen_video.py --prompt "..." --mode t2v
    python gen_video.py --image input.png --mode i2v

    # Or specify URL manually
    python gen_video.py --url https://xxx.trycloudflare.com --prompt "..."

How it works:
    1. Reads ComfyUI tunnel URL from cache or --url flag
    2. For I2V: uploads image → submits AnimateDiff workflow
       For T2V: submits AnimateDiff workflow directly (no image)
    3. Polls until done, downloads video to local directory
"""

import argparse
import os
import random
import sys
import time
import json
import requests


# --- Workflow builders (matches the working Colab version) ---

def make_workflow_i2v(image_name, prompt, negative, ckpt, motion,
                       seed, steps, cfg, denoise, width, height, frames):
    """Build image-to-video workflow. Requires reference image."""
    return {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg,
            "denoise": denoise, "sampler_name": "euler",
            "scheduler": "normal",
            "model": ["31", 0], "positive": ["6", 0],
            "negative": ["7", 0], "latent_image": ["29", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["8", 0], "frame_rate": 8, "loop_count": 0,
            "format": "video/h264-mp4", "pingpong": False,
            "save_output": True, "filename_prefix": "AnimateDiff"}},
        "29": {"class_type": "ADE_EmptyLatentImageLarge",
               "inputs": {"width": width, "height": height, "batch_size": frames}},
        "31": {"class_type": "AnimateDiffLoaderV1", "inputs": {
            "model": ["4", 0], "model_name": motion,
            "beta_schedule": "sqrt_linear (AnimateDiff)",
            "context_length": frames, "context_stride": 1,
            "context_overlap": 4, "latents": ["29", 0],
            "noise_type": "per_step", "seed": 0,
            "unlimited_area_hack": False}},
        "35": {"class_type": "LoadImage",
               "inputs": {"image": image_name}},
    }


def make_workflow_t2v(prompt, negative, ckpt, motion,
                       seed, steps, cfg, width, height, frames):
    """Build text-to-video workflow. No image required."""
    return {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg,
            "denoise": 1.0, "sampler_name": "euler",
            "scheduler": "normal",
            "model": ["31", 0], "positive": ["6", 0],
            "negative": ["7", 0], "latent_image": ["29", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["8", 0], "frame_rate": 8, "loop_count": 0,
            "format": "video/h264-mp4", "pingpong": False,
            "save_output": True, "filename_prefix": "AnimateDiff"}},
        "29": {"class_type": "ADE_EmptyLatentImageLarge",
               "inputs": {"width": width, "height": height, "batch_size": frames}},
        "31": {"class_type": "AnimateDiffLoaderV1", "inputs": {
            "model": ["4", 0], "model_name": motion,
            "beta_schedule": "sqrt_linear (AnimateDiff)",
            "context_length": frames, "context_stride": 1,
            "context_overlap": 4, "latents": ["29", 0],
            "noise_type": "per_step", "seed": 0,
            "unlimited_area_hack": False}},
    }


# --- Tunnel URL retrieval ---

def get_tunnel_url(args_url):
    """Get tunnel URL from args, cache, or Drive."""
    # 1. Command line
    if args_url:
        return args_url.rstrip("/")
    # 2. Local cache (set by previous run or manually)
    cache_path = os.path.expanduser("~/.comfyui_tunnel_url")
    if os.path.exists(cache_path):
        url = open(cache_path).read().strip()
        if url.startswith("https://") and "trycloudflare.com" in url:
            print(f"Using cached tunnel URL: {url}")
            return url
    print("ERROR: No tunnel URL found. Use --url or save to ~/.comfyui_tunnel_url")
    sys.exit(1)


# --- ComfyUI API calls ---

def upload_image(url, image_path):
    """Upload image to ComfyUI server."""
    filename = os.path.basename(image_path)
    with open(image_path, "rb") as f:
        resp = requests.post(f"{url}/upload/image",
                             files={"image": (filename, f)}, timeout=60)
    resp.raise_for_status()
    print(f"Uploaded: {filename}")
    return filename


def submit_workflow(url, workflow):
    """Submit workflow to ComfyUI and return prompt_id."""
    resp = requests.post(f"{url}/prompt", json={"prompt": workflow}, timeout=30)
    if resp.status_code == 400:
        print(f"\n--- COMFYUI ERROR 400 ---")
        print(resp.text[:2000])
        print("---\n")
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def poll_result(url, prompt_id, timeout=600):
    """Poll ComfyUI history until generation is done."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            hist = requests.get(f"{url}/history/{prompt_id}", timeout=10).json()
            if prompt_id in hist:
                elapsed = int(time.time() - start)
                print(f"Done in {elapsed}s")
                return hist[prompt_id]
        except Exception:
            pass
        time.sleep(2)
    print(f"Timeout after {timeout}s")
    return None


def download_output(url, history, output_dir):
    """Download generated video from ComfyUI."""
    os.makedirs(output_dir, exist_ok=True)
    for node_out in history.get("outputs", {}).values():
        for key in ("videos", "images", "gifs"):
            for item in node_out.get(key, []):
                filename = item["filename"]
                params = {
                    "filename": filename,
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                }
                resp = requests.get(f"{url}/view", params=params, timeout=60)
                resp.raise_for_status()
                out_path = os.path.join(output_dir, filename)
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                print(f"Downloaded: {out_path}")
                return out_path
    print("No output found in history")
    return None


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate video via Colab ComfyUI server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text → video (no image needed)
  %(prog)s --prompt "a cat walking in the rain, cinematic lighting"

  # Image → video (uses reference image)
  %(prog)s --image photo.png --prompt "cute cartoon, gentle motion"

  # With all options
  %(prog)s --prompt "..." --frames 32 --width 768 --height 512
""")

    parser.add_argument("--prompt", default="cute cartoon, warm colors, gentle float, subtle motion",
                        help="Positive prompt")
    parser.add_argument("--negative", default="ugly, blurry, low quality, deformed, text, watermark",
                        help="Negative prompt")
    parser.add_argument("--image", help="Input image path (for I2V mode)")
    parser.add_argument("--mode", choices=["auto", "t2v", "i2v"], default="auto",
                        help="auto=t2v if no --image, i2v if --image given (default: auto)")
    parser.add_argument("--ckpt", default="v1-5-pruned-emaonly.safetensors")
    parser.add_argument("--motion", default="mm_sd_v15_v2.ckpt")
    parser.add_argument("--seed", type=int, default=-1, help="-1 for random")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--denoise", type=float, default=0.75,
                        help="Only for I2V. 0.7=close to input, 0.85=more creative")
    parser.add_argument("--frames", type=int, default=16, help="16=~2s, 32=~4s")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--url", help="ComfyUI tunnel URL (or save to ~/.comfyui_tunnel_url)")
    parser.add_argument("--output", default="./output", help="Output directory")
    args = parser.parse_args()

    # Resolve mode
    if args.mode == "auto":
        mode = "i2v" if args.image else "t2v"
    else:
        mode = args.mode

    # I2V mode must have image
    if mode == "i2v":
        if not args.image:
            print("ERROR: --image is required for I2V mode")
            sys.exit(1)
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            sys.exit(1)

    # Get tunnel URL
    url = get_tunnel_url(args.url)
    print(f"Server: {url}")

    # Seed
    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31)
    print(f"Mode: {mode.upper()}  |  Seed: {seed}")

    # Build & submit workflow
    duration_s = args.frames / 8
    print(f"Prompt: {args.prompt[:60]}...")
    print(f"Generating {args.frames} frames ({duration_s:.0f}s) at {args.width}x{args.height}...")

    if mode == "i2v":
        # Upload image first
        print("Uploading image...")
        image_name = upload_image(url, args.image)
        wf = make_workflow_i2v(
            image_name, args.prompt, args.negative,
            args.ckpt, args.motion, seed,
            args.steps, args.cfg, args.denoise,
            args.width, args.height, args.frames,
        )
    else:
        wf = make_workflow_t2v(
            args.prompt, args.negative,
            args.ckpt, args.motion, seed,
            args.steps, args.cfg,
            args.width, args.height, args.frames,
        )

    prompt_id = submit_workflow(url, wf)
    print(f"Task submitted: {prompt_id}")

    # Poll & download
    result = poll_result(url, prompt_id)
    if result:
        out = download_output(url, result, args.output)
        if out:
            print(f"\nVideo saved: {out}")
            # Cache tunnel URL
            cache_path = os.path.expanduser("~/.comfyui_tunnel_url")
            with open(cache_path, "w") as f:
                f.write(url)
        else:
            sys.exit(1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
