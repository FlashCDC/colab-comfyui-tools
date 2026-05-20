#!/usr/bin/env python3
"""
gen_video.py — Local client for Colab ComfyUI server

Usage:
    # Auto-read tunnel URL from Google Drive
    python gen_video.py --image input.png --prompt "cute cartoon"

    # Or specify URL manually
    python gen_video.py --url https://xxx.trycloudflare.com --image input.png

    # With all options
    python gen_video.py --image input.png --prompt "cute cartoon" \
        --negative "ugly, blurry" --steps 20 --cfg 7.0 --denoise 0.75 \
        --frames 16 --width 512 --height 768 --seed 42

How it works:
    1. Reads ComfyUI tunnel URL from Drive (tunnel_url.txt) or --url flag
    2. Uploads your image to ComfyUI
    3. Submits AnimateDiff workflow
    4. Polls until done, downloads video to local directory
"""

import argparse
import os
import random
import sys
import time
import json
import requests

# Try to import gdown for reading tunnel URL from Drive
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False


# --- Workflow builder (matches the working Colab version) ---

def make_workflow(image_name, prompt, negative, ckpt, motion,
                  seed, steps, cfg, denoise, width, height, frames):
    """Build AnimateDiff workflow JSON for ComfyUI API."""
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


# --- Tunnel URL retrieval ---

TUNNEL_FILE_ID = None  # Set this after uploading tunnel_url.txt to Drive as public

def get_tunnel_url_from_drive():
    """Read tunnel URL from Google Drive via gdown."""
    if not HAS_GDOWN:
        print("gdown not installed. Install with: pip install gdown")
        return None
    if TUNNEL_FILE_ID is None:
        # Try reading from local cache first
        cache_path = os.path.expanduser("~/.comfyui_tunnel_url")
        if os.path.exists(cache_path):
            url = open(cache_path).read().strip()
            if url.startswith("https://") and "trycloudflare.com" in url:
                return url
        print("No tunnel URL found. Either:")
        print("  1. Use --url to specify manually")
        print("  2. Set TUNNEL_FILE_ID in this script after making tunnel_url.txt public on Drive")
        print("  3. Save URL to ~/.comfyui_tunnel_url")
        return None
    try:
        url = gdown.download(f"https://drive.google.com/uc?id={TUNNEL_FILE_ID}",
                             quiet=True, fuzzy=True)
        with open(url) as f:
            tunnel_url = f.read().strip()
        os.remove(url)
        return tunnel_url
    except Exception as e:
        print(f"Failed to read tunnel URL from Drive: {e}")
        return None


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
    # 3. Google Drive
    url = get_tunnel_url_from_drive()
    if url:
        print(f"Got tunnel URL from Drive: {url}")
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
    parser = argparse.ArgumentParser(description="Generate video via Colab ComfyUI")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", default="cute cartoon, warm colors, gentle float, subtle motion")
    parser.add_argument("--negative", default="ugly, blurry, low quality, deformed, text, watermark")
    parser.add_argument("--ckpt", default="v1-5-pruned-emaonly.safetensors")
    parser.add_argument("--motion", default="mm_sd_v15_v2.ckpt")
    parser.add_argument("--seed", type=int, default=-1, help="-1 for random")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--denoise", type=float, default=0.75, help="0.7=close to input, 0.85=more creative")
    parser.add_argument("--frames", type=int, default=16, help="16=~2s, 32=~4s")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--url", help="ComfyUI tunnel URL (or save to ~/.comfyui_tunnel_url)")
    parser.add_argument("--output", default="./output", help="Output directory")
    args = parser.parse_args()

    # Get tunnel URL
    url = get_tunnel_url(args.url)
    print(f"Server: {url}")

    # Validate image
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    # Seed
    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31)
    print(f"Seed: {seed}")

    # Upload image
    print("Uploading image...")
    image_name = upload_image(url, args.image)

    # Build & submit workflow
    duration_s = args.frames / 8
    print(f"Generating {args.frames} frames ({duration_s:.0f}s video)...")
    wf = make_workflow(
        image_name, args.prompt, args.negative,
        args.ckpt, args.motion, seed,
        args.steps, args.cfg, args.denoise,
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
            # Cache tunnel URL for next time
            cache_path = os.path.expanduser("~/.comfyui_tunnel_url")
            with open(cache_path, "w") as f:
                f.write(url)
        else:
            sys.exit(1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
