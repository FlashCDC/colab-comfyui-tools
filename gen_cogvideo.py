#!/usr/bin/env python3
"""
gen_cogvideo.py — Local client for Colab ComfyUI CogVideoX server

Generates video using CogVideoX-5B-I2V (image-to-video) or T2V.
Understands specific actions like "washing hair", "walking", etc.

Usage:
    # I2V: image + prompt → video (recommended)
    python gen_cogvideo.py --url https://xxx.trycloudflare.com \
        --image template.png --prompt "a mother washing her baby's hair"

    # T2V: text → video (no image)
    python gen_cogvideo.py --prompt "a cute cat walking in a garden"
"""

import argparse
import os
import random
import sys
import time
import requests


# --- Tunnel URL ---

def get_tunnel_url(args_url):
    if args_url:
        return args_url.rstrip("/")
    cache_path = os.path.expanduser("~/.comfyui_tunnel_url")
    if os.path.exists(cache_path):
        url = open(cache_path).read().strip()
        if url.startswith("https://"):
            print(f"Using cached tunnel URL: {url}")
            return url
    print("ERROR: No tunnel URL. Use --url or save to ~/.comfyui_tunnel_url")
    sys.exit(1)


# --- Workflow builders (API-verified) ---

def make_workflow_i2v(image_name, prompt, negative, seed, steps, cfg, frames):
    return {
        "20": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors",
            "type": "sd3"}},
        "30": {"class_type": "CogVideoTextEncode", "inputs": {
            "clip": ["20", 0], "prompt": prompt}},
        "31": {"class_type": "CogVideoTextEncode", "inputs": {
            "clip": ["20", 0], "prompt": negative}},
        "36": {"class_type": "LoadImage", "inputs": {
            "image": image_name}},
        "37": {"class_type": "ImageResizeKJ", "inputs": {
            "width": 720, "height": 480, "upscale_method": "lanczos",
            "keep_proportion": False, "divisible_by": 16,
            "image": ["36", 0]}},
        "59": {"class_type": "DownloadAndLoadCogVideoModel", "inputs": {
            "model": "THUDM/CogVideoX-5b-I2V",
            "precision": "bf16", "quantization": "disabled",
            "attention_mode": "sdpa", "load_device": "main_device",
            "enable_sequential_cpu_offload": True}},
        "62": {"class_type": "CogVideoImageEncode", "inputs": {
            "vae": ["59", 1], "start_image": ["37", 0]}},
        "63": {"class_type": "CogVideoSampler", "inputs": {
            "model": ["59", 0], "positive": ["30", 0],
            "negative": ["31", 0],
            "num_frames": frames, "steps": steps, "cfg": cfg, "seed": seed,
            "scheduler": "CogVideoXDDIM",
            "image_cond_latents": ["62", 0]}},
        "60": {"class_type": "CogVideoDecode", "inputs": {
            "vae": ["59", 1], "samples": ["63", 0],
            "enable_vae_tiling": True,
            "tile_sample_min_height": 240, "tile_sample_min_width": 360,
            "tile_overlap_factor_height": 0.2, "tile_overlap_factor_width": 0.2,
            "auto_tile_size": True}},
        "44": {"class_type": "VHS_VideoCombine", "inputs": {
            "frame_rate": 8, "loop_count": 0,
            "filename_prefix": "CogVideoX", "format": "video/h264-mp4",
            "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True,
            "pingpong": False, "save_output": True,
            "images": ["60", 0]}},
    }


def make_workflow_t2v(prompt, negative, seed, steps, cfg, frames):
    return {
        "20": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors",
            "type": "sd3"}},
        "30": {"class_type": "CogVideoTextEncode", "inputs": {
            "clip": ["20", 0], "prompt": prompt}},
        "31": {"class_type": "CogVideoTextEncode", "inputs": {
            "clip": ["20", 0], "prompt": negative}},
        "37": {"class_type": "EmptyLatentImage", "inputs": {
            "width": 720, "height": 480, "batch_size": frames}},
        "59": {"class_type": "DownloadAndLoadCogVideoModel", "inputs": {
            "model": "THUDM/CogVideoX-5b-I2V",
            "precision": "bf16", "quantization": "disabled",
            "attention_mode": "sdpa", "load_device": "main_device",
            "enable_sequential_cpu_offload": True}},
        "63": {"class_type": "CogVideoSampler", "inputs": {
            "model": ["59", 0], "positive": ["30", 0],
            "negative": ["31", 0],
            "num_frames": frames, "steps": steps, "cfg": cfg, "seed": seed,
            "scheduler": "CogVideoXDDIM"}},
        "60": {"class_type": "CogVideoDecode", "inputs": {
            "vae": ["59", 1], "samples": ["63", 0],
            "enable_vae_tiling": True,
            "tile_sample_min_height": 240, "tile_sample_min_width": 360,
            "tile_overlap_factor_height": 0.2, "tile_overlap_factor_width": 0.2,
            "auto_tile_size": True}},
        "44": {"class_type": "VHS_VideoCombine", "inputs": {
            "frame_rate": 8, "loop_count": 0,
            "filename_prefix": "CogVideoX", "format": "video/h264-mp4",
            "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True,
            "pingpong": False, "save_output": True,
            "images": ["60", 0]}},
    }


# --- API helpers ---

def upload_image(url, image_path):
    filename = os.path.basename(image_path)
    with open(image_path, "rb") as f:
        resp = requests.post(f"{url}/upload/image",
                             files={"image": (filename, f)}, timeout=60)
    resp.raise_for_status()
    print(f"Uploaded: {filename}")
    return filename


def submit_workflow(url, workflow):
    resp = requests.post(f"{url}/prompt", json={"prompt": workflow}, timeout=30)
    if resp.status_code == 400:
        print(f"\n--- COMFYUI ERROR 400 ---")
        print(resp.text[:2000])
        print("---\n")
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def poll_result(url, prompt_id, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        try:
            hist = requests.get(f"{url}/history/{prompt_id}", timeout=30).json()
            if prompt_id in hist:
                elapsed = int(time.time() - start)
                print(f"Done in {elapsed}s")
                return hist[prompt_id]
        except Exception:
            pass
        time.sleep(5)
    print(f"Timeout after {timeout}s")
    return None


def download_output(url, history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for node_out in history.get("outputs", {}).values():
        for key in ("videos", "images", "gifs"):
            for item in node_out.get(key, []):
                filename = item["filename"]
                params = {"filename": filename, "subfolder": item.get("subfolder",""),
                          "type": item.get("type", "output")}
                resp = requests.get(f"{url}/view", params=params, timeout=120)
                resp.raise_for_status()
                out_path = os.path.join(output_dir, filename)
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                print(f"Downloaded: {out_path}")
                return out_path
    print("No output found")
    return None


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate video via Colab ComfyUI CogVideoX")
    parser.add_argument("--image", help="Input image path (I2V mode)")
    parser.add_argument("--prompt", default="gentle floating, warm colors, soft animation",
                        help="Prompt describing the action")
    parser.add_argument("--negative", default="ugly, blurry, low quality, deformed, text",
                        help="Negative prompt")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--frames", type=int, default=49, help="49=~6s")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--url", help="Tunnel URL")
    parser.add_argument("--output", default="./output", help="Output dir")
    args = parser.parse_args()

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31)
    url = get_tunnel_url(args.url)
    print(f"Server: {url}")
    print(f"Seed: {seed}")

    mode = "I2V" if args.image else "T2V"
    print(f"Mode: {mode}")
    print(f"Prompt: {args.prompt[:80]}...")

    duration_s = args.frames / 8
    print(f"Generating {args.frames} frames ({duration_s:.0f}s at 720x480)...")

    if mode == "I2V":
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            sys.exit(1)
        print("Uploading image...")
        image_name = upload_image(url, args.image)
        wf = make_workflow_i2v(image_name, args.prompt, args.negative,
                               seed, args.steps, args.cfg, args.frames)
    else:
        wf = make_workflow_t2v(args.prompt, args.negative,
                                seed, args.steps, args.cfg, args.frames)

    print("Submitting (first run downloads ~2GB model)...")
    prompt_id = submit_workflow(url, wf)
    print(f"Task: {prompt_id}")

    result = poll_result(url, prompt_id)
    if result:
        out = download_output(url, result, args.output)
        if out:
            print(f"\nVideo: {out}")
            cache_path = os.path.expanduser("~/.comfyui_tunnel_url")
            with open(cache_path, "w") as f:
                f.write(url)
        else:
            sys.exit(1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
