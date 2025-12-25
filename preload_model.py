"""
premodel_download.py

Build-time ONLY script.
- Downloads Wan 2.2 I2V base model files
- Downloads Lightx2v LoRA weights
- NO model loading
- NO GPU / CUDA usage
"""

from huggingface_hub import snapshot_download
from pathlib import Path
import os

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

WAN_REPO_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
LORA_REPO_ID = "Kijai/WanVideo_comfy"

BASE_MODEL_DIR = "/models/Wan2.2-I2V-A14B-Diffusers"
LORA_DIR = "/models/lora/WanVideo_comfy"

# Ensure base dirs exist
Path(BASE_MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(LORA_DIR).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# DOWNLOAD WAN 2.2 BASE MODEL (exact tree, symlinks OK)
# ---------------------------------------------------------------------

print("📥 Downloading Wan 2.2 I2V base model...")

wan_allow_patterns = [
    "model_index.json",

    "scheduler/*",

    "text_encoder/*",
    "tokenizer/*",

    "transformer/*",
    "transformer_2/*",

    "vae/*",
]

wan_snapshot_path = snapshot_download(
    repo_id=WAN_REPO_ID,
    repo_type="model",
    local_dir=BASE_MODEL_DIR,
    local_dir_use_symlinks=True,   # keep HF blob layout (efficient)
    allow_patterns=wan_allow_patterns,
)

print(f"✅ Wan base model downloaded to: {wan_snapshot_path}")

# ---------------------------------------------------------------------
# DOWNLOAD LIGHTX2V LoRA (real file, NO symlinks)
# ---------------------------------------------------------------------

print("📥 Downloading Lightx2v LoRA...")

lora_allow_patterns = [
    "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"
]

lora_snapshot_path = snapshot_download(
    repo_id=LORA_REPO_ID,
    repo_type="model",
    local_dir=LORA_DIR,
    local_dir_use_symlinks=False,  # LoRA must be a real file
    allow_patterns=lora_allow_patterns,
)

print(f"✅ LoRA downloaded to: {lora_snapshot_path}")

print("🎉 All models downloaded successfully (download-only, offline-ready)")
