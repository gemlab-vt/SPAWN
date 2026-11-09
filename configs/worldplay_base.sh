#!/bin/bash
# ---------------------------------------------------------------------------
# Base WorldPlay generation config.
#
# These are the STOCK HY-WorldPlay generate.py settings (model paths + sampling
# knobs). The injection wrapper reuses them unchanged and only adds the
# concept-injection inputs (entity image + when). Source this file, then set the
# injection inputs, then run (see ../run_inject.sh).
#
# Nothing here is injection-specific — it mirrors how upstream WorldPlay is run.
# ---------------------------------------------------------------------------

# Where the upstream HY-WorldPlay checkout lives (the base the wrapper patches).
export WORLDPLAY_DIR="${WORLDPLAY_DIR:-$HOME/HY-WorldPlay}"

# --- model weights (HunyuanVideo-1.5 base + HY-WorldPlay AR-distilled action head) ---
# Defaults resolve inside the HuggingFace hub cache by globbing the snapshot dir, since the
# snapshot hash differs per download. Override MODEL_PATH / AR_DISTILL if your weights live elsewhere.
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
_hunyuan_snap="$(ls -d "$HF_HUB"/models--tencent--HunyuanVideo-1.5/snapshots/*/ 2>/dev/null | tail -1)"
_worldplay_snap="$(ls -d "$HF_HUB"/models--tencent--HY-WorldPlay/snapshots/*/ 2>/dev/null | tail -1)"
MODEL_PATH="${MODEL_PATH:-$_hunyuan_snap}"
AR_DISTILL="${AR_DISTILL:-${_worldplay_snap}ar_distilled_action_model/diffusion_pytorch_model.safetensors}"

# --- sampling / output geometry (standard WorldPlay few-step AR settings) ---
RESOLUTION="480p"
ASPECT_RATIO="16:9"
WIDTH=832
HEIGHT=480
MODEL_TYPE="ar"
FEW_STEP="true"
NUM_INFERENCE_STEPS=4
REWRITE="false"
SR="false"
SEED="${SEED:-42}"

# perf toggles (left off, as in the reference runs)
USE_VAE_PARALLEL="false"
USE_SAGEATTN="false"
USE_FP8_GEMM="false"

# --- camera / rollout ---
# pose "w-63" = 63 forward steps. video_length is inferred from POSE by run_inject.sh
# (length = 4 * total_steps + 1); set VIDEO_LENGTH only to override.
POSE="${POSE:-w-63}"
