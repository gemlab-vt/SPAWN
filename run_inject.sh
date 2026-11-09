#!/bin/bash
# ---------------------------------------------------------------------------
# Run WorldPlay with concept injection by ANCHOR REPLACEMENT (replace mode).
#
# Design: the wrapper (inject/wp_recache_inject.py) does NOT reimplement
# WorldPlay. It installs the recache patch (entity_recache.patch_pipeline) by
# lazily wrapping ar_rollout, then delegates to WorldPlay's own generate.py, so
# every non-injection argument is just the standard WorldPlay config
# (configs/worldplay_base.sh). This script supplies only the injection inputs:
#     WHAT to inject -> ENTITY            (a single concept image)
#     WHERE/WHEN      -> TRIGGER_CHUNK + INJECT_POSITIONS
#                        (which chunk injection starts at, and which context
#                         frame indices to overwrite with the entity)
#
# Usage:
#   ENTITY=concept.png START_IMAGE=... POSE="left-47" \
#   TRIGGER_CHUNK=1 WINDOW=5 INJECT_POSITIONS=0,1 \
#   PROMPT="A road." bash run_inject.sh   # VIDEO_LENGTH inferred from POSE
# ---------------------------------------------------------------------------
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/configs/worldplay_base.sh"

# Fail fast with a clear message if the base/weights aren't where the defaults expect.
for _v in WORLDPLAY_DIR MODEL_PATH AR_DISTILL; do
  _p="${!_v}"
  if [ ! -e "$_p" ]; then
    echo "ERROR: $_v does not exist: $_p" >&2
    echo "  Set it to your own path and re-run, e.g.:  $_v=/your/path bash run_inject.sh ..." >&2
    echo "  (see configs/worldplay_base.sh for all overridable paths)" >&2
    exit 1
  fi
done

: "${ENTITY:?set ENTITY=path/to/concept.png}"
: "${PROMPT:?set PROMPT=\"...\"}"
START_IMAGE="${START_IMAGE:-$HERE/data/reference/8.png}"   # first-frame reference for the rollout

# Infer VIDEO_LENGTH from POSE unless given: length = 4 * (sum of per-move step counts) + 1.
# e.g. POSE="left-47" -> 189 ; POSE="w-12, left-43" -> 221.
if [ -z "${VIDEO_LENGTH:-}" ]; then
  _sum=0
  IFS=',' read -ra _toks <<< "$POSE"
  for _t in "${_toks[@]}"; do
    _n="${_t##*-}"; _n="${_n//[^0-9]/}"            # digits after the last '-'
    _sum=$(( _sum + ${_n:-0} ))
  done
  VIDEO_LENGTH=$(( 4 * _sum + 1 ))
fi

TRIGGER_CHUNK="${TRIGGER_CHUNK:-1}"
WINDOW="${WINDOW:-5}"                       # injection-window length in chunks
INJECT_POSITIONS="${INJECT_POSITIONS:-0,1}" # context frames overwritten in each chunk
OUT_DIR="${OUT_DIR:-$HERE/outputs}"
SAVE="${SAVE:-spawn_$(basename "${ENTITY%.*}").mp4}"
mkdir -p "$OUT_DIR"

echo "=============================================================="
echo "RUN $SAVE  entity=$(basename "$ENTITY")"
echo "    replace context pos [$INJECT_POSITIONS] over $WINDOW chunks from chunk $TRIGGER_CHUNK"
echo "=============================================================="

WP_ENTITY="$ENTITY" WP_TRIGGER_CHUNK="$TRIGGER_CHUNK" \
WP_INJECT_WINDOW="$WINDOW" WP_INJECT_POSITIONS="$INJECT_POSITIONS" \
WORLDPLAY_DIR="$WORLDPLAY_DIR" \
torchrun --nproc_per_node=1 --master_port $(( 20000 + RANDOM % 20000 )) \
  "$HERE/inject/wp_recache_inject.py" \
  --prompt "$PROMPT" \
  --image_path "$START_IMAGE" \
  --resolution "$RESOLUTION" \
  --aspect_ratio "$ASPECT_RATIO" \
  --video_length "$VIDEO_LENGTH" \
  --seed "$SEED" \
  --rewrite "$REWRITE" \
  --sr "$SR" \
  --pose "$POSE" \
  --output_path "$OUT_DIR" \
  --output_filename "$SAVE" \
  --model_path "$MODEL_PATH" \
  --action_ckpt "$AR_DISTILL" \
  --few_step "$FEW_STEP" \
  --num_inference_steps "$NUM_INFERENCE_STEPS" \
  --model_type "$MODEL_TYPE" \
  --width "$WIDTH" --height "$HEIGHT" \
  --use_vae_parallel "$USE_VAE_PARALLEL" \
  --use_sageattn "$USE_SAGEATTN" \
  --use_fp8_gemm "$USE_FP8_GEMM"

echo "DONE -> $OUT_DIR/$SAVE"
