#!/usr/bin/env python
"""
WorldPlay concept injection by ANCHOR REPLACEMENT, as a standalone wrapper --
no edits to the repo source.

It installs entity_recache.patch_pipeline (replace mode) onto the pipeline by
lazily wrapping `ar_rollout`: the first time the pipeline runs its AR rollout,
we apply patch_pipeline (which re-binds ar_rollout to the recache version) and
call it. Then we delegate to WorldPlay's normal hyvideo/generate.py CLI, so every
non-injection argument passes through unchanged.

Config via env vars:
  WP_ENTITY             path to the concept image (required to activate injection)
  WP_TRIGGER_CHUNK      chunk index where injection starts (default 1)
  WP_INJECT_WINDOW      injection-window length in chunks (default 5)
  WP_INJECT_POSITIONS   context frame indices to overwrite in each chunk, ',' separated
                        (default "0,1"); broadcast across WP_INJECT_WINDOW chunks. For
                        per-chunk control, pass ';'-separated groups (one per chunk), in
                        which case the number of groups sets the window and WP_INJECT_WINDOW
                        is ignored.

Usage (under torchrun, with the SAME args you'd give generate.py):
  WP_ENTITY=/path/concept.png WP_TRIGGER_CHUNK=1 WP_INJECT_WINDOW=5 WP_INJECT_POSITIONS=0,1 \
  torchrun --nproc_per_node=1 wp_recache_inject.py --prompt "..." --image_path 8.png ...
"""
import os
import sys

# Point this at a checkout of the upstream HY-WorldPlay repo (the "base").
WORLDPLAY_DIR = os.environ.get("WORLDPLAY_DIR", os.path.expanduser("~/HY-WorldPlay"))
sys.path.insert(0, WORLDPLAY_DIR)
sys.path.insert(0, os.path.join(WORLDPLAY_DIR, "hyvideo"))  # generate.py's local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import entity_recache` resolves

from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from entity_recache import patch_pipeline

ENT_PATH = os.environ.get("WP_ENTITY")
TRIGGER_CHUNK = int(os.environ.get("WP_TRIGGER_CHUNK", "1"))
WINDOW = int(os.environ.get("WP_INJECT_WINDOW", "5"))
POS_SPEC = os.environ.get("WP_INJECT_POSITIONS", "0,1")
if ";" in POS_SPEC:
    # explicit per-chunk positions; the number of groups sets the window
    INJECTION_POSITIONS = [
        [int(x) for x in grp.split(",") if x != ""]
        for grp in POS_SPEC.split(";") if grp != ""
    ]
else:
    # single per-chunk spec broadcast across WINDOW chunks
    _per_chunk = [int(x) for x in POS_SPEC.split(",") if x != ""]
    INJECTION_POSITIONS = [list(_per_chunk) for _ in range(WINDOW)]

_orig_ar_rollout = HunyuanVideo_1_5_Pipeline.ar_rollout
_patched_instances = set()


def _lazy_ar_rollout(self, *args, **kwargs):
    # On first rollout of this pipeline instance, install the recache patch.
    if ENT_PATH and id(self) not in _patched_instances:
        _patched_instances.add(id(self))
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[recache-inject] entity={os.path.basename(ENT_PATH)} "
                  f"trigger_chunk={TRIGGER_CHUNK} positions={INJECTION_POSITIONS}", flush=True)
        patch_pipeline(
            self,
            entity_image_paths=ENT_PATH,
            trigger_chunk=TRIGGER_CHUNK,
            injection_positions=INJECTION_POSITIONS,
        )
        # patch_pipeline re-bound self.ar_rollout (instance attr) to the recache version.
        return self.ar_rollout(*args, **kwargs)
    return _orig_ar_rollout(self, *args, **kwargs)


HunyuanVideo_1_5_Pipeline.ar_rollout = _lazy_ar_rollout

# Delegate to the normal generate.py CLI (args pass through via sys.argv).
import hyvideo.generate as G
G.main()
