"""
entity_recache.py
─────────────────
Training-free concept injection into HY-WorldPlay by ANCHOR REPLACEMENT in the
AR rollout's context memory.

Drop-in patch for HunyuanVideo_1_5_Pipeline: it monkey-patches `ar_rollout` on a
live pipeline instance so that, starting at `trigger_chunk`, the concept image's
latent OVERWRITES chosen context frames in-place (context length unchanged) before
the KV cache is rebuilt. The model then carries the concept forward through the
rollout. No training, no change to the model weights.

USAGE
-----
    from entity_recache import patch_pipeline

    patch_pipeline(
        pipeline,
        entity_image_paths="concept.png",   # str or list[str], one per injection chunk
        trigger_chunk=1,                           # chunk index where injection starts
        injection_positions=[[0, 1]] * 5,          # one list per chunk; here a 5-chunk window of frames 0,1
    )
"""

import os
import types
import torch
from PIL import Image
import torchvision.transforms as transforms


# Per-frame denoise levels (fixed in the final design): context (memory) frames vs. the
# overwritten entity slots. Higher entity level = softer conditioning.
STABILIZATION_LEVEL = 15
ENTITY_STABILIZATION_LEVEL = 500


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def patch_pipeline(
    pipeline,
    entity_image_paths,
    trigger_chunk: int,
    injection_positions: list,
):
    """
    Monkey-patches ar_rollout on a live pipeline instance (replace mode).

    entity_image_paths: str or list[str].
        Single path → same image used for every injection chunk.
        List of paths → one per injection chunk; len must match injection_positions.

    trigger_chunk: chunk index where injection starts; injection runs on
        contiguous chunks trigger_chunk .. trigger_chunk + len(injection_positions) - 1.

    injection_positions: list[list[int]], one inner list per injection chunk.
        Each inner list holds the context frame indices to OVERWRITE with the entity
        (0 = oldest frame in the selected context window).
    """
    if isinstance(entity_image_paths, str):
        entity_image_paths = [entity_image_paths]

    assert trigger_chunk is not None, "trigger_chunk is required"
    assert injection_positions is not None and len(injection_positions) > 0, \
        "injection_positions must not be empty"
    if len(entity_image_paths) == 1:
        entity_image_paths = entity_image_paths * len(injection_positions)
    assert len(entity_image_paths) == len(injection_positions), (
        f"entity_image_paths length ({len(entity_image_paths)}) must match "
        f"injection_positions length ({len(injection_positions)})"
    )
    exp_chunks = list(range(trigger_chunk, trigger_chunk + len(injection_positions)))

    for path in entity_image_paths:
        assert os.path.exists(path), f"Entity image not found: {path}"
    for i, slots in enumerate(injection_positions):
        assert len(slots) > 0, f"injection {i} has empty slot list"
        for pos in slots:
            assert isinstance(pos, int) and pos >= 0, \
                f"position values must be non-negative ints, got {pos}"

    chunk_to_idx = {c: i for i, c in enumerate(exp_chunks)}

    pipeline._recache_entity_image_paths         = entity_image_paths
    pipeline._recache_trigger_chunk              = trigger_chunk
    pipeline._recache_injection_positions        = injection_positions
    pipeline._recache_ext_latents                = {}  # keyed by image path
    pipeline._recache_chunk_to_idx               = chunk_to_idx

    pipeline._encode_entity_image = types.MethodType(_encode_entity_image, pipeline)
    pipeline._splice_entity_slots = types.MethodType(_splice_entity_slots, pipeline)
    pipeline._recache_with_entity = types.MethodType(_recache_with_entity, pipeline)
    pipeline.ar_rollout           = types.MethodType(_ar_rollout_with_recache, pipeline)

    print("[entity_recache] Patched pipeline (replace mode).")
    for chunk_i, slots, path in zip(exp_chunks, injection_positions, entity_image_paths):
        print(f"  chunk {chunk_i}: pos={slots}, image={path}")


# ─────────────────────────────────────────────────────────────
# ENTITY ENCODE
# ─────────────────────────────────────────────────────────────

def _encode_entity_image(self, height, width, path):
    """Encode entity image to a 1-frame latent. Cached by path."""
    if path in self._recache_ext_latents:
        return self._recache_ext_latents[path]

    img = Image.open(path).convert("RGB")
    to_tensor = transforms.Compose([
        transforms.Resize(
            (height, width),
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    img_tensor = (
        to_tensor(img).unsqueeze(0).unsqueeze(2)
        .to(self.execution_device, dtype=next(self.vae.parameters()).dtype)
    )

    vae_param_dtype = next(self.vae.parameters()).dtype
    use_autocast = (
        bool(getattr(self, "vae_autocast_enabled", False))
        and vae_param_dtype in (torch.float16, torch.bfloat16)
    )

    from hyvideo.commons import auto_offload_model
    with (
        torch.autocast(device_type="cuda", dtype=vae_param_dtype, enabled=use_autocast),
        auto_offload_model(self.vae, self.execution_device, enabled=self.enable_offloading),
    ):
        ext_latent = self.vae.encode(img_tensor).latent_dist.mode()
        ext_latent = ext_latent * self.vae.config.scaling_factor

    ext_latent = ext_latent.to(dtype=self.target_dtype)
    self._recache_ext_latents[path] = ext_latent
    return ext_latent


# ─────────────────────────────────────────────────────────────
# SPLICE ENTITY SLOTS (replace: overwrite in-place, context length unchanged)
# ─────────────────────────────────────────────────────────────

def _splice_entity_slots(
    self,
    context_latents_in,
    context_viewmats,
    context_Ks,
    context_action,
    positions,
    ext_latent,
    entity_viewmat,
    entity_K,
    entity_action,
    device,
):
    """Overwrite the entity frame at each given context index (full strength)."""
    B = ext_latent.shape[0]
    C_in  = context_latents_in.shape[1]
    C_lat = ext_latent.shape[1]
    C_cond = C_in - C_lat

    ext_cond = torch.zeros(
        B, C_cond, 1, ext_latent.shape[3], ext_latent.shape[4],
        device=device, dtype=ext_latent.dtype,
    )
    ext_input = torch.cat([ext_latent, ext_cond], dim=1)  # (B, C_in, 1, H_l, W_l)

    insert_positions = []
    for pos in positions:
        F_ctx = context_latents_in.shape[2]
        replace_at = max(0, min(pos, max(0, F_ctx - 1)))
        insert_positions.append(replace_at)
        context_latents_in[:, :, replace_at : replace_at + 1] = ext_input
        context_viewmats[:, replace_at : replace_at + 1] = entity_viewmat
        context_Ks[:, replace_at : replace_at + 1] = entity_K
        context_action[:, replace_at : replace_at + 1] = entity_action

    return context_latents_in, context_viewmats, context_Ks, context_action, insert_positions


# ─────────────────────────────────────────────────────────────
# RECACHE / INJECT PASS
# ─────────────────────────────────────────────────────────────

def _recache_with_entity(
    self,
    latents,
    cond_latents,
    current_start_frame,
    positions,
    entity_image_path,
    prompt_embeds,
    prompt_mask,
    vision_states,
    task_type,
    extra_kwargs,
    viewmats,
    Ks,
    action,
    device,
    height,
    width,
):
    from hyvideo.commons import auto_offload_model
    from hyvideo.utils.retrieval_context import select_aligned_memory_frames

    stabilization_level = STABILIZATION_LEVEL

    # ── Select memory frames using the same retrieval as the non-inject path ──
    # Keeps the inject context within training distribution (up to 20 frames:
    # 12 temporal + 8 spatial), so injection positions index directly into context.
    if current_start_frame == 0:
        return

    selected_frame_indices = []
    for chunk_start_idx in range(
        current_start_frame,
        current_start_frame + self.chunk_latent_frames,
        4,
    ):
        selected_history_frame_id = select_aligned_memory_frames(
            viewmats[0].cpu().detach().numpy(),
            chunk_start_idx,
            memory_frames=20, temporal_context_size=12,
            pred_latent_size=4, points_local=self.points_local, device=device,
        )
        selected_frame_indices += selected_history_frame_id

    selected_frame_indices = sorted(list(set(selected_frame_indices)))
    to_remove = list(range(current_start_frame, current_start_frame + self.chunk_latent_frames))
    selected_frame_indices = [x for x in selected_frame_indices if x not in to_remove]

    if len(selected_frame_indices) == 0:
        return

    # build context window from selected frames
    context_latents    = latents[:, :, selected_frame_indices]
    context_cond       = cond_latents[:, :, selected_frame_indices]
    context_latents_in = torch.cat([context_latents, context_cond], dim=1)
    context_viewmats   = viewmats[:, selected_frame_indices].to(device)
    context_Ks         = Ks[:, selected_frame_indices].to(device)
    context_action     = action[:, selected_frame_indices].to(device)

    # encode entity (cached by path)
    ext_latent = self._encode_entity_image(height, width, entity_image_path)
    B = latents.shape[0]
    ext_latent = ext_latent.expand(B, -1, -1, -1, -1).contiguous()

    # camera params for entity slots: use frame 0's camera — entity image was edited
    # from the initial viewpoint, so its extrinsics match frame 0.
    entity_viewmat = viewmats[:, 0:1].to(device)
    entity_K       = Ks[:, 0:1].to(device)
    entity_action  = action[:, 0:1].to(device)

    # overwrite the chosen context slots with the entity (in-place; length unchanged)
    context_latents_in, context_viewmats, context_Ks, context_action, insert_positions = (
        self._splice_entity_slots(
            context_latents_in=context_latents_in,
            context_viewmats=context_viewmats,
            context_Ks=context_Ks,
            context_action=context_action,
            positions=positions,
            ext_latent=ext_latent,
            entity_viewmat=entity_viewmat,
            entity_K=entity_K,
            entity_action=entity_action,
            device=device,
        )
    )

    # per-frame timestep: context frames at stabilization_level, entity slots at
    # entity_stabilization_level (higher = more noise = softer conditioning).
    T_ctx_new = context_latents_in.shape[2]
    entity_stab = ENTITY_STABILIZATION_LEVEL
    context_timestep = torch.full(
        (T_ctx_new,), stabilization_level - 1,
        device=device, dtype=torch.long,
    )
    for idx in insert_positions:
        if 0 <= idx < T_ctx_new:
            context_timestep[idx] = entity_stab - 1

    # rebuild KV with the entity-overwritten context
    with (
        torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled),
        auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading),
    ):
        self._kv_cache = self.transformer(
            bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
            hidden_states=context_latents_in,
            timestep=context_timestep, timestep_r=None,
            mask_type=task_type, return_dict=False,
            viewmats=context_viewmats.to(self.target_dtype),
            Ks=context_Ks.to(self.target_dtype),
            action=context_action.to(self.target_dtype),
            kv_cache=self._kv_cache, cache_vision=True,
            rope_temporal_size=context_latents_in.shape[2],
            start_rope_start_idx=0,
        )
        if self.do_classifier_free_guidance:
            self._kv_cache_neg = self.transformer(
                bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                hidden_states=context_latents_in,
                timestep=context_timestep, timestep_r=None,
                mask_type=task_type, return_dict=False,
                viewmats=context_viewmats.to(self.target_dtype),
                Ks=context_Ks.to(self.target_dtype),
                action=context_action.to(self.target_dtype),
                kv_cache=self._kv_cache_neg, cache_vision=True,
                rope_temporal_size=context_latents_in.shape[2],
                start_rope_start_idx=0,
            )


# ─────────────────────────────────────────────────────────────
# PATCHED ar_rollout
# ─────────────────────────────────────────────────────────────

def _ar_rollout_with_recache(
    self,
    latents, timesteps, prompt_embeds, prompt_mask,
    vision_states, cond_latents, task_type, extra_kwargs,
    viewmats, Ks, action, device,
    transformer_resident=False,   # accepted for pipeline compatibility; recache does its own offloading
):
    from hyvideo.commons import auto_offload_model
    from hyvideo.utils.retrieval_context import select_aligned_memory_frames

    self.init_kv_cache()

    stabilization_level = STABILIZATION_LEVEL
    positive_idx = 1 if self.do_classifier_free_guidance else 0

    spatial_ratio = self.vae_spatial_compression_ratio
    pixel_h = latents.shape[3] * spatial_ratio
    pixel_w = latents.shape[4] * spatial_ratio

    chunk_to_idx        = self._recache_chunk_to_idx
    injection_positions = self._recache_injection_positions

    # ── text KV cache init ──
    with (
        torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled),
        auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading),
    ):
        extra_kwargs_pos = {
            "byt5_text_states": extra_kwargs["byt5_text_states"][positive_idx, None, ...],
            "byt5_text_mask":   extra_kwargs["byt5_text_mask"][positive_idx, None, ...],
        }
        t_expand_txt = torch.tensor([0]).to(device).to(latents.dtype)
        self._kv_cache = self.transformer(
            bi_inference=False, ar_txt_inference=True, ar_vision_inference=False,
            timestep_txt=t_expand_txt,
            text_states=prompt_embeds[positive_idx, None, ...],
            encoder_attention_mask=prompt_mask[positive_idx, None, ...],
            vision_states=vision_states[positive_idx, None, ...],
            mask_type=task_type,
            extra_kwargs=extra_kwargs_pos,
            kv_cache=self._kv_cache, cache_txt=True,
        )
        if self.do_classifier_free_guidance:
            extra_kwargs_neg = {
                "byt5_text_states": extra_kwargs["byt5_text_states"][0, None, ...],
                "byt5_text_mask":   extra_kwargs["byt5_text_mask"][0, None, ...],
            }
            self._kv_cache_neg = self.transformer(
                bi_inference=False, ar_txt_inference=True, ar_vision_inference=False,
                timestep_txt=t_expand_txt,
                text_states=prompt_embeds[0, None, ...],
                encoder_attention_mask=prompt_mask[0, None, ...],
                vision_states=vision_states[0, None, ...],
                mask_type=task_type,
                extra_kwargs=extra_kwargs_neg,
                kv_cache=self._kv_cache_neg, cache_txt=True,
            )

    selected_frame_indices = []

    for chunk_i in range(self.chunk_num):

        # ── inject decision ──
        schedule_idx = chunk_to_idx.get(chunk_i)
        should_inject = schedule_idx is not None and chunk_i > 0
        context_len = len(selected_frame_indices)

        if should_inject:
            slots = injection_positions[schedule_idx]
            entity_path = self._recache_entity_image_paths[schedule_idx]

            self._recache_with_entity(
                latents=latents, cond_latents=cond_latents,
                current_start_frame=chunk_i * self.chunk_latent_frames,
                positions=slots,
                entity_image_path=entity_path,
                prompt_embeds=prompt_embeds, prompt_mask=prompt_mask,
                vision_states=vision_states, task_type=task_type,
                extra_kwargs=extra_kwargs,
                viewmats=viewmats, Ks=Ks, action=action,
                device=device, height=pixel_h, width=pixel_w,
            )

        if chunk_i > 0:
            current_frame_idx = chunk_i * self.chunk_latent_frames
            selected_frame_indices = []

            for chunk_start_idx in range(
                current_frame_idx, current_frame_idx + self.chunk_latent_frames, 4
            ):
                selected_history_frame_id = select_aligned_memory_frames(
                    viewmats[0].cpu().detach().numpy(),
                    chunk_start_idx,
                    memory_frames=20, temporal_context_size=12,
                    pred_latent_size=4, points_local=self.points_local, device=device,
                )
                selected_frame_indices += selected_history_frame_id

            selected_frame_indices = sorted(list(set(selected_frame_indices)))
            to_remove = list(range(current_frame_idx, current_frame_idx + self.chunk_latent_frames))
            selected_frame_indices = [x for x in selected_frame_indices if x not in to_remove]

            context_latents_input = torch.concat([
                latents[:, :, selected_frame_indices],
                cond_latents[:, :, selected_frame_indices],
            ], dim=1)
            context_viewmats = viewmats[:, selected_frame_indices].to(device)
            context_Ks       = Ks[:, selected_frame_indices].to(device)
            context_action   = action[:, selected_frame_indices].to(device)

            context_len = len(selected_frame_indices)

            context_timestep = torch.full(
                (context_latents_input.shape[2],), stabilization_level - 1,
                device=device, dtype=timesteps.dtype,
            )

            with (
                torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled),
                auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading),
            ):
                if not should_inject:
                    self._kv_cache = self.transformer(
                        bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                        hidden_states=context_latents_input,
                        timestep=context_timestep, timestep_r=None,
                        mask_type=task_type, return_dict=False,
                        viewmats=context_viewmats.to(self.target_dtype),
                        Ks=context_Ks.to(self.target_dtype),
                        action=context_action.to(self.target_dtype),
                        kv_cache=self._kv_cache, cache_vision=True,
                        rope_temporal_size=context_latents_input.shape[2],
                        start_rope_start_idx=0,
                    )
                    if self.do_classifier_free_guidance:
                        self._kv_cache_neg = self.transformer(
                            bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                            hidden_states=context_latents_input,
                            timestep=context_timestep, timestep_r=None,
                            mask_type=task_type, return_dict=False,
                            viewmats=context_viewmats.to(self.target_dtype),
                            Ks=context_Ks.to(self.target_dtype),
                            action=context_action.to(self.target_dtype),
                            kv_cache=self._kv_cache_neg, cache_vision=True,
                            rope_temporal_size=context_latents_input.shape[2],
                            start_rope_start_idx=0,
                        )

            self.scheduler.set_timesteps(self.num_inference_steps, device=device)

        start_idx = chunk_i * self.chunk_latent_frames
        end_idx   = start_idx + self.chunk_latent_frames

        with (
            self.progress_bar(total=self.num_inference_steps) as progress_bar,
            auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading),
        ):
            for i, t in enumerate(timesteps):
                timestep_input     = torch.full((self.chunk_latent_frames,), t, device=device, dtype=timesteps.dtype)
                latent_model_input = latents[:, :, start_idx:end_idx]
                cond_latents_input = cond_latents[:, :, start_idx:end_idx]
                viewmats_input     = viewmats[:, start_idx:end_idx].to(device)
                Ks_input           = Ks[:, start_idx:end_idx].to(device)
                action_input       = action[:, start_idx:end_idx].to(device)

                latents_concat = torch.concat([latent_model_input, cond_latents_input], dim=1)
                latents_concat = self.scheduler.scale_model_input(latents_concat, t)

                rope_total = latents_concat.shape[2] + context_len

                with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                    noise_pred = self.transformer(
                        bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                        hidden_states=latents_concat, timestep=timestep_input, timestep_r=None,
                        mask_type=task_type, return_dict=False,
                        viewmats=viewmats_input.to(self.target_dtype),
                        Ks=Ks_input.to(self.target_dtype),
                        action=action_input.to(self.target_dtype),
                        kv_cache=self._kv_cache, cache_vision=False,
                        rope_temporal_size=rope_total,
                        start_rope_start_idx=context_len,
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond = self.transformer(
                            bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                            hidden_states=latents_concat, timestep=timestep_input, timestep_r=None,
                            mask_type=task_type, return_dict=False,
                            viewmats=viewmats_input.to(self.target_dtype),
                            Ks=Ks_input.to(self.target_dtype),
                            action=action_input.to(self.target_dtype),
                            kv_cache=self._kv_cache_neg, cache_vision=False,
                            rope_temporal_size=rope_total,
                            start_rope_start_idx=context_len,
                        )[0]

                if self.do_classifier_free_guidance:
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred - noise_pred_uncond
                    )

                latent_model_input = self.scheduler.step(
                    noise_pred, t, latent_model_input, return_dict=False
                )[0]
                latents[:, :, start_idx:end_idx] = latent_model_input[:, :, -self.chunk_latent_frames:]

                if i == len(timesteps) - 1 or (
                    (i + 1) > self.num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()

    return latents
