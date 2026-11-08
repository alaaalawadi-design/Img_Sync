#!/usr/bin/env python
"""LongCat-Video-Avatar-1.5 persistent worker for a single RTX 5090 (32 GB).

* Runs the ORIGINAL meituan-longcat code path (LongCatVideoAvatarPipeline.generate_ai2v for segment 1,
  generate_avc with KV cache for every following segment) with the exact settings of the official
  run_demo_avatar_single_audio_to_video.py (seed 42, 93 frames, 13 cond frames, 25 fps,
  ref_img_index=10, mask_frame_range=3, distill => 8 steps / guidance 1.0, official negative prompt).
* The number of segments is derived from the audio length, so the whole audio is generated.
* Models stay loaded in RAM (and the active DiT in VRAM) between jobs.
* Only lossless placement changes are made to fit 32 GB (see dit_streaming.py):
    - T5 text encoder / Whisper live in RAM and visit the GPU only while they run
      (prompt embeddings are memoised - same prompt -> same tensors).
    - BF16 DiT: blocks that do not fit stay in pinned RAM and are streamed in just in time.
    - KV cache of the conditioning frames: the original `offload_kv_cache` option.
* Every finished segment is written immediately (mp4 + its audio slice) and a resume checkpoint
  (latents + generator RNG state) is kept, so a crash never loses finished work and a resumed job
  is bit-identical to an uninterrupted one.

Usage:
  worker (queue mode):  python avatar_worker.py --serve [--preload int8,bf16]
  one shot:             python avatar_worker.py --job job.json
  submit to the worker: python submit_job.py --audio a.wav --image i.png --quant int8 --distill
"""
import gc
import os
import sys
import json
import math
import time
import glob
import shutil
import random
import argparse
import subprocess
import traceback
import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("LONGCAT_REPO", HERE / "LongCat-Video"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import PIL.Image
import librosa
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.utils import load_image

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.modules.quantization import load_quantized_dit
from longcat_video.context_parallel import context_parallel_util
from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from audio_separator.separator import Separator

from dit_streaming import BlockStreamer
from crop_paste import Compositor, PersonSegmenter

# ---------------------------------------------------------------- official constants (avatar-v1.5)
NUM_FRAMES = 93
NUM_COND_FRAMES = 13
SAVE_FPS = 25
AUDIO_STRIDE = 1
REF_IMG_INDEX = 10
MASK_FRAME_RANGE = 3
GLOBAL_SEED = 42
NEGATIVE_PROMPT = "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

DEFAULT_PROMPT = ("A TV presenter talking to the camera. Do not change anything in the background. "
                  "Preserve the identity of the person. The head and eyes are naturally focused on the camera. "
                  "Realistic facial features.")

DEFAULT_CKPT = str(REPO / "weights" / "LongCat-Video-Avatar-1.5")

# VRAM layout defaults per DiT precision (tuned for 32 GB; OOM => automatic, lossless fallback)
LAYOUT = {
    "int8": {"resident_blocks": 0, "offload_kv_cache": True},
    "bf16": {"resident_blocks": 0, "offload_kv_cache": True},
}
CAPS_FILE = HERE / "jobs" / "layout_caps.json"


def load_caps():
    try:
        return json.loads(CAPS_FILE.read_text())
    except Exception:
        return {}
# after the first segments the layout is re-tuned from the measured activation peak so that as many
# blocks as possible stay resident while total allocated VRAM stays under this budget
VRAM_BUDGET_GIB = float(os.environ.get("LONGCAT_VRAM_BUDGET_GIB") or "29.3")  # lower it on shared GPUs
AUTOTUNE_SEGMENTS = 3


import threading
import psutil


class RamMonitor:
    """Samples this process' RSS (incl. pinned/shared) and system RAM use every second; keeps peaks."""
    def __init__(self):
        self.proc = psutil.Process()
        self.reset()
        threading.Thread(target=self._run, daemon=True).start()

    def reset(self):
        self.peak_rss = 0
        self.peak_sys = 0

    def _run(self):
        while True:
            try:
                self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
                vm = psutil.virtual_memory()
                self.peak_sys = max(self.peak_sys, vm.total - vm.available)
            except Exception:
                pass
            time.sleep(1)


RAM = RamMonitor()


def log(*a):
    print(f"[{datetime.datetime.now():%H:%M:%S}]", *a, flush=True)


def gpu_gib():
    return torch.cuda.max_memory_allocated() / 2**30


def num_segments_for(duration_s):
    first = NUM_FRAMES / SAVE_FPS
    step = (NUM_FRAMES - NUM_COND_FRAMES) / SAVE_FPS
    if duration_s <= first:
        return 1
    return 1 + math.ceil((duration_s - first) / step - 1e-9)


def seg_time_window(k):
    """(start_s, dur_s) of the NEW frames contributed by segment k (1-based)."""
    if k == 1:
        return 0.0, NUM_FRAMES / SAVE_FPS
    return (NUM_FRAMES + (k - 2) * (NUM_FRAMES - NUM_COND_FRAMES)) / SAVE_FPS, (NUM_FRAMES - NUM_COND_FRAMES) / SAVE_FPS


def write_mp4(frames_u8, path, fps=SAVE_FPS, audio=None, a_start=0.0, a_dur=None, crf=12):
    """frames_u8: [T,H,W,3] uint8. Visually lossless H.264 (crf 12) + optional AAC audio slice."""
    T, H, W, _ = frames_u8.shape
    tmp = str(path) + ".part.mp4"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0"]
    if audio:
        cmd += ["-ss", f"{a_start:.4f}", "-t", f"{(a_dur or T / fps):.4f}", "-i", audio,
                "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames_u8).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    os.replace(tmp, path)


def smooth_lower_face(frames, sigma):
    """Conditioning-only cleanup against texture drift: Gaussian-smooth the lower face (cheeks, beard, chin) of the
    13 overlap frames handed to the next segment. Face box from OpenCV's Haar detector on the last frame."""
    import cv2
    if sigma <= 0 or not frames:
        return frames
    arr = [np.asarray(f) for f in frames]
    det = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    g = cv2.cvtColor(arr[-1], cv2.COLOR_RGB2GRAY)
    faces = det.detectMultiScale(g, 1.1, 5, minSize=(g.shape[0] // 10, g.shape[0] // 10))
    if len(faces) == 0:
        return frames
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    H, W = g.shape
    m = np.zeros((H, W), np.float32)
    x0, x1 = max(0, int(x - 0.12 * w)), min(W, int(x + 1.12 * w))
    y0, y1 = max(0, int(y + 0.45 * h)), min(H, int(y + 1.30 * h))
    m[y0:y1, x0:x1] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), 0.08 * w)[..., None]
    out = []
    for a in arr:
        b = cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma)
        out.append(PIL.Image.fromarray(np.clip(a * (1 - m) + b * m + 0.5, 0, 255).astype(np.uint8)))
    return out


class AvatarWorker:
    def __init__(self, ckpt=DEFAULT_CKPT, device="cuda:0", preload=()):
        self.ckpt = ckpt
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        base = os.path.join(ckpt, "..", "LongCat-Video")
        t0 = time.time()
        # ---- exactly the official loaders ----
        self.tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer", torch_dtype=torch.bfloat16)
        self.text_encoder = UMT5EncoderModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.bfloat16)  # RAM
        self.vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae", torch_dtype=torch.bfloat16).to(self.device)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ckpt, subfolder="scheduler", torch_dtype=torch.bfloat16)
        wpath = os.path.join(ckpt, "whisper-large-v3")
        self.audio_encoder = get_audio_encoder(wpath, "avatar-v1.5")  # RAM, visits GPU for embedding
        self.audio_feature_extractor = get_audio_feature_extractor(wpath, "avatar-v1.5")
        self.tmp_root = HERE / "tmp"
        self.tmp_root.mkdir(exist_ok=True)
        self.audio_tmp = self.tmp_root / "audio_temp_file"
        self.audio_tmp.mkdir(exist_ok=True)
        sep_path = os.path.join(ckpt, "vocal_separator/Kim_Vocal_2.onnx")
        self.vocal_separator = Separator(output_dir=self.audio_tmp / "vocals", output_single_stem="vocals",
                                         model_file_dir=os.path.dirname(sep_path))
        self.vocal_separator.load_model(os.path.basename(sep_path))
        self.dits = {}
        self.active = None  # dict(key, dit, streamer, pipe)
        self.prompt_cache = {}
        self.dit_peak = 0
        self.vae_peak = 0
        self.lora_multiplier = 1.0
        self.use_vcond = False
        self._wrap_vae()
        for q in preload:
            self.get_dit(q)
        log(f"worker ready in {time.time() - t0:.0f}s (preloaded: {list(self.dits)})")

    def _wrap_vae(self):
        """The ORIGINAL vae.encode/decode, run with the DiT's resident blocks parked in pinned RAM
        (720p untiled decode needs a lot of VRAM). Also records per-phase VRAM peaks for auto-tuning."""
        vae = self.vae
        for name in ("encode", "decode"):
            orig = getattr(vae, name)

            def wrapped(*args, _orig=orig, **kw):
                self.dit_peak = max(self.dit_peak, torch.cuda.max_memory_allocated())
                st = self.active["streamer"] if self.active else None
                if st:
                    st.park()
                torch.cuda.reset_peak_memory_stats()
                try:
                    return _orig(*args, **kw)
                finally:
                    self.vae_peak = max(self.vae_peak, torch.cuda.max_memory_allocated())
                    if st:
                        st.unpark()
                    torch.cuda.reset_peak_memory_stats()

            setattr(vae, name, wrapped)

    # ------------------------------------------------------------ models
    def get_dit(self, quant):
        if quant not in self.dits:
            t0 = time.time()
            split = context_parallel_util.get_optimal_split(1)
            if quant == "int8":
                dit = load_quantized_dit(self.ckpt, subfolder="base_model_int8", cp_split_hw=split)
            elif quant == "bf16":
                dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
                    self.ckpt, subfolder="base_model", cp_split_hw=split, torch_dtype=torch.bfloat16)
            else:
                raise ValueError(f"quant must be int8 or bf16, got {quant}")
            lora = os.path.join(self.ckpt, "lora", "dmd_lora.safetensors")
            dit.load_lora(lora, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
            self.dits[quant] = dit
            log(f"loaded DiT {quant} into RAM in {time.time() - t0:.0f}s")
        return self.dits[quant]

    def _apply_lora_multiplier(self):
        """Per-job settings on the active pipeline:
        LoRA strength via the original LoRANetwork.set_multiplier; continuation mode via the original `use_vcond`."""
        a = self.active
        if a is not None and a["distill"]:
            a["dit"].lora_dict["dmd"].set_multiplier(self.lora_multiplier)
        if a is not None:
            a["pipe"].use_vcond = self.use_vcond

    def deactivate(self):
        if self.active is None:
            return
        a = self.active
        a["streamer"].close()
        if a["distill"]:
            for lora in a["dit"].lora_dict["dmd"].loras:
                lora.to("cpu")
        self.active = None
        gc.collect()
        torch.cuda.empty_cache()

    def activate(self, quant, distill, resident_blocks, offload_kv_cache):
        key = (quant, distill, resident_blocks)
        if self.active is not None and self.active["key"] == key:
            self.active["offload_kv_cache"] = offload_kv_cache
            self._apply_lora_multiplier()
            return self.active
        self.deactivate()
        for q in [q for q in self.dits if q != quant]:  # only the active DiT stays in RAM (both = >122 GB peak)
            del self.dits[q]
            gc.collect()
            log(f"released DiT {q} from RAM")
        dit = self.get_dit(quant)
        # official: load_lora + enable_loras on the CPU model, then pipe.to(gpu) (moves dit + loras)
        if distill:
            dit.enable_loras(["dmd"])
        else:
            dit.disable_all_loras()
        streamer = BlockStreamer(dit, self.device, resident_blocks)
        if distill:
            for lora in dit.lora_dict["dmd"].loras:
                lora.to(self.device, non_blocking=True)
        pipe = LongCatVideoAvatarPipeline(
            tokenizer=self.tokenizer, text_encoder=self.text_encoder, vae=self.vae, scheduler=self.scheduler,
            dit=dit, audio_encoder=self.audio_encoder, audio_feature_extractor=self.audio_feature_extractor,
            model_type="avatar-v1.5")
        pipe.device = self.device  # what pipe.to() sets; T5/Whisper are placed on demand instead
        self._split_cfg_batch(dit)
        self._patch_encode_prompt(pipe)
        torch.cuda.synchronize()
        self.active = dict(key=key, quant=quant, distill=distill, dit=dit, streamer=streamer, pipe=pipe,
                           resident_blocks=resident_blocks, offload_kv_cache=offload_kv_cache,
                           base_alloc=torch.cuda.memory_allocated(), block_bytes=streamer.block_bytes)
        self._apply_lora_multiplier()
        log(f"active DiT={quant} distill={distill} lora_multiplier={self.lora_multiplier if distill else None} use_vcond={self.use_vcond} resident_blocks={resident_blocks}/48 "
            f"offload_kv_cache={offload_kv_cache} | VRAM allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
        return self.active

    @staticmethod
    def _split_cfg_batch(dit):
        """CFG (non-distill) sends a batch of 2 [negative-text, positive] through the DiT, which needs 2x activation
        memory at 720p. Run the two samples one after another instead - exactly what the original pipeline already
        does in its multi-person CFG branch (self.dit(...[:1]) / self.dit(...[1:]) + torch.cat). The KV cache is
        built for one sample and repeated per batch inside attention, so per-sample calls see the same context."""
        if getattr(dit, "_lc_split", False):
            return
        orig = dit.forward
        keys = ("hidden_states", "timestep", "encoder_hidden_states", "encoder_attention_mask", "audio_embs")

        def forward(*args, **kw):
            hs = kw.get("hidden_states")
            if args or hs is None or hs.shape[0] < 2 or kw.get("return_kv"):
                return orig(*args, **kw)
            B = hs.shape[0]
            outs = []
            for i in range(B):
                kwi = {k: (v[i:i + 1] if k in keys and torch.is_tensor(v) and v.shape[0] == B else v) for k, v in kw.items()}
                outs.append(orig(**kwi))
            return torch.cat(outs, 0)

        dit.forward = forward
        dit._lc_split = True

    def _patch_encode_prompt(self, pipe):
        """Memoised, on-demand-GPU wrapper around the ORIGINAL encode_prompt (identical tensors)."""
        orig = pipe.encode_prompt
        worker = self

        def encode_prompt(prompt, negative_prompt=None, do_classifier_free_guidance=True,
                          num_videos_per_prompt=1, max_sequence_length=512, device=None, dtype=None):
            k = (json.dumps(prompt), json.dumps(negative_prompt), bool(do_classifier_free_guidance),
                 num_videos_per_prompt, max_sequence_length, str(device), str(dtype))
            if k not in worker.prompt_cache:
                worker._with_gpu_room(lambda: worker.text_encoder.to(device))
                try:
                    with torch.no_grad():  # as in the original flow (always called inside @torch.no_grad generate_*)
                        out = orig(prompt=prompt, negative_prompt=negative_prompt,
                                   do_classifier_free_guidance=do_classifier_free_guidance,
                                   num_videos_per_prompt=num_videos_per_prompt,
                                   max_sequence_length=max_sequence_length, device=device, dtype=dtype)
                    out = tuple(t.detach() if torch.is_tensor(t) else t for t in out)
                finally:
                    worker.text_encoder.to("cpu")
                    torch.cuda.empty_cache()
                    worker._restore_room()
                worker.prompt_cache[k] = out
            return tuple(t.clone() if torch.is_tensor(t) else t for t in worker.prompt_cache[k])

        pipe.encode_prompt = encode_prompt

    def _with_gpu_room(self, fn):
        # T5 (11.4 GB) / Whisper (6 GB) need room: park the resident DiT part in RAM while they run
        if self.active is not None:
            self.active["streamer"].park()
        fn()

    def _restore_room(self):
        if self.active is not None:
            self.active["streamer"].unpark()
        torch.cuda.reset_peak_memory_stats()

    def autotune(self, act_peak_bytes):
        """Most resident blocks that fit: fixed + rb*block + activations <= budget (placement only, lossless)."""
        a = self.active
        blk = a["block_bytes"]
        fixed = a["base_alloc"] - a["resident_blocks"] * blk
        budget = VRAM_BUDGET_GIB * 2**30
        rb = int((budget - fixed - act_peak_bytes) // blk)
        return max(0, min(48, rb))

    # ------------------------------------------------------------ audio
    def extract_vocal(self, src, dst):
        outputs = self.vocal_separator.separate(src)
        if len(outputs) <= 0:
            return None
        default_vocal_path = (self.audio_tmp / "vocals" / outputs[0]).resolve().as_posix()
        shutil.move(default_vocal_path, dst)
        return dst

    def audio_embedding(self, pipe, speech_array, sr):
        self._with_gpu_room(lambda: self.audio_encoder.to(self.device))
        try:
            emb = pipe.get_audio_embedding(speech_array, fps=SAVE_FPS * AUDIO_STRIDE, device=self.device,
                                           sample_rate=sr, model_type="avatar-v1.5")
        finally:
            self.audio_encoder.to("cpu")
            torch.cuda.empty_cache()
            self._restore_room()
        return emb

    # ------------------------------------------------------------ job
    @torch.no_grad()
    def run_job(self, job):
        job = dict(job)
        quant = job.get("quant", "int8")
        distill = bool(job.get("distill", True))
        resolution = job.get("resolution", "720p")
        prompt = job.get("prompt") or DEFAULT_PROMPT
        neg_prompt = job.get("negative_prompt") or NEGATIVE_PROMPT  # only used when CFG is active (non-distill)
        image_path, raw_speech_path = job["image"], job["audio"]
        crop_box = [int(v) for v in job["crop_box"]] if job.get("crop_box") else None  # crop -> generate -> paste back
        seed = int(job.get("seed", GLOBAL_SEED))
        lora_mult = float(job.get("lora_multiplier", 1.0)) if distill else None  # official default 1.0
        self.lora_multiplier = lora_mult if lora_mult is not None else 1.0
        # True = condition each continuation segment on the previous segment's latents (original pipeline attribute
        # `use_vcond`, the v1.0 behaviour); False (default, official v1.5) = re-encode the last decoded frames
        latent_cont = bool(job.get("latent_continuation", False))
        # re-anchor: every N segments start a fresh image->video chain from the ORIGINAL image (official generate_ai2v)
        # at the current audio position, crossfading its 13 overlapping frames into the previous segment
        reanchor = int(job.get("reanchor_every") or 0)
        # start_segment > 1: segments 1..start-1 are copied from `segments_from` (another run of the same inputs) and
        # generation starts at start_segment with a fresh chain from the original image; re-anchors count from there
        start_segment = int(job.get("start_segment") or 1)
        start_mode = job.get("start_mode", "fresh")  # fresh: new chain from the image; continue: from the previous segment's 13 frames
        # LoRA schedule (distill only): strength = max(min, start - step * ((seg - from_segment) // every))
        lsched = job.get("lora_schedule")  # dict(start, step, every, min, from_segment)
        cond_smooth = float(job.get("cond_face_smooth") or 0)  # sigma (px) for the lower-face smoothing of the 13 overlap frames
        if (start_mode == "continue" or cond_smooth > 0) and latent_cont:
            latent_cont = False  # both need conditioning from the 13 real frames (official v1.5 re-encode path)
            self.use_vcond = False
        xfade = max(0, min(NUM_COND_FRAMES, int(job.get("reanchor_crossfade", NUM_COND_FRAMES))))  # 0 = hard cut
        self.use_vcond = latent_cont
        layout = dict(LAYOUT[quant])
        layout["resident_blocks"] = 0  # segment 1 (image->video) needs the most VRAM: start safe, autotune raises it after
        layout.update({k: job[k] for k in ("resident_blocks", "offload_kv_cache") if k in job})

        if distill:  # official: forced for avatar-v1.5 + distill
            steps, tg, ag = 8, 1.0, 1.0
        else:
            steps = int(job.get("num_inference_steps", 50))
            tg = float(job.get("text_guidance_scale", 4.0))
            ag = float(job.get("audio_guidance_scale", 4.0))

        duration = librosa.get_duration(path=raw_speech_path)
        nseg = int(job.get("num_segments") or num_segments_for(duration))
        name = job.get("name") or (f"{Path(image_path).stem}_{quant}_{'distill' if distill else 'base'}"
                                   f"{'' if not distill or lora_mult == 1.0 else f'_lora{lora_mult:g}'}"
                                   f"{'_latentcont' if latent_cont else ''}_{resolution}")
        out_dir = Path(job.get("output_dir") or (HERE / "outputs" / name))
        seg_dir = out_dir / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)
        comp, gen_image_path = None, image_path
        if crop_box:
            comp = Compositor(image_path, crop_box, resolution)
            gen_image_path = str(out_dir / "crop_input.png")
            comp.crop_image(crop_box).save(gen_image_path)
        paste_mode = job.get("paste_mode", "rect")  # rect: paste the whole crop; person: paste only the person
        paste_kw = dict(dilate_px=int(job.get("paste_dilate", 10)), feather_px=int(job.get("paste_feather", 12)))
        segm = None
        if comp is not None and paste_mode == "person":
            if getattr(self, "_segmenter", None) is None:
                self._segmenter = PersonSegmenter(self.device)
            segm = self._segmenter
            comp.set_reference_person(segm(comp.bg[None])[0])
        if comp is not None:
            (out_dir / "segments_crop").mkdir(exist_ok=True)
        state_path = out_dir / "resume_state.pt"
        cfg = dict(name=name, quant=quant, distill=distill, lora_multiplier=lora_mult, latent_continuation=latent_cont, reanchor_every=reanchor, reanchor_crossfade=xfade if reanchor else None, start_segment=start_segment, lora_schedule=lsched, start_mode=start_mode if start_segment > 1 else None, cond_face_smooth=cond_smooth, segments_from=job.get("segments_from"), crop_box=crop_box, paste_mode=job.get("paste_mode", "rect") if crop_box else None,
                   crop_generation_size=[comp.gw, comp.gh] if comp else None, output_size=[comp.Wc, comp.Hc] if comp else None,
                   resolution=resolution, prompt=prompt,
                   negative_prompt=neg_prompt, image=image_path, audio=raw_speech_path, seed=seed,
                   audio_duration_s=duration, num_segments=nseg, num_inference_steps=steps,
                   text_guidance_scale=tg, audio_guidance_scale=ag, num_frames=NUM_FRAMES,
                   num_cond_frames=NUM_COND_FRAMES, fps=SAVE_FPS, ref_img_index=REF_IMG_INDEX,
                   mask_frame_range=MASK_FRAME_RANGE, layout=layout, repo=str(REPO), ckpt=self.ckpt)
        (out_dir / "job_config.json").write_text(json.dumps(cfg, indent=2))
        shutil.copy(image_path, out_dir / ("input_image" + Path(image_path).suffix))
        log(f"JOB {name}: audio {duration:.2f}s -> {nseg} segments, {resolution}, DiT={quant}, "
            f"distill={distill}, lora_multiplier={lora_mult}, latent_continuation={latent_cont}, steps={steps}, guidance=({tg},{ag})")

        SIG_KEYS = ("quant", "distill", "resolution", "prompt", "seed", "num_segments", "image", "audio") + \
            (("lora_multiplier",) if distill and lora_mult != 1.0 else ()) + (("latent_continuation",) if latent_cont else ()) + (("crop_box",) if crop_box else ()) + (("reanchor_every",) if reanchor else ()) + (("start_segment",) if start_segment > 1 else ()) + (("lora_schedule",) if lsched else ())
        a = self.activate(quant, distill, layout["resident_blocks"], layout["offload_kv_cache"])
        pipe = a["pipe"]

        # ---- audio: vocal separation + padding + whisper embedding (official) ----
        # drive_audio = what the model listens to (default: the job audio); the job audio is always the soundtrack
        drive_path = job.get("drive_audio") or raw_speech_path
        temp_vocal = str(self.tmp_root / f"temp_speech_{int(time.time())}_{random.randint(100000, 999999)}_vocal.wav")
        if job.get("skip_vocal_separation"):  # e.g. a pure-silence drive: nothing to separate
            shutil.copy(drive_path, temp_vocal)
        else:
            temp_vocal = self.extract_vocal(drive_path, temp_vocal)
        log(f"VRAM after vocal separation: allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB, reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB, device used {(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/2**30:.2f} GiB")
        assert temp_vocal is not None and os.path.exists(temp_vocal), "No vocal detected"
        generate_duration = NUM_FRAMES / SAVE_FPS + (nseg - 1) * (NUM_FRAMES - NUM_COND_FRAMES) / SAVE_FPS
        speech_array, sr = librosa.load(temp_vocal, sr=16000)
        added = math.ceil((generate_duration - len(speech_array) / sr) * sr)
        if added > 0:
            speech_array = np.append(speech_array, [0.] * added)
        full_audio_emb = self.audio_embedding(pipe, speech_array, sr)
        os.remove(temp_vocal)
        log(f"VRAM after whisper embedding: allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB, reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB, device used {(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/2**30:.2f} GiB")
        if torch.isnan(full_audio_emb).any():
            raise ValueError("broken audio embedding with nan values")

        # ---- pre-warm the prompt cache with the exact args the pipeline will use ----
        pipe._text_guidance_scale, pipe._audio_guidance_scale = tg, ag
        pipe.encode_prompt(prompt=prompt, negative_prompt=neg_prompt,
                           do_classifier_free_guidance=pipe.do_classifier_free_guidance,
                           num_videos_per_prompt=1, max_sequence_length=512,
                           device=self.device, dtype=pipe.dit.dtype)

        log(f"VRAM after prompt encoding: allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB, reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB, device used {(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/2**30:.2f} GiB")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        indices = torch.arange(2 * 2 + 1) - 2

        def audio_window(start_idx):
            end_idx = start_idx + AUDIO_STRIDE * NUM_FRAMES
            ci = torch.arange(start_idx, end_idx, AUDIO_STRIDE).unsqueeze(1) + indices.unsqueeze(0)
            ci = torch.clamp(ci, min=0, max=full_audio_emb.shape[0] - 1)
            return full_audio_emb[ci][None, ...].to(self.device)

        # ---- continue from another (shorter) run of the same inputs: copy its segments + state once ----
        src = job.get("resume_from")
        if src and not state_path.exists() and Path(src, "resume_state.pt").exists():
            src = Path(src)
            for sub in ("segments", "segments_crop"):
                if (src / sub).exists():
                    (out_dir / sub).mkdir(exist_ok=True)
                    for f in sorted((src / sub).glob("seg_*.mp4")):
                        shutil.copy(f, out_dir / sub / f.name)
            st = torch.load(src / "resume_state.pt", map_location="cpu", weights_only=False)
            st["cfg_sig"] = json.dumps({k: cfg[k] for k in SIG_KEYS})
            torch.save(st, state_path)
            log(f"continuing from {src.name}: copied segments 1..{st['next_segment'] - 1} and its state")
        # ---- resume ----
        start_seg, current_video, latent, ref_latent, audio_start_idx = 1, None, None, None, 0
        if state_path.exists() and job.get("resume", True):
            st = torch.load(state_path, map_location="cpu", weights_only=False)
            if st["cfg_sig"] == json.dumps({k: cfg[k] for k in SIG_KEYS}):
                start_seg = st["next_segment"]
                current_video = [PIL.Image.fromarray(f) for f in st["current_video"]]
                latent, ref_latent = st["latent"].to(self.device), st["ref_latent"].to(self.device)
                audio_start_idx = st["audio_start_idx"]
                generator.set_state(st["generator_state"])
                log(f"resuming at segment {start_seg}/{nseg}")
        cfg_sig = json.dumps({k: cfg[k] for k in SIG_KEYS})
        prev_seg = None
        if start_seg == 1 and start_segment > 1:
            src = Path(job["segments_from"])
            for sub in ("segments", "segments_crop"):
                if (src / sub).exists():
                    (out_dir / sub).mkdir(exist_ok=True)
                    for k in range(1, start_segment):
                        shutil.copy(src / sub / f"seg_{k:03d}.mp4", out_dir / sub / f"seg_{k:03d}.mp4")
            start_seg = start_segment
            audio_start_idx = (start_segment - 2) * AUDIO_STRIDE * (NUM_FRAMES - NUM_COND_FRAMES) if start_segment > 2 else 0
            # previous segment's frames (in generation space) for the crossfade into the fresh chain
            prev_file = out_dir / ("segments_crop" if comp is not None else "segments") / f"seg_{start_segment - 1:03d}.mp4"
            import cv2
            cap, fr = cv2.VideoCapture(str(prev_file)), []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                fr.append(f[:, :, ::-1])
            pa, pd = seg_time_window(start_segment - 1)
            if start_mode == "continue":
                # continue the chain from the previous segment's frames (its last 13 are the conditioning overlap);
                # the latent is only a placeholder (conditioning is re-encoded from these frames), the reference
                # latent is the original image encoded exactly like generate_ai2v does
                current_video = [PIL.Image.fromarray(f) for f in fr]
                gw, gh = current_video[0].size
                latent = torch.zeros(1, 16, (NUM_FRAMES - 1) // 4 + 1, gh // 8, gw // 8, device=self.device, dtype=torch.bfloat16)
                img = pipe.video_processor.preprocess(load_image(gen_image_path), height=gh, width=gw, resize_mode="crop")
                img = img.to(self.device, torch.bfloat16).unsqueeze(2)
                from longcat_video.pipeline_longcat_video_avatar import retrieve_latents
                ref_latent = pipe.normalize_latents(retrieve_latents(self.vae.encode(img), None, sample_mode="argmax")).to(torch.bfloat16)
                log(f"start_segment={start_segment}: copied segments 1..{start_segment - 1} from {src.name}; continuing the chain from segment {start_segment - 1}'s last 13 frames (no crossfade)")
            else:
                prev_seg = dict(seg=start_segment - 1, frames=np.ascontiguousarray(np.stack(fr)), a_start=pa, a_dur=pd)
                log(f"start_segment={start_segment}: copied segments 1..{start_segment - 1} from {src.name}; fresh chain from the image at segment {start_segment}")

        t_job = time.time()
        timings = []
        max_act = 0
        rb_cap = load_caps().get(quant, 48)
        for seg in range(start_seg, nseg + 1):
            g_state = generator.get_state()
            if lsched and distill:
                f0 = int(lsched.get("from_segment", start_segment))
                k = max(0, seg - f0) // max(1, int(lsched.get("every", 1)))
                m = max(float(lsched.get("min", 0.0)), round(float(lsched["start"]) - float(lsched["step"]) * k, 4))
                if m != self.lora_multiplier:
                    self.lora_multiplier = m
                    self._apply_lora_multiplier()
                    log(f"segment {seg}: LoRA multiplier -> {m}")
            for attempt in range(1, 10):
                self.dit_peak = self.vae_peak = 0
                RAM.reset()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                oom = None
                try:
                    base = start_segment if start_segment > 1 else 1
                    is_anchor = seg > 1 and ((seg == start_segment and start_mode == "fresh") or (reanchor > 0 and seg > base and (seg - base) % reanchor == 0))
                    if seg == 1 or is_anchor:
                        new_start = 0 if seg == 1 else audio_start_idx + AUDIO_STRIDE * (NUM_FRAMES - NUM_COND_FRAMES)
                        audio_emb = audio_window(new_start)
                        output, latent_new = pipe.generate_ai2v(
                            image=load_image(gen_image_path), prompt=prompt, negative_prompt=neg_prompt,
                            resolution=resolution, num_frames=NUM_FRAMES, num_inference_steps=steps,
                            text_guidance_scale=tg, audio_guidance_scale=ag, output_type="both",
                            generator=generator, audio_emb=audio_emb, use_distill=distill)
                    else:
                        new_start = audio_start_idx + AUDIO_STRIDE * (NUM_FRAMES - NUM_COND_FRAMES)
                        audio_emb = audio_window(new_start)
                        width, height = current_video[0].size
                        cond_video = current_video
                        if cond_smooth > 0:
                            cond_video = current_video[:-NUM_COND_FRAMES] + smooth_lower_face(current_video[-NUM_COND_FRAMES:], cond_smooth)
                        output, latent_new = pipe.generate_avc(
                            video=cond_video, video_latent=latent, prompt=prompt, negative_prompt=neg_prompt,
                            height=height, width=width, num_frames=NUM_FRAMES, num_cond_frames=NUM_COND_FRAMES,
                            num_inference_steps=steps, text_guidance_scale=tg, audio_guidance_scale=ag,
                            generator=generator, output_type="both", use_kv_cache=True,
                            offload_kv_cache=self.active["offload_kv_cache"],
                            enhance_hf=True if not distill else False, audio_emb=audio_emb,
                            ref_latent=ref_latent, ref_img_index=REF_IMG_INDEX,
                            mask_frame_range=MASK_FRAME_RANGE, use_distill=distill)
                except torch.OutOfMemoryError as e:
                    tb = traceback.extract_tb(e.__traceback__)
                    where = " <- ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in tb[-3:][::-1])
                    oom = f"{' '.join(str(e).split())[:600]} | at {where}"
                    del e, tb
                if oom is None:
                    break
                # VRAM is not reliably released in-process after an OOM, so: lower the layout cap, persist it,
                # and restart the process (start_worker.sh loops on exit code 75). The job resumes from the last
                # finished segment with the saved RNG state -> identical output.
                log(f"OOM on segment {seg}: {oom}")
                if self.active["resident_blocks"] == 0:
                    raise RuntimeError(f"segment {seg}: OOM even with all blocks streamed: {oom}")
                caps = load_caps()
                caps[quant] = max(0, self.active["resident_blocks"] - 4)
                CAPS_FILE.write_text(json.dumps(caps))
                log(f"restarting worker with resident_blocks cap {quant}={caps[quant]} (job resumes)")
                sys.stdout.flush()
                os._exit(75)
            else:
                raise RuntimeError(f"segment {seg} failed after OOM retries")

            output = output[0]
            new_video = [PIL.Image.fromarray((output[i] * 255).astype(np.uint8)) for i in range(output.shape[0])]
            del output
            if seg == 1:
                ref_latent = latent_new[:, :, :1].clone()
                frames_new = new_video
            elif is_anchor:
                audio_start_idx = new_start
                frames_new = new_video[NUM_COND_FRAMES:]
                if ref_latent is None:  # chain started at start_segment > 1
                    ref_latent = latent_new[:, :, :1].clone()
                if prev_seg is not None:  # crossfade: the anchor's first 13 frames cover the same time as the previous tail
                    pf = prev_seg["frames"].astype(np.float32)
                    for i in range(NUM_COND_FRAMES):  # last `xfade` overlapping frames blend linearly; earlier ones keep the old chain
                        j = i - (NUM_COND_FRAMES - xfade)
                        al = 0.0 if j < 0 else (j + 1) / (xfade + 1)
                        pf[-NUM_COND_FRAMES + i] = (1 - al) * pf[-NUM_COND_FRAMES + i] + al * np.asarray(new_video[i], np.float32)
                    pf = np.clip(pf + 0.5, 0, 255).astype(np.uint8)
                    ps, pa, pd = prev_seg["seg"], prev_seg["a_start"], prev_seg["a_dur"]
                    if comp is not None:
                        write_mp4(pf, out_dir / "segments_crop" / f"seg_{ps:03d}.mp4", audio=raw_speech_path, a_start=pa, a_dur=pd)
                        pf = comp(pf, person_masks=segm(pf) if segm else None, **paste_kw)
                    write_mp4(pf, seg_dir / f"seg_{ps:03d}.mp4", audio=raw_speech_path, a_start=pa, a_dur=pd)
                    log(f"re-anchored at segment {seg} from the original image; crossfaded into segment {ps}")
            else:
                audio_start_idx = new_start
                frames_new = new_video[NUM_COND_FRAMES:]
            latent, current_video = latent_new, new_video

            dt = time.time() - t0
            timings.append(dt)
            self.dit_peak = max(self.dit_peak, torch.cuda.max_memory_allocated())
            act = self.dit_peak - self.active["base_alloc"]
            max_act = max(max_act, act)
            peak_msg = (f"peak VRAM DiT-phase {self.dit_peak / 2**30:.2f} GiB (activations {act / 2**30:.2f}) "
                        f"VAE-phase {self.vae_peak / 2**30:.2f} GiB | resident {self.active['resident_blocks']}/48 | "
                        f"peak RAM process {RAM.peak_rss / 2**30:.1f} GiB, system {RAM.peak_sys / 2**30:.1f}/"
                        f"{psutil.virtual_memory().total / 2**30:.0f} GiB")
            RAM.reset()
            a_start, a_dur = seg_time_window(seg)
            seg_path = seg_dir / f"seg_{seg:03d}.mp4"
            seg_frames = np.stack([np.asarray(f) for f in frames_new])
            prev_seg = dict(seg=seg, frames=seg_frames.copy(), a_start=a_start, a_dur=a_dur) if reanchor else None
            if comp is not None:  # keep the raw crop too (full generation detail), then paste back into the full frame
                write_mp4(seg_frames, out_dir / "segments_crop" / seg_path.name, audio=raw_speech_path, a_start=a_start, a_dur=a_dur)
                seg_frames = comp(seg_frames, person_masks=segm(seg_frames) if segm else None, **paste_kw)
            write_mp4(seg_frames, seg_path, audio=raw_speech_path,
                      a_start=a_start, a_dur=a_dur)
            torch.save(dict(cfg_sig=cfg_sig, next_segment=seg + 1, latent=latent.cpu(), ref_latent=ref_latent.cpu(),
                            current_video=np.stack([np.asarray(f) for f in current_video]),
                            audio_start_idx=audio_start_idx, generator_state=generator.get_state()),
                       str(state_path) + ".tmp")
            os.replace(str(state_path) + ".tmp", state_path)
            eta = (nseg - seg) * (sum(timings) / len(timings))
            log(f"segment {seg}/{nseg} done in {dt:.0f}s | lora {self.lora_multiplier if distill else '-'} | {peak_msg} | "
                f"{frames_new[0].size[0]}x{frames_new[0].size[1]} | ETA {eta / 60:.1f} min -> {seg_path.name}")
            with open(out_dir / "progress.json", "w") as f:
                json.dump(dict(done=seg, total=nseg, last_seg_s=dt, eta_min=eta / 60), f)
            # ---- auto-tune the (lossless) VRAM layout from the measured activation peak ----
            if seg < nseg and seg - start_seg < AUTOTUNE_SEGMENTS:
                rb_new = min(rb_cap, self.autotune(max_act))
                if rb_new != self.active["resident_blocks"]:
                    log(f"autotune: activations {max_act / 2**30:.2f} GiB -> resident_blocks "
                        f"{self.active['resident_blocks']} -> {rb_new}")
                    a = self.activate(quant, distill, rb_new, self.active["offload_kv_cache"])
                    pipe = a["pipe"]
                    LAYOUT[quant]["resident_blocks"] = rb_new

        # ---- final: concat segments, trim to the audio length, mux the original audio ----
        lst = out_dir / "segments.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in sorted(seg_dir.glob("seg_*.mp4"))))
        silent = out_dir / "video_silent.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-map", "0:v", "-c", "copy", str(silent)], check=True)
        final = out_dir / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent),
                        "-i", raw_speech_path, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "192k", "-t", f"{min(duration, (NUM_FRAMES + (nseg - 1) * (NUM_FRAMES - NUM_COND_FRAMES)) / SAVE_FPS):.3f}", "-movflags", "+faststart", str(final)], check=True)
        silent.unlink()
        total = time.time() - t_job
        (out_dir / "DONE.json").write_text(json.dumps(dict(final=str(final), total_min=total / 60,
                                                           sec_per_segment=timings), indent=2))
        log(f"JOB {name} DONE in {total / 60:.1f} min -> {final}")
        return str(final)


# ---------------------------------------------------------------- queue mode
def serve(worker, qdir):
    for sub in ("pending", "running", "done", "failed"):
        (qdir / sub).mkdir(parents=True, exist_ok=True)
    for f in (qdir / "running").glob("*.json"):  # jobs interrupted by a restart -> resume them
        f.rename(qdir / "pending" / f.name)
    log(f"serving queue {qdir}")
    while True:
        jobs = sorted((qdir / "pending").glob("*.json"), key=lambda p: p.name)
        if not jobs:
            time.sleep(5)
            continue
        jf = jobs[0]
        run = qdir / "running" / jf.name
        jf.rename(run)
        try:
            worker.run_job(json.loads(run.read_text()))
            run.rename(qdir / "done" / jf.name)
        except Exception:
            traceback.print_exc()
            run.rename(qdir / "failed" / jf.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--queue", default=str(HERE / "jobs"))
    ap.add_argument("--job", help="run one job json and exit")
    ap.add_argument("--preload", default="", help="comma list of DiTs to load into RAM at start: int8,bf16")
    ap.add_argument("--checkpoint_dir", default=DEFAULT_CKPT)
    args = ap.parse_args()
    preload = [p for p in args.preload.split(",") if p]
    worker = AvatarWorker(args.checkpoint_dir, preload=preload)
    if args.job:
        worker.run_job(json.loads(Path(args.job).read_text()))
    if args.serve:
        serve(worker, Path(args.queue))


if __name__ == "__main__":
    main()
