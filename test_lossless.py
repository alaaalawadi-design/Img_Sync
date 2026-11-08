#!/usr/bin/env python
"""Proves the VRAM layout is lossless: the same short ai2v generation (13 frames, 720p by default, distill,
seed 42) must give bit-identical latents whether DiT blocks are GPU-resident or streamed."""
import sys
import torch
import librosa
import numpy as np
from diffusers.utils import load_image
sys.path.insert(0, "/home/administrator/imgsync")
import avatar_worker as aw

RES = sys.argv[1] if len(sys.argv) > 1 else "720p"
w = aw.AvatarWorker(preload=[])
img = "/home/administrator/imgsync/inputs/woman2.png"
wav = "/home/administrator/imgsync/inputs/pi_female_egy_v5_full.wav"
speech, sr = librosa.load(wav, sr=16000, duration=3.0)


def run(quant, resident):
    a = w.activate(quant, True, resident, True)
    pipe = a["pipe"]
    emb = w.audio_embedding(pipe, speech, sr)
    ci = torch.clamp(torch.arange(0, 13).unsqueeze(1) + (torch.arange(5) - 2).unsqueeze(0), 0, emb.shape[0] - 1)
    g = torch.Generator(device=w.device); g.manual_seed(42)
    _, lat = pipe.generate_ai2v(image=load_image(img), prompt=aw.DEFAULT_PROMPT, negative_prompt=aw.NEGATIVE_PROMPT,
                                resolution=RES, num_frames=13, num_inference_steps=8, text_guidance_scale=1.0,
                                audio_guidance_scale=1.0, output_type="both", generator=g,
                                audio_emb=emb[ci][None].to(w.device), use_distill=True)
    return lat.float().cpu()


results = {}
for quant, layouts in (("int8", (48, 30)), ("bf16", (22, 8))):
    l0, l1 = run(quant, layouts[0]), run(quant, layouts[1])
    same = torch.equal(l0, l1)
    results[quant] = same
    print(f"LOSSLESS_CHECK {RES} {quant}: resident={layouts[0]} vs resident={layouts[1]} -> bit-identical={same} "
          f"max|diff|={(l0 - l1).abs().max().item():.3e}", flush=True)
    w.deactivate()
print("LOSSLESS_ALL_OK" if all(results.values()) else "LOSSLESS_FAILED", flush=True)
