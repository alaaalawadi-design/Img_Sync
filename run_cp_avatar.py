#!/usr/bin/env python
"""LongCat-Video-Avatar-1.5 - one command for the "crop -> generate -> paste back" runs.

This is exactly the configuration used for cp_full_new_ref_teeth_gaza_lora0.32_latentcont on server 22:
  original LongCat pipeline (generate_ai2v for segment 1, generate_avc + KV cache for the rest), INT8 DiT,
  distill (8 steps, guidance 1.0) with the DMD LoRA at strength 0.32, latent-space continuation (use_vcond=True),
  official 720p bucket, seed 42, segments = enough to cover the whole audio, lossless VRAM layout.
Crop -> generate -> paste back: the model generates a full-height crop around the person (face gets ~1.4x more
pixels), every frame is pasted back into the original full frame, so the output keeps the input framing.

Outputs in --output_dir: segments/seg_XXX.mp4 (final frame), segments_crop/seg_XXX.mp4 (raw crop),
<name>.mp4 (full video), job_config.json, DONE.json, resume_state.pt (re-run the same command to resume).

Examples (activate the env first:  conda activate /home/administrator/miniconda3/envs/longcat-avatar15)
  # the server-22 run
  python run_cp_avatar.py --image inputs/new_ref_teeth.png --audio inputs/gaza_all_01_08.wav \\
      --prompt "A TV presenter talking to the camera. Do not change anything in the background. Preserve the identity of the person. The head and eyes are naturally focused on the camera. Realistic facial features dont zoom in or out ." \\
      --crop_box 0,0,805,860 --name cp_full_new_ref_teeth_gaza_lora0.32_latentcont
  # automatic crop box, only the first 30 s of the audio
  python run_cp_avatar.py --image face.png --audio talk.wav --seconds 30
  # silent listener (model hears silence, video is silent), non-distill 15 steps with a negative prompt
  python run_cp_avatar.py --image man1.png --silence 15 --no-distill --steps 15 \\
      --prompt_file prompt_listen_v3.txt --negative_prompt_file negative_listen_v3.txt
  # only print the crop box that would be used
  python run_cp_avatar.py --image face.png --audio talk.wav --plan_only
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def auto_crop_box(image_path, resolution="720p", margin=0.035):
    """Full-height crop around the person: person mask (DeepLabV3) -> horizontal extent + margin ->
    narrowest official 720p bucket that contains it -> centred on the person, clamped to the image."""
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    sys.path.insert(0, str(HERE / "LongCat-Video"))
    from longcat_video.utils.bukcet_config import get_bucket_config

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    m = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT").eval()
    s = 520 / min(H, W)
    x = torch.from_numpy(np.asarray(img.resize((round(W * s), round(H * s))))).permute(2, 0, 1)[None].float() / 255
    x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        person = (m(x)["out"][0].argmax(0) == 15).numpy()  # VOC class 15 = person
    person = np.asarray(Image.fromarray(person.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
    cols = np.where(person.mean(0) > 0.02)[0]
    if cols.size == 0:
        raise SystemExit("no person found in the image - pass --crop_box x0,y0,x1,y1 or --no_crop")
    left, right = int(cols.min()), int(cols.max())
    need = (right - left) + 2 * int(margin * W)
    cfg = get_bucket_config(resolution, 16)
    widths = sorted(round(H / (h / w)) for (h, w), _ in cfg.values())
    cw = min(W, next((w for w in widths if w >= need), W))
    x0 = int(round(min(max(0, (left + right) / 2 - cw / 2), W - cw)))
    return [x0, 0, x0 + cw, H]


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", help="speech audio (also the soundtrack)")
    ap.add_argument("--silence", type=float, default=None, help="no audio: generate N seconds of pure silence (input + soundtrack)")
    ap.add_argument("--drive_audio", default=None, help="audio the model listens to, if different from --audio")
    ap.add_argument("--seconds", type=float, default=None, help="use only the first N seconds of the audio")
    ap.add_argument("--prompt", default=None, help="default: the TV-presenter prompt")
    ap.add_argument("--prompt_file", default=None)
    ap.add_argument("--negative_prompt_file", default=None, help="only used without distill (CFG)")
    ap.add_argument("--lora_multiplier", type=float, default=0.32, help="DMD distill LoRA strength (official 1.0)")
    ap.add_argument("--lora_schedule", default=None,
                    help="start,step,every,min[,from_segment]: LoRA strength goes down by <step> every <every> segments, "
                         "never below <min> (e.g. 0.30,0.01,3,0.20 = the cp_full_gazaman3 run)")
    ap.add_argument("--no-distill", dest="distill", action="store_false", default=True, help="base model with CFG 4/4 (slow)")
    ap.add_argument("--steps", type=int, default=50, help="steps without distill (official 50)")
    ap.add_argument("--quant", choices=["int8", "bf16"], default="int8")
    ap.add_argument("--reencode_continuation", action="store_true", help="official v1.5 pixel re-encode instead of latent continuation")
    ap.add_argument("--reanchor_every", type=int, default=0, help="fresh image->video chain from the original image every N segments (0 = off)")
    ap.add_argument("--reanchor_crossfade", type=int, default=13, help="frames (0-13) crossfaded at each re-anchor; 0 = hard cut")
    ap.add_argument("--crop_box", default=None, help="x0,y0,x1,y1 in input-image pixels (default: automatic)")
    ap.add_argument("--paste_mode", choices=["rect", "person"], default="rect", help="paste back the whole crop or only the person (background stays original)")
    ap.add_argument("--paste_dilate", type=int, default=10)
    ap.add_argument("--paste_feather", type=int, default=12)
    ap.add_argument("--no_crop", action="store_true", help="plain full-frame generation")
    ap.add_argument("--resolution", choices=["480p", "720p"], default="720p")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None)
    ap.add_argument("--output_dir", default=None, help="default: outputs/<name>")
    ap.add_argument("--plan_only", action="store_true", help="print the crop box and exit")
    a = ap.parse_args()

    image = str(Path(a.image).resolve())
    name = a.name or f"cp_{Path(image).stem.replace(' ', '_')}_{a.quant}_{'distill_lora%g' % a.lora_multiplier if a.distill else 'nodistill%d' % a.steps}"
    out_dir = Path(a.output_dir or HERE / "outputs" / name).resolve()

    box = None if a.no_crop else ([int(v) for v in a.crop_box.split(",")] if a.crop_box else auto_crop_box(image, a.resolution))
    print("crop box:", box)
    if a.plan_only:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if a.silence:
        audio = out_dir / f"silence_{a.silence:g}s.wav"
        ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", str(a.silence), "-c:a", "pcm_s16le", str(audio))
        audio, drive, skip_sep = str(audio), str(audio), True
    else:
        if not a.audio:
            raise SystemExit("give --audio or --silence N")
        audio = str(Path(a.audio).resolve())
        if a.seconds:
            trimmed = out_dir / f"audio_first{a.seconds:g}s.wav"
            ffmpeg("-t", str(a.seconds), "-i", audio, "-ar", "24000", str(trimmed))
            audio = str(trimmed)
        drive, skip_sep = (str(Path(a.drive_audio).resolve()) if a.drive_audio else None), False

    job = dict(name=name, image=image, audio=audio, quant=a.quant, distill=a.distill, resolution=a.resolution, seed=a.seed,
               latent_continuation=not a.reencode_continuation, output_dir=str(out_dir))
    if a.distill:
        job["lora_multiplier"] = a.lora_multiplier
        if a.lora_schedule:
            v = [float(x) for x in a.lora_schedule.split(",")]
            job["lora_schedule"] = dict(start=v[0], step=v[1], every=int(v[2]), min=v[3],
                                        **({"from_segment": int(v[4])} if len(v) > 4 else {}))
            job["lora_multiplier"] = v[0]
    else:
        job["num_inference_steps"] = a.steps
    if box:
        job["crop_box"] = box
        job.update(paste_mode=a.paste_mode, paste_dilate=a.paste_dilate, paste_feather=a.paste_feather)
    if a.reanchor_every:
        job["reanchor_every"], job["reanchor_crossfade"] = a.reanchor_every, a.reanchor_crossfade
    if drive:
        job["drive_audio"] = drive
    if skip_sep:
        job["skip_vocal_separation"] = True
    if a.prompt_file:
        job["prompt"] = Path(a.prompt_file).read_text().strip()
    elif a.prompt:
        job["prompt"] = a.prompt
    if a.negative_prompt_file:
        job["negative_prompt"] = Path(a.negative_prompt_file).read_text().strip()

    print(json.dumps({k: v for k, v in job.items() if k not in ("prompt", "negative_prompt")}, indent=2))
    from avatar_worker import AvatarWorker  # loads the original LongCat pipeline + models
    final = AvatarWorker(preload=[a.quant]).run_job(job)
    print("FINAL VIDEO:", final)


if __name__ == "__main__":
    main()
