#!/usr/bin/env python
"""Queue a job for the persistent LongCat-Avatar-1.5 worker (avatar_worker.py --serve).

Examples
  python submit_job.py --audio a.wav --image face.png                      # INT8 + distill, 720p, default prompt
  python submit_job.py --audio a.wav --image face.png --quant bf16         # BF16 + distill
  python submit_job.py --audio a.wav --image face.png --no-distill         # base model, 50 steps, CFG 4/4 (very slow)
  python submit_job.py --audio a.wav --image face.png --prompt "..." --resolution 480p
"""
import json
import time
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--audio", required=True)
ap.add_argument("--image", required=True)
ap.add_argument("--prompt", default=None, help="default: the TV-presenter prompt in avatar_worker.DEFAULT_PROMPT")
ap.add_argument("--quant", choices=["int8", "bf16"], default="int8")
ap.add_argument("--distill", dest="distill", action="store_true", default=True)
ap.add_argument("--no-distill", dest="distill", action="store_false")
ap.add_argument("--resolution", choices=["480p", "720p"], default="720p")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--lora_multiplier", type=float, default=None, help="distill LoRA strength (official 1.0)")
ap.add_argument("--latent_continuation", action="store_true", default=None, help="use_vcond=True: continue from latents (no VAE re-encode)")
ap.add_argument("--name", default=None)
ap.add_argument("--output_dir", default=None)
ap.add_argument("--num_segments", type=int, default=None, help="default: derived from the audio length")
ap.add_argument("--num_inference_steps", type=int, default=None, help="non-distill only (official default 50)")
ap.add_argument("--resident_blocks", type=int, default=None, help="VRAM layout override (lossless)")
ap.add_argument("--crop_box", default=None, help="x0,y0,x1,y1 in input-image pixels: crop -> generate -> paste back")
ap.add_argument("--resume_from", default=None, help="output dir of a shorter run with identical inputs to continue from")
ap.add_argument("--drive_audio", default=None, help="audio the model listens to (default: --audio); --audio stays the soundtrack")
ap.add_argument("--skip_vocal_separation", action="store_true", default=None)
ap.add_argument("--prompt_file", default=None)
ap.add_argument("--negative_prompt_file", default=None, help="used only when CFG is active (non-distill)")
ap.add_argument("--reanchor_every", type=int, default=None, help="fresh image->video chain from the original image every N segments")
ap.add_argument("--reanchor_crossfade", type=int, default=None, help="frames (0-13) blended at each re-anchor; 0 = hard cut, default 13")
ap.add_argument("--paste_mode", choices=["rect", "person"], default=None, help="with --crop_box: paste the whole crop (rect) or only the person")
ap.add_argument("--paste_dilate", type=int, default=None)
ap.add_argument("--paste_feather", type=int, default=None)
ap.add_argument("--start_segment", type=int, default=None, help="begin generation here with a fresh chain from the image")
ap.add_argument("--segments_from", default=None, help="output dir whose segments 1..start_segment-1 are reused")
ap.add_argument("--start_mode", choices=["fresh", "continue"], default=None, help="at --start_segment: fresh chain from the image, or continue from the previous segment's 13 frames")
ap.add_argument("--cond_face_smooth", type=float, default=None, help="sigma (px) to smooth the lower face of the 13 conditioning frames (0 = off)")
ap.add_argument("--lora_schedule", default=None, help="start,step,every,min[,from_segment] e.g. 0.30,0.05,2,0.0")
ap.add_argument("--queue", default=str(HERE / "jobs"))
a = ap.parse_args()

job = {k: v for k, v in vars(a).items() if v is not None and k != "queue"}
if a.lora_schedule:
    v = [float(x) for x in a.lora_schedule.split(",")]
    job["lora_schedule"] = dict(start=v[0], step=v[1], every=int(v[2]), min=v[3], **({"from_segment": int(v[4])} if len(v) > 4 else {}))
if a.crop_box:
    job["crop_box"] = [int(v) for v in a.crop_box.split(",")]
job["audio"] = str(Path(a.audio).resolve())
job["image"] = str(Path(a.image).resolve())
if a.drive_audio:
    job["drive_audio"] = str(Path(a.drive_audio).resolve())
if a.prompt_file:
    job["prompt"] = Path(a.prompt_file).read_text().strip()
    job.pop("prompt_file", None)
if a.negative_prompt_file:
    job["negative_prompt"] = Path(a.negative_prompt_file).read_text().strip()
    job.pop("negative_prompt_file", None)
q = Path(a.queue) / "pending"
q.mkdir(parents=True, exist_ok=True)
f = q / f"{time.strftime('%Y%m%d_%H%M%S')}_{a.name or Path(a.image).stem}_{a.quant}{'_distill' if a.distill else ''}{'' if a.lora_multiplier is None else f'_lora{a.lora_multiplier:g}'}{'_latentcont' if a.latent_continuation else ''}.json"
f.write_text(json.dumps(job, indent=2))
print("queued", f)
