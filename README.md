# Img_Sync — LongCat-Video-Avatar-1.5 production runner

Audio + image + prompt → a full-length 720p talking-avatar video on **one RTX 5090 (32 GB)**.

The project uses the **original** [meituan-longcat/LongCat-Video](https://github.com/meituan-longcat/LongCat-Video)
code at commit `6b3f4b8582a8bc3f20f795735f5383716c4ba794`, without modifying it, together with the original
[LongCat-Video-Avatar-1.5](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5) weights.
Everything in this repo is a layer **around** the original pipeline. That layer adds the features listed in
[What is new compared to the original LongCat code](#what-is-new-compared-to-the-original-longcat-code).

---

## Best configuration (reference run)

The best result so far is `cp_full_gazaman3_gaza_lora0.30_minus0.01x3_min0.20`. It is 6:25 long (121 segments), 1312x704, at 25 fps.

| Setting | Value |
|---|---|
| Model | DiT **INT8** (official `base_model_int8`) |
| Distill | yes: 8 steps, text/audio guidance 1.0 (the negative prompt is unused) |
| DMD LoRA | **0.30**, minus **0.01 every 3 segments**, floor **0.20** (`--lora_schedule 0.30,0.01,3,0.20`) |
| Continuation | official v1.5 (re-encode the last 13 decoded frames) |
| Crop → generate → paste back | crop box `0,0,879,940`, generated at 928x992, pasted back (`rect`) into the full 1312x704 frame |
| Prompt | [`prompt_best.txt`](prompt_best.txt) (TV presenter) |
| Seed / resolution | 42 / 720p |
| Speed | ~351 s per segment (one segment ≈ 3.2 s of video), so **~1 h 50 min per minute of audio** |
| Memory | peak 27.7 GiB VRAM (37 of 48 DiT blocks resident), ~80 GiB RAM |

Exact job files: [`configs/best_gazaman3_job.json`](configs/best_gazaman3_job.json) (what was queued) and
[`configs/best_gazaman3_job_resolved.json`](configs/best_gazaman3_job_resolved.json) (every value the worker resolved).

```bash
./run_best_queue.sh                                   # through the worker queue (how the reference run was made)
screen -S best ./run_best_direct.sh                   # one command, no queue
IMAGE=inputs/x.png AUDIO=inputs/x.wav CROP_BOX=x0,0,x1,H NAME=my_run ./run_best_queue.sh   # another person
python run_cp_avatar.py --image inputs/x.png --audio inputs/x.wav --plan_only               # automatic crop box
```

---

## What is new compared to the original LongCat code

The original repo provides `run_demo_avatar_single_audio_to_video.py`. That script is a demo: it runs once,
generates a fixed number of segments, keeps everything in GPU memory (it needs more than 32 GB at 720p with the
BF16 DiT), and has no resume. None of the following exists in the original repo.

### 1. Persistent worker and job queue — `avatar_worker.py`, `submit_job.py`
- **Models stay loaded.** T5 and Whisper stay in RAM, the VAE stays on the GPU, and the active DiT (INT8 or BF16) stays in RAM and VRAM between jobs. The next job starts without reloading anything.
- **Job queue.** Jobs go into `jobs/pending/*.json`, move to `jobs/running/`, then to `jobs/done/`. Use `submit_job.py` to queue a job with any option below.
- **Whole audio.** The number of segments comes from the audio length, so the whole audio is always generated: `1 + ceil((duration − 3.72) / 3.2)`.
- **Segments are saved as soon as they finish.** Each one is written to `segments/seg_XXX.mp4` with its own audio slice.
  - The final `<name>.mp4` is trimmed to `min(audio, video)`.
  - `progress.json` holds live progress and the ETA, and `DONE.json` holds the per-segment timings.
- **Resume.** `resume_state.pt` stores the latents and the generator RNG state. A job that resumes after a crash or reboot is **bit-identical** to one that ran without interruption.
- **Detailed log line per segment.** Each segment's log line shows the time, LoRA strength, VRAM peaks (DiT phase and VAE phase), resident blocks, RAM peak and ETA.
- **CUDA out-of-memory handling.**
  - The worker writes `jobs/layout_caps.json` with a safer layout and exits with code 75.
  - The loop in `start_worker.sh` then restarts it, which fully frees VRAM, and the job continues.
- **Same settings as the official demo.** Seed 42, 93 frames, 13 conditioning frames, 25 fps, `ref_img_index=10`, `mask_frame_range=3` and the official negative prompt.

### 2. Lossless fit into 32 GB VRAM — `dit_streaming.py`, `test_lossless.py`
These changes only move data between devices; the computation and its results stay the same.
- **`BlockStreamer`.**
  - DiT blocks that do not fit in VRAM stay in **pinned RAM**.
  - They are streamed to the GPU just in time, with double-buffered prefetch.
  - Resident blocks never move.
- **Auto-tune.** After segment 1 the worker measures the activation peak and picks the largest number of resident blocks that fits the VRAM budget.
  - Set the budget with `LONGCAT_VRAM_BUDGET_GIB`; the default is 29.3. Use a lower value on a shared GPU.
  - Or force a value with `--resident_blocks`.
- **VAE phase.** The DiT is parked during VAE encode and decode.
- **KV cache.** The original `offload_kv_cache` option is used for the conditioning frames' KV cache.
- **Text encoder.** T5 visits the GPU only while it encodes, and prompt embeddings are memoised under `no_grad`.
- **Non-distill CFG.** The batch of 2 is run as two batch-1 passes, the same way the original multitalk branch does it.
- **Proof.** `test_lossless.py` produces bit-identical latents with blocks resident and with blocks streamed.

### 3. Quality controls
| Option (`submit_job.py` / job key) | What it does |
|---|---|
| `--quant int8\|bf16` | official INT8 or BF16 DiT (only the active one is kept in RAM) |
| `--distill` / `--no-distill` | distill: 8 steps, guidance 1.0, with the DMD LoRA. Base model: CFG 4/4, `--num_inference_steps` (official 50) |
| `--lora_multiplier X` | DMD LoRA strength (official 1.0). **0.30 gives a more natural face and teeth** |
| `--lora_schedule start,step,every,min[,from]` | **new**: the LoRA strength goes down during the video to limit texture drift ("artistic" beard) in long runs |
| `--latent_continuation` | continue from latents (v1.0 style, `use_vcond=True`) instead of the official v1.5 pixel re-encode |
| `--prompt` / `--prompt_file`, `--negative_prompt_file` | prompt; the negative prompt is only used without distill |
| `--seed`, `--resolution 480p\|720p`, `--num_segments` | as in the original |

### 4. Crop → generate → paste back — `crop_paste.py`
- **Crop.** The model generates a **full-height crop around the person**, snapped to an official 720p bucket. The face gets **about 1.4–1.5× more pixels** for the same token budget, which gives better teeth and face detail.
- **Paste back.** Every generated frame is pasted into the original full frame. That frame is prepared exactly as the pipeline would prepare it: Lanczos resize to cover, then centre crop.
- **Output.** The output keeps the input's size and framing.
- `--crop_box x0,y0,x1,y1`: the crop in input-image pixels. `run_cp_avatar.py` can also find it automatically with a DeepLabV3 person mask.
- `--paste_mode rect`: pastes the whole crop, with a 24 px feather where the crop edge lies inside the frame.
- `--paste_mode person`: pastes only the generated person (DeepLabV3 mask, `--paste_dilate`, `--paste_feather`). The background stays **pixel-identical** to the input.
- `segments_crop/` keeps the raw crop clips.

### 5. Long-video and segment control
| Option | What it does |
|---|---|
| `--start_segment N --segments_from DIR` | reuse segments 1..N-1 of an earlier run and generate from segment N |
| `--start_mode continue\|fresh` | at segment N: continue from the previous segment's 13 frames, or start a fresh chain from the image |
| `--reanchor_every N`, `--reanchor_crossfade F` | experimental: a fresh chain from the original image every N segments. The crossfade can show a visible jump |
| `--resume_from DIR` | continue a shorter run that had identical inputs |
| `--cond_face_smooth S` | experimental: smooth the lower face of the 13 conditioning frames (0 = off) |

### 6. Audio control
- `--drive_audio`: the audio the model listens to, when it is different from the soundtrack (`--audio`).
- `--skip_vocal_separation`: feed the drive audio as it is. Use it, for example, for a silent "listener" video. By default the vocals are separated with Kim_Vocal_2, as in the original.
- `run_cp_avatar.py --silence N`: N seconds of pure silence, used both as the model input and as the soundtrack.
- `run_cp_avatar.py --seconds N`: use only the first N seconds of the audio.

### 7. One-command runner — `run_cp_avatar.py`
Runs one job without the queue. It supports automatic crop boxes, silence, trimming and `--plan_only`, plus all the
options above, including `--lora_schedule`.

### 8. Operations
| File | What |
|---|---|
| `start_worker.sh` | starts the worker in `screen longcat_worker`; restarts it on exit code 75; never starts two workers |
| `watchdog.sh` | runs in `screen longcat_watchdog`: restarts a dead worker while jobs are queued, kills a hung one (log silent > 15 min), gives up after 4 restarts without progress |
| crontab `@reboot` | `sleep 90; cd /home/administrator/imgsync && screen -dmS longcat_watchdog ./watchdog.sh`, so runs resume after a reboot |
| `fix_final_len.sh <job>` | re-mux a final mp4 to `min(video, audio)` length (stream copy) |
| `verify_weights.py` | checks every local weight file against the Hugging Face tree (size + sha256) |

### 9. Automatic evaluation — `eval/`
- `eval/eval_watcher.sh`: runs in `screen longcat_eval` and evaluates every finished job with `eval/eval_avatar.py`.
- Output: `outputs/<job>/eval/report.md`, `metrics.json` and `contact_sheet.jpg`.

The report covers these metrics, overall and per segment:

| Aspect | Metrics |
|---|---|
| Lip sync | SyncNet LSE-C / LSE-D and A/V offset |
| Identity | SFace cosine against the input |
| Background | PSNR / SSIM / LPIPS on the non-person area |
| Realism | MUSIQ and face sharpness |
| Artifacts | flicker, segment-seam jump and missing-face frames |
| Framing | zoom and face drift |

Setup: `eval/setup_eval.sh`, which uses its own env `avatar-eval` and downloads YuNet, SFace and SyncNet.

### 10. Multi-server tools — `tools/`
| File | What |
|---|---|
| `tools/push_eval_to22.py` | runs on a worker server; sends finished runs to the evaluation server |
| `tools/sync_outputs.py job [16:job] [12:job]` | downloads segments to a local machine as they finish |
| `tools/download_all.py` | robust full download of every run from all servers |

The tools read credentials from the environment and contain no passwords: `export SRV_PASS=...` (and optionally
`SRV_USER`, `EVAL_HOST` and `LOCAL_OUT`).

---

## Install (one server)
```bash
# 1. this repo as the project folder (the shell scripts use /home/administrator/imgsync)
git clone https://github.com/alaaalawadi-design/Img_Sync.git /home/administrator/imgsync
cd /home/administrator/imgsync

# 2. the original LongCat code, pinned
git clone https://github.com/meituan-longcat/LongCat-Video.git
git -C LongCat-Video checkout 6b3f4b8582a8bc3f20f795735f5383716c4ba794

# 3. weights (official). setup/setup_env.sh links them from an existing folder; or download:
huggingface-cli download meituan-longcat/LongCat-Video          --local-dir LongCat-Video/weights/LongCat-Video
huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 --local-dir LongCat-Video/weights/LongCat-Video-Avatar-1.5

# 4. env (RTX 5090 = sm_120: torch 2.8.0 cu128 + flash-attn 2.8.3 wheel, sympy 1.13.3)
bash setup/setup_env.sh && bash setup/fix_torch.sh
#    exact package versions of the reference server: setup/requirements_lock.txt

# 5. optional: verify weights, prove the lossless layout, evaluation env
python verify_weights.py ; python test_lossless.py ; bash eval/setup_eval.sh

# 6. run
mkdir -p inputs logs jobs ; ./start_worker.sh ; screen -dmS longcat_watchdog ./watchdog.sh
python submit_job.py --audio inputs/a.wav --image inputs/i.png --lora_multiplier 0.30 --lora_schedule 0.30,0.01,3,0.20 \
       --crop_box 0,0,879,940 --paste_mode rect --prompt_file prompt_best.txt --name my_run
```
- **Why not the official env pins:** `requirements_avatar.txt` pins `sympy==1.13.1`, which downgrades torch to 2.6, and torch 2.6 has no RTX 5090 kernels. That's why the env is installed with `setup_env.sh` and `fix_torch.sh`.
- **Paths:** the Python code finds the original repo next to itself, or through `LONGCAT_REPO`. The shell scripts assume `/home/administrator/imgsync` and the env `/home/administrator/miniconda3/envs/longcat-avatar15`; edit them if you deploy elsewhere.

## Layout
```
avatar_worker.py      worker / generation engine (wraps the original pipeline)
dit_streaming.py      lossless DiT block streaming RAM <-> VRAM
crop_paste.py         crop -> generate -> paste back, person masks
submit_job.py         queue a job                run_cp_avatar.py   one-command run
run_best_queue.sh     re-run the best config     run_best_direct.sh same, without the queue
start_worker.sh       worker in screen           watchdog.sh        auto-restart / resume
fix_final_len.sh  verify_weights.py  test_lossless.py
prompt_best.txt       best prompt                prompts/           TV-presenter + silent-listener (v7) prompts
configs/              exact job files of the best run
setup/                env setup, torch fix for sm_120, requirements_lock.txt, reference system info
eval/                 automatic evaluation       tools/             multi-server push / download
(not committed)       LongCat-Video/  inputs/  outputs/  jobs/  logs/
```
