#!/usr/bin/env python
"""Diagnostic evaluation of a LongCat-Avatar video against its input image + audio (CPU).

Metrics (all diagnostic - always also watch the video):
  sync        SyncNet v2 (joonson/syncnet_python weights) on 8 s windows over the whole video:
              LSE-C (confidence, higher = better), LSE-D (distance at best offset, lower = better), offset (0 = in sync)
              Face crops from YuNet (official LSE uses S3FD crops -> absolute values are close but not identical).
  identity    SFace cosine between each sampled frame's face and the input-image face (higher = same person;
              OpenCV's same-identity threshold is 0.363).
  background  person mask (DeepLabV3) on input + frame; on the remaining background pixels vs the input image
              (preprocessed exactly like the pipeline: resize-to-cover + center crop): PSNR, SSIM, LPIPS.
  realism     MUSIQ (no-reference IQA, 0-100) of frames vs the input image; face sharpness (Laplacian var) vs input.
  artifacts   temporal flicker (flow-compensated), jump at segment seams vs inside segments, frames without a face.
  framing     face-size ratio vs input (zoom) and face-center drift (fraction of frame width).
"""
import os
import sys
import json
import math
import time
import argparse
import subprocess
from pathlib import Path

import numpy as np
import cv2
import torch

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
sys.path.insert(0, str(HERE / "vendor" / "syncnet_python"))
torch.set_num_threads(int(os.environ.get("EVAL_THREADS", "10")))

SEG_FIRST, SEG_NEXT, FPS = 93, 80, 25


# ------------------------------------------------------------------ helpers
def stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)), n=int(v.size))


def seg_of(frame_idx):
    return 1 if frame_idx < SEG_FIRST else 2 + (frame_idx - SEG_FIRST) // SEG_NEXT


def preprocess_ref(img_bgr, W, H):
    """Resize-to-cover + center crop, like the pipeline's resize_mode='crop'."""
    h0, w0 = img_bgr.shape[:2]
    s = max(W / w0, H / h0)
    r = cv2.resize(img_bgr, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_LANCZOS4)
    y0, x0 = (r.shape[0] - H) // 2, (r.shape[1] - W) // 2
    return r[y0:y0 + H, x0:x0 + W]


class Faces:
    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (640, 640), 0.7, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(MODELS / "face_recognition_sface_2021dec.onnx"), "")

    def detect(self, f):
        s = min(1.0, 640 / max(f.shape[:2]))
        small = cv2.resize(f, None, fx=s, fy=s) if s < 1 else f
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        if faces is None or len(faces) == 0:
            return None
        faces = faces.copy(); faces[:, :14] /= s
        return max(faces, key=lambda r: r[2] * r[3])

    def embed(self, f, face):
        v = self.rec.feature(self.rec.alignCrop(f, face)).reshape(-1)
        return v / max(np.linalg.norm(v), 1e-12)


class PersonSeg:
    def __init__(self):
        import torchvision
        self.m = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT").eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, f_bgr):
        H, W = f_bgr.shape[:2]
        s = 520 / min(H, W)
        x = cv2.resize(f_bgr, (round(W * s), round(H * s)))[:, :, ::-1].copy()
        t = (torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255 - self.mean) / self.std
        out = self.m(t)["out"][0].argmax(0).numpy() == 15  # VOC class 15 = person
        return cv2.resize(out.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def masked_psnr(a, b, m):
    d = (a.astype(np.float64) - b.astype(np.float64))[m]
    mse = float((d ** 2).mean()) if d.size else float("nan")
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def masked_ssim(a, b, m):
    from skimage.metrics import structural_similarity
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    _, smap = structural_similarity(ga, gb, data_range=255, full=True)
    return float(smap[m].mean())


def flicker(a, b):
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (320, 172))
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (320, 172))
    flow = cv2.calcOpticalFlowFarneback(gb, ga, None, .5, 3, 15, 3, 5, 1.2, 0)
    yy, xx = np.mgrid[:172, :320].astype("float32")
    warped = cv2.remap(ga, xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_LINEAR)
    return float(np.abs(gb.astype(float) - warped).mean() / 255)


# ------------------------------------------------------------------ SyncNet (same math as SyncNetInstance.evaluate, CPU)
class Sync:
    def __init__(self):
        from SyncNetModel import S
        self.net = S(num_layers_in_fc_layers=1024)
        sd = torch.load(MODELS / "syncnet_v2.model", map_location="cpu")
        own = self.net.state_dict()
        for k, v in sd.items():
            own[k].copy_(v)
        self.net.eval()

    @staticmethod
    def crop(frame, box, cs=0.4):
        """syncnet_python run_pipeline crop: square around the face, extended towards the mouth, 224x224."""
        x, y, w, h = box
        bs = max(w, h) / 2
        my, mx = y + h / 2, x + w / 2
        bsi = int(bs * (1 + 2 * cs))
        pad = np.pad(frame, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110)
        my, mx = my + bsi, mx + bsi
        c = pad[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        return cv2.resize(c, (224, 224))

    @torch.no_grad()
    def window(self, crops, mfcc, vshift=15):
        import torch.nn.functional as F
        im = torch.from_numpy(np.stack(crops, 3)[None].transpose(0, 3, 4, 1, 2).astype(np.float32))  # 1,3,T,224,224
        cc = torch.from_numpy(mfcc[None, None].astype(np.float32))  # 1,1,13,Ta
        last = min(len(crops), mfcc.shape[1] // 4) - 5
        imf, ccf = [], []
        for i in range(0, last, 20):
            idx = range(i, min(last, i + 20))
            imf.append(self.net.forward_lip(torch.cat([im[:, :, v:v + 5] for v in idx], 0)))
            ccf.append(self.net.forward_aud(torch.cat([cc[:, :, :, v * 4:v * 4 + 20] for v in idx], 0)))
        imf, ccf = torch.cat(imf, 0), torch.cat(ccf, 0)
        win = vshift * 2 + 1
        ccp = F.pad(ccf, (0, 0, vshift, vshift))
        dists = [F.pairwise_distance(imf[[i]].repeat(win, 1), ccp[i:i + win]) for i in range(len(imf))]
        md = torch.stack(dists, 1).mean(1)
        mv, mi = torch.min(md, 0)
        return dict(lse_c=float(torch.median(md) - mv), lse_d=float(mv), offset_frames=int(vshift - mi))


# ------------------------------------------------------------------ main
def evaluate(video, image, out, sample_every=12, quality_every=25, sync_window_s=8.0):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref = preprocess_ref(cv2.imread(str(image)), W, H)
    cv2.imwrite(str(out / "reference_preprocessed.jpg"), ref)
    faces, seg = Faces(), PersonSeg()
    import lpips, pyiqa
    lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).eval()
    musiq = pyiqa.create_metric("musiq", device="cpu")

    rface = faces.detect(ref)
    assert rface is not None, "no face in the input image"
    remb = faces.embed(ref, rface)
    rmask = cv2.dilate(seg(ref).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    rsharp = cv2.Laplacian(cv2.cvtColor(ref[int(rface[1]):int(rface[1] + rface[3]), int(rface[0]):int(rface[0] + rface[2])], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    tmp = out / "_ref.png"; cv2.imwrite(str(tmp), ref)
    rmusiq = float(musiq(str(tmp)).item())

    def to_t(x):
        return torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 127.5 - 1

    # audio for SyncNet (from the video's own audio track)
    wav = out / "_audio16k.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", str(wav)], check=True)
    from scipy.io import wavfile
    import python_speech_features
    sr, audio = wavfile.read(str(wav))
    mfcc_all = python_speech_features.mfcc(audio, sr).T  # 13 x (100 per s)
    sync = Sync()

    rows, sync_rows, crops, prev, seam, inner = [], [], [], None, [], []
    last_box = None
    win_frames = int(sync_window_s * FPS)
    contact = []
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        face = faces.detect(f)
        box = tuple(face[:4]) if face is not None else last_box
        last_box = box if box is not None else last_box
        crops.append(Sync.crop(f, box) if box is not None else None)
        if prev is not None:
            d = float(np.abs(f.astype(np.int16) - prev.astype(np.int16)).mean())
            (seam if (i == SEG_FIRST or (i > SEG_FIRST and (i - SEG_FIRST) % SEG_NEXT == 0)) else inner).append(d)
        if i % max(1, n // 8) == 0 and len(contact) < 9:
            contact.append((i, f.copy()))
        if i % sample_every == 0 or i % quality_every == 0:
            r = dict(frame=i, t=i / FPS, segment=seg_of(i), face=face is not None)
            if face is not None:
                r["identity_cos"] = float(faces.embed(f, face) @ remb)
                r["zoom_ratio"] = float(face[3] / rface[3])
                r["center_shift"] = float(math.hypot((face[0] + face[2] / 2) - (rface[0] + rface[2] / 2),
                                                     (face[1] + face[3] / 2) - (rface[1] + rface[3] / 2)) / W)
                fc = f[int(max(0, face[1])):int(face[1] + face[3]), int(max(0, face[0])):int(face[0] + face[2])]
                if fc.size:
                    r["face_sharpness_ratio"] = float(cv2.Laplacian(cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() / max(rsharp, 1e-6))
            if i % quality_every == 0:
                fmask = cv2.dilate(seg(f).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
                bg = ~(fmask | rmask)
                r["bg_fraction"] = float(bg.mean())
                if bg.mean() > 0.02:
                    r["bg_psnr"] = masked_psnr(f, ref, bg)
                    r["bg_ssim"] = masked_ssim(f, ref, bg)
                    with torch.no_grad():  # spatial LPIPS map averaged over background pixels only
                        lmap = lp(to_t(f), to_t(ref))[0, 0].numpy()
                    r["bg_lpips"] = float(lmap[bg].mean())
                cv2.imwrite(str(tmp), f)
                r["musiq"] = float(musiq(str(tmp)).item())
                if prev is not None:
                    r["flicker"] = flicker(f, prev)
            rows.append(r)
        prev = f
        if len(crops) == win_frames or (i == n - 1 and len(crops) >= 50):
            start = i + 1 - len(crops)
            valid = [c for c in crops if c is not None]
            if len(valid) == len(crops):
                a0, a1 = int(start / FPS * 100), int((start + len(crops)) / FPS * 100)
                res = sync.window(crops, mfcc_all[:, a0:a1])
                res.update(start_s=start / FPS, dur_s=len(crops) / FPS, segment=seg_of(start + len(crops) // 2))
                sync_rows.append(res)
            crops = []
        if i % 500 == 0:
            print(f"  frame {i}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    cap.release()
    for p in (tmp, wav):
        p.unlink(missing_ok=True)

    # ---- contact sheet: reference + 8 frames
    tiles = [cv2.putText(cv2.resize(ref, (W // 3, H // 3)), "input", (10, 30), 0, 1, (0, 255, 255), 2)]
    for i, f in contact[:8]:
        tiles.append(cv2.putText(cv2.resize(f, (W // 3, H // 3)), f"t={i / FPS:.1f}s seg {seg_of(i)}", (10, 30), 0, 1, (0, 255, 255), 2))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)])
    cv2.imwrite(str(out / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    col = lambda k: [r.get(k) for r in rows]
    segs = sorted({r["segment"] for r in rows})
    per_seg = []
    for s in segs:
        g = [r for r in rows if r["segment"] == s]
        sg = [x for x in sync_rows if x["segment"] == s]
        per_seg.append(dict(segment=s,
                            identity=(stats([r.get("identity_cos") for r in g]) or {}).get("mean"),
                            bg_psnr=(stats([r.get("bg_psnr") for r in g]) or {}).get("mean"),
                            bg_ssim=(stats([r.get("bg_ssim") for r in g]) or {}).get("mean"),
                            musiq=(stats([r.get("musiq") for r in g]) or {}).get("mean"),
                            zoom=(stats([r.get("zoom_ratio") for r in g]) or {}).get("mean"),
                            lse_c=(stats([x["lse_c"] for x in sg]) or {}).get("mean"),
                            lse_d=(stats([x["lse_d"] for x in sg]) or {}).get("mean")))
    m = dict(
        video=str(video), image=str(image), frames=n, duration_s=n / FPS, resolution=f"{W}x{H}",
        sync=dict(windows=len(sync_rows), lse_c=stats([x["lse_c"] for x in sync_rows]), lse_d=stats([x["lse_d"] for x in sync_rows]),
                  offset_frames=stats([x["offset_frames"] for x in sync_rows]),
                  windows_in_sync_pct=100 * float(np.mean([abs(x["offset_frames"]) <= 1 for x in sync_rows])) if sync_rows else None),
        identity=dict(cosine=stats(col("identity_cos")),
                      below_threshold_pct=100 * float(np.mean([c < 0.363 for c in col("identity_cos") if c is not None])) if any(c is not None for c in col("identity_cos")) else None,
                      first_seg=per_seg[0]["identity"], last_seg=per_seg[-1]["identity"]),
        background=dict(psnr=stats(col("bg_psnr")), ssim=stats(col("bg_ssim")), lpips=stats(col("bg_lpips")), fraction=stats(col("bg_fraction")),
                        first_seg_psnr=per_seg[0]["bg_psnr"], last_seg_psnr=per_seg[-1]["bg_psnr"]),
        realism=dict(musiq=stats(col("musiq")), musiq_input_image=rmusiq, face_sharpness_ratio=stats(col("face_sharpness_ratio"))),
        artifacts=dict(flicker=stats(col("flicker")), seam_jump=stats(seam), inner_frame_diff=stats(inner),
                       seam_to_inner_ratio=(float(np.mean(seam)) / max(float(np.mean(inner)), 1e-6)) if seam and inner else None,
                       frames_without_face_pct=100 * float(np.mean([not r["face"] for r in rows]))),
        framing=dict(zoom_ratio=stats(col("zoom_ratio")), center_shift=stats(col("center_shift"))),
        per_segment=per_seg, eval_seconds=round(time.time() - t0, 1))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    (out / "per_frame.json").write_text(json.dumps(rows))
    (out / "sync_windows.json").write_text(json.dumps(sync_rows, indent=2))
    (out / "report.md").write_text(report_md(m))
    print(f"EVAL_DONE {out} in {m['eval_seconds']}s", flush=True)
    return m


def f(x, k="mean", d=3):
    return "n/a" if not x else f"{x[k]:.{d}f}"


def report_md(m):
    s, i, b, r, a, z = m["sync"], m["identity"], m["background"], m["realism"], m["artifacts"], m["framing"]
    L = [f"# Evaluation: {Path(m['video']).name}", "",
         f"{m['resolution']}, {m['frames']} frames, {m['duration_s']:.1f} s. Diagnostic metrics; always also watch the video and `contact_sheet.jpg`.", "",
         "| Aspect | Metric | Value | How to read |", "|---|---|---|---|",
         f"| Lip sync | LSE-C (confidence) | **{f(s['lse_c'], d=2)}** (median {f(s['lse_c'],'median',2)}, worst window {f(s['lse_c'],'min',2)}) | higher = better; real talking video ≈ 6–8, < 3 = weak sync |",
         f"| Lip sync | LSE-D (distance) | **{f(s['lse_d'], d=2)}** | lower = better; real video ≈ 6.5–8 |",
         f"| Lip sync | A/V offset | mean {f(s['offset_frames'], d=1)} frames; windows within ±1 frame: {s['windows_in_sync_pct'] if s['windows_in_sync_pct'] is None else round(s['windows_in_sync_pct'])}% | 0 = in sync (1 frame = 40 ms) |",
         f"| Identity | SFace cosine vs input | **{f(i['cosine'])}** (min {f(i['cosine'],'min')}); first seg {i['first_seg'] and round(i['first_seg'],3)} → last seg {i['last_seg'] and round(i['last_seg'],3)} | ≥ 0.363 = same person (OpenCV threshold); a drop over segments = identity drift |",
         f"| Identity | frames below threshold | {i['below_threshold_pct'] if i['below_threshold_pct'] is None else round(i['below_threshold_pct'],1)}% | ideally 0% |",
         f"| Background | PSNR vs input (non-person area) | **{f(b['psnr'], d=2)} dB**; first seg {b['first_seg_psnr'] and round(b['first_seg_psnr'],2)} → last {b['last_seg_psnr'] and round(b['last_seg_psnr'],2)} | higher = better preserved; > 30 dB ≈ unchanged, < 25 dB visible change |",
         f"| Background | SSIM / LPIPS | **{f(b['ssim'])}** / **{f(b['lpips'])}** | SSIM → 1 and LPIPS → 0 = unchanged |",
         f"| Realism | MUSIQ (frames) vs input image | **{f(r['musiq'], d=1)}** vs {r['musiq_input_image']:.1f} | 0–100, higher = better; close to the input's score = no quality loss |",
         f"| Realism | face sharpness vs input | **{f(r['face_sharpness_ratio'], d=2)}×** | ≈ 1 = as sharp as input; ≪ 1 blurry/plastic, ≫ 1 noisy |",
         f"| Artifacts | temporal flicker | **{f(a['flicker'], d=4)}** | lower = steadier |",
         f"| Artifacts | segment-seam jump / inside-segment change | **{a['seam_to_inner_ratio'] and round(a['seam_to_inner_ratio'],2)}×** | ≈ 1 = invisible seams; > 2 = visible jumps at segment joins |",
         f"| Artifacts | frames without detectable face | {round(a['frames_without_face_pct'],1)}% | 0% expected |",
         f"| Framing | face size vs input (zoom) | **{f(z['zoom_ratio'])}** (range {f(z['zoom_ratio'],'min')}–{f(z['zoom_ratio'],'max')}) | 1.00 = no zoom; ±5% normal head motion |",
         f"| Framing | face-center drift | {f(z['center_shift'])} of frame width (max {f(z['center_shift'],'max')}) | small = presenter stays in place |",
         "", "## Per segment", "", "| seg | identity | bg PSNR | bg SSIM | MUSIQ | zoom | LSE-C | LSE-D |", "|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, d=3: "" if v is None else f"{v:.{d}f}"
    for p in m["per_segment"]:
        L.append(f"| {p['segment']} | {fmt(p['identity'])} | {fmt(p['bg_psnr'],2)} | {fmt(p['bg_ssim'])} | {fmt(p['musiq'],1)} | {fmt(p['zoom'])} | {fmt(p['lse_c'],2)} | {fmt(p['lse_d'],2)} |")
    L += ["", f"Evaluation took {m['eval_seconds']} s on CPU. SyncNet crops use YuNet faces (official LSE uses S3FD), so compare runs with each other rather than with published tables."]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample_every", type=int, default=12, help="identity/zoom sampling (frames)")
    ap.add_argument("--quality_every", type=int, default=25, help="background/MUSIQ/flicker sampling (frames)")
    a = ap.parse_args()
    evaluate(a.video, a.image, a.out, a.sample_every, a.quality_every)
