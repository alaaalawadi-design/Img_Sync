"""Crop -> generate -> paste back.

The model generates a full-height crop around the presenter (so the face gets ~1.5x more pixels at the same
720p token budget). Every generated frame is then scaled back into the crop's footprint inside the ORIGINAL
full frame (prepared exactly like the pipeline would prepare the full image: PIL Lanczos resize-to-cover +
center crop to the official 720p bucket). Only footprint edges that lie inside the frame are feathered, so the
output has the same size and framing as a normal full-frame run.
"""
import numpy as np
import cv2
from PIL import Image

from longcat_video.utils.bukcet_config import get_bucket_config

SCALE_FACTOR_SPATIAL = 16  # = vae_scale_factor_spatial(8) * 2, as in LongCatVideoAvatarPipeline.generate_ai2v


def bucket_for(h, w, resolution="720p"):
    cfg = get_bucket_config(resolution, SCALE_FACTOR_SPATIAL)
    key = sorted(cfg, key=lambda k: abs(float(k) - h / w))[0]
    return cfg[key][0]  # (H, W)


def cover_crop(img: Image.Image, W, H):
    """Same as diffusers VaeImageProcessor resize_mode='crop' (Lanczos resize to cover, center crop)."""
    w0, h0 = img.size
    s = max(W / w0, H / h0)
    rw, rh = round(w0 * s), round(h0 * s)
    r = img.resize((rw, rh), Image.LANCZOS)
    ox, oy = (rw - W) // 2, (rh - H) // 2
    return np.asarray(r.crop((ox, oy, ox + W, oy + H))), s, ox, oy


class Compositor:
    def __init__(self, image_path, box, resolution="720p", feather=24):
        self.img = Image.open(image_path).convert("RGB")
        w0, h0 = self.img.size
        self.Hc, self.Wc = bucket_for(h0, w0, resolution)            # final (full-frame) size
        self.bg, s, ox, oy = cover_crop(self.img, self.Wc, self.Hc)
        x0, y0, x1, y1 = box
        self.gh, self.gw = bucket_for(y1 - y0, x1 - x0, resolution)  # size the model generates the crop at
        # crop footprint in final-frame coordinates (float -> int)
        self.fx0, self.fy0 = round(x0 * s - ox), round(y0 * s - oy)
        self.fx1, self.fy1 = round(x1 * s - ox), round(y1 * s - oy)
        fw, fh = self.fx1 - self.fx0, self.fy1 - self.fy0
        # visible part of the footprint
        self.cx0, self.cy0 = max(0, self.fx0), max(0, self.fy0)
        self.cx1, self.cy1 = min(self.Wc, self.fx1), min(self.Hc, self.fy1)
        self.fw, self.fh = fw, fh
        m = np.ones((self.cy1 - self.cy0, self.cx1 - self.cx0), np.float32)
        ramp = np.linspace(0, 1, feather, dtype=np.float32)
        if self.fx1 < self.Wc:  # right edge inside the frame -> feather
            m[:, -feather:] *= ramp[::-1][None]
        if self.fx0 > 0:
            m[:, :feather] *= ramp[None]
        if self.fy1 < self.Hc:
            m[-feather:, :] *= ramp[::-1][:, None]
        if self.fy0 > 0:
            m[:feather, :] *= ramp[:, None]
        self.mask = m[..., None]

    def crop_image(self, box):
        return self.img.crop(tuple(box))

    def set_reference_person(self, ref_mask_full):
        """Person mask of the original image in final-frame coordinates (for paste_mode='person')."""
        self.ref_person = ref_mask_full[self.cy0:self.cy1, self.cx0:self.cx1]

    def __call__(self, frames_u8, person_masks=None, dilate_px=10, feather_px=12):
        """frames_u8: [T, gh, gw, 3] generated crop frames -> [T, Hc, Wc, 3] full frames.
        person_masks (optional, [T, gh, gw] bool): paste only the person (+ soft edge); the rest stays the original."""
        out = np.empty((len(frames_u8), self.Hc, self.Wc, 3), np.uint8)
        bg = self.bg.astype(np.float32)
        for i, f in enumerate(frames_u8):
            g = cv2.resize(f, (self.fw, self.fh), interpolation=cv2.INTER_AREA).astype(np.float32)
            g = g[self.cy0 - self.fy0:self.cy1 - self.fy0, self.cx0 - self.fx0:self.cx1 - self.fx0]
            alpha = self.mask
            if person_masks is not None:
                pm = cv2.resize(person_masks[i].astype(np.uint8), (self.fw, self.fh), interpolation=cv2.INTER_NEAREST) > 0
                pm = pm[self.cy0 - self.fy0:self.cy1 - self.fy0, self.cx0 - self.fx0:self.cx1 - self.fx0]
                alpha = self.mask * person_alpha(pm, self.ref_person, dilate_px, feather_px)[..., None]
            o = bg.copy()
            reg = o[self.cy0:self.cy1, self.cx0:self.cx1]
            o[self.cy0:self.cy1, self.cx0:self.cx1] = reg * (1 - alpha) + g * alpha
            out[i] = np.clip(o + 0.5, 0, 255).astype(np.uint8)
        return out


class PersonSegmenter:
    """DeepLabV3-ResNet101 person masks (torchvision, VOC class 15). Kept in RAM, visits the GPU only while it runs
    (between segments, when the DiT is idle)."""
    def __init__(self, device="cuda"):
        import torch
        import torchvision
        self.torch, self.device = torch, device
        self.m = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT").eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, frames_u8, short_side=520, batch=8):
        torch = self.torch
        T, H, W, _ = frames_u8.shape
        s = short_side / min(H, W)
        h, w = round(H * s), round(W * s)
        out = np.empty((T, H, W), bool)
        self.m.to(self.device)
        try:
            with torch.no_grad():
                for i in range(0, T, batch):
                    x = torch.from_numpy(np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames_u8[i:i + batch]]))
                    x = ((x.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std).to(self.device)
                    lab = self.m(x)["out"].argmax(1).eq(15).to(torch.uint8).cpu().numpy()
                    for j, l in enumerate(lab):
                        out[i + j] = cv2.resize(l, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        finally:
            self.m.to("cpu")
            torch.cuda.empty_cache()
        return out


def person_alpha(gen_mask, ref_mask, dilate_px=10, feather_px=12):
    """Soft alpha for pasting the generated person: union of the person in the generated frame and in the original
    image (so the original person never shows where the generated one moved away), grown a little and feathered.
    Everything outside it stays the ORIGINAL background, pixel for pixel."""
    m = (gen_mask | ref_mask).astype(np.uint8)
    if dilate_px:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)))
    a = m.astype(np.float32)
    if feather_px:
        a = cv2.GaussianBlur(a, (0, 0), feather_px / 2.0)
        a = np.maximum(a, m * 0.0)  # keep range [0,1]
    return np.clip(a, 0, 1)
