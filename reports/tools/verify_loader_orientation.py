"""§9.1 evidence: B loader (OODTestDataset) orientation vs dataset PNG, mask vs dataset mask PNG.

Run: uv run python reports/tools/verify_loader_orientation.py
Result on 2026-11-23 (PTX-498-v2-fix): 498/498 orientation match; mask IoU 1.0 for 498/498.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sharpness_matters.ptx.datasets.ptx498_dataset import OODTestDataset


def small(a, s=128):
    t = torch.tensor(np.asarray(a, dtype=np.float32))[None, None]
    return F.interpolate(t, size=(s, s), mode="bilinear", antialias=True, align_corners=False)[0, 0].numpy()


def z(t):
    return (t - t.mean()) / (t.std() + 1e-8)


ds = OODTestDataset(img_size=128)
good, ious = 0, []
for i in range(len(ds)):
    item = ds[i]
    site, n = item["image_id"].split("_")
    ref = z(small(np.array(Image.open(f"{ds.root_dir}/Site{site}/{n}.1.img.png").convert("L"))))
    img = item["data"][0].numpy()
    good += float((z(img) * ref).mean()) > float((z(np.fliplr(img).copy()) * ref).mean())
    png_mask = np.array(Image.open(f"{ds.root_dir}/Site{site}/{n}.2.mask.png").convert("L")) > 0
    gt = item["mask"] > 0
    ious.append((png_mask & gt).sum() / max((png_mask | gt).sum(), 1) if png_mask.shape == gt.shape else float("nan"))
print(f"orientation matches dataset PNG: {good}/{len(ds)}")
print(f"mask IoU vs dataset mask PNG: min {np.nanmin(ious):.4f}, median {np.nanmedian(ious):.4f}, shape mismatches {int(np.isnan(ious).sum())}")
