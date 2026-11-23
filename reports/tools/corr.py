import glob, os
import nibabel as nib, numpy as np, torch
import torch.nn.functional as F
from PIL import Image

root = "/Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix"


def small(a):
    t = torch.tensor(np.asarray(a, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(128, 128), mode="bilinear", antialias=True, align_corners=False)[0, 0].numpy()
    return (t - t.mean()) / (t.std() + 1e-8)


res = {"author_rot90": 0, "rot90_then_fliplr": 0, "ambiguous": 0, "no_png": 0}
margins = []
for site in "ABC":
    for p in sorted(glob.glob(f"{root}/Site{site}/*.4.img.nii.gz")):
        n = os.path.basename(p).split(".")[0]
        png = f"{root}/Site{site}/{n}.1.img.png"
        if not os.path.exists(png):
            res["no_png"] += 1
            continue
        img = nib.load(p).get_fdata()
        img = img[..., 0] if img.ndim == 3 else img
        a = torch.rot90(torch.tensor(img), 1).numpy()
        ref = small(np.array(Image.open(png).convert("L")))
        ca = float((small(a) * ref).mean())
        cf = float((small(np.fliplr(a).copy()) * ref).mean())
        margins.append(cf - ca)
        if cf - ca > 0.05:
            res["rot90_then_fliplr"] += 1
        elif ca - cf > 0.05:
            res["author_rot90"] += 1
        else:
            res["ambiguous"] += 1
print(res, "min margin", min(margins), "median", float(np.median(margins)))
