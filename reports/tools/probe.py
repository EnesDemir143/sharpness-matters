import glob, os
import nibabel as nib, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

root = "/Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix"
rows = []
for site in ["A", "B", "C"]:
    ids = sorted(int(os.path.basename(p).split(".")[0]) for p in glob.glob(f"{root}/Site{site}/*.4.img.nii.gz"))
    for n in ids[:: max(1, len(ids) // 3)][:3]:
        rows.append((site, n))
fig, ax = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
for r, (site, n) in enumerate(rows):
    img = nib.load(f"{root}/Site{site}/{n}.4.img.nii.gz").get_fdata().astype(np.float32)
    msk = nib.load(f"{root}/Site{site}/{n}.5.mask.nii.gz").get_fdata()
    img2 = img[..., 0] if img.ndim == 3 else img
    msk2 = msk[..., 0] if msk.ndim == 3 else msk
    rot = torch.rot90(torch.tensor(img2), 1).numpy()
    rotm = torch.rot90(torch.tensor(msk2), 1).numpy()
    pngs = glob.glob(f"{root}/Site{site}/{n}.1.img.png")
    ax[r, 0].imshow(img2, cmap="gray"); ax[r, 0].set_title(f"Site{site}/{n} raw nifti {img2.shape}", fontsize=8)
    ax[r, 1].imshow(rot, cmap="gray"); ax[r, 1].contour(rotm > 0, colors="r", linewidths=0.5)
    ax[r, 1].set_title("author rot90 k=1 + mask", fontsize=8)
    if pngs:
        ax[r, 2].imshow(np.array(Image.open(pngs[0]).convert("L")), cmap="gray"); ax[r, 2].set_title("dataset .1.img.png", fontsize=8)
    for a in ax[r]:
        a.axis("off")
plt.tight_layout()
plt.savefig("/tmp/ptx_orient/sites.png", dpi=70)
print(rows)
