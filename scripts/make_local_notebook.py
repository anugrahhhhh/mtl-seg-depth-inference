import json, sys, nbformat
src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
C = nb.cells
def setsrc(i, s): C[i].source = s
setsrc(1, "# [LOCAL RUN] Colab drive mount removed - data is read from the local folder\nimport os")
setsrc(3, C[3].source.replace(
  'rootdir = "/content/drive/My Drive/UGM/07 TRAINING/TRAINER/2025/Pelatihan Python DL Jogja/Materi/Day 3 - DL dan AI/code and data/seg_dep/"',
  'rootdir = "C:/Users/ASUS/Downloads/seg_dep/"  # [LOCAL RUN]'))
# cell 22: keep imports + dataset_dir; no training objects needed for inference-only
setsrc(22, """# [LOCAL RUN] inference-only: training config (optimizer, train/val loaders, SummaryWriter) skipped
from torch import torch
from collections import OrderedDict
from torch.utils.data import DataLoader
import pandas as pd
import os
dataset_dir = rootdir+"cityscapes/"
""")
setsrc(24, """# [LOCAL RUN] model graph / torchsummary skipped; only set the device
import time
from torch import torch, nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "| torch", torch.__version__)
""")
setsrc(30, C[30].source.replace("torch.load(save_dir+'best_model.pth')",
                                "torch.load(save_dir+'best_model.pth', map_location=device)"))
setsrc(32, C[32].source.replace("num_workers=2", "num_workers=0"))
# extra cell: save a qualitative grid of all test images for the report
extra = nbformat.v4.new_code_cell("""# [LOCAL RUN] extra: qualitative results for all test samples (for the report)
fig, axes = plt.subplots(len(test_set), 5, figsize=(17, 1.75*len(test_set)))
titles = ['Input RGB', 'GT segmentation', 'Pred segmentation', 'GT depth (log)', 'Pred depth (log)']
model.eval()
with torch.no_grad():
    for r in range(len(test_set)):
        d = test_set[r]
        x = torch.from_numpy(d['rgb'][None]).to(device, dtype=torch.float)
        ps, pd_ = model(x)
        rgb = d['rgb'].transpose(1,2,0)[:, :, ::-1]
        ims = [rgb, vis_seg(np.argmax(d['seg'], 0), SEG_MAP).astype(np.uint8),
               vis_seg(np.argmax(ps.cpu().numpy()[0], 0), SEG_MAP).astype(np.uint8),
               logdepth(d['dep'][0]).astype(np.uint8),
               logdepth(np.clip(pd_.cpu().numpy()[0,0], 1e-6, None)).astype(np.uint8)]
        for c in range(5):
            axes[r, c].imshow(ims[c]); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0: axes[r, c].set_title(titles[c], fontsize=10)
        axes[r, 0].set_ylabel(os.path.basename(test_set.rgb[r])[:20], fontsize=7)
        print(os.path.basename(test_set.rgb[r]), 'GT labelled px %:', round(100*(np.argmax(d['seg'],0)>0).mean(), 2),
              '| pred classes:', [SEG_MAP['classes'][k] for k in np.unique(np.argmax(ps.cpu().numpy()[0],0))])
plt.tight_layout(); plt.savefig(rootdir+'inference_results/qualitative_test.png', dpi=150); plt.show()
print(pd.read_csv(save_dir+'test_log.csv').to_string(index=False))
""")
# execute only: 0-12 (data inspection), 13-19 (model+utils), 22, 24, 28-32 (test), extra. skip 26, 27 (training)
keep = [i for i in range(len(C)) if i not in (26, 27, 33)]
nb.cells = [C[i] for i in keep] + [extra]
nbformat.write(nb, dst)
nb = nbformat.read(dst, as_version=4)
for c in nb.cells:
    if 'models.resnet34(pretrained=True)' in c.source:
        c.source = c.source.replace('models.resnet34(pretrained=True) #resnet18',
            'models.resnet34(pretrained=False) #[LOCAL RUN] ImageNet weights not needed: all weights are loaded from best_model.pth')
nbformat.write(nb, dst)
