# Supplementary: run the notebook's own test()/IOU() on the labelled train & val splits.
import nbformat, numpy as np, pandas as pd, os
nb = nbformat.read(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mtl_seg_dep_local.ipynb"), 4)
code = [c.source for c in nb.cells if c.cell_type == 'code']
g = {}
exec("import matplotlib; matplotlib.use('Agg')\nimport cv2, numpy as np, os\nfrom matplotlib import pyplot as plt", g)
for s in code:
    if 'class resunet_mtl' in s or 'class myData' in s or 'class AverageMeter' in s:
        exec(s.replace('print(resunet_mtl())', ''), g)
    if 'def IOU' in s:
        exec(s.split('#BUAT DATA BATCH')[0], g)
exec("from torch import torch\nfrom collections import OrderedDict\nfrom torch.utils.data import DataLoader\nimport pandas as pd, time", g)
torch = g['torch']
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/") + "/"
model = g['resunet_mtl']()
model.load_state_dict(torch.load(root + 'trained_model/best_model.pth', map_location='cpu'))
model.eval()
rows = []
for part in ['test', 'val', 'train']:
    ds = g['myData'](root_dir=root + 'cityscapes/', part=part)
    g['save_dir'] = root + f'inference_results/{part}_'
    res = g['test'](g['DataLoader'](ds, batch_size=1, shuffle=False, num_workers=0), model, torch.device('cpu'))
    inter = np.zeros(20); union = np.zeros(20); correct = tot = 0; mae_valid = []
    with torch.no_grad():
        for i in range(len(ds)):
            d = ds[i]
            ps, pdp = model(torch.from_numpy(d['rgb'][None]).float())
            p = ps[0].argmax(0).numpy(); t = np.argmax(d['seg'], 0)
            m = t > 0  # ignore unlabelled / 'None'
            for k in range(1, 20):
                inter[k] += ((p == k) & (t == k) & m).sum(); union[k] += (((p == k) | (t == k)) & m).sum()
            correct += (p[m] == t[m]).sum(); tot += m.sum()
            gt = d['dep'][0]; v = gt < 1.0  # pixels with valid disparity (not clipped at 100 m)
            mae_valid.append(np.abs(pdp[0, 0].numpy()[v] - gt[v]).mean() * 100)
    present = union[1:] > 0
    ciou = inter[1:][present] / union[1:][present]
    rows.append(dict(split=part, n=len(ds), iou_nb=res['iou'], mae_nb=res['mae'], mae_m=res['mae'] * 100,
                     mae_valid_m=np.mean(mae_valid), miou=ciou.mean() if tot else np.nan,
                     pix_acc=correct / tot if tot else np.nan, labelled_pct=100 * tot / (len(ds) * 256 * 512)))
    if part in ('train', 'val'):
        pd.DataFrame({'class': np.array(g['SEG_MAP']['classes'][1:])[present], 'iou': ciou}).to_csv(root + 'inference_results/' + part + '_per_class_iou.csv', index=False)
df = pd.DataFrame(rows); print(df.to_string(index=False)); df.to_csv(root + 'inference_results/summary_all_splits.csv', index=False)
print(pd.read_csv(root+'inference_results/val_per_class_iou.csv').round(3).to_string(index=False))
