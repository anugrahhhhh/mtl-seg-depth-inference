import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/") + "/"
OUT = ROOT + "Report_Inference_MTL_Seg_Depth.pdf"
NAME = "Anugrah Putra Al Fatih"
NIM = "26/590120/PPA/07385"

ss = getSampleStyleSheet()
body = ParagraphStyle('b', parent=ss['Normal'], fontName='Helvetica', fontSize=9.4, leading=12.6,
                      alignment=TA_JUSTIFY, spaceAfter=4)
h1 = ParagraphStyle('h1', parent=ss['Heading2'], fontName='Helvetica-Bold', fontSize=11.5,
                    leading=14, spaceBefore=7, spaceAfter=3, textColor=colors.HexColor('#1F3A5F'))
title = ParagraphStyle('t', parent=ss['Title'], fontName='Helvetica-Bold', fontSize=15, leading=19,
                       spaceAfter=2, textColor=colors.HexColor('#1F3A5F'))
sub = ParagraphStyle('s', parent=body, alignment=TA_CENTER, fontSize=9.4, textColor=colors.HexColor('#444444'))
cap = ParagraphStyle('c', parent=body, fontSize=8.3, leading=10.5, alignment=TA_CENTER,
                     textColor=colors.HexColor('#444444'), spaceBefore=2, spaceAfter=6)
bullet = ParagraphStyle('bl', parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)
cell = ParagraphStyle('cell', parent=body, fontSize=8.3, leading=10, alignment=TA_CENTER, spaceAfter=0)

HEAD = colors.HexColor('#1F3A5F')
GRID = colors.HexColor('#B8C2CC')
ZEBRA = colors.HexColor('#F2F5F8')


def table(data, widths, bold_last=False, align_first_left=True):
    t = Table(data, colWidths=widths, hAlign='CENTER')
    st = [('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 8.3),
          ('FONT', (0, 1), (-1, -1), 'Helvetica', 8.3),
          ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
          ('BACKGROUND', (0, 0), (-1, 0), HEAD),
          ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
          ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('GRID', (0, 0), (-1, -1), 0.4, GRID),
          ('TOPPADDING', (0, 0), (-1, -1), 2.2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2)]
    if align_first_left:
        st.append(('ALIGN', (0, 1), (0, -1), 'LEFT'))
    for r in range(1, len(data)):
        if r % 2 == 0:
            st.append(('BACKGROUND', (0, r), (-1, r), ZEBRA))
    if bold_last:
        st += [('FONT', (0, -1), (-1, -1), 'Helvetica-Bold', 8.3),
               ('LINEABOVE', (0, -1), (-1, -1), 0.9, HEAD)]
    t.setStyle(TableStyle(st))
    return t


def header_footer(c, doc):
    w, h = A4
    c.saveState()
    c.setStrokeColor(HEAD); c.setLineWidth(0.6)
    c.setFont('Helvetica', 8); c.setFillColor(colors.HexColor('#333333'))
    c.drawString(2 * cm, h - 1.25 * cm, "Inference Testing Report: Multi-task Segmentation & Depth (ResUNet-MTL)")
    c.drawRightString(w - 2 * cm, h - 1.25 * cm, f"{NAME}  |  NIM {NIM}")
    c.line(2 * cm, h - 1.4 * cm, w - 2 * cm, h - 1.4 * cm)
    c.line(2 * cm, 1.45 * cm, w - 2 * cm, 1.45 * cm)
    c.drawString(2 * cm, 1.05 * cm, f"Name: {NAME}   NIM: {NIM}")
    c.drawRightString(w - 2 * cm, 1.05 * cm, f"Page {doc.page}")
    c.restoreState()


# ------------------------------------------------------------------ data
test = pd.read_csv(ROOT + "trained_model/test_log.csv")
summ = pd.read_csv(ROOT + "inference_results/summary_all_splits.csv").set_index('split')
pc_tr = pd.read_csv(ROOT + "inference_results/train_per_class_iou.csv").set_index('class')['iou']
pc_va = pd.read_csv(ROOT + "inference_results/val_per_class_iou.csv").set_index('class')['iou']
tv = pd.read_csv(ROOT + "resunet_mtl_0/trainval_log.csv")
avg = test[test.batch == 'avg'].iloc[0]

S = []
S.append(Paragraph("Inference Testing of a Multi-task CNN for Semantic Segmentation and Depth Estimation on Cityscapes", title))
S.append(Paragraph(f"{NAME} &nbsp;&nbsp;|&nbsp;&nbsp; NIM {NIM}", sub))
S.append(Spacer(1, 6))

S.append(Paragraph("1. Objective", h1))
S.append(Paragraph(
    "This report covers the <b>inference (testing) stage</b> of the notebook <i>mtl_seg_dep_colab.ipynb</i>. "
    "The trained weights <i>trained_model/best_model.pth</i> were loaded into the ResUNet-MTL network, and the network "
    "predicted a 20-class semantic segmentation map and a dense depth map for every image in the Cityscapes test split. "
    "Training was <b>not</b> re-run. The training/validation log from the original Colab session is included only for "
    "context (Section 5).", body))

S.append(Paragraph("2. Model, Data and Setup", h1))
S.append(Paragraph(
    "<b>Architecture.</b> A shared ResNet-34 encoder (ImageNet-normalised RGB input) feeds two U-Net-style decoders. "
    "Each decoder has four ConvBlocks (Conv-BN-ReLU x2) with bilinear upsampling and skip connections from encoder "
    "stages 0-3. The segmentation head ends in a 1x1 conv to 20 channels with a sigmoid. The depth head ends in a 1x1 "
    "conv to 1 channel with a ReLU (depth normalised to 0-1, where 1 = 100 m). Total: <b>27.78 M parameters</b>.", body))
S.append(Paragraph(
    "<b>Data.</b> Cityscapes subset with 5 images per split (train = Aachen, val = Frankfurt, test = Berlin). Images "
    "are downsampled 4x to 256x512. The 34 Cityscapes label IDs are mapped to 20 classes (19 + <i>None</i>). "
    "Disparity is converted to depth with depth = (0.209313 x 2262.52) / disparity, clipped at 100 m.", body))
S.append(Paragraph(
    "<b>Environment and changes for local execution.</b> The notebook ran locally on CPU (PyTorch 2.14.1, Windows) "
    "instead of Colab. The only changes were: Google-Drive mount and paths replaced with the local folder; "
    "<i>device</i> falls back to CPU; <i>map_location</i> added to <i>torch.load</i>; "
    "<i>resnet34(pretrained=False)</i> (the ImageNet weights are overwritten by <i>best_model.pth</i> anyway); "
    "<i>num_workers=0</i> (Windows); and the training, model-graph and TensorBoard cells were skipped. "
    "The four per-image results stored in the original Colab <i>test_log.csv</i> (images 0-3) are reproduced to "
    "7 decimal places, so the local run matches the original.", body))

S.append(Paragraph("3. Evaluation Metrics", h1))
for b in [
    "<b>IoU</b> (notebook definition): the sigmoid output and the one-hot ground truth are binarised at 0.5 over "
    "<i>all</i> 20 channels, then IoU = |P AND G| / |P OR G|. Higher is better.",
    "<b>MAE</b>: mean absolute error between predicted and ground-truth normalised depth (x100 = metres). Lower is better.",
    "<b>TM</b> (total metric) = (1 - IoU) + MAE. Lower is better.",
    "Supplementary (labelled splits only): <b>mIoU</b> over classes present (argmax prediction, <i>None</i> pixels "
    "ignored), <b>pixel accuracy</b>, and MAE restricted to pixels with valid disparity.",
]:
    S.append(Paragraph(b, bullet, bulletText='\u2022'))

S.append(Paragraph("4. Results", h1))
S.append(Paragraph("<b>4.1 Test split (notebook section E, \"Test ke seluruh sample\")</b>", body))
rows = [["Sample", "IoU", "MAE (norm.)", "MAE (m)", "TM", "Inference time (s)"]]
names = ["berlin_000000", "berlin_000001", "berlin_000002", "berlin_000003", "berlin_000004"]
for i, r in test[test.batch != 'avg'].reset_index(drop=True).iterrows():
    rows.append([names[i], f"{r.test_seg:.4f}", f"{r.test_mae:.4f}", f"{r.test_mae*100:.2f}",
                 f"{r.test_tm:.4f}", f"{r.elapsed_time:.3f}"])
rows.append(["Average", f"{avg.test_seg:.4f}", f"{avg.test_mae:.4f}", f"{avg.test_mae*100:.2f}",
             f"{avg.test_tm:.4f}", f"{avg.elapsed_time:.3f}"])
S.append(KeepTogether([table(rows, [3.3*cm, 1.8*cm, 2.2*cm, 1.8*cm, 1.8*cm, 3.0*cm], bold_last=True),
                       Paragraph("Table 1. Per-image test results (CPU, batch size 1, 256x512 input).", cap)]))
S.append(Paragraph(
    f"Average test depth MAE is <b>{avg.test_mae:.4f}</b> (about <b>{avg.test_mae*100:.2f} m</b>). Restricted to pixels "
    f"with a valid disparity measurement, it is <b>{summ.loc['test','mae_valid_m']:.2f} m</b>. A forward pass takes "
    f"about <b>{avg.elapsed_time:.2f} s per image</b> on CPU (about {1/avg.elapsed_time:.1f} FPS). The test IoU is "
    f"only <b>{avg.test_seg:.4f}</b>, but this is <b>not</b> a sign of a bad model. Cityscapes does not publish test "
    "labels: every Berlin <i>gtFine_labelIds</i> file is 100% unlabelled (class <i>None</i>), so the IoU only measures how "
    "many pixels the model calls <i>None</i>. Figure 1 shows that the predicted segmentations are in fact sensible. "
    "To get a meaningful segmentation score, the same <i>test()</i> function was also run on the labelled splits "
    "(Table 2).", body))

img_w = 17 * cm
S.append(KeepTogether([
    Image(ROOT + "inference_results/qualitative_test.png", width=img_w, height=img_w * 1312 / 2550),
    Paragraph("Figure 1. Qualitative inference on all 5 test images. From left: input RGB, ground-truth segmentation "
              "(empty: test labels are withheld), predicted segmentation, ground-truth depth from disparity "
              "(log scale, white = far, holes = missing disparity), predicted depth (log scale).", cap)]))

S.append(Paragraph("<b>4.2 Summary across splits (same weights, same metric code)</b>", body))
rows = [["Split (city)", "Seg. labels", "IoU (nb)", "mIoU", "Pixel acc.", "MAE (m)", "MAE valid px (m)"]]
for sp, city in [("test", "Berlin"), ("val", "Frankfurt"), ("train", "Aachen")]:
    r = summ.loc[sp]
    lab = "none" if r.labelled_pct == 0 else f"{r.labelled_pct:.0f}% px"
    f = (lambda v, k=3: "n/a" if pd.isna(v) else f"{v:.{k}f}")
    rows.append([f"{sp} ({city})", lab, f(r.iou_nb, 4), f(r.miou), f(r.pix_acc * 100 if not pd.isna(r.pix_acc) else r.pix_acc, 1) + ("%" if not pd.isna(r.pix_acc) else ""),
                 f"{r.mae_m:.2f}", f"{r.mae_valid_m:.2f}"])
S.append(KeepTogether([table(rows, [2.9*cm, 2.0*cm, 1.8*cm, 1.6*cm, 2.0*cm, 1.8*cm, 2.8*cm]),
                       Paragraph("Table 2. Inference on all three 5-image splits. The val split (Frankfurt) is the "
                                 "most reliable held-out estimate for segmentation (see Section 6).", cap)]))

classes = [c for c in pc_va.index.union(pc_tr.index)
           if c in ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                    'motorcycle', 'bicycle']]
order = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
         'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
classes = [c for c in order if c in classes]
half = (len(classes) + 1) // 2
g = lambda s, c: f"{s[c]:.2f}" if c in s.index else "-"
rows = [["Class", "IoU val", "IoU train", "Class", "IoU val", "IoU train"]]
for i in range(half):
    a = classes[i]; b = classes[i + half] if i + half < len(classes) else None
    rows.append([a, g(pc_va, a), g(pc_tr, a)] + ([b, g(pc_va, b), g(pc_tr, b)] if b else ["", "", ""]))
t = table(rows, [2.6*cm, 1.6*cm, 1.7*cm, 2.6*cm, 1.6*cm, 1.7*cm])
t.setStyle(TableStyle([('ALIGN', (3, 1), (3, -1), 'LEFT'), ('LINEBEFORE', (3, 0), (3, -1), 1.0, HEAD)]))
S.append(KeepTogether([t, Paragraph("Table 3. Per-class IoU on the labelled splits (\"-\" = class absent "
                                    "from both GT and prediction in that split).", cap)]))

S.append(Paragraph("5. Training/Validation Log (optional, from the original Colab run)", h1))
S.append(Paragraph(
    f"The notebook's demo training (2 epochs, Adam lr 0.0025 with cosine annealing, batch 1, trained on the 5 val "
    f"images) reduced training loss from {tv.train_loss_total[0]:.3f} to {tv.train_loss_total[1]:.3f} "
    f"(BCE {tv.train_loss_seg[0]:.3f} to {tv.train_loss_seg[1]:.3f}; L1 depth {tv.train_loss_dep[0]:.3f} to "
    f"{tv.train_loss_dep[1]:.3f}). Validation loss fell from {tv.val_loss_total[0]:.2f} to "
    f"{tv.val_loss_total[1]:.2f}. Its very large BCE term shows that 2 epochs on 5 images is far from converged. "
    "That checkpoint (<i>resunet_mtl_0/model.pth</i>) is a demo only; this report evaluates the provided fully trained "
    "<i>best_model.pth</i>.", body))

S.append(Paragraph("6. Discussion", h1))
for b in [
    "<b>Depth estimation works well.</b> Average errors are 3.6-6.8 m on a 0-100 m range. Predicted depth maps are smooth "
    "and complete; they even fill the holes and noisy speckle of the stereo-disparity ground truth (Fig. 1). The "
    "largest errors are on far background and sky, where the ground truth is clipped at 100 m or missing.",
    "<b>Segmentation is strong for large \"stuff\" classes</b> (road 0.95, sky 0.96, building 0.92, vegetation 0.90, "
    "car 0.91 IoU on val) and weak for thin or rare classes (pole, traffic light, rider, fence). This is expected "
    "with a 4x-downsampled input and per-channel BCE loss without class weighting. Several val classes score 0 "
    "because they cover only a few pixels in the 5 images (wall, terrain, truck, bus, train).",
    "<b>The train split scores higher than val</b> (mIoU 0.61 vs 0.38, MAE 3.6 m vs 6.8 m). The provided weights were most "
    "likely trained on the official Cityscapes train set, which includes Aachen, so Frankfurt (val) is the fairer "
    "held-out estimate. With only 5 images per split, all numbers have high variance.",
    "<b>The notebook IoU is a coarse metric.</b> Because it pools all 20 binarised channels into one score, it is dominated "
    "by large classes and is meaningless on the label-less test split. mIoU and pixel accuracy on labelled data give a "
    "more informative picture.",
]:
    S.append(Paragraph(b, bullet, bulletText='\u2022'))

S.append(Paragraph("7. Conclusion", h1))
S.append(Paragraph(
    f"The trained ResUNet-MTL ran successfully in inference-only mode on CPU (about {avg.elapsed_time:.2f} s/image) and "
    f"reproduces the results stored in the original Colab test log. On the Cityscapes test split it achieves a depth MAE of "
    f"{avg.test_mae:.4f} (about {avg.test_mae*100:.1f} m) and TM = {avg.test_tm:.4f}. The test IoU "
    f"({avg.test_seg:.4f}) cannot be interpreted because test labels are withheld. On labelled held-out data (val) the "
    f"model reaches {summ.loc['val','pix_acc']*100:.1f}% pixel accuracy and notebook-IoU "
    f"{summ.loc['val','iou_nb']:.3f}, and the qualitative results confirm coherent scene parsing and depth. Main "
    "improvements would be higher input resolution, class-balanced or cross-entropy/Dice loss for small classes, and "
    "evaluation on the full 500-image val set.", body))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.9*cm,
                        bottomMargin=1.9*cm, title="Inference Testing Report - ResUNet-MTL", author=NAME)
doc.build(S, onFirstPage=header_footer, onLaterPages=header_footer)
print("written", OUT)
