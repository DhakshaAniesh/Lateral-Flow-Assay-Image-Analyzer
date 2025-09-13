#!/usr/bin/env python3
"""
lfa_toolkit.py

Modes:
  --mode repair     : Convert polygon-style label lines to YOLO bbox lines.
                       Usage: python lfa_toolkit.py --mode repair --images /path/images --labels /path/labels
  --mode visualize  : Draw label bboxes on images so you can inspect annotations.
                       Usage: python lfa_toolkit.py --mode visualize --images /path/images --labels /path/labels --out ./annot_vis
  --mode predict    : Run model inference with improved postprocessing and heuristics.
                       Usage: python lfa_toolkit.py --mode predict --model /path/to/best.pt --image /path/to/image_or_folder

Notes:
 - Repair will backup each label file to <file>.bak before overwriting.
 - Predict writes per-model predict outputs to the same exp folder (exp_xxx/runs/predict/).
"""

import argparse
from pathlib import Path
import shutil
import json
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
import math
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import csv
from collections import Counter

# try imports
try:
    import cv2
except Exception:
    print("ERROR: missing opencv. Install: pip install opencv-python")
    raise

try:
    from ultralytics import YOLO
except Exception:
    print("ERROR: missing ultralytics. Install: pip install ultralytics")
    raise

# ----------------------------
# Config (tweak if needed)
# ----------------------------
# Default per-class confidence thresholds (you can override via CLI in predict mode)
CLASS_CONF_THRESH_DEFAULT = {
    0: 0.14,   # control zone (relaxed)
    1: 0.40,   # test zone  (stricter)
    2: 0.25    # wicking pad
}

# Per-class area fraction ranges (min_frac, max_frac) relative to image area.
# These remove absurdly small or huge boxes for each class.
CLASS_AREA_FRACS = {
    0: (0.0002, 0.06),   # control zone small
    1: (0.0002, 0.06),   # test zone small
    2: (0.0008, 0.6)     # wicking pad larger but not full image
}

# postprocessing defaults
IOU_THR = 0.35
MAX_PER_CLASS = 1
USE_DEVICE_DEFAULT = 0  # GPU device index or "cpu"
DRAW_THICKNESS = 3
FONT_SCALE = 1.0
# ----------------------------

def parse_floats(tokens: List[str]) -> List[float]:
    return [float(t) for t in tokens]

# ----------------------------
# LABEL REPAIR: polygon -> bbox
# ----------------------------
def repair_labels_for_image(label_path: Path, img_path: Path, overwrite: bool = True) -> bool:
    """
    Reads a label file; if lines are not standard YOLO 5-token format (class cx cy w h),
    tries to convert polygon-style lines (class x1 y1 x2 y2 ...) -> bbox.
    If number values appear > 1, we treat them as absolute pixel coords and normalize
    using img size. Backups original to .bak if overwriting.
    Returns True if file was modified.
    """
    if not label_path.exists():
        return False

    img = cv2.imread(str(img_path))
    if img is None:
        # cannot load image, skip
        return False
    H, W = img.shape[:2]

    with label_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    new_lines = []
    modified = False

    for ln in lines:
        parts = ln.split()
        if len(parts) == 5:
            # assume correct already (class cx cy w h). Just keep.
            new_lines.append(" ".join(parts))
            continue
        # otherwise, try to interpret as polygon / multi-point or unexpected line
        try:
            cls_id = int(parts[0])
            coords = parse_floats(parts[1:])
            # coords should be pairs (x,y). If odd count, drop last.
            if len(coords) < 6:
                # suspicious small number of coords, just skip line
                continue
            if len(coords) % 2 == 1:
                coords = coords[:-1]
            xs = coords[0::2]
            ys = coords[1::2]
            # detect if coords are normalized (<=1) or absolute (>1)
            as_abs = any(c > 1.0 for c in xs+ys)
            if as_abs:
                # assume absolute px coords -> normalize
                xs_norm = [float(x)/W for x in xs]
                ys_norm = [float(y)/H for y in ys]
            else:
                xs_norm = xs
                ys_norm = ys
            minx = max(0.0, min(xs_norm))
            miny = max(0.0, min(ys_norm))
            maxx = min(1.0, max(xs_norm))
            maxy = min(1.0, max(ys_norm))
            cx = (minx + maxx) / 2.0
            cy = (miny + maxy) / 2.0
            w = maxx - minx
            h = maxy - miny
            # Sanity: if bbox degenerate, skip
            if w <= 0 or h <= 0:
                continue
            new_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            modified = True
        except Exception:
            # couldn't parse line; skip it to be safe
            continue

    if modified and overwrite:
        # backup
        bak = label_path.with_suffix(label_path.suffix + ".bak")
        if not bak.exists():
            shutil.copyfile(label_path, bak)
        with label_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
    return modified

def repair_labels_folder(images_dir: Path, labels_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    """
    Iterate over images and corresponding label txts, attempt repair. Returns summary dict.
    """
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]])
    summary = {"processed": 0, "modified": 0, "errors": []}
    for img in images:
        lbl = labels_dir / (img.stem + ".txt")
        try:
            modified = repair_labels_for_image(lbl, img, overwrite=not dry_run)
            summary["processed"] += 1
            if modified:
                summary["modified"] += 1
        except Exception as e:
            summary["errors"].append({"image": str(img), "error": str(e)})
    return summary

# ----------------------------
# VISUALIZE ANNOTATIONS
# ----------------------------
def visualize_annotations(images_dir: Path, labels_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]])
    for img in imgs:
        lbl = labels_dir / (img.stem + ".txt")
        img_cv = cv2.imread(str(img))
        if img_cv is None:
            continue
        H, W = img_cv.shape[:2]
        if lbl.exists():
            with lbl.open("r", encoding="utf-8") as f:
                for ln in f:
                    parts = ln.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = int(parts[0])
                        cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
                        x1 = int((cx - w/2.0) * W); y1 = int((cy - h/2.0) * H)
                        x2 = int((cx + w/2.0) * W); y2 = int((cy + h/2.0) * H)
                        color = (0,255,0) if cid==2 else (255,0,0)
                        cv2.rectangle(img_cv, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(img_cv, str(cid), (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception:
                        continue
        out_p = out_dir / img.name
        cv2.imwrite(str(out_p), img_cv)
    return out_dir

# ----------------------------
# PREDICTION / postprocessing
# ----------------------------
def iou_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA); interH = max(0.0, yB - yA)
    interA = interW * interH
    areaA = max(0.0, (boxA[2]-boxA[0])) * max(0.0, (boxA[3]-boxA[1]))
    areaB = max(0.0, (boxB[2]-boxB[0])) * max(0.0, (boxB[3]-boxB[1]))
    union = areaA + areaB - interA
    if union <= 0.0:
        return 0.0
    return interA / union

def greedy_nms_keep_top(boxes: np.ndarray, scores: np.ndarray, max_keep: int, iou_thresh: float):
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    for idx in order:
        b = boxes[idx]
        skip = False
        for k in keep:
            if iou_xyxy(b, boxes[k]) > iou_thresh:
                skip = True
                break
        if not skip:
            keep.append(idx)
        if len(keep) >= max_keep:
            break
    return keep

def postprocess_one_image(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, img_wh: Tuple[int,int],
                          class_conf_thresh: Dict[int,float], class_area_fracs: Dict[int,Tuple[float,float]],
                          iou_thr: float = IOU_THR, max_per_class: int = MAX_PER_CLASS):
    """
   Apply per-class confidence thresholds, class area filters, per-class greedy NMS,
   then additional spatial consistency with wicking-pad.
   boxes: Nx4 in xyxy pixel coords
   classes, scores: N
   img_wh: (W,H)
   """

    W, H = img_wh
    area_img = max(1.0, W*H)

    # 1) Confidence filter
    keep_idx = [i for i in range(len(scores)) if scores[i] >= class_conf_thresh.get(int(classes[i]), 0.25)]
    if not keep_idx:
        return []
    boxes_f = boxes[keep_idx]; scores_f = scores[keep_idx]; classes_f = classes[keep_idx].astype(int)

    # 2) Area filter
    areas = (boxes_f[:,2]-boxes_f[:,0]) * (boxes_f[:,3]-boxes_f[:,1])
    keep_mask = np.ones(len(areas), dtype=bool)
    for i,cls in enumerate(classes_f):
        min_frac,max_frac = class_area_fracs.get(int(cls),(0.0,1.0))
        if areas[i] < (min_frac * area_img) or areas[i] > (max_frac * area_img):
            keep_mask[i] = False
    boxes_f = boxes_f[keep_mask]; scores_f = scores_f[keep_mask]; classes_f = classes_f[keep_mask]
    if len(boxes_f) == 0:
        return []

    # 3) Per-class NMS
    final_boxes, final_scores, final_classes = [], [], []
    for cls in np.unique(classes_f):
        cls_mask = np.where(classes_f==cls)[0]
        cls_boxes = boxes_f[cls_mask]; cls_scores = scores_f[cls_mask]
        local_keep = greedy_nms_keep_top(cls_boxes, cls_scores, max_keep=max_per_class, iou_thresh=iou_thr)
        for lk in local_keep:
            final_boxes.append(cls_boxes[lk])
            final_scores.append(cls_scores[lk])
            final_classes.append(cls)
    if not final_boxes:
        return []
    final_boxes = np.array(final_boxes); final_scores = np.array(final_scores); final_classes = np.array(final_classes)

    # 4) Wicking pad alignment
    pad_idx = None
    for i, c in enumerate(final_classes):
        if c == 2:
            if pad_idx is None or final_scores[i] > final_scores[pad_idx]:
                pad_idx = i
    if pad_idx is not None:
        pad_box = final_boxes[pad_idx]
        pad_cy = (pad_box[1] + pad_box[3]) / 2.0

        # 5) Fix control vs test overlap
        ctrl_idx = [i for i, c in enumerate(final_classes) if c == 0]
        test_idx = [i for i, c in enumerate(final_classes) if c == 1]

        if ctrl_idx and test_idx:
            ci, ti = ctrl_idx[0], test_idx[0]
            if iou_xyxy(final_boxes[ci], final_boxes[ti]) > 0.6:
                cy_c = (final_boxes[ci][1] + final_boxes[ci][3]) / 2.0
                cy_t = (final_boxes[ti][1] + final_boxes[ti][3]) / 2.0
                dist_c, dist_t = abs(cy_c - pad_cy), abs(cy_t - pad_cy)
                if dist_c < dist_t:
                    final_classes[ti] = -1  # drop test
                else:
                    final_classes[ci] = -1  # drop control
        keep_mask = final_classes != -1
        final_boxes = final_boxes[keep_mask]; final_scores = final_scores[keep_mask]; final_classes = final_classes[keep_mask]

    # 6) Convert to dict
    out = []
    order = final_scores.argsort()[::-1]
    for oi in order:
        out.append({"cls": int(final_classes[oi]), "xyxy": final_boxes[oi].tolist(), "conf": float(final_scores[oi])})
    return out

def draw_and_save(img_path: Path, detections: List[Dict[str,Any]], save_path: Path, class_names: Dict[int,str]):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    H,W = img.shape[:2]
    for d in detections:
        x1,y1,x2,y2 = map(int, d['xyxy'])
        name = class_names.get(d['cls'], str(d['cls']))
        label = f"{name} {d['conf']:.2f}"
        color = (0,200,0) if d['cls']==2 else (255,50,50)  # green for pad, red for others
        cv2.rectangle(img, (x1,y1), (x2,y2), color, DRAW_THICKNESS)
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)
        ty = max(0, y1 - 6)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw, ty + 4), color, -1)
        cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,0), 2, cv2.LINE_AA)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)

# ----------------------------
# Evaluation function
# ----------------------------
def evaluate_predictions(results_map: Dict[str,str], ground_truth_csv: Path, save_base: Path):
    # Load ground truth labels
    gt = {}
    with open(ground_truth_csv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            fname, label = parts
            gt[fname.strip()] = label.strip()

    y_true, y_pred = [], []
    for fname, pred in results_map.items():
        if fname in gt:
            y_true.append(gt[fname])
            y_pred.append(pred)

    if not y_true:
        print("⚠️ No matching ground truth found for evaluation.")
        return

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n📊 Accuracy: {acc*100:.2f}%")

    # Detailed report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))

    # Confusion Matrix
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels],
                            columns=[f"Pred_{l}" for l in labels])
    print("\nConfusion Matrix:")
    print(cm_df)

    # Save metrics
    cm_df.to_csv(save_base / "confusion_matrix.csv")
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(save_base / "detailed_predictions.csv", index=False)
    print(f"\n✅ Metrics saved: {save_base / 'confusion_matrix.csv'} & {save_base / 'detailed_predictions.csv'}")

# ----------------------------
# Predict wrapper
# ----------------------------
def predict_with_model(model_path: Path, source: Path, class_conf_thresh_override: Dict[int, float] = None, device=USE_DEVICE_DEFAULT):
    import re

    # --------------------------
    # Setup
    # --------------------------
    BASE_PRED_DIR = Path(r"C:\Users\Dhaksha Aniesh\Desktop\Misc\YOLO Predictions")

    # Load model
    model = YOLO(str(model_path))
    names_map = {i: n for i, n in model.names.items()}

    # 🔹 Extract model type (v8n, v8s, v8m) from path string
    match = re.search(r"yolov8([nsm])", str(model_path).lower())
    if match:
        model_name = f"v8{match.group(1)}"
    else:
        raise ValueError(f"Could not determine model type from path: {model_path}")

    # Use correct model directory
    model_dir = BASE_PRED_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Auto-increment predict folder number
    existing = [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("predict")]
    run_num = len(existing) + 1
    save_base = model_dir / f"predict{run_num}-{model_name}"
    save_base.mkdir(parents=True, exist_ok=True)

    # CSV paths
    confusion_matrix_path = save_base / "confusion_matrix.csv"
    detailed_csv_path = save_base / "detailed_predictions.csv"

    # Thresholds
    class_conf = CLASS_CONF_THRESH_DEFAULT.copy()
    if class_conf_thresh_override:
        class_conf.update(class_conf_thresh_override)

    # Prepare image files
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted([p for p in source.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]])
    else:
        raise FileNotFoundError(f"{source} not found")

    # --------------------------
    # Prediction Loop
    # --------------------------
    results_map = {}
    detailed_rows = []

    for img_p in files:
        print(f"Predicting: {img_p.name} ...")
        res_list = model.predict(source=str(img_p), conf=0.001, device=device, verbose=False)
        img_cv = cv2.imread(str(img_p))
        H, W = img_cv.shape[:2]

        # Extract detections
        if len(res_list) == 0 or not hasattr(res_list[0], "boxes") or len(res_list[0].boxes) == 0:
            boxes_np = np.zeros((0, 4), dtype=np.float32)
            scores_np = np.zeros((0,), dtype=np.float32)
            classes_np = np.zeros((0,), dtype=np.int32)
        else:
            r = res_list[0]
            try:
                boxes_np = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                scores_np = r.boxes.conf.cpu().numpy().astype(np.float32)
                classes_np = r.boxes.cls.cpu().numpy().astype(np.int32)
            except Exception:
                bxs, scs, cls = [], [], []
                for b in r.boxes:
                    try:
                        xy = b.xyxy[0].cpu().numpy().astype(np.float32)
                        bxs.append(xy)
                        scs.append(float(b.conf[0].cpu().numpy()))
                        cls.append(int(b.cls[0].cpu().numpy()))
                    except Exception:
                        continue
                if len(bxs) == 0:
                    boxes_np = np.zeros((0, 4), dtype=np.float32)
                    scores_np = np.zeros((0,), dtype=np.float32)
                    classes_np = np.zeros((0,), dtype=np.int32)
                else:
                    boxes_np = np.vstack(bxs); scores_np = np.array(scs); classes_np = np.array(cls)

        # Postprocess
        final = postprocess_one_image(
            boxes_np, scores_np, classes_np, (W, H),
            class_conf, CLASS_AREA_FRACS, iou_thr=IOU_THR, max_per_class=MAX_PER_CLASS
        )

        # Save annotated image
        out_img_path = save_base / img_p.name
        draw_and_save(img_p, final, out_img_path, names_map)

        # Decide result
        det_names = [names_map.get(d['cls'], str(d['cls'])) for d in final]
        has_c = any("control" in dn.lower() for dn in det_names)
        has_t = any("test" in dn.lower() for dn in det_names)

        if not has_c and not has_t:
            final_result = "Invalid (No valid bands)"
        elif not has_c and has_t:
            final_result = "Invalid (Test line without control)"
        elif has_c and has_t:
            final_result = "Positive Result"
        elif has_c and not has_t:
            final_result = "Negative Result"
        else:
            final_result = "Invalid (Unexpected)"

        results_map[img_p.name] = final_result
        detailed_rows.append([img_p.name, final_result, [(names_map.get(d['cls'], str(d['cls'])), d['conf']) for d in final]])

    # --------------------------
    # Save CSV Reports
    # --------------------------
    with open(detailed_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Final Result", "Detections (class, conf)"])
        for row in detailed_rows:
            writer.writerow(row)

    counts = Counter(results_map.values())
    with open(confusion_matrix_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Count"])
        for k, v in counts.items():
            writer.writerow([k, v])

    print(f"\n✅ Results saved to {save_base}")
    print(f"   - Annotated images inside this folder")
    print(f"   - Detailed CSV: {detailed_csv_path}")
    print(f"   - Confusion CSV: {confusion_matrix_path}")

    return results_map, save_base

# ----------------------------
# CLI / main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="LFA dataset repair / visualize / predict toolkit")
    p.add_argument("--mode", required=True, choices=["repair","visualize","predict"])
    p.add_argument("--images", help="Path to images folder (required for repair/visualize)")
    p.add_argument("--labels", help="Path to labels folder (required for repair/visualize)")
    p.add_argument("--out", help="Out dir for visualizations (optional)")
    p.add_argument("--dry", action="store_true", help="Dry-run for repair")
    p.add_argument("--model", help="Path to trained model .pt (for predict)")
    p.add_argument("--image", help="Path to image file or folder to predict on (for predict)")
    p.add_argument("--device", default=None, help="Device for prediction: gpu index (0) or 'cpu'")
    p.add_argument("--ground_truth", help="CSV file with ground truth labels for evaluation")

    args = p.parse_args()

    if args.mode in ("repair","visualize"):
        if not args.images or not args.labels:
            print("ERROR: --images and --labels required for repair/visualize")
            sys.exit(1)
        images_dir = Path(args.images)
        labels_dir = Path(args.labels)
        if args.mode == "repair":
            print("Repairing labels... (backup created as .bak)")
            summary = repair_labels_folder(images_dir, labels_dir, dry_run=args.dry)
            print("Done. Summary:", summary)
        else:
            out_dir = Path(args.out) if args.out else Path("annot_vis")
            print(f"Visualizing annotations -> {out_dir}")
            visualize_annotations(images_dir, labels_dir, out_dir)
            print("Visualization complete.")
    elif args.mode == "predict":
        if not args.model or not args.image:
            print("ERROR: --model and --image required for predict")
            sys.exit(1)
        model_p = Path(args.model)
        src = Path(args.image)
        device = USE_DEVICE_DEFAULT
        if args.device:
            device = args.device
        else:
            try:
                import torch
                if not torch.cuda.is_available():
                    device = "cpu"
                else:
                    device = USE_DEVICE_DEFAULT
            except Exception:
                device = "cpu"

        results, save_base = predict_with_model(model_p, src, device=device)

        # 👇 Add this to see per-image results in CMD
        print("\nSummary:")
        for k, v in results.items():
            print(f"  {k} -> {v}")

        if args.ground_truth:
            evaluate_predictions(results, Path(args.ground_truth), save_base)

if __name__ == "__main__":
    main()
