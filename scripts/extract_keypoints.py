# scripts/extract_keypoints.py
import numpy as np, cv2, argparse, os
from ultralytics import YOLO
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--images', default='../data/images', help='folder with labelled image subfolders')
parser.add_argument('--model', default='../models/yolov8_best.pt', help='yolov8 pose/keypoint model')
parser.add_argument('--out', default='../outputs/keypoints_npz')
args = parser.parse_args()

model = YOLO(args.model)
images_root = Path(__file__).resolve().parents[1] / args.images
out_root = Path(__file__).resolve().parents[1] / args.out
out_root.mkdir(parents=True, exist_ok=True)

# Assumes images are organized in subfolders per label
for label_dir in sorted(images_root.iterdir()):
    if not label_dir.is_dir():
        continue
    label = label_dir.name
    dst = out_root / label
    dst.mkdir(parents=True, exist_ok=True)
    for img_f in sorted(label_dir.glob('*.jpg')):
        img = cv2.imread(str(img_f))
        if img is None:
            continue
        # run model (pose/keypoints mode)
        res = model(img)
        # ultralytics returns .keypoints if pose model; else .masks / boxes
        kp_list = []
        for r in res:
            # r.keypoints: list of keypoints arrays per detection. Newer ultralytics returns r.keypoints or r.keypoints.xy
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                # r.keypoints is an array (n, k, 2) or have .xy attributes
                kps = np.array(r.keypoints.xy) if hasattr(r.keypoints, 'xy') else np.array(r.keypoints)
                # flatten  (k*2)
                if kps.size == 0:
                    continue
                kp_list.append(kps.flatten())
            else:
                # Try boxes -> skip
                pass
        if len(kp_list) == 0:
            # save zeros if no detection
            kparr = np.zeros((1, 21*2), dtype=np.float32)
        else:
            kparr = np.array(kp_list)
        # average if multiple detections — take the highest confidence later; for now take first
        arr = kparr[0]
        # normalize by image size
        h, w = img.shape[:2]
        arr_norm = arr.copy()
        if arr.size > 0:
            for i in range(0, arr.size, 2):
                arr_norm[i] = arr[i] / w
                arr_norm[i+1] = arr[i+1] / h
        # save
        outp = dst / (img_f.stem + '.npz')
        np.savez_compressed(outp, keypoints=arr_norm, imgshape=(h,w))
        print('Saved', outp)