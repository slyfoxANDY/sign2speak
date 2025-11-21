# scripts/capture_images.py
# Quick tool to capture labelled images for annotation/training.
import cv2, argparse, os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='../data/images', help='output folder relative to scripts')
parser.add_argument('--label', required=True, help='label name (subfolder)')
parser.add_argument('--count', type=int, default=100)
args = parser.parse_args()

out = Path(__file__).resolve().parents[1] / args.out
label_dir = out / args.label
label_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
print('Press space to capture one frame. q to quit.')
idx = len(list(label_dir.glob('*.jpg')))
while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.putText(frame, f'Label: {args.label} Count: {idx}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Capture', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == 32:  # space
        fname = label_dir / f'{args.label}_{idx:04d}.jpg'
        cv2.imwrite(str(fname), frame)
        print('Saved', fname)
        idx += 1
        if idx >= args.count:
            break
cap.release()
cv2.destroyAllWindows()