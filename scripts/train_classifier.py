# scripts/train_classifier.py
import numpy as np, joblib, argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--kpdir', default='../outputs/keypoints_npz')
parser.add_argument('--out', default='../models/classifier.joblib')
args = parser.parse_args()

kp_root = Path(__file__).resolve().parents[1] / args.kpdir
X, y, labels = [], [], []
for label_dir in sorted(kp_root.iterdir()):
    if not label_dir.is_dir(): continue
    labels.append(label_dir.name)
for idx, label_dir in enumerate(sorted(kp_root.iterdir())):
    if not label_dir.is_dir(): continue
    for f in label_dir.glob('*.npz'):
        d = np.load(f)
        kp = d['keypoints']
        kp = kp.flatten()
        X.append(kp)
        y.append(idx)

X = np.array(X)
y = np.array(y)
print('Samples:', X.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
print('Train done')
# eval
pred = clf.predict(X_val)
print(classification_report(y_val, pred, target_names=labels))
joblib.dump({'model':clf, 'labels':labels}, Path(__file__).resolve().parents[1] / args.out)
print('Saved classifier to', args.out)