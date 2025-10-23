import json, pandas as pd, numpy as np
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib, os

def load_npz_dict(path):
    z = np.load(path, allow_pickle=True)
    feats = pd.DataFrame(z["feats"].item())
    return feats

LABELS = json.load(open("../data/label_map.json"))
df_index = pd.read_csv("../data/dataset.csv")

rows = []
for _,r in df_index.iterrows():
    feats = load_npz_dict(r["npz_path"])
    feats["label_str"] = r["label"]
    feats["label"] = LABELS[r["label"]]
    rows.append(feats)
df = pd.concat(rows, ignore_index=True)

X = df[["I_mean","I_std","RB_mean","GB_mean","flow_mean","bp_0_10","bp_10_30","bp_30_90"]].values
y = df["label"].values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=0)
clf.fit(Xtr, ytr)

pred = clf.predict(Xte)
print(classification_report(yte, pred, target_names=LABELS.keys()))
print(confusion_matrix(yte, pred))

os.makedirs(".", exist_ok=True)
joblib.dump(clf, "model.pkl")
print("Saved ml/model.pkl")

# Feature importances
imp = clf.feature_importances_
cols = ["I_mean","I_std","RB_mean","GB_mean","flow_mean","bp_0_10","bp_10_30","bp_30_90"]
print("Feature importance:")
for c,w in sorted(zip(cols,imp), key=lambda x:-x[1]):
    print(f"{c:12s}  {w:.3f}")
