import joblib, json, numpy as np, pandas as pd
from pathlib import Path
import sys, os
sys.path.append(str(Path(__file__).resolve().parents[1] / "features"))
from extract_from_video import extract

LABELS = json.load(open("../data/label_map.json"))
IDX2LBL = {v:k for k,v in LABELS.items()}
model = joblib.load("model.pkl")

vid = sys.argv[1] if len(sys.argv)>1 else "../videos/plasma_fixed.mp4"
tmp_npz = "../data/features/tmp_predict.npz"
os.makedirs(os.path.dirname(tmp_npz), exist_ok=True)
extract(vid, tmp_npz)   # make segments

z = np.load(tmp_npz, allow_pickle=True)
feats = pd.DataFrame(z["feats"].item())
X = feats[["I_mean","I_std","RB_mean","GB_mean","flow_mean","bp_0_10","bp_10_30","bp_30_90"]].values
proba = model.predict_proba(X)
pred  = model.predict(X)

# Majority vote over segments
from collections import Counter
counts = Counter(pred)
major = counts.most_common(1)[0][0]
print("Predicted class:", IDX2LBL[major], "counts:", {IDX2LBL[k]:v for k,v in counts.items()})

# Save per-segment CSV
feats["pred"] = [IDX2LBL[i] for i in pred]
np.savetxt("../data/predict_segments.txt", np.array(feats[["start_frame","end_frame","pred"]]), fmt='%s')
print("Saved per-segment predictions.")
