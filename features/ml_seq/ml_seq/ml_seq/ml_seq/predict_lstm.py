import numpy as np, yaml, sys, os
import tensorflow as tf

cfg = yaml.safe_load(open("ml_seq/seq_config.yaml"))
model = tf.keras.models.load_model("ml_seq/model_lstm.h5")

# 1) make per-frame features for the new video
vid = sys.argv[1] if len(sys.argv)>1 else "videos/new_run.mp4"
tmp_frames = "data/features/tmp_frames.npz"
os.system(f"python features/extract_frameseries.py {vid} {tmp_frames}")

# 2) build sequences (unlabeled)
z = np.load(tmp_frames, allow_pickle=True)
Xf, names = z["frames"], list(z["feat_names"])
F_idx = [names.index(f) for f in cfg["features"]]
Xf = Xf[:, F_idx].astype(np.float32)

seqs = []
starts = []
for s in range(0, len(Xf)-cfg["seq_len"]+1, cfg["seq_stride"]):
    seqs.append(Xf[s:s+cfg["seq_len"], :])
    starts.append(s)
X = np.stack(seqs) if seqs else np.zeros((0, cfg["seq_len"], len(cfg["features"])), np.float32)
if len(X)==0: raise SystemExit("Not enough frames for one sequence window")

proba = model.predict(X, verbose=0)
pred  = proba.argmax(axis=1)

# Majority vote
from collections import Counter
counts = Counter(pred)
major = counts.most_common(1)[0][0]
inv_labels = {v:k for k,v in cfg["labels"].items()}
print("Predicted:", inv_labels[major], "window votes:", {inv_labels[k]:v for k,v in counts.items()})
