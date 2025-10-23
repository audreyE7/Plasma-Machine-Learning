import numpy as np, pandas as pd, yaml, json, os, sys

cfg = yaml.safe_load(open("ml_seq/seq_config.yaml"))
LBLMAP = cfg["labels"]

def load_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    X = z["frames"]        # [T, F]
    names = list(z["feat_names"])
    return X, names, float(z["fps"])

def build_sequences(index_csv, out_npz):
    idx = pd.read_csv(index_csv)
    Xs, ys = [], []
    for _, r in idx.iterrows():
        X, names, fps = load_npz(r["npz_path"])
        F_idx = [names.index(f) for f in cfg["features"]]
        X = X[:, F_idx]  # select features

        L = X.shape[0]
        for start in range(0, L - cfg["seq_len"] + 1, cfg["seq_stride"]):
            seq = X[start:start+cfg["seq_len"], :]
            Xs.append(seq.astype(np.float32))
            ys.append(LBLMAP[r["label"]])
    Xs = np.stack(Xs)  # [N, T, F]
    ys = np.array(ys, dtype=np.int64)
    np.savez(out_npz, X=Xs, y=ys, feat_names=np.array(cfg["features"]))
    print("Saved", out_npz, "shape:", Xs.shape)

if __name__ == "__main__":
    build_sequences("data/frames_index.csv", "data/seq_dataset.npz")
