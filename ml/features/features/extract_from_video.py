import cv2, os, json, yaml
import numpy as np, pandas as pd
from tqdm import tqdm
from utils_signal import bandpower_welch, norm01

def get_cfg(path="../ml/feature_config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def roi_rect(h, w, roi):
    xc,yc,ww,hh = roi
    W, H = int(ww*w), int(hh*h)
    x0 = max(0, int(xc*w - W/2)); y0 = max(0, int(yc*h - H/2))
    return x0, y0, min(x0+W, w), min(y0+H, h)

def extract(video_path, out_npz="out.npz", cfg=None):
    if cfg is None: cfg = get_cfg()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise SystemExit("Video open failed")

    fps = cap.get(cv2.CAP_PROP_FPS) or cfg["target_fps"]
    segN = int(cfg["segment_seconds"]*fps)
    hop  = int((cfg["segment_seconds"]-cfg["overlap_seconds"])*fps)

    # Read all frames (grayscale + RGB ROI)
    frames_gray, frames_rgb = [], []
    ret, f0 = cap.read()
    if not ret: raise SystemExit("No frames")
    h,w = f0.shape[:2]
    x0,y0,x1,y1 = roi_rect(h,w,cfg["roi"])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, fr = cap.read()
        if not ret: break
        roi_rgb = fr[y0:y1, x0:x1]
        frames_rgb.append(roi_rgb)
        frames_gray.append(cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY))
    cap.release()

    frames_gray = np.array(frames_gray)      # [T, h, w]
    frames_rgb  = np.array(frames_rgb)

    # Precompute optical flow magnitudes
    flows = []
    for i in range(1, len(frames_gray)):
        flow = cv2.calcOpticalFlowFarneback(frames_gray[i-1], frames_gray[i], None,
                                            0.5, 3, 21, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()
        flows.append(mag)
    flows = np.array([0.0] + flows)  # align length to frames

    # Time series scalars per frame
    meanI = frames_gray.reshape(len(frames_gray), -1).mean(axis=1)
    stdI  = frames_gray.reshape(len(frames_gray), -1).std(axis=1)
    R = frames_rgb[...,2].reshape(len(frames_rgb), -1).mean(axis=1)
    G = frames_rgb[...,1].reshape(len(frames_rgb), -1).mean(axis=1)
    B = frames_rgb[...,0].reshape(len(frames_rgb), -1).mean(axis=1)
    RB = R/(B+1e-9); GB = G/(B+1e-9)

    # segment
    feats, idx = [], []
    for start in range(0, len(meanI)-segN+1, max(1,hop)):
        end = start + segN
        Iseg = meanI[start:end]
        if Iseg.mean() < cfg["min_brightness"]: continue

        seg = {
            "I_mean": float(Iseg.mean()),
            "I_std": float(Iseg.std()),
            "RB_mean": float(RB[start:end].mean()),
            "GB_mean": float(GB[start:end].mean()),
            "flow_mean": float(flows[start:end].mean()),
            "bp_0_10": bandpower_welch(Iseg, fps, 0, 10),
            "bp_10_30": bandpower_welch(Iseg, fps, 10, 30),
            "bp_30_90": bandpower_welch(Iseg, fps, 30, 90),
            "start_frame": start, "end_frame": end
        }
        feats.append(seg); idx.append((start,end))

    feats = pd.DataFrame(feats)
    np.savez(out_npz, feats=feats.to_dict("list"), fps=fps, roi=[x0,y0,x1,y1])
    print("Saved", out_npz, "segments:", len(feats))
    return out_npz

if __name__ == "__main__":
    import sys, os
    vid = sys.argv[1] if len(sys.argv)>1 else "../videos/plasma_fixed.mp4"
    out = sys.argv[2] if len(sys.argv)>2 else "../data/features/example.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    extract(vid, out)
