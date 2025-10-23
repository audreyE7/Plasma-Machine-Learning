import cv2, numpy as np, yaml, os, sys
from tqdm import tqdm
from utils_signal import norm01

def get_cfg(path="../ml/feature_config.yaml"):
    import yaml
    with open(path) as f: return yaml.safe_load(f)

def roi_rect(h, w, roi):
    xc,yc,ww,hh = roi
    W,H = int(ww*w), int(hh*h)
    x0 = max(0, int(xc*w - W/2)); y0 = max(0, int(yc*h - H/2))
    return x0, y0, min(x0+W, w), min(y0+H, h)

def extract_frameseries(video_path, out_npz, cfg=None):
    if cfg is None: cfg = get_cfg()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise SystemExit("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg["target_fps"]
    ret, f0 = cap.read()
    if not ret: raise SystemExit("No frames")
    h,w = f0.shape[:2]
    x0,y0,x1,y1 = roi_rect(h,w,cfg["roi"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    meanI, stdI, RB, GB, FLOW = [], [], [], [], []
    prev_gray = None

    while True:
        ok, fr = cap.read()
        if not ok: break
        roi = fr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        b,g,r = cv2.split(roi)
        meanI.append(float(gray.mean()))
        stdI.append(float(gray.std()))
        RB.append(float(r.mean()/(b.mean()+1e-9)))
        GB.append(float(g.mean()/(b.mean()+1e-9)))

        if prev_gray is None:
            FLOW.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,21,3,5,1.2,0)
            FLOW.append(float(np.sqrt(flow[...,0]**2+flow[...,1]**2).mean()))
        prev_gray = gray

    cap.release()

    arr = np.column_stack([meanI, stdI, RB, GB, FLOW])  # shape [T, 5]
    np.savez(out_npz, frames=arr, fps=fps, roi=[x0,y0,x1,y1], feat_names=["I_mean","I_std","RB","GB","flow"])
    print("Saved", out_npz, "frames:", arr.shape[0])

if __name__ == "__main__":
    vid = sys.argv[1]
    out = sys.argv[2]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    extract_frameseries(vid, out)
