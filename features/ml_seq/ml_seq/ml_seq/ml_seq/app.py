import streamlit as st
import numpy as np
import pandas as pd
import cv2, tempfile, yaml, os, io
import matplotlib.pyplot as plt
from pathlib import Path

# ========= Config defaults =========
DEFAULT_SEQ_CFG = {
    "seq_len": 180,        # 3s @ 60 fps
    "seq_stride": 30,      # slide 0.5s
    "features": ["I_mean","I_std","RB","GB","flow"],
    "labels": {"stable":0, "unstable":1, "extinguish":2}
}

# ========= Feature extraction (per-frame) =========
def roi_rect(h, w, roi):
    xc, yc, ww, hh = roi
    W, H = int(ww*w), int(hh*h)
    x0 = max(0, int(xc*w - W/2)); y0 = max(0, int(yc*h - H/2))
    return x0, y0, min(x0+W, w), min(y0+H, h)

def extract_frameseries(video_path, roi_frac, target_fps=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or (target_fps or 60.0)
    ret, f0 = cap.read()
    if not ret: raise RuntimeError("No frames in video")
    h, w = f0.shape[:2]
    x0, y0, x1, y1 = roi_rect(h, w, roi_frac)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    I_mean, I_std, RB, GB, FLOW = [], [], [], [], []
    prev_gray = None
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prog = st.progress(0, text="Extracting features…")
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        roi = fr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        b,g,r = cv2.split(roi)

        I_mean.append(float(gray.mean()))
        I_std.append(float(gray.std()))
        RB.append(float(r.mean()/(b.mean()+1e-9)))
        GB.append(float(g.mean()/(b.mean()+1e-9)))

        if prev_gray is None:
            FLOW.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 21, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()
            FLOW.append(float(mag))
        prev_gray = gray

        i += 1
        if frames > 0 and i % 10 == 0:
            prog.progress(min(i/frames,1.0), text="Extracting features…")
    cap.release()

    arr = np.column_stack([I_mean, I_std, RB, GB, FLOW]) # [T,5]
    feat_names = ["I_mean","I_std","RB","GB","flow"]
    return arr, feat_names, fps, (x0,y0,x1,y1)

def build_windows(frames_arr, seq_len, seq_stride, feat_idx):
    Xf = frames_arr[:, feat_idx].astype(np.float32)
    seqs, starts = [], []
    L = Xf.shape[0]
    for s in range(0, L - seq_len + 1, seq_stride):
        seqs.append(Xf[s:s+seq_len, :])
        starts.append(s)
    if not seqs:
        return np.zeros((0, seq_len, len(feat_idx)), np.float32), []
    return np.stack(seqs), starts

# ========= Visualization helpers =========
def plot_signals(time_s, I, RB):
    fig, ax1 = plt.subplots(figsize=(8,3))
    ax1.plot(time_s, I, label="Intensity (norm)", linewidth=1.7)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Intensity (a.u.)")
    ax2 = ax1.twinx()
    ax2.plot(time_s, RB, color="crimson", alpha=0.7, label="R/B ratio")
    ax2.set_ylabel("R/B ratio (a.u.)")
    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l; labels += lab
    ax1.legend(lines, labels, loc="upper right")
    fig.tight_layout()
    return fig

def render_timeline(pred_classes, starts, fps, colors, label_names):
    # Build a color bar across time using rectangles
    if not pred_classes: return None
    t0 = np.array(starts)/fps
    t1 = t0 + (t0[1]-t0[0] if len(t0)>1 else (1.0))
    fig, ax = plt.subplots(figsize=(8,1.0))
    for i, c in enumerate(pred_classes):
        cname = label_names[c]
        ax.barh(0, t1[i]-t0[i], left=t0[i], color=colors[cname], edgecolor='none', height=0.8)
    ax.set_yticks([]); ax.set_xlabel("Time (s)")
    ax.set_title("Predicted state timeline")
    fig.tight_layout()
    return fig

# ========= Streamlit UI =========
st.set_page_config(page_title="Plasma Video Analyzer", layout="wide")
st.title("📹 Plasma Video → ML Analyzer")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Keras model path", "ml_seq/model_lstm.h5")
    seq_len = st.number_input("Sequence length (frames)", 60, 1000, DEFAULT_SEQ_CFG["seq_len"], step=30)
    seq_stride = st.number_input("Stride (frames)", 1, 300, DEFAULT_SEQ_CFG["seq_stride"], step=5)
    fps_override = st.number_input("FPS override (0 = use video FPS)", 0.0, 500.0, 0.0, step=1.0)

    st.caption("ROI (fractions of frame):")
    roi_xc = st.slider("ROI center X", 0.0, 1.0, 0.5, 0.01)
    roi_yc = st.slider("ROI center Y", 0.0, 1.0, 0.5, 0.01)
    roi_w  = st.slider("ROI width",    0.05, 1.0, 0.4, 0.01)
    roi_h  = st.slider("ROI height",   0.05, 1.0, 0.4, 0.01)

    analyze_btn = st.button("Analyze")

uploaded = st.file_uploader("Upload a video (MP4/MOV)", type=["mp4","mov","MP4","MOV"])

# Color map for timeline
label_names = {0:"stable", 1:"unstable", 2:"extinguish"}
colors = {"stable":"#2ecc71", "unstable":"#e67e22", "extinguish":"#e74c3c"}

if uploaded:
    st.subheader("Preview")
    st.video(uploaded)

if analyze_btn and uploaded:
    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Load / check model
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model at {model_path}\n{e}")
        st.stop()

    # Extract features
    roi_frac = (roi_xc, roi_yc, roi_w, roi_h)
    frames_arr, feat_names, fps, roi_px = extract_frameseries(tmp_path, roi_frac, target_fps=(fps_override or None))
    time_s = np.arange(frames_arr.shape[0])/(fps if fps>0 else 60.0)

    # Normalize intensity for plotting
    I = frames_arr[:,0]
    I_norm = (I - I.min())/(I.max()-I.min()+1e-12)
    RB = frames_arr[:,2]

    st.markdown("### Signals")
    fig1 = plot_signals(time_s, I_norm, RB)
    st.pyplot(fig1, use_container_width=True)

    # Build sequences → predict
    feat_idx = [feat_names.index(f) for f in DEFAULT_SEQ_CFG["features"] if f in feat_names]
    X, starts = build_windows(frames_arr, int(seq_len), int(seq_stride), feat_idx)
    if X.shape[0] == 0:
        st.warning("Video too short for one sequence window with current settings.")
        st.stop()

    proba = model.predict(X, verbose=0)
    pred  = proba.argmax(axis=1).tolist()

    # Majority vote
    from collections import Counter
    counts = Counter(pred)
    maj = counts.most_common(1)[0][0]
    maj_name = label_names.get(maj, str(maj))

    st.markdown(f"### Prediction: **{maj_name.upper()}**  (majority over {len(pred)} windows)")
    # Timeline
    fig2 = render_timeline(pred, starts, fps if fps>0 else 60.0, colors, label_names)
    if fig2: st.pyplot(fig2, use_container_width=True)

    # Details table
    st.markdown("#### Per-window probabilities (last 10)")
    last = min(10, len(pred))
    tbl = pd.DataFrame({
        "start_frame": starts[-last:],
        "start_time_s": (np.array(starts[-last:])/(fps if fps>0 else 60.0)).round(2),
        "pred": [label_names.get(i, str(i)) for i in pred[-last:]],
        "p_stable": proba[-last:, 0].round(3) if proba.shape[1] > 0 else 0,
        "p_unstable": proba[-last:, 1].round(3) if proba.shape[1] > 1 else 0,
        "p_extinguish": proba[-last:, 2].round(3) if proba.shape[1] > 2 else 0
    })
    st.dataframe(tbl, use_container_width=True)

    # Clean temp
    try:
        os.remove(tmp_path)
    except Exception:
        pass

else:
    st.info("Upload a video and press **Analyze**. Tip: convert iPhone clips to SDR 60 fps CFR with ffmpeg for best results.")
