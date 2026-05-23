import streamlit as st
import tempfile
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import time
from pathlib import Path

st.set_page_config(page_title="Drone Crowd Panic Monitor", layout="wide")
st.title("Drone Crowd Panic Monitor")

with st.sidebar:
    st.header("Settings")
    yolo_path = st.text_input("YOLO weights path", "yolo.pt")
    use_rtfm = st.checkbox("Use RTFM anomaly head (if rtfm_head.pt present)", value=True)
    frame_stride = st.slider("Process every Nth frame (speed)", 1, 8, 3)
    conf_th = st.slider("YOLO confidence", 0.1, 0.9, 0.35, 0.05)
    count_jump_thresh = st.slider("Sudden count jump (people)", 1, 100, 12)
    flow_mag_thresh = st.slider("Flow magnitude threshold", 0.1, 10.0, 2.0, 0.1)
    dir_agree_thresh = st.slider("Direction agreement (0-1)", 0.1, 1.0, 0.6, 0.05)
    ma_win = st.number_input("Moving avg window (frames)", 5, 200, 30)
    anomaly_sample_seconds = st.number_input("RTFM clip duration (s)", 2, 8, 4)
    anomaly_sample_stride = st.number_input("RTFM sample stride (frames)", 5, 30, 15)
    risk_warning = st.slider("Warning risk threshold", 0.0, 1.0, 0.35)
    risk_critical = st.slider("Critical risk threshold", 0.0, 1.0, 0.6)

col1, col2 = st.columns([2,1])
u = col2.empty()
uploaded = col1.file_uploader("Upload drone video (mp4,avi,mkv)", type=["mp4","avi","mkv"])
start_button = col2.button("Start")

if not uploaded:
    st.info("Upload a video file to begin. The app will run detection, optical flow, and (optionally) anomaly inference.")
    st.stop()

tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded.read())

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_exists = Path(yolo_path).exists()
if not yolo_exists:
    st.error(f"YOLO weights not found at {yolo_path}. Place a weights file named 'yolo.pt' or change the path.")
    st.stop()

yolo = YOLO(yolo_path)
rtfm_path = Path("rtfm_head.pt")
use_rtfm = use_rtfm and rtfm_path.exists()
rtfm_model = None
if use_rtfm:
    try:
        rtfm_model = torch.load(str(rtfm_path), map_location=device)
        rtfm_model.eval()
    except Exception as e:
        st.warning("Failed to load rtfm_head.pt. Falling back to heuristics.")
        use_rtfm = False

cap = cv2.VideoCapture(tfile.name)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

ph_img = st.empty()
ph_metrics = st.empty()
ph_alerts = st.empty()

if start_button:
    prev_gray = None
    frame_idx = 0
    counts = []
    flow_vals = []
    dir_agrees = []
    times = []
    alerts = []
    count_ma = []
    last_anom_time = -9999
    start_time = time.time()
    progress = st.progress(0.0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        if frame_idx > total_frames:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = yolo.predict(source=rgb, conf=conf_th, imgsz=640, verbose=False)
        people = []
        for r in res:
            if getattr(r, "boxes", None) is None:
                continue
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            for i,c in enumerate(cls):
                if c == 0:
                    people.append(xyxy[i])
        count = len(people)
        if prev_gray is None:
            flow_mag, dir_agree = 0.0, 0.0
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            flow_mag = float(np.mean(mag))
            hist, _ = np.histogram(ang, bins=12, range=(0,2*np.pi))
            dir_agree = float(np.max(hist)/(np.sum(hist)+1e-6))
        counts.append(count)
        flow_vals.append(flow_mag)
        dir_agrees.append(dir_agree)
        t = frame_idx / fps
        times.append(t)
        count_ma.append(count)
        if len(count_ma) > ma_win:
            count_ma.pop(0)
        moving_avg = float(np.mean(count_ma)) if len(count_ma) else 0.0
        sudden_jump = (count - moving_avg) > count_jump_thresh
        run_event = (flow_mag > flow_mag_thresh) and (dir_agree > dir_agree_thresh)
        anomaly_score = 0.0
        if use_rtfm and (frame_idx - last_anom_time) >= anomaly_sample_stride:
            try:
                clip_frames = []
                sample_frame_count = int(anomaly_sample_seconds * fps)
                seek_frame = max(0, frame_idx - sample_frame_count//2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
                for _ in range(sample_frame_count):
                    r2, f2 = cap.read()
                    if not r2:
                        break
                    f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                    clip_frames.append(cv2.resize(f2, (112,112)))
                clip = np.array(clip_frames).astype(np.float32)/255.0
                if clip.shape[0] >= 1:
                    clip_tensor = torch.tensor(clip).permute(0,3,1,2).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = rtfm_model(clip_tensor)
                        anomaly_score = float(torch.sigmoid(torch.max(out)).item())
                last_anom_time = frame_idx
            except Exception:
                anomaly_score = 0.0
        risk = 0.0
        if moving_avg > 0:
            cnt_sig = max(0.0, (count - moving_avg) / moving_avg)
        else:
            cnt_sig = 0.0
        motion_sig = flow_mag / (flow_mag + 10.0)
        anomaly_sig = anomaly_score
        risk = 0.5 * cnt_sig + 0.35 * motion_sig + 0.15 * anomaly_sig
        level = "NORMAL"
        if risk >= risk_critical:
            level = "CRITICAL"
        elif risk >= risk_warning:
            level = "WARNING"
        flags = []
        if sudden_jump:
            flags.append("sudden_crowd_formation")
        if run_event:
            flags.append("sudden_crowd_run")
        if level != "NORMAL":
            flags.append("risk_"+level.lower())
        if flags:
            alerts.append({"time_s": round(t,2), "frame": frame_idx, "count": int(count), "flow": round(flow_mag,3), "dir_agree": round(dir_agree,3), "anomaly": round(anomaly_score,3), "risk": round(risk,3), "flags": "|".join(flags)})
        for x1,y1,x2,y2 in people:
            cv2.rectangle(rgb, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(rgb, f"count={count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(rgb, f"flow={flow_mag:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(rgb, f"risk={risk:.2f} {level}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if level=="CRITICAL" else (0,255,255), 2)
        ph_img.image(rgb, use_column_width=True)
        df_metrics = pd.DataFrame({"time_s": times, "count": counts, "flow": flow_vals, "dir_agree": dir_agrees}).set_index("time_s")
        ph_metrics.dataframe(df_metrics.tail(10))
        ph_alerts.dataframe(pd.DataFrame(alerts).tail(10))
        prev_gray = gray
        frame_idx += 1
        progress.progress(min(1.0, frame_idx / max(1, total_frames)))
    cap.release()
    elapsed = time.time() - start_time
    st.success(f"Processing finished in {elapsed:.1f}s")
    if alerts:
        adf = pd.DataFrame(alerts)
        st.warning(f"Alerts: {len(adf)}")
        st.dataframe(adf)
        st.download_button("Download alerts CSV", adf.to_csv(index=False).encode(), file_name="alerts.csv", mime="text/csv")
    else:
        st.info("No alerts detected with current thresholds.")
