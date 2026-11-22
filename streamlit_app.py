import os
import sqlite3
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from inference import WasteClassifier

st.set_page_config(page_title="Edge Waste Segregation", layout="wide")

MODELS_DIR = "models"
CAPTURES_DIR = "captures"
DB_PATH = "history.db"
os.makedirs(CAPTURES_DIR, exist_ok=True)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                category TEXT,
                confidence REAL,
                latency_ms REAL,
                model TEXT,
                image_path TEXT
            )
        """)
        # Migrate existing history.db if image_path column is missing
        cols = [r[1] for r in conn.execute("PRAGMA table_info(classifications)").fetchall()]
        if "image_path" not in cols:
            conn.execute("ALTER TABLE classifications ADD COLUMN image_path TEXT")


def save_and_log(frame_bgr: np.ndarray, category: str, confidence: float, latency_ms: float, model: str):
    ts_iso = datetime.now().isoformat(timespec="seconds")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    img_path = os.path.join(CAPTURES_DIR, f"{ts_file}_{category}.jpg")
    cv2.imwrite(img_path, frame_bgr)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO classifications
               (timestamp, category, confidence, latency_ms, model, image_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ts_iso, category, confidence, latency_ms, model, img_path),
        )
    return img_path


def load_history() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT timestamp, category, confidence, latency_ms, model, image_path FROM classifications ORDER BY id DESC",
            conn,
        )


@st.cache_resource
def get_classifier(model_name: str) -> WasteClassifier:
    return WasteClassifier(models_dir=MODELS_DIR, default_model=model_name)


init_db()

# --- Sidebar ---
available_models = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".onnx"))
if not available_models:
    st.error(f"No .onnx models found in `{MODELS_DIR}/`.")
    st.stop()

default_idx = (
    available_models.index("yolo11n_static_int8.onnx")
    if "yolo11n_static_int8.onnx" in available_models
    else 0
)

with st.sidebar:
    st.header("Edge Runtime")
    selected_model = st.selectbox("Active ONNX Model", available_models, index=default_idx)
    classifier = get_classifier(selected_model)
    st.caption(f"Provider: `{classifier.providers[0]}`")

    st.divider()
    st.subheader("Live Inference Settings")
    cam_index = st.number_input("Camera Index", min_value=0, max_value=4, value=0, step=1)
    auto_log = st.checkbox("Auto-save frames to DB", value=True)
    log_interval = st.slider("Log interval (seconds)", 1.0, 10.0, 3.0, 0.5)
    min_conf = st.slider("Min confidence to log", 0.50, 0.99, 0.75, 0.05)

    st.divider()
    if st.button("Clear History & Captures", use_container_width=True):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM classifications")
        for f in os.listdir(CAPTURES_DIR):
            os.remove(os.path.join(CAPTURES_DIR, f))
        st.rerun()

# --- Main UI ---
st.title("Intelligent Waste Segregation (Local Edge)")
tab_live, tab_upload = st.tabs(["Live Webcam Inference", "Upload Image"])

with tab_live:
    run_live = st.toggle("Start Live Inference", value=False)
    col_vid, col_info = st.columns([1.4, 1], gap="large")
    frame_placeholder = col_vid.empty()
    info_placeholder = col_info.empty()

    if run_live:
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error(f"Could not open camera index {cam_index}.")
        else:
            last_log_time = 0.0
            try:
                while run_live:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to grab frame from webcam.")
                        break

                    t0 = time.perf_counter()
                    result = classifier.predict_frame(frame)
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                    # Auto-log frame + image_path at chosen interval
                    now = time.time()
                    if auto_log and result["confidence"] >= min_conf and (now - last_log_time) >= log_interval:
                        save_and_log(frame, result["category"], result["confidence"], latency_ms, result["model"])
                        last_log_time = now

                    # Draw overlay on display frame
                    display = frame.copy()
                    label = f"{result['category'].upper()} {result['confidence']*100:.1f}% | {latency_ms:.1f} ms"
                    cv2.rectangle(display, (10, 10), (460, 52), (15, 23, 42), -1)
                    cv2.putText(display, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (16, 185, 129), 2)

                    frame_placeholder.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), use_container_width=True)
                    with info_placeholder.container():
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Category", result["category"].upper())
                        m2.metric("Confidence", f"{result['confidence']*100:.1f}%")
                        m3.metric("Latency", f"{latency_ms} ms")
                        st.success(f"**Disposal:** {result['recommendation']}")
            finally:
                cap.release()
    else:
        st.info("Toggle **Start Live Inference** to stream real-time predictions from your webcam.")

with tab_upload:
    col_in, col_out = st.columns([1, 1], gap="large")
    with col_in:
        uploaded = st.file_uploader("Choose a waste image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            st.image(uploaded.getvalue(), caption="Input Image", use_container_width=True)

    with col_out:
        st.subheader("Prediction")
        if uploaded is not None and st.button("Classify & Save", type="primary", use_container_width=True):
            arr = np.frombuffer(uploaded.getvalue(), np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            t0 = time.perf_counter()
            result = classifier.predict_frame(frame_bgr)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            saved_path = save_and_log(frame_bgr, result["category"], result["confidence"], latency_ms, result["model"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Category", result["category"].upper())
            m2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
            m3.metric("Latency", f"{latency_ms} ms")
            st.success(f"**Disposal:** {result['recommendation']}")
            st.caption(f"Saved image: `{saved_path}`")

# --- Dashboard: Stats, History & Saved Captures ---
st.divider()
df = load_history()

if df.empty:
    st.caption("No classifications recorded yet.")
else:
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        st.subheader("Category-Wise Statistics")
        st.bar_chart(df["category"].value_counts())

    with c2:
        st.subheader("Classification History")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    # Visual gallery of the latest 6 saved captures
    saved_rows = df[df["image_path"].notna()].head(6)
    if not saved_rows.empty:
        st.subheader("Recent Saved Captures")
        cols = st.columns(len(saved_rows))
        for col, (_, row) in zip(cols, saved_rows.iterrows()):
            if os.path.exists(row["image_path"]):
                col.image(
                    row["image_path"],
                    caption=f"{row['category'].upper()} ({row['confidence']*100:.0f}%)",
                    use_container_width=True,
                )