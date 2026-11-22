import time
import cv2
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Edge Waste Segregation", layout="wide")

API_URL = st.sidebar.text_input("Edge API URL", value="http://localhost:8000")

# --- Sidebar: Model Switcher ---
try:
    m_info = requests.get(f"{API_URL}/models", timeout=3).json()
except Exception as e:
    st.error(f"Cannot connect to Docker Edge API at {API_URL}: {e}")
    st.stop()

with st.sidebar:
    st.header("Docker Edge Runtime")
    available = m_info["available"]
    active_idx = available.index(m_info["active"]) if m_info["active"] in available else 0
    selected = st.selectbox("Active ONNX Model", available, index=active_idx)

    if selected != m_info["active"]:
        requests.post(f"{API_URL}/models/select", json={"model_name": selected}, timeout=5)
        st.rerun()

    st.caption(f"Provider: `{m_info['provider']}`")
    st.divider()
    if st.button("Clear History", use_container_width=True):
        requests.delete(f"{API_URL}/history", timeout=5)
        st.rerun()

# --- Main UI ---
st.title("Intelligent Waste Segregation (Docker Edge Client)")
tab_upload, tab_live = st.tabs(["Upload Image", "Live Webcam"])

with tab_upload:
    col_in, col_out = st.columns([1, 1], gap="large")
    with col_in:
        uploaded = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            st.image(uploaded.getvalue(), caption="Input Image", use_container_width=True)

    with col_out:
        st.subheader("Prediction")
        if uploaded is not None and st.button("Send to Edge API", type="primary", use_container_width=True):
            resp = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=10,
            )
            res = resp.json()

            m1, m2, m3 = st.columns(3)
            m1.metric("Category", res["category"].upper())
            m2.metric("Confidence", f"{res['confidence'] * 100:.1f}%")
            m3.metric("Edge Latency", f"{res['latency_ms']} ms")

            if res["alert"]:
                st.warning("Low confidence — verify item placement or lighting.")
            st.success(f"**Disposal:** {res['recommendation']}")

with tab_live:
    run_live = st.toggle("Start Live Webcam", value=False)
    col_vid, col_info = st.columns([1.4, 1], gap="large")
    frame_box = col_vid.empty()
    info_box = col_info.empty()

    if run_live:
        cap = cv2.VideoCapture(0)
        last_save = 0.0
        try:
            while run_live:
                ret, frame = cap.read()
                if not ret:
                    break

                # Center ROI crop
                h, w, _ = frame.shape
                box = min(h, w) // 2
                x1, y1 = (w - box) // 2, (h - box) // 2
                roi = frame[y1 : y1 + box, x1 : x1 + box]

                _, jpg = cv2.imencode(".jpg", roi)
                now = time.time()
                should_save = (now - last_save) >= 3.0

                res = requests.post(
                    f"{API_URL}/predict",
                    params={"save_record": should_save},
                    files={"file": ("frame.jpg", jpg.tobytes(), "image/jpeg")},
                    timeout=5,
                ).json()
                if should_save:
                    last_save = now

                display = frame.copy()
                cv2.rectangle(display, (x1, y1), (x1 + box, y1 + box), (16, 185, 129), 2)
                label = f"{res['category'].upper()} {res['confidence']*100:.1f}% | {res['latency_ms']} ms"
                cv2.putText(display, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (16, 185, 129), 2)

                frame_box.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), use_container_width=True)
                with info_box.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Category", res["category"].upper())
                    m2.metric("Confidence", f"{res['confidence']*100:.1f}%")
                    m3.metric("Edge Latency", f"{res['latency_ms']} ms")
                    if res["alert"]:
                        st.warning("Low confidence detection.")
                    st.success(f"**Disposal:** {res['recommendation']}")
        finally:
            cap.release()

# --- Dashboard ---
st.divider()
stats = requests.get(f"{API_URL}/stats", timeout=3).json()
history = requests.get(f"{API_URL}/history", timeout=3).json()

if not history:
    st.caption("No classifications recorded yet.")
else:
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        st.subheader("Category-Wise Statistics")
        st.bar_chart(pd.Series(stats))
    with c2:
        st.subheader("Classification History")
        st.dataframe(pd.DataFrame(history).drop(columns=["image_file"]), use_container_width=True, hide_index=True)

    st.subheader("Recent Saved Captures")
    cols = st.columns(min(len(history), 6))
    for col, row in zip(cols, history[:6]):
        if row.get("image_file"):
            col.image(
                f"{API_URL}/captures/{row['image_file']}",
                caption=f"{row['category'].upper()} ({row['confidence']*100:.0f}%)",
                use_container_width=True,
            )