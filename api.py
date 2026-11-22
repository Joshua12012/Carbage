import os
import sqlite3
import time
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from inference import WasteClassifier

app = FastAPI(title="Edge Waste Inference API")
classifier = WasteClassifier(models_dir="models", default_model="yolo11n_static_int8.onnx")

DB_PATH = "data/history.db"
CAPTURES_DIR = "data/captures"
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
                image_file TEXT
            )
        """)


init_db()


class ModelSwitch(BaseModel):
    model_name: str


@app.get("/models")
def get_models():
    return {
        "active": classifier.active_model,
        "available": classifier.list_models(),
        "provider": classifier.providers[0],
    }


@app.post("/models/select")
def select_model(req: ModelSwitch):
    try:
        classifier.load_model(req.model_name)
        return {"status": "ok", "active": classifier.active_model}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict")
async def predict(file: UploadFile = File(...), save_record: bool = True):
    arr = np.frombuffer(await file.read(), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    t0 = time.perf_counter()
    res = classifier.predict_frame(frame)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    res["latency_ms"] = latency_ms
    res["alert"] = res["confidence"] < 0.65

    if save_record:
        ts_iso = datetime.now().isoformat(timespec="seconds")
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{res['category']}.jpg"
        cv2.imwrite(os.path.join(CAPTURES_DIR, fname), frame)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO classifications
                   (timestamp, category, confidence, latency_ms, model, image_file)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ts_iso, res["category"], res["confidence"], latency_ms, res["model"], fname),
            )
        res["image_file"] = fname

    return res


@app.get("/history")
def get_history(limit: int = 20):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT timestamp, category, confidence, latency_ms, model, image_file FROM classifications ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/stats")
def get_stats():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT category, COUNT(*) FROM classifications GROUP BY category"
        ).fetchall()
    return {r[0]: r[1] for r in rows}


@app.delete("/history")
def clear_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM classifications")
    for f in os.listdir(CAPTURES_DIR):
        os.remove(os.path.join(CAPTURES_DIR, f))
    return {"status": "cleared"}


app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")