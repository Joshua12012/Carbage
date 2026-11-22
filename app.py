import os
import sqlite3
import time
from datetime import datetime
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from inference import WasteClassifier

app = FastAPI(title="Edge Waste Segregation API")
classifier = WasteClassifier(models_dir="models", default_model="yolo11n_static_int8.onnx")

DB_PATH = "history.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                category TEXT,
                confidence REAL,
                latency_ms REAL,
                model TEXT
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
async def predict_waste(file: UploadFile = File(...)):
    image_bytes = await file.read()
    t0 = time.perf_counter()
    try:
        result = classifier.predict_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    result["latency_ms"] = latency_ms

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO classifications (timestamp, category, confidence, latency_ms, model) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                result["category"],
                result["confidence"],
                latency_ms,
                result["model"],
            ),
        )
    return result

@app.get("/history")
def get_history(limit: int = 20):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT timestamp, category, confidence, latency_ms, model FROM classifications ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"timestamp": r[0], "category": r[1], "confidence": r[2], "latency_ms": r[3], "model": r[4]}
        for r in rows
    ]

@app.get("/stats")
def get_stats():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT category, COUNT(*) FROM classifications GROUP BY category"
        ).fetchall()
    return {r[0]: r[1] for r in rows}

os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
