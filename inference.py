import os
import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["glass", "metal", "organic", "paper", "plastic"]

DISPOSAL_GUIDE = {
    "glass": "Blue Bin (Dry Recyclable) : Rinse and remove metal caps.",
    "metal": "Blue Bin (Dry Recyclable) : Empty, rinse, and crush cans if possible.",
    "organic": "Green Bin (Wet/Compostable) : Suitable for composting.",
    "paper": "Blue Bin (Dry Recyclable) : Keep dry and flatten cardboard.",
    "plastic": "Blue Bin (Dry Recyclable) : Clean, dry, and check resin code.",
}

class WasteClassifier:
    def __init__(self, models_dir: str = "models", default_model: str = "yolo11n_static_int8.onnx", threads: int = 4):
        self.models_dir = models_dir
        self.threads = threads
        self.providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.active_model = None
        self.session = None
        self.input_name = None

        available = self.list_models()
        if not available:
            raise FileNotFoundError(f"No .onnx models found in {models_dir}/")
        target = default_model if default_model in available else available[0]
        self.load_model(target)

    def list_models(self) -> list[str]:
        return sorted(f for f in os.listdir(self.models_dir) if f.endswith(".onnx"))

    def load_model(self, model_name: str):
        path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(path, sess_options=opts, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.active_model = model_name

    def predict_frame(self, frame_bgr: np.ndarray) -> dict:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (224, 224), swapRB=False, crop=True)

        out = self.session.run(None, {self.input_name: blob.astype(np.float32)})[0][0]
        if np.min(out) < 0 or not np.isclose(np.sum(out), 1.0, atol=1e-2):
            exp = np.exp(out - np.max(out))
            probs = exp / exp.sum()
        else:
            probs = out

        idx = int(np.argmax(probs))
        category = CLASSES[idx]

        return {
            "category": category,
            "confidence": round(float(probs[idx]), 4),
            "recommendation": DISPOSAL_GUIDE[category],
            "model": self.active_model,
            "provider": self.providers[0],
        }

    def predict_bytes(self, image_bytes: bytes) -> dict:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
        return self.predict_frame(img)