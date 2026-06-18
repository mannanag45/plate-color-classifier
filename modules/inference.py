from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASSES = ["black", "blue", "green", "white", "yellow"]
IMG_SIZE = 96
CNN_TRUST_THRESHOLD = 0.98
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PlateColorInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_type = self.session.get_inputs()[0].type
        self.dtype = np.float16 if "float16" in self.input_type else np.float32

    def softmax(self, logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(logits)

        return exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )

    def preprocess_image(self, image_path):
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        return image

    def hsv_sanity_check(self, image_path):
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.resize(image, (280, 80))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        mask = (v > 70) & (s > 30)

        if np.sum(mask) < 100:
            return "unknown"

        h_mean = np.mean(h[mask])
        s_mean = np.mean(s[mask])
        v_mean = np.mean(v[mask])

        if s_mean < 45 and v_mean > 120:
            return "white"

        if v_mean < 80:
            return "black"

        if 15 <= h_mean <= 40 and s_mean > 50:
            return "yellow"

        if 40 < h_mean <= 85 and s_mean > 40:
            return "green"

        if 85 < h_mean <= 135 and s_mean > 40:
            return "blue"

        return "unknown"

    def predict_image(self, image_path):
        image_path = Path(image_path)

        image = self.preprocess_image(image_path)
        batch = np.expand_dims(image, axis=0).astype(self.dtype)

        logits = self.session.run(
            None,
            {self.input_name: batch}
        )[0]

        probabilities = self.softmax(logits)

        pred_idx = int(np.argmax(probabilities, axis=1)[0])
        cnn_color = CLASSES[pred_idx]
        cnn_confidence = float(probabilities[0][pred_idx])

        hsv_color = self.hsv_sanity_check(image_path)

        if cnn_color == hsv_color:
            final_color = cnn_color
            message = "CNN and HSV agree"

        elif cnn_confidence >= CNN_TRUST_THRESHOLD:
            final_color = cnn_color
            message = f"High confidence CNN ({cnn_confidence:.4f})"

        else:
            final_color = "unknown"
            message = f"CNN/HSV disagreement: CNN={cnn_color}, HSV={hsv_color}"

        final_result = {
            "image": image_path.name,
            "color": final_color,
            "cnn_color": cnn_color,
            "cnn_confidence": round(cnn_confidence, 4),
            "hsv_color": hsv_color,
            "message": message
        }

        return final_result

    def predict_folder(self, image_folder, batch_size=1):
        folder = Path(image_folder)

        image_paths = sorted([
            path for path in folder.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ])

        results = []

        for image_path in image_paths:
            result = self.predict_image(image_path)
            results.append(result)

        return results