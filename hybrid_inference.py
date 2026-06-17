import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASSES = ["black", "blue", "green", "white", "yellow"]
IMG_SIZE = 96
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_for_cnn(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    return image


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits)

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )


def cnn_predict(session, image_path):
    image = preprocess_for_cnn(image_path)

    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type

    dtype = np.float16 if "float16" in input_type else np.float32

    batch = np.expand_dims(image, axis=0).astype(dtype)

    logits = session.run(None, {input_name: batch})[0]
    probabilities = softmax(logits)

    pred_idx = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(probabilities[0][pred_idx])

    return CLASSES[pred_idx], confidence


def hsv_predict(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (280, 80))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Ignore very dark text/shadow pixels
    mask = (v > 70) & (s > 30)

    if np.sum(mask) < 100:
        return "unknown"

    h_mean = np.mean(h[mask])
    s_mean = np.mean(s[mask])
    v_mean = np.mean(v[mask])

    if s_mean < 45 and v_mean > 120:
        return "white"

    elif v_mean < 80:
        return "black"

    elif 15 <= h_mean <= 40 and s_mean > 50:
        return "yellow"

    elif 40 < h_mean <= 85 and s_mean > 40:
        return "green"

    elif 85 < h_mean <= 135 and s_mean > 40:
        return "blue"

    else:
        return "unknown"


def hybrid_predict(session, image_path):
    cnn_color, cnn_confidence = cnn_predict(session, image_path)
    hsv_color = hsv_predict(image_path)

    if cnn_color == hsv_color:
        final_color = cnn_color
        message = "CNN and HSV agree"

    elif cnn_confidence > 0.98:
        final_color = cnn_color
        message = f"High confidence CNN ({cnn_confidence:.4f})"

    else:
        final_color = "unknown"
        message = (
            f"CNN/HSV disagreement: "
            f"CNN={cnn_color}, HSV={hsv_color}"
        )

    return {
        "image": Path(image_path).name,
        "color": final_color,
        "cnn_color": cnn_color,
        "cnn_confidence": round(cnn_confidence, 4),
        "hsv_color": hsv_color,
        "message": message
    }


def get_image_paths(image_folder):
    folder = Path(image_folder)

    return sorted([
        path for path in folder.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid CNN + HSV inference for plate color classification."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to ONNX model."
    )

    parser.add_argument(
        "--image",
        help="Path to a single image."
    )

    parser.add_argument(
        "--image_folder",
        help="Folder containing images."
    )

    parser.add_argument(
        "--output",
        default="hybrid_predictions.json",
        help="Path to save JSON output."
    )

    args = parser.parse_args()

    if args.image:
        image_paths = [Path(args.image)]
    elif args.image_folder:
        image_paths = get_image_paths(args.image_folder)
    else:
        raise ValueError("Please provide either --image or --image_folder.")

    session = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"]
    )

    results = [
        hybrid_predict(session, image_path)
        for image_path in image_paths
    ]

    print(json.dumps(results, indent=2))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()