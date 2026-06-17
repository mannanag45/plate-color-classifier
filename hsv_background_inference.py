import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def classify_hsv_color(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (280, 80))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Remove text/shadow influence by ignoring very dark pixels
    mask = (v > 70) & (s > 30)

    if np.sum(mask) < 100:
        return {
            "image": Path(image_path).name,
            "color": "unknown",
            "confidence": 0.0,
            "reason": "not enough background pixels"
        }

    h_mean = np.mean(h[mask])
    s_mean = np.mean(s[mask])
    v_mean = np.mean(v[mask])

    if s_mean < 45 and v_mean > 120:
        color = "white"
    elif v_mean < 80:
        color = "black"
    elif 15 <= h_mean <= 40 and s_mean > 50:
        color = "yellow"
    elif 40 < h_mean <= 85 and s_mean > 40:
        color = "green"
    elif 85 < h_mean <= 135 and s_mean > 40:
        color = "blue"
    else:
        color = "unknown"

    return {
        "image": Path(image_path).name,
        "color": color,
        "h_mean": round(float(h_mean), 2),
        "s_mean": round(float(s_mean), 2),
        "v_mean": round(float(v_mean), 2),
        "background_pixels": int(np.sum(mask))
    }


def get_image_paths(image_folder):
    folder = Path(image_folder)

    return sorted([
        path for path in folder.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ])


def main():
    parser = argparse.ArgumentParser(
        description="HSV background color classifier for number plates."
    )

    parser.add_argument(
        "--image",
        help="Path to a single image."
    )

    parser.add_argument(
        "--image_folder",
        help="Folder containing plate images."
    )

    parser.add_argument(
        "--output",
        default="hsv_background_predictions.json",
        help="Output JSON file."
    )

    args = parser.parse_args()

    if args.image:
        image_paths = [Path(args.image)]
    elif args.image_folder:
        image_paths = get_image_paths(args.image_folder)
    else:
        raise ValueError("Please provide either --image or --image_folder.")

    results = []

    for image_path in image_paths:
        result = classify_hsv_color(image_path)
        results.append(result)

    print(json.dumps(results, indent=2))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()