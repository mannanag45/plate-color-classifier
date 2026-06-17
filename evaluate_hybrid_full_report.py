import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


CLASSES = ["black", "blue", "green", "white", "yellow"]
IMG_SIZE = 96
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CNN_TRUST_THRESHOLD = 0.98


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
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def hsv_predict(image_path):
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


def get_images(dataset_dir):
    paths = []
    labels = []

    root = Path(dataset_dir)

    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        class_name = folder.name.lower()

        if class_name not in CLASSES:
            continue

        label = CLASSES.index(class_name)

        for path in folder.iterdir():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
                labels.append(label)

    return paths, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    session = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type
    dtype = np.float16 if "float16" in input_type else np.float32

    image_paths, true_labels = get_images(args.dataset_dir)

    pure_preds = []
    hybrid_preds = []
    hybrid_labels_classified = []
    hybrid_preds_classified = []

    unknown_count = 0

    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_labels = true_labels[i:i + args.batch_size]

        batch = np.stack(
            [preprocess_for_cnn(path) for path in batch_paths],
            axis=0
        ).astype(dtype)

        logits = session.run(None, {input_name: batch})[0]
        probs = softmax(logits)

        cnn_pred_indices = np.argmax(probs, axis=1)
        cnn_confidences = np.max(probs, axis=1)

        for path, true_label, cnn_pred_idx, cnn_conf in zip(
            batch_paths,
            batch_labels,
            cnn_pred_indices,
            cnn_confidences
        ):
            cnn_pred_idx = int(cnn_pred_idx)
            cnn_conf = float(cnn_conf)

            cnn_color = CLASSES[cnn_pred_idx]
            hsv_color = hsv_predict(path)

            pure_preds.append(cnn_pred_idx)

            if cnn_color == hsv_color:
                final_color = cnn_color
            elif cnn_conf >= CNN_TRUST_THRESHOLD:
                final_color = cnn_color
            else:
                final_color = "unknown"

            if final_color == "unknown":
                unknown_count += 1

                # For full report, keep CNN prediction as fallback
                hybrid_preds.append(cnn_pred_idx)
            else:
                final_idx = CLASSES.index(final_color)
                hybrid_preds.append(final_idx)

                hybrid_labels_classified.append(true_label)
                hybrid_preds_classified.append(final_idx)

    total = len(true_labels)

    print("\nPure CNN Evaluation")
    print("-------------------")
    print(f"Total images: {total}")
    print(f"Accuracy: {accuracy_score(true_labels, pure_preds) * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            pure_preds,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            true_labels,
            pure_preds,
            labels=list(range(len(CLASSES)))
        )
    )

    print("\nHybrid CNN + HSV Evaluation")
    print("---------------------------")
    print(f"Total images: {total}")
    print(f"Unknown / rejected: {unknown_count}")
    print(f"Coverage: {((total - unknown_count) / total) * 100:.2f}%")

    if len(hybrid_labels_classified) > 0:
        print(
            f"Accuracy on classified images: "
            f"{accuracy_score(hybrid_labels_classified, hybrid_preds_classified) * 100:.2f}%"
        )

        print("\nClassification Report on classified images only:")
        print(
            classification_report(
                hybrid_labels_classified,
                hybrid_preds_classified,
                labels=list(range(len(CLASSES))),
                target_names=CLASSES,
                zero_division=0
            )
        )

        print("\nConfusion Matrix on classified images only:")
        print(
            confusion_matrix(
                hybrid_labels_classified,
                hybrid_preds_classified,
                labels=list(range(len(CLASSES)))
            )
        )

    print("\nHybrid Full Report using CNN fallback for unknowns")
    print("------------------------------------------------")
    print(f"Accuracy: {accuracy_score(true_labels, hybrid_preds) * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            hybrid_preds,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            true_labels,
            hybrid_preds,
            labels=list(range(len(CLASSES)))
        )
    )


if __name__ == "__main__":
    main()