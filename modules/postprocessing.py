import numpy as np

CLASSES = ["black", "blue", "green", "white", "yellow"]


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)

    exp_values = np.exp(logits)

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )


def postprocess_logits(logits, image_names):
    probabilities = softmax(logits)

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    results = []

    for image_name, pred_idx, probs in zip(
        image_names,
        predictions,
        probabilities
    ):
        results.append({
            "image": image_name,
            "color": CLASSES[pred_idx],
            "confidence": round(
                float(probs[pred_idx]),
                4
            )
        })

    return results