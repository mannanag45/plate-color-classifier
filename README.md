# Plate Color Classifier

Lightweight number plate color classifier using an Improved Tiny CNN with an HSV-based sanity check during inference.

Supported classes:

- Black
- Blue
- Green
- White
- Yellow

---

## Installation

```bash
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

---

## Model Weights

Download ONNX model:

https://drive.google.com/file/d/1ElRyY_bfAYk991UkfMDieT04ViKfu1zM/view?usp=drive_link

Place the model at:

```text
weights/improved_tiny_cnn_dynamic_fp16.onnx
```

---

## Example Usage

Run hybrid inference on a single image:

```bash
python hybrid_inference.py \
  --model weights/improved_tiny_cnn_dynamic_fp16.onnx \
  --image "/path/to/image.jpg"
```

---

## Preprocessing

- Read image using OpenCV (`cv2`)
- Convert BGR → RGB
- Resize to 96 × 96
- Normalize pixel values to [0,1]
- Convert HWC → CHW
- Create NumPy tensor

---

## HSV Sanity Check

During inference, HSV color analysis is also performed.

Hybrid decision logic:

- If CNN and HSV agree → return that color
- If CNN confidence ≥ 0.98 → return CNN prediction
- Otherwise → return `unknown`

This safety mechanism helps reduce false color classifications on difficult images.

---

## Postprocessing

- Apply softmax
- Select class with highest probability
- Compute confidence score
- Run HSV sanity check
- Produce final hybrid decision

---

## Output

### Example: Normal Prediction

```json
{
  "image": "image1.jpg",
  "color": "blue",
  "cnn_color": "blue",
  "cnn_confidence": 0.9996,
  "hsv_color": "blue",
  "message": "CNN and HSV agree"
}
```

### Example: Rejected as Unknown

```json
{
  "image": "image2.jpg",
  "color": "unknown",
  "cnn_color": "yellow",
  "cnn_confidence": 0.6699,
  "hsv_color": "white",
  "message": "CNN/HSV disagreement: CNN=yellow, HSV=white"
}
```

### Why "unknown" is returned

If the CNN and HSV predictions disagree and the CNN confidence is below the trust threshold (0.98), the system returns `unknown` instead of forcing a potentially incorrect plate color prediction.

---

## Returned Fields

- `image` – input image filename
- `color` – final predicted plate color
- `cnn_color` – CNN prediction
- `cnn_confidence` – CNN confidence score
- `hsv_color` – HSV sanity-check prediction
- `message` – explanation of the hybrid decision

---

## Test Client

```bash
python test_client_tensor.py
```

---

## Performance

```text
External Dataset (numberplate_colour)

Pure CNN:
Accuracy: 99.87%
Wrong Predictions: 27

Hybrid CNN + HSV:
Coverage: 98.85%
Unknown / Rejected: 231
Wrong Predictions: 11
Accuracy on Classified Images: 99.94%

All India Dataset:
Accuracy: 99.99%
```

---

## Model Information

Architecture:

```text
Improved Tiny CNN
RGB Input
5 Output Classes
```

Training Model:

```text
improved_tiny_cnn_best.pth
```

Deployment Model:

```text
improved_tiny_cnn_dynamic_fp16.onnx
```

Model Size:

```text
FP32 ONNX: 772 KB
FP16 ONNX: 398 KB
```

---

## Deployment

The exported ONNX model supports:

- Dynamic batch sizes
- ONNX Runtime
- TensorRT conversion
- CPU and GPU deployment

The deployed inference pipeline combines CNN predictions with an HSV-based sanity check to reduce false classifications on difficult images.