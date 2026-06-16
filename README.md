# Plate Color Classifier

Lightweight number plate color classifier using an Improved Tiny CNN.

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

```python
from modules import PlateColorInference

model = PlateColorInference(
    model_path="weights/improved_tiny_cnn_dynamic_fp16.onnx"
)

results = model.predict_folder(
    image_folder="test_images",
    batch_size=1
)

print(results)
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

## Postprocessing

- Apply softmax
- Select class with highest probability
- Return predicted color and confidence score

---

## Output

Example output:

```json
{
  "image": "image1.jpg",
  "color": "blue",
  "confidence": 1.0
}
```

Returns:

- `image` – input image filename
- `color` – predicted plate color
- `confidence` – prediction confidence score

---

## Test Client

```bash
python test_client_tensor.py
```

---

## Performance

```text
Internal Test Accuracy: 99.96%
External Dataset Accuracy: 98.95%

Black Recall: 0.97
Green Recall: 1.00

Model Size:
FP32 ONNX: 772 KB
FP16 ONNX: 398 KB
```

---

## Deployment

Training format:

```text
improved_tiny_cnn_best.pth
```

Deployment format:

```text
improved_tiny_cnn_dynamic_fp16.onnx
```

The exported ONNX model supports dynamic batch sizes and can be converted to TensorRT in the deployment environment.