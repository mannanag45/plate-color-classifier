

## Installation

```bash
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

---

## Model

Download ONNX model:

https://drive.google.com/file/d/1ElRyY_bfAYk991UkfMDieT04ViKfu1zM/view?usp=drive_link

Place the model at:

```text
weights/improved_tiny_cnn_dynamic_fp16.onnx
```

---

## Input

The model accepts an image containing a cropped number plate.

Supported formats:

```text
jpg
jpeg
png
bmp
webp
```

---

## Preprocessing

1. Read image using OpenCV
2. Convert BGR → RGB
3. Resize to 96 × 96
4. Normalize pixel values to [0,1]
5. Convert HWC → CHW
6. Create ONNX input tensor

---

## Postprocessing

1. Apply softmax to model output
2. Obtain CNN prediction and confidence
3. Run HSV sanity check
4. Produce final result

Decision logic:

```text
If CNN and HSV agree:
    return color

Else if CNN confidence >= 0.98:
    return CNN prediction

Else:
    return unknown
```

---

## Usage

```python
from modules import PlateColorInference

model = PlateColorInference(
    model_path="weights/improved_tiny_cnn_dynamic_fp16.onnx"
)

result = model.predict_image(
    "image.jpg"
)

print(result)
```

---

## Output

Example:

```json
{
  "image": "image.jpg",
  "color": "yellow",
  "cnn_color": "yellow",
  "cnn_confidence": 0.9992,
  "hsv_color": "yellow",
  "message": "CNN and HSV agree"
}
```