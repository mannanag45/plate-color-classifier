# Plate Color Classifier

Lightweight number plate color classifier using an Improved Tiny CNN.

Supported classes:

- Black
- Blue
- Green
- White
- Yellow

## Installation

```bash
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

## Model Weights

Download ONNX model:

https://drive.google.com/file/d/1uqSTbOZyFp3g4x-0cXui8d1EU3LosnYl/view?usp=drive_link

Place the model at:

```text
weights/improved_tiny_cnn_single.onnx
```

## Example Usage

```python
from modules import PlateColorInference

model = PlateColorInference(
    model_path="weights/improved_tiny_cnn_single.onnx"
)

results = model.predict_folder(
    image_folder="test_images",
    batch_size=1
)

print(results)
```

## Preprocessing

- Read image using OpenCV (`cv2`)
- Convert BGR → RGB
- Resize to `96 × 96`
- Normalize pixel values to `[0,1]`
- Convert HWC → CHW
- Create NumPy tensor

## Postprocessing

- Apply softmax
- Select class with highest probability
- Return predicted color and confidence score

## Test Client

```bash
python test_client_tensor.py
```

## Performance

```text
Test Accuracy: 100.00%
Model Size: 771 KB (ONNX)
Latency: 3.58 ms/image
```

## Deployment

Training format:

```text
improved_tiny_cnn_best.pth
```

Deployment format:

```text
improved_tiny_cnn_single.onnx
```

The exported ONNX model can be converted to TensorRT in the deployment environment.