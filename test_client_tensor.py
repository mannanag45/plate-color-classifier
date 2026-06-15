from modules import PlateColorInference


model = PlateColorInference(
    model_path="models/improved_tiny_cnn_single.onnx"
)

results = model.predict_folder(
    image_folder="real_blue_test/blue",
    batch_size=1
)

for result in results:
    print(result)