import torch
import torch.nn as nn

MODEL_PATH = "models/improved_tiny_cnn_best.pth"
ONNX_PATH = "models/improved_tiny_cnn_single.onnx"

IMG_SIZE = 96


class ImprovedTinyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 160, 3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(160, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


checkpoint = torch.load(MODEL_PATH, map_location="cpu")
classes = checkpoint["classes"]

model = ImprovedTinyCNN(len(classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    do_constant_folding=True,
    external_data=False
)

print("Exported single-file ONNX:", ONNX_PATH)
print("Classes:", classes)