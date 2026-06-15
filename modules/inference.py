from pathlib import Path

import numpy as np
import onnxruntime as ort

from modules.preprocessing import preprocess_image
from modules.postprocessing import postprocess_logits


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


class PlateColorInference:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = (
            self.session.get_inputs()[0].name
        )

    def get_image_paths(self, image_folder):
        folder = Path(image_folder)

        return sorted([
            path
            for path in folder.iterdir()
            if path.suffix.lower()
            in IMAGE_EXTENSIONS
        ])

    def predict_folder(
        self,
        image_folder,
        batch_size=1
    ):
        image_paths = self.get_image_paths(
            image_folder
        )

        if len(image_paths) == 0:
            raise ValueError(
                "No valid images found."
            )

        all_results = []

        for i in range(
            0,
            len(image_paths),
            batch_size
        ):
            batch_paths = image_paths[
                i:i + batch_size
            ]

            batch = np.stack(
                [
                    preprocess_image(path)
                    for path in batch_paths
                ],
                axis=0
            ).astype(np.float32)

            logits = self.session.run(
                None,
                {self.input_name: batch}
            )[0]

            image_names = [
                path.name
                for path in batch_paths
            ]

            results = postprocess_logits(
                logits,
                image_names
            )

            all_results.extend(results)

        return all_results