import io
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes

from models.detector import create_detection_model
#from utils.device import get_device


CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@dataclass
class DetectionResult:
    image_bytes: bytes
    detections: list[dict]
    inference_time_ms: float


class DetectionAgent:
    def __init__(
        self,
        weights_path: str | Path,
        score_threshold: float = 0.70,
    ) -> None:
        self.device = torch.device("cpu")
        self.score_threshold = score_threshold

        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"No existe el checkpoint: {self.weights_path}"
            )

        self.model = create_detection_model(
            num_classes=len(CLASS_NAMES),
            pretrained_backbone=False,
        ).to(self.device)

        state_dict = torch.load(
            self.weights_path,
            map_location=self.device,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, image_bytes: bytes) -> DetectionResult:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image_tensor = (
            pil_to_tensor(image).float() / 255.0
        )

        start_time = perf_counter()

        with torch.inference_mode():
            prediction = self.model(
                [image_tensor.to(self.device)]
            )[0]

        elapsed_time_ms = (
            perf_counter() - start_time
        ) * 1000

        boxes = prediction["boxes"].detach().cpu()
        labels = prediction["labels"].detach().cpu()
        scores = prediction["scores"].detach().cpu()

        keep = scores >= self.score_threshold

        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        detections = []

        for box, label, score in zip(
            boxes,
            labels,
            scores,
        ):
            class_index = int(label.item())

            detections.append(
                {
                    "class_id": class_index,
                    "class_name": CLASS_NAMES[class_index],
                    "score": float(score.item()),
                    "box": [
                        round(float(value), 2)
                        for value in box.tolist()
                    ],
                }
            )

        annotated_image = draw_bounding_boxes(
            (image_tensor * 255).to(torch.uint8),
            boxes=boxes,
            labels=[
                f"{item['class_name']}: "
                f"{item['score']:.2f}"
                for item in detections
            ],
            colors="red",
            width=3,
        )

        output_image = Image.fromarray(
            annotated_image.permute(1, 2, 0).numpy()
        )

        output_buffer = io.BytesIO()
        output_image.save(
            output_buffer,
            format="JPEG",
            quality=90,
        )

        return DetectionResult(
            image_bytes=output_buffer.getvalue(),
            detections=detections,
            inference_time_ms=elapsed_time_ms,
        )
