import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from PIL import Image
from torchvision import transforms


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass
class InferenceResult:
    predicted_class: int
    class_name: str
    confidence: float
    inference_time_ms: float
    logits: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


class InferenceAgent:
    def __init__(
        self,
        triton_url: str = "http://localhost:8000", #dirección de triton
        model_name: str = "vgg11", # nombre del modelo
        timeout: float = 30.0, #velocidad maxima de espera
    ) -> None:
        self.triton_url = triton_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

        self.inference_url = (
            f"{self.triton_url}/v2/models/"
            f"{self.model_name}/infer"
        )

        self.transform = transforms.Compose(  #aqui reproduce CIFAR10
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def load_image(self, image_source) -> Image.Image:
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")

        image_path = Path(image_source)

        if not image_path.exists():
            raise FileNotFoundError(
                f"No existe la imagen: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"La ruta no corresponde a un archivo: {image_path}"
            )

        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except Exception as error:
            raise ValueError(
                f"No se pudo abrir la imagen {image_path}: {error}"
            ) from error

    def preprocess(self, image_source) -> torch.Tensor:
        image = self.load_image(image_source)
        tensor = self.transform(image)

        # Triton espera batch:
        # [canales, alto, ancho] -> [1, canales, alto, ancho]
        return tensor.unsqueeze(0)

    def build_request(self, input_tensor: torch.Tensor) -> dict:
        input_data = input_tensor.detach().cpu().numpy()

        return {
            "inputs": [
                {
                    "name": "input",
                    "shape": list(input_data.shape),
                    "datatype": "FP32",
                    "data": input_data.reshape(-1).tolist(),
                }
            ],
            "outputs": [
                {
                    "name": "output",
                }
            ],
        }

    def send_request(self, request_payload: dict) -> dict:
        body = json.dumps(request_payload).encode("utf-8")

        request = Request(
            self.inference_url,
            data=body,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                response_body = response.read().decode("utf-8")

        except HTTPError as error:
            error_body = error.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"Triton respondió HTTP {error.code}: "
                f"{error_body}"
            ) from error

        except URLError as error:
            raise ConnectionError(
                f"No se pudo conectar con Triton en "
                f"{self.inference_url}. "
                f"¿Está ejecutándose Docker Compose? "
                f"Detalle: {error.reason}"
            ) from error

        except TimeoutError as error:
            raise TimeoutError(
                f"Triton superó el timeout de "
                f"{self.timeout} segundos."
            ) from error

        try:
            return json.loads(response_body)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Triton devolvió una respuesta que no es JSON válido."
            ) from error

    def parse_logits(self, response: dict) -> list[float]:
        outputs = response.get("outputs")

        if not outputs:
            raise ValueError(
                "La respuesta de Triton no contiene 'outputs'."
            )

        output = outputs[0]

        if "data" not in output:
            raise ValueError(
                "La salida de Triton no contiene 'data'."
            )

        logits = output["data"]

        if len(logits) != len(CLASS_NAMES):
            raise ValueError(
                f"Se esperaban {len(CLASS_NAMES)} logits, "
                f"pero Triton devolvió {len(logits)}."
            )

        return [float(value) for value in logits]

    def predict(self, image_source) -> InferenceResult:
        input_tensor = self.preprocess(image_source)
        request_payload = self.build_request(input_tensor)

        start_time = perf_counter()
        response = self.send_request(request_payload)
        elapsed_time_ms = (perf_counter() - start_time) * 1000

        logits = self.parse_logits(response)

        logits_tensor = torch.tensor(logits)
        probabilities = torch.softmax(logits_tensor, dim=0)

        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())

        return InferenceResult(
            predicted_class=predicted_class,
            class_name=CLASS_NAMES[predicted_class],
            confidence=confidence,
            inference_time_ms=elapsed_time_ms,
            logits=logits,
        )
