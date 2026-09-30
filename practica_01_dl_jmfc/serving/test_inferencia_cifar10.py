import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import torch
from torchvision import datasets, transforms


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


def main() -> None:
    dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        ),
    )

    image, label = dataset[0]

    input_data = image.unsqueeze(0).numpy().astype(np.float32)

    payload = {
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

    request = Request(
        "http://localhost:8000/v2/models/vgg11/infer",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as error:
        print(f"Error HTTP {error.code}:")
        print(error.read().decode("utf-8"))
        sys.exit(1)

    except URLError as error:
        print(f"Error de conexión con Triton: {error}")
        sys.exit(1)

    logits = np.asarray(
        response_data["outputs"][0]["data"],
        dtype=np.float32,
    )

    predicted_class = int(np.argmax(logits))

    probabilities = torch.softmax(
        torch.from_numpy(logits),
        dim=0,
    )

    confidence = float(probabilities[predicted_class])

    print("Inferencia CIFAR-10 correcta")
    print("Etiqueta real:", label, CLASS_NAMES[label])
    print(
        "Predicción:",
        predicted_class,
        CLASS_NAMES[predicted_class],
    )
    print("Confianza:", f"{confidence:.4f}")
    print("Logits:", logits.tolist())


if __name__ == "__main__":
    main()
