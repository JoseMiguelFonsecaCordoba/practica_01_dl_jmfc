import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


def main() -> None:
    url = "http://localhost:8000/v2/models/vgg11/infer"

    rng = np.random.default_rng(seed=42)

    input_data = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(1, 3, 224, 224),
    ).astype(np.float32)

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
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as error:
        error_body = error.read().decode("utf-8")
        print(f"Error HTTP {error.code}: {error_body}")
        sys.exit(1)

    except URLError as error:
        print(f"Error de conexión con Triton: {error}")
        sys.exit(1)

    output = response_data["outputs"][0]
    output_data = np.asarray(
        output["data"],
        dtype=np.float32,
    )

    expected_shape = (1, 10)

    if tuple(output["shape"]) != expected_shape:
        raise RuntimeError(
            f"Forma inesperada: {output['shape']}. "
            f"Se esperaba {expected_shape}."
        )

    predicted_class = int(np.argmax(output_data))

    print("Inferencia HTTP correcta")
    print("Entrada:", input_data.shape)
    print("Salida:", tuple(output["shape"]))
    print("Clase predicha:", predicted_class)
    print("Logits:", output_data.tolist())


if __name__ == "__main__":
    main()
