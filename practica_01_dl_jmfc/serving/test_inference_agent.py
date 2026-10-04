import sys
from pathlib import Path

from agent import InferenceAgent


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Uso:\n"
            "  python -m serving.test_inference_agent "
            "ruta/a/imagen.png"
        )
        raise SystemExit(1)

    image_path = Path(sys.argv[1])

    agent = InferenceAgent()

    try:
        result = agent.predict(image_path)
    except Exception as error:
        print(f"Error durante la inferencia: {error}")
        raise SystemExit(1) from error

    print("Inferencia realizada correctamente")
    print(f"Clase predicha: {result.predicted_class}")
    print(f"Nombre: {result.class_name}")
    print(f"Confianza: {result.confidence:.4f}")
    print(
        f"Tiempo de inferencia: "
        f"{result.inference_time_ms:.2f} ms"
    )
    print(f"Logits: {result.logits}")


if __name__ == "__main__":
    main()
