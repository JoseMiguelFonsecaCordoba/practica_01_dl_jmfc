from pathlib import Path

import torch

from models.factory import create_model


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    weights_path = project_root / "artifacts" / "best_model.pth"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"No se encontraron pesos en {weights_path}. "
            "Primero ejecuta: python train.py"
        )

    model = create_model("vgg11", pretrained=False).to("cpu")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    sample = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(sample)

    if logits.shape != (1, 10):
        raise RuntimeError(f"Forma inesperada de logits: {tuple(logits.shape)}")

    print("Inferencia local OK. logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
