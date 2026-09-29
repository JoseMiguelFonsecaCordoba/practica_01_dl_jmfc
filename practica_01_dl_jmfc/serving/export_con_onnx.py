from pathlib import Path

import onnx
import onnxruntime as ort
import torch

from models.factory import create_model


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    weights_path = project_root / "artifacts" / "best_model.pth"
    onnx_output_path = (
        project_root / "serving" / "model_repository" / "vgg11" / "1" / "model.onnx"
    )
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"No se encontraron pesos en {weights_path}. "
            "Primero ejecuta: python train.py"
        )

    model = create_model("vgg11", pretrained=False).to("cpu")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(2, 1, 28, 28, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )

    onnx_model = onnx.load(str(onnx_output_path))
    onnx.checker.check_model(onnx_model)

    with torch.inference_mode():
        torch_output = model(dummy_input)

    session = ort.InferenceSession(
        str(onnx_output_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_output = session.run(
        ["output"],
        {"input": dummy_input.numpy()},
    )[0]

    torch.testing.assert_close(
        torch_output,
        torch.tensor(onnx_output),
        rtol=1e-3,
        atol=1e-4,
    )

    print(f"Modelo ONNX exportado y validado correctamente: {onnx_output_path}")


if __name__ == "__main__":
    main()
