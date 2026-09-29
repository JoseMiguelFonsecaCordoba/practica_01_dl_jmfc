#descargar
# python -m pip install onnx onnxscript onnxruntime

import torch
import onnx
import onnxruntime as ort

import models.cnn as CNN


def main() -> None:
    model = CNN()

    state_dict = torch.load(
        "..artifacts/best_model.pth", 
        map_location='cpu',
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(
        8, 1, 28, 28
    ) #imagen de prueba

    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={
            "x": {0: "batch_size"},
        },
        dynamo=True,
    )

    onnx_program.save(
        "serving/model_repository/cnn/1/model.onnx"
    )

    onnx_model = onnx.load(
        "serving/model_repository/cnn/1/model.onnx"
    )

    #de aqui hacia arriba es guardar modelo

    #hacia abajo es validar que lo que guardamos este bien
    ###validar que lo que guardamos este bien ####
    onnx.checker.check_model(onnx_model)
    with torch.inference_mode():
        torch_output = model(dummy_input)

    session = ort.InferenceSession(
        "serving/model_repository/cnn/1/model.onnx",
        providers=["CPUExecutionProvider"]
    )
    onnx_output = session.run(
        ["output"], {"input": dummy_input.numpy()}
    )[0]

    torch.testing.assert_close(
        torch_output,
        torch.tensor(onnx_output),
        rtol=1e-3,
        atol=1e-5,
    )

    print("Modelo ONNX exportado y validado correctamente!")

if __name__ == "__main__":
    main()



