import html
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from agent import InferenceAgent


app = FastAPI(
    title="CIFAR-10 Inference Server",
    description="Servidor web para clasificar imágenes mediante Triton y VGG11.",
)

agent = InferenceAgent()


HTML_PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>CIFAR-10 - Inferencia</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 40px auto;
            padding: 0 20px;
        }

        h1 {
            color: #333;
        }

        form {
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }

        input[type="file"] {
            margin-bottom: 15px;
        }

        button {
            padding: 10px 18px;
            cursor: pointer;
        }

        .result {
            background-color: #f0f8f0;
            border: 1px solid #72a772;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }

        .error {
            background-color: #fff0f0;
            border: 1px solid #cc7777;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            color: #8b0000;
        }

        .field {
            margin: 8px 0;
        }
    </style>
</head>
<body>
    <h1>Clasificador CIFAR-10</h1>

    <p>
        Selecciona una imagen para enviarla al modelo VGG11
        servido por Triton.
    </p>

    <form action="/predict" method="post" enctype="multipart/form-data">
        <input
            type="file"
            name="file"
            accept=".png,.jpg,.jpeg,.bmp"
            required
        >
        <br>
        <button type="submit">Clasificar imagen</button>
    </form>

    {result}
</body>
</html>
"""
def render_page(result: str = "") -> str:
    return HTML_PAGE.replace("{result}", result)


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    return HTMLResponse(
        content=render_page(),
        status_code=200,
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        content={
            "status": "ok",
            "service": "cifar10-inference-server",
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)) -> HTMLResponse:
    if not file.filename:
        return HTMLResponse(
            content=render_page(
                result=(
                    '<div class="error">'
                    "No se recibió ningún archivo."
                    "</div>"
                )
            ),
            status_code=400,
        )

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        return HTMLResponse(
            content=render_page(
                result=(
                    '<div class="error">'
                    "Formato no permitido. Utiliza PNG, JPG, "
                    "JPEG o BMP."
                    "</div>"
                )
            ),
            status_code=400,
        )

    file_content = await file.read()

    if not file_content:
        return HTMLResponse(
            content=render_page(
                result=(
                    '<div class="error">'
                    "El archivo recibido está vacío."
                    "</div>"
                )
            ),
            status_code=400,
        )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_file.write(file_content)
            temporary_path = Path(temporary_file.name)

        result = agent.predict(temporary_path)

        safe_filename = html.escape(file.filename)

        result_html = f"""
        <div class="result">
            <h2>Resultado</h2>

            <div class="field">
                <strong>Archivo:</strong> {safe_filename}
            </div>

            <div class="field">
                <strong>Clase predicha:</strong>
                {html.escape(result.class_name)}
            </div>

            <div class="field">
                <strong>Índice de clase:</strong>
                {result.predicted_class}
            </div>

            <div class="field">
                <strong>Confianza:</strong>
                {result.confidence * 100:.2f}%
            </div>

            <div class="field">
                <strong>Latencia:</strong>
                {result.inference_time_ms:.2f} ms
            </div>
        </div>
        """

        return HTMLResponse(
            content=render_page(result=result_html),
            status_code=200,
        )

    except Exception as error:
        error_message = html.escape(str(error))

        return HTMLResponse(
            content=render_page(
                result=(
                    '<div class="error">'
                    "<strong>Error durante la inferencia:</strong><br>"
                    f"{error_message}"
                    "</div>"
                )
            ),
            status_code=500,
        )

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
