import base64
import html
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from detection_agent import DetectionAgent

from detection_metrics import DetectionMetrics


PROJECT_ROOT = Path(__file__).resolve().parent

WEIGHTS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "best_detection_model.pth"
)


app = FastAPI(
    title="Detector de objetos",
    description=(
        "Detección de objetos con Faster R-CNN "
        "y ResNet-50-FPN."
    ),
)


agent = DetectionAgent(
    weights_path=WEIGHTS_PATH,
    score_threshold=0.70,
)

metrics = DetectionMetrics()

HTML_PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Detector de objetos</title>

    <style>
        :root {
            --background: #f4f7fb;
            --card: #ffffff;
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --text: #172033;
            --muted: #64748b;
            --border: #e2e8f0;
            --success: #15803d;
            --success-bg: #f0fdf4;
            --error: #b91c1c;
            --error-bg: #fef2f2;
            --shadow: 0 15px 40px rgba(15, 23, 42, 0.08);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Arial, Helvetica, sans-serif;
            color: var(--text);
            background:
                linear-gradient(
                    135deg,
                    #e0ecff 0%,
                    var(--background) 45%,
                    #eefcf4 100%
                );
        }

        .container {
            width: min(1100px, 92%);
            margin: 0 auto;
            padding: 35px 0 50px;
        }

        .hero {
            color: white;
            background:
                linear-gradient(
                    135deg,
                    #1e3a8a,
                    #2563eb 55%,
                    #0891b2
                );
            border-radius: 24px;
            padding: 32px;
            box-shadow: var(--shadow);
            margin-bottom: 24px;
        }

        .hero h1 {
            margin: 0 0 10px;
            font-size: clamp(2rem, 5vw, 3.4rem);
        }

        .hero p {
            margin: 0;
            max-width: 760px;
            line-height: 1.6;
            color: #dbeafe;
        }

        .badges {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 22px;
        }

        .badge {
            padding: 8px 13px;
            border: 1px solid rgba(255,255,255,.35);
            border-radius: 999px;
            background: rgba(255,255,255,.12);
            font-size: .9rem;
        }

        .grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(280px, .7fr);
            gap: 24px;
            align-items: start;
        }

        .card {
            background: var(--card);
            border: 1px solid rgba(226, 232, 240, .9);
            border-radius: 20px;
            padding: 25px;
            box-shadow: var(--shadow);
        }

        .card h2 {
            margin-top: 0;
            margin-bottom: 8px;
        }

        .muted {
            color: var(--muted);
            line-height: 1.6;
        }

        .upload-area {
            border: 2px dashed #93c5fd;
            border-radius: 16px;
            padding: 28px;
            text-align: center;
            background: #eff6ff;
            transition: .2s ease;
        }

        .upload-area:hover {
            border-color: var(--primary);
            background: #dbeafe;
        }

        input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: white;
            cursor: pointer;
        }

        .preview {
            display: none;
            max-width: 100%;
            max-height: 260px;
            margin: 20px auto 0;
            border-radius: 14px;
            box-shadow: 0 8px 25px rgba(15, 23, 42, .15);
        }

        .button {
            width: 100%;
            margin-top: 18px;
            border: 0;
            border-radius: 11px;
            padding: 14px 20px;
            color: white;
            background: var(--primary);
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: .2s ease;
        }

        .button:hover {
            background: var(--primary-dark);
            transform: translateY(-1px);
        }

        .feature {
            display: flex;
            gap: 13px;
            margin: 18px 0;
        }

        .feature-icon {
            display: grid;
            place-items: center;
            flex: 0 0 38px;
            height: 38px;
            border-radius: 10px;
            color: #1d4ed8;
            background: #dbeafe;
            font-weight: bold;
        }

        .feature strong {
            display: block;
            margin-bottom: 4px;
        }

        .result {
            margin-top: 24px;
            padding: 25px;
            border: 1px solid #86efac;
            border-radius: 20px;
            background: var(--success-bg);
            box-shadow: var(--shadow);
        }

        .result h2 {
            margin-top: 0;
            color: var(--success);
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin: 18px 0;
        }

        .stat {
            padding: 15px;
            border-radius: 13px;
            background: white;
            border: 1px solid #dcfce7;
            text-align: center;
        }

        .stat strong {
            display: block;
            font-size: 1.35rem;
            color: #166534;
        }

        .stat span {
            display: block;
            margin-top: 5px;
            color: var(--muted);
            font-size: .85rem;
        }

        .result-image {
            display: block;
            width: 100%;
            max-height: 650px;
            object-fit: contain;
            margin: 22px auto 0;
            border-radius: 14px;
            background: #e2e8f0;
        }

        .detections {
            margin-top: 20px;
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 12px;
            overflow: hidden;
        }

        th, td {
            padding: 11px 12px;
            border-bottom: 1px solid var(--border);
            text-align: left;
            font-size: .92rem;
        }

        th {
            color: #1e3a8a;
            background: #dbeafe;
        }

        .error {
            margin-top: 24px;
            padding: 18px;
            border: 1px solid #fecaca;
            border-radius: 14px;
            color: var(--error);
            background: var(--error-bg);
        }

        footer {
            margin-top: 25px;
            text-align: center;
            color: var(--muted);
            font-size: .9rem;
        }

        @media (max-width: 780px) {
            .grid {
                grid-template-columns: 1fr;
            }

            .stats {
                grid-template-columns: 1fr;
            }

            .hero {
                padding: 25px;
            }
        }
    </style>
</head>

<body>
    <main class="container">
        <section class="hero">
            <h1>Detector de objetos</h1>
            <p>
                Sube una imagen y el sistema localizará y clasificará
                múltiples objetos mediante aprendizaje profundo.
            </p>

            <div class="badges">
                <span class="badge">Faster R-CNN</span>
                <span class="badge">ResNet-50-FPN</span>
                <span class="badge">Pascal VOC</span>
                <span class="badge">CUDA / PyTorch</span>
            </div>
        </section>

        <section class="grid">
            <div class="card">
                <h2>Analizar una imagen</h2>

                <p class="muted">
                    Selecciona una imagen en formato JPG, JPEG, PNG o BMP.
                    El resultado mostrará las cajas, las clases y la confianza.
                </p>

                <form
                    action="/predict"
                    method="post"
                    enctype="multipart/form-data"
                >
                    <div class="upload-area">
                        <input
                            id="fileInput"
                            type="file"
                            name="file"
                            accept=".png,.jpg,.jpeg,.bmp"
                            required
                        >

                        <img
                            id="preview"
                            class="preview"
                            alt="Vista previa"
                        >
                    </div>

                    <button class="button" type="submit">
                        Detectar objetos
                    </button>
                </form>
            </div>

            <aside class="card">
                <h2>Acerca del sistema</h2>

                <div class="feature">
                    <div class="feature-icon">1</div>
                    <div>
                        <strong>Preprocesamiento</strong>
                        <span class="muted">
                            La imagen se prepara para el modelo.
                        </span>
                    </div>
                </div>

                <div class="feature">
                    <div class="feature-icon">2</div>
                    <div>
                        <strong>Detección</strong>
                        <span class="muted">
                            Se localizan objetos mediante bounding boxes.
                        </span>
                    </div>
                </div>

                <div class="feature">
                    <div class="feature-icon">3</div>
                    <div>
                        <strong>Resultado</strong>
                        <span class="muted">
                            Se muestran las clases y scores de confianza.
                        </span>
                    </div>
                </div>
            </aside>
        </section>

        {result}

        <footer>
            Segunda fase de Deep Learning · Detección de objetos
        </footer>
    </main>

    <script>
        const input = document.getElementById("fileInput");
        const preview = document.getElementById("preview");

        input.addEventListener("change", function () {
            const file = this.files[0];

            if (!file) {
                preview.style.display = "none";
                return;
            }

            preview.src = URL.createObjectURL(file);
            preview.style.display = "block";
        });
    </script>
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
            "service": "object-detection-server",
            "model": "fasterrcnn_resnet50_fpn",
            "device": str(agent.device),
        }
    )

@app.get("/api/metrics")
async def api_metrics() -> JSONResponse:
    return JSONResponse(
        content=metrics.snapshot()
    )


####
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    data = metrics.snapshot()

    class_counts = data["class_counts"]

    if class_counts:
        maximum = max(class_counts.values())
    else:
        maximum = 1

    class_rows = ""

    for class_name, count in class_counts.items():
        percentage = (
            count / maximum
        ) * 100

        class_rows += f"""
        <tr>
            <td>{html.escape(class_name)}</td>
            <td>{count}</td>
            <td>
                <div class="bar-background">
                    <div
                        class="bar"
                        style="width: {percentage:.1f}%"
                    ></div>
                </div>
            </td>
        </tr>
        """

    if not class_rows:
        class_rows = """
        <tr>
            <td colspan="3">
                Todavía no hay detecciones registradas.
            </td>
        </tr>
        """

    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >

        <meta
            http-equiv="refresh"
            content="15"
        >

        <title>Dashboard de detección</title>

        <style>
            :root {{
                --blue: #2563eb;
                --dark-blue: #1e3a8a;
                --background: #f4f7fb;
                --card: #ffffff;
                --text: #172033;
                --muted: #64748b;
                --border: #e2e8f0;
                --green: #15803d;
                --red: #b91c1c;
            }}

            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                background: var(--background);
                color: var(--text);
                font-family: Arial, sans-serif;
            }}

            .container {{
                width: min(1150px, 92%);
                margin: 0 auto;
                padding: 30px 0 50px;
            }}

            header {{
                padding: 30px;
                color: white;
                border-radius: 20px;
                background:
                    linear-gradient(
                        135deg,
                        var(--dark-blue),
                        var(--blue)
                    );
                margin-bottom: 25px;
            }}

            header h1 {{
                margin: 0 0 10px;
            }}

            header p {{
                margin: 0;
                color: #dbeafe;
            }}

            .actions {{
                margin-top: 20px;
            }}

            .actions a {{
                display: inline-block;
                padding: 10px 15px;
                color: white;
                border: 1px solid rgba(255,255,255,.5);
                border-radius: 9px;
                text-decoration: none;
            }}

            .cards {{
                display: grid;
                grid-template-columns:
                    repeat(auto-fit, minmax(190px, 1fr));
                gap: 16px;
                margin-bottom: 25px;
            }}

            .card {{
                padding: 22px;
                border: 1px solid var(--border);
                border-radius: 16px;
                background: var(--card);
                box-shadow:
                    0 8px 25px rgba(15, 23, 42, .06);
            }}

            .metric-label {{
                color: var(--muted);
                font-size: .9rem;
            }}

            .metric-value {{
                margin-top: 10px;
                font-size: 2rem;
                font-weight: bold;
            }}

            .success {{
                color: var(--green);
            }}

            .failure {{
                color: var(--red);
            }}

            .section {{
                margin-top: 25px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                overflow: hidden;
                border-radius: 12px;
                background: white;
            }}

            th,
            td {{
                padding: 13px;
                border-bottom: 1px solid var(--border);
                text-align: left;
            }}

            th {{
                color: var(--dark-blue);
                background: #dbeafe;
            }}

            .bar-background {{
                width: 100%;
                height: 14px;
                overflow: hidden;
                border-radius: 999px;
                background: #e2e8f0;
            }}

            .bar {{
                height: 100%;
                border-radius: 999px;
                background:
                    linear-gradient(
                        90deg,
                        #2563eb,
                        #06b6d4
                    );
            }}

            .note {{
                margin-top: 20px;
                color: var(--muted);
                font-size: .9rem;
            }}

            @media (max-width: 650px) {{
                th:nth-child(3),
                td:nth-child(3) {{
                    display: none;
                }}
            }}
        </style>
    </head>

    <body>
        <main class="container">
            <header>
                <h1>Dashboard de detección</h1>

                <p>
                    Métricas acumuladas del servidor
                    Faster R-CNN.
                </p>

                <div class="actions">
                    <a href="/dashboard">
                        Actualizar dashboard
                    </a>
                    <a href="/">
                        Volver al detector
                    </a>
                </div>
            </header>

            <section class="cards">
                <article class="card">
                    <div class="metric-label">
                        Consultas totales
                    </div>
                    <div class="metric-value">
                        {data["total_requests"]}
                    </div>
                </article>

                <article class="card">
                    <div class="metric-label">
                        Consultas exitosas
                    </div>
                    <div class="metric-value success">
                        {data["successful_requests"]}
                    </div>
                </article>

                <article class="card">
                    <div class="metric-label">
                        Consultas fallidas
                    </div>
                    <div class="metric-value failure">
                        {data["failed_requests"]}
                    </div>
                </article>

                <article class="card">
                    <div class="metric-label">
                        Detecciones acumuladas
                    </div>
                    <div class="metric-value">
                        {data["total_detections"]}
                    </div>
                </article>

                <article class="card">
                    <div class="metric-label">
                        Promedio por imagen
                    </div>
                    <div class="metric-value">
                        {data[
                            "average_detections_per_image"
                        ]}
                    </div>
                </article>

                <article class="card">
                    <div class="metric-label">
                        Latencia promedio
                    </div>
                    <div class="metric-value">
                        {data["average_latency_ms"]:.2f} ms
                    </div>
                </article>
            </section>

            <section class="card section">
                <h2>Detecciones por clase</h2>

                <table>
                    <thead>
                        <tr>
                            <th>Clase</th>
                            <th>Cantidad</th>
                            <th>Distribución</th>
                        </tr>
                    </thead>

                    <tbody>
                        {class_rows}
                    </tbody>
                </table>
            </section>

            <p class="note">
                Las métricas se almacenan en memoria y se reinician
                cuando se reinicia el proceso o el contenedor.
                La página se actualiza automáticamente cada 15 segundos.
            </p>
        </main>
    </body>
    </html>
    """

    return HTMLResponse(
        content=dashboard_html,
        status_code=200,
    )
####


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    file: UploadFile = File(...),
) -> HTMLResponse:
    if not file.filename:
        return HTMLResponse(
            content=render_page(
                '<div class="error">'
                "No se recibió ningún archivo."
                "</div>"
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
                '<div class="error">'
                "Formato no permitido. Utiliza JPG, JPEG, PNG o BMP."
                "</div>"
            ),
            status_code=400,
        )

    file_content = await file.read()

    if not file_content:
        return HTMLResponse(
            content=render_page(
                '<div class="error">'
                "El archivo está vacío."
                "</div>"
            ),
            status_code=400,
        )

    try:
        result = agent.predict(file_content)

        metrics.record_success(
            detections=result.detections,
            latency_ms=result.inference_time_ms,
        )

        encoded_image = base64.b64encode(
            result.image_bytes
        ).decode("utf-8")

        safe_filename = html.escape(file.filename)

        rows = ""

        for detection in result.detections:
            rows += f"""
            <tr>
                <td>
                    {html.escape(detection["class_name"])}
                </td>
                <td>
                    {detection["score"] * 100:.2f}%
                </td>
                <td>
                    {html.escape(str(detection["box"]))}
                </td>
            </tr>
            """

        if not rows:
            rows = """
            <tr>
                <td colspan="3">
                    No se detectaron objetos con suficiente confianza.
                </td>
            </tr>
            """

        result_html = f"""
        <section class="result">
            <h2>Resultado de la detección</h2>

            <p>
                Archivo analizado:
                <strong>{safe_filename}</strong>
            </p>

            <div class="stats">
                <div class="stat">
                    <strong>{len(result.detections)}</strong>
                    <span>objetos detectados</span>
                </div>

                <div class="stat">
                    <strong>
                        {result.inference_time_ms:.0f} ms
                    </strong>
                    <span>latencia</span>
                </div>

                <div class="stat">
                    <strong>0.70</strong>
                    <span>umbral utilizado</span>
                </div>
            </div>

            <div class="detections">
                <table>
                    <thead>
                        <tr>
                            <th>Clase</th>
                            <th>Confianza</th>
                            <th>Bounding box</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>

            <img
                class="result-image"
                src="data:image/jpeg;base64,{encoded_image}"
                alt="Imagen con objetos detectados"
            >
        </section>
        """

        return HTMLResponse(
            content=render_page(result_html),
            status_code=200,
        )

    except Exception as error:
        metrics.record_failure()
        
        return HTMLResponse(
            content=render_page(
                '<div class="error">'
                "<strong>Error durante la inferencia:</strong><br>"
                f"{html.escape(str(error))}"
                "</div>"
            ),
            status_code=500,
        )