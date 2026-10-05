import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from agent import InferenceAgent


class InferenceApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CIFAR-10 - Inferencia con Triton")
        self.root.geometry("700x650")
        self.root.resizable(False, False)

        self.agent = InferenceAgent()

        self.selected_image_path: Path | None = None
        self.preview_image = None

        self.create_widgets()

    def create_widgets(self) -> None:
        title_label = tk.Label(
            self.root,
            text="Clasificador CIFAR-10",
            font=("Arial", 20, "bold"),
        )
        title_label.pack(pady=15)

        description_label = tk.Label(
            self.root,
            text="Selecciona una imagen para enviarla a Triton",
            font=("Arial", 11),
        )
        description_label.pack(pady=5)

        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=15)

        select_button = tk.Button(
            buttons_frame,
            text="Seleccionar imagen",
            command=self.select_image,
            width=20,
        )
        select_button.grid(row=0, column=0, padx=5)

        predict_button = tk.Button(
            buttons_frame,
            text="Clasificar imagen",
            command=self.predict_image,
            width=20,
        )
        predict_button.grid(row=0, column=1, padx=5)

        self.image_label = tk.Label(
            self.root,
            text="Aquí aparecerá la imagen",
            width=50,
            height=20,
            relief=tk.SUNKEN,
        )
        self.image_label.pack(pady=10)

        self.path_label = tk.Label(
            self.root,
            text="Ninguna imagen seleccionada",
            wraplength=600,
            font=("Arial", 10),
        )
        self.path_label.pack(pady=5)

        result_frame = tk.LabelFrame(
            self.root,
            text="Resultado",
            padx=15,
            pady=15,
        )
        result_frame.pack(fill="x", padx=30, pady=15)

        self.prediction_label = tk.Label(
            result_frame,
            text="Predicción: --",
            font=("Arial", 14, "bold"),
        )
        self.prediction_label.pack(anchor="w")

        self.confidence_label = tk.Label(
            result_frame,
            text="Confianza: --",
            font=("Arial", 12),
        )
        self.confidence_label.pack(anchor="w", pady=5)

        self.latency_label = tk.Label(
            result_frame,
            text="Latencia: --",
            font=("Arial", 12),
        )
        self.latency_label.pack(anchor="w")

        self.status_label = tk.Label(
            self.root,
            text="Estado: esperando una imagen",
            fg="gray",
            font=("Arial", 10),
        )
        self.status_label.pack(pady=10)

    def select_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                (
                    "Imágenes",
                    "*.png *.jpg *.jpeg *.bmp",
                ),
                ("Todos los archivos", "*.*"),
            ],
        )

        if not file_path:
            return

        self.selected_image_path = Path(file_path)
        self.path_label.config(
            text=f"Imagen seleccionada: {self.selected_image_path}"
        )

        try:
            image = Image.open(self.selected_image_path)
            image.thumbnail((400, 300))

            self.preview_image = ImageTk.PhotoImage(image)
            self.image_label.config(
                image=self.preview_image,
                text="",
            )

            self.status_label.config(
                text="Estado: imagen lista para clasificar",
                fg="blue",
            )

            self.clear_result()

        except Exception as error:
            messagebox.showerror(
                "Error al abrir la imagen",
                f"No se pudo abrir la imagen:\n{error}",
            )

    def predict_image(self) -> None:
        if self.selected_image_path is None:
            messagebox.showwarning(
                "Imagen requerida",
                "Primero selecciona una imagen.",
            )
            return

        self.status_label.config(
            text="Estado: realizando inferencia...",
            fg="orange",
        )
        self.root.update_idletasks()

        try:
            result = self.agent.predict(self.selected_image_path)

            self.prediction_label.config(
                text=(
                    f"Predicción: {result.class_name} "
                    f"(clase {result.predicted_class})"
                )
            )

            self.confidence_label.config(
                text=(
                    f"Confianza: "
                    f"{result.confidence * 100:.2f}%"
                )
            )

            self.latency_label.config(
                text=(
                    f"Latencia: "
                    f"{result.inference_time_ms:.2f} ms"
                )
            )

            self.status_label.config(
                text="Estado: inferencia completada",
                fg="green",
            )

        except Exception as error:
            self.status_label.config(
                text="Estado: error durante la inferencia",
                fg="red",
            )

            messagebox.showerror(
                "Error de inferencia",
                str(error),
            )

    def clear_result(self) -> None:
        self.prediction_label.config(
            text="Predicción: --"
        )
        self.confidence_label.config(
            text="Confianza: --"
        )
        self.latency_label.config(
            text="Latencia: --"
        )


def main() -> None:
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
