from pathlib import Path

import matplotlib.pyplot as plt


def plot_forecast(targets, predictions, output_path, num_points=200):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(targets[:num_points], label="Real")
    plt.plot(predictions[:num_points], label="Predicción")
    plt.xlabel("Tiempo")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
