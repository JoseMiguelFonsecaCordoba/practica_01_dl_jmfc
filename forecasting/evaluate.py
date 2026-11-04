import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn

from forecasting.data.series import get_jena_loaders
from forecasting.models import LSTMForecaster, RNNForecaster
from forecasting.plotting import plot_forecast


MODEL_CLASSES = {"rnn": RNNForecaster, "lstm": LSTMForecaster}


def evaluate_checkpoint(checkpoint_path, test_loader, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MODEL_CLASSES[checkpoint["model_name"]](
        hidden_size=checkpoint["hidden_size"]
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    targets = []
    predictions = []
    with torch.no_grad():
        for features, batch_targets in test_loader:
            batch_predictions = model(features.to(device))
            losses.append(
                criterion(batch_predictions, batch_targets.to(device)).item()
            )
            targets.extend(batch_targets.squeeze(-1).cpu().numpy())
            predictions.extend(batch_predictions.squeeze(-1).cpu().numpy())
    return {
        "model": checkpoint["model_name"],
        "test_mse": float(np.mean(losses)),
        "targets": targets,
        "predictions": predictions,
        "mean": checkpoint["mean"],
        "std": checkpoint["std"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate forecasting checkpoints")
    parser.add_argument("--artifacts-dir", default="forecasting/artifacts")
    parser.add_argument("--data-dir", default="forecasting/data")
    parser.add_argument("--output-dir", default="forecasting/reports")
    parser.add_argument("--mlruns-dir", default="mlruns")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    mlflow.set_tracking_uri(Path(args.mlruns_dir).resolve().as_uri())
    loaders = get_jena_loaders(args.data_dir)
    test_loader = loaders[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for checkpoint_path in sorted(Path(args.artifacts_dir).glob("*_best.pt")):
        result = evaluate_checkpoint(checkpoint_path, test_loader, device)
        model_name = result["model"]
        with mlflow.start_run(run_name=f"{model_name}-test-evaluation"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("checkpoint", checkpoint_path.name)
            mlflow.log_metric("test_mse_normalized", result["test_mse"])
            mlflow.log_metric(
                "test_rmse_celsius",
                float(np.sqrt(result["test_mse"]) * result["std"]),
            )
        plot_forecast(
            np.asarray(result["targets"]) * result["std"] + result["mean"],
            np.asarray(result["predictions"]) * result["std"] + result["mean"],
            output_dir / f"{model_name}_forecast.png",
        )
        results[model_name] = {
            "test_mse_normalized": result["test_mse"],
            "test_rmse_celsius": float(np.sqrt(result["test_mse"]) * result["std"]),
        }
    (output_dir / "evaluation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
