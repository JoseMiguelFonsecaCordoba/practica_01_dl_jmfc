import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn

from forecasting.data.series import get_jena_loaders
from forecasting.models import LSTMForecaster, RNNForecaster


ROOT = Path(__file__).resolve().parent
MODEL_CLASSES = {"rnn": RNNForecaster, "lstm": LSTMForecaster}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, targets)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size
    return total_loss / max(count, 1)


def train_model(args, model_name, loaders, mean, std, device):
    train_loader, val_loader, _, _, _ = loaders
    model = MODEL_CLASSES[model_name](
        input_size=1,
        hidden_size=args.hidden_size,
        output_size=1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    with mlflow.start_run(
        run_name=f"{model_name}-{args.run_name_suffix}"
    ):
        mlflow.log_params(
            {
                "model": model_name,
                "sequence_length": args.sequence_length,
                "hidden_size": args.hidden_size,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "device": str(device),
                "normalization_mean": mean,
                "normalization_std": std,
            }
        )
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
            with torch.no_grad():
                val_loss = run_epoch(model, val_loader, criterion, device)
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss},
                step=epoch,
            )
            improved = val_loss < best_val_loss - args.min_delta
            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if args.patience > 0 and epochs_without_improvement >= args.patience:
                    mlflow.log_param("stopped_epoch", epoch)
                    break

        model.load_state_dict(best_state)
        mlflow.log_metric("best_val_loss", best_val_loss)
        input_example = torch.zeros(
            1,
            args.sequence_length,
            1,
            dtype=torch.float32,
        )
        mlflow.pytorch.log_model(
            model,
            "model",
            input_example=input_example,
            serialization_format="pickle",
        )
        run_id = mlflow.active_run().info.run_id

    output_dir = Path(args.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_name}_best.pt"
    torch.save(
        {
            "model_name": model_name,
            "state_dict": best_state,
            "hidden_size": args.hidden_size,
            "sequence_length": args.sequence_length,
            "mean": mean,
            "std": std,
            "val_loss": best_val_loss,
            "run_id": run_id,
        },
        checkpoint_path,
    )
    return {
        "model": model_name,
        "val_loss": best_val_loss,
        "checkpoint": checkpoint_path.name,
        "run_id": run_id,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train RNN and LSTM with MLflow")
    parser.add_argument(
        "--models", nargs="+", choices=("rnn", "lstm"), default=("rnn", "lstm")
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment-name",
        default="jena-temperature-forecasting",
    )
    parser.add_argument("--run-name-suffix", default="baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    device = resolve_device(args.device)
    loaders = get_jena_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    mean, std = loaders[-2:]
    results = [
        train_model(args, name, loaders, mean, std, device) for name in args.models
    ]
    best = min(results, key=lambda result: result["val_loss"])
    output_dir = Path(args.artifacts_dir)
    (output_dir / "model_metadata.json").write_text(
        json.dumps({"best": best, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"best_model": best, "results": results}, indent=2))


if __name__ == "__main__":
    main()
