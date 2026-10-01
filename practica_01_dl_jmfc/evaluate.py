from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn

from datasets.cifar10 import get_cifar10_loaders
from models.factory import create_model
from utils.device import get_device


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def calculate_metrics(confusion_matrix):
    true_positive = confusion_matrix.diag()

    support = confusion_matrix.sum(dim=1)
    predicted_count = confusion_matrix.sum(dim=0)

    accuracy_per_class = torch.zeros_like(true_positive, dtype=torch.float64)
    precision = torch.zeros_like(true_positive, dtype=torch.float64)
    recall = torch.zeros_like(true_positive, dtype=torch.float64)
    f1_score = torch.zeros_like(true_positive, dtype=torch.float64)

    valid_support = support > 0
    valid_predicted = predicted_count > 0

    accuracy_per_class[valid_support] = (
        true_positive[valid_support].double()
        / support[valid_support].double()
    )

    precision[valid_predicted] = (
        true_positive[valid_predicted].double()
        / predicted_count[valid_predicted].double()
    )

    recall[valid_support] = (
        true_positive[valid_support].double()
        / support[valid_support].double()
    )

    denominator = precision + recall
    valid_denominator = denominator > 0

    f1_score[valid_denominator] = (
        2
        * precision[valid_denominator]
        * recall[valid_denominator]
        / denominator[valid_denominator]
    )

    return (
        accuracy_per_class,
        precision,
        recall,
        f1_score,
        support,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    weights_path = project_root / "artifacts" / "best_model.pth"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"No se encontraron los pesos en: {weights_path}\n"
            "Ejecuta primero el entrenamiento."
        )

    device = get_device()
    print(f"Evaluating on {device}", flush=True)

    _, _, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=32,
        val_split=0.2,
        num_workers=0,
    )

    model = create_model(
        "vgg11",
        num_classes=10,
        pretrained=False,
    ).to(device)

    state_dict = torch.load(
        weights_path,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_inference_time = 0.0

    confusion_matrix = torch.zeros(
        (len(CLASS_NAMES), len(CLASS_NAMES)),
        dtype=torch.int64,
    )

    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            start_time = perf_counter()
            outputs = model(inputs)
            total_inference_time += perf_counter() - start_time

            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)
            total_correct += (predictions == labels).sum().item()

            for true_label, predicted_label in zip(
                labels.cpu(),
                predictions.cpu(),
            ):
                confusion_matrix[true_label, predicted_label] += 1

    test_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    average_batch_time_ms = (
        total_inference_time / len(test_loader) * 1000
    )
    average_sample_time_ms = (
        total_inference_time / total_samples * 1000
    )

    (
        accuracy_per_class,
        precision,
        recall,
        f1_score,
        support,
    ) = calculate_metrics(confusion_matrix)

    print("\n===== Test metrics =====")
    print(f"Test samples: {total_samples}")
    print(f"Correct: {total_correct}")
    print(f"Incorrect: {total_samples - total_correct}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(
        f"Average inference time per batch: "
        f"{average_batch_time_ms:.2f} ms"
    )
    print(
        f"Average inference time per sample: "
        f"{average_sample_time_ms:.2f} ms"
    )

    print("\n===== Per-class metrics =====")
    print(
        f"{'Class':<12}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>10}"
    )

    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name:<12}"
            f"{accuracy_per_class[index].item() * 100:>11.2f}%"
            f"{precision[index].item() * 100:>11.2f}%"
            f"{recall[index].item() * 100:>11.2f}%"
            f"{f1_score[index].item() * 100:>11.2f}%"
            f"{support[index].item():>10d}"
        )

    print("\n===== Confusion matrix =====")
    print(confusion_matrix)


if __name__ == "__main__":
    main()
