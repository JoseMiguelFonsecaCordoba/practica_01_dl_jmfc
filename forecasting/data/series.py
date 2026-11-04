import csv
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DATA_URL = (
    "https://storage.googleapis.com/"
    "tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)


class SequenceDataset(Dataset):
    def __init__(self, series, sequence_length=24):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.series = torch.as_tensor(series, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.series) - self.sequence_length)

    def __getitem__(self, index):
        start = index
        end = start + self.sequence_length
        return (
            self.series[start:end].unsqueeze(-1),
            self.series[end].unsqueeze(0),
        )


def download_jena(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "jena_climate_2009_2016.csv"
    if csv_path.exists():
        return csv_path

    zip_path = data_dir / "jena_climate_2009_2016.csv.zip"
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)
    return csv_path


def load_temperature(csv_path, sample_every=6):
    values = []
    with Path(csv_path).open(newline="") as file:
        for row in csv.DictReader(file):
            values.append(float(row["T (degC)"]))
    return np.asarray(values[::sample_every], dtype=np.float32)


def get_jena_loaders(
    data_dir,
    batch_size=64,
    sequence_length=24,
    num_workers=0,
):
    values = load_temperature(download_jena(data_dir))
    train_end = int(len(values) * 0.70)
    val_end = int(len(values) * 0.85)
    train_values = values[:train_end]
    val_values = values[train_end:val_end]
    test_values = values[val_end:]

    mean = float(train_values.mean())
    std = float(train_values.std())
    if std == 0:
        raise ValueError("Training series has zero variance")

    def normalize(split):
        return (split - mean) / std
    datasets = [
        SequenceDataset(normalize(split), sequence_length)
        for split in (train_values, val_values, test_values)
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for dataset in datasets
    ]
    return (*loaders, mean, std)
