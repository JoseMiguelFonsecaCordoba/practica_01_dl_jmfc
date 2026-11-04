import torch

from forecasting.models import LSTMForecaster, RNNForecaster
from forecasting.data.series import SequenceDataset


def test_sequence_dataset_shapes():
    dataset = SequenceDataset([1, 2, 3, 4, 5], sequence_length=3)
    features, target = dataset[0]
    assert features.shape == (3, 1)
    assert target.shape == (1,)


def test_recurrent_models_output_one_value():
    inputs = torch.randn(4, 12, 1)
    assert RNNForecaster()(inputs).shape == (4, 1)
    assert LSTMForecaster()(inputs).shape == (4, 1)
