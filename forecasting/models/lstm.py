import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        return self.output(sequence_output[:, -1, :])
