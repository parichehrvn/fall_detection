import torch
import torch.nn as nn


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, num_classes=2):
        super(TemporalConvNet, self).__init__()

        # TCN layers
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size,
                                 stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)

        # Add global average pooling to reduce dimensionality
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer for classification
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # Pass through TCN layers
        x = self.network(x)

        # Apply global average pooling (pooling across the time dimension)
        x = self.pool(x)

        # Flatten the output to pass it into the fully connected layer
        x = x.squeeze(-1)  # Remove the last dimension (time dimension after pooling)
        # Pass through fully connected layer for classification
        x = self.fc(x)

        return x
