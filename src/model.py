import torch.nn as nn

class BaseRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        # 1-channel input for FMNIST
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

    def forward(self, x):
        return self.features(x)

class ClientHead(nn.Module):
    def __init__(self, input_dim=16*14*14, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)