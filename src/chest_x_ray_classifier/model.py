import torch
from torch import nn

from torchvision.models import densenet121, resnet18, ResNet18_Weights

class ChestDenseNet121(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = densenet121(weights="IMAGENET1K_V1")

        self.model.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )
        self.model.features.pool0 = nn.Identity()
        self.model.classifier = nn.Linear(1024, 14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ChestResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

        self.model.conv1.stride = (1, 1)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            14
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)