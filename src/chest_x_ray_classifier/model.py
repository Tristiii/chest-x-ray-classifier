import torch
from torch import nn

from torchvision.models import densenet121

class VGGStyleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        self.block_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(
                in_features=128,
                out_features=14
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)
        x_4 = self.block_4(x_3)
        return x_4


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