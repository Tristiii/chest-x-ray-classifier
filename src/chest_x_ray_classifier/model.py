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

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            14
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize to 224x224
        x = nn.functional.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        # Convert grayscale to RGB
        x = x.repeat(1, 3, 1, 1)

        # ImageNet normalization
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        x = (x - mean) / std

        return self.model(x)