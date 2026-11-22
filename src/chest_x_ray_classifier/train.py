from model import VGGStyleCNN

from torchvision import transforms

from medmnist import ChestMNIST

transformation = transforms.Compose([
    transforms.ToTensor()
])

train_mnist = ChestMNIST(
    split='train',
    size=28,
    transform=transformation
)

test_mnist = ChestMNIST(
    split='test',
    size=28,
    transform=transformation
)