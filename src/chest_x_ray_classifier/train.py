from model import VGGStyleCNN

from torchvision.transforms import ToTensor

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchmetrics import Accuracy

from medmnist import ChestMNIST

train_mnist = ChestMNIST(
    split='train',
    size=28,
    transform=ToTensor()
)

test_mnist = ChestMNIST(
    split='test',
    size=28,
    transform=ToTensor()
)

train_dataloader = DataLoader(train_mnist, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_mnist, batch_size=64, shuffle=True)

torch.manual_seed(42)

model = VGGStyleCNN()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
accuracy_fn = Accuracy(task='multilabel', num_labels=14)

for epoch in range(1000):
    total_loss = 0

    model.train()
    for X_batch, y_batch in train_dataloader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    total_loss /= len(train_dataloader)

    model.eval()
    with torch.inference_mode():
        for X_batch, y_batch in test_dataloader:
            y_pred = model(X_batch)
            accuracy_fn.update(y_pred, y_batch)

        accuracy = accuracy_fn.compute()
        accuracy_fn.reset()

    print(f'Epoch {epoch} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2%}')
