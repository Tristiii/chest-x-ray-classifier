from model import ChestResNet18
from utils import get_model

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.transforms import (
    Compose,
    Grayscale,
    ToTensor,
    Normalize
)

from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score, MultilabelAUROC

from medmnist import ChestMNIST

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}.')

transformation = Compose([
    Grayscale(num_output_channels=3),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_mnist = ChestMNIST(
    split='train',
    size=28,
    transform=transformation
)

val_mnist = ChestMNIST(
    split='val',
    size=28,
    transform=transformation
)

test_mnist = ChestMNIST(
    split='test',
    size=28,
    transform=transformation
)

train_dataloader = DataLoader(train_mnist, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_mnist, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_mnist, batch_size=64, shuffle=False)

labels = train_mnist.labels

positive = labels.sum(axis=0)
negative = len(labels) - positive

pos_weight = torch.tensor(
    negative / positive,
    dtype=torch.float32,
    device=device
)

torch.manual_seed(42)

model = ChestResNet18().to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
accuracy_fn = MultilabelAccuracy(num_labels=14, average='macro').to(device)
f1_fn = MultilabelF1Score(num_labels=14, average='macro').to(device)
AUROC_fn = MultilabelAUROC(num_labels=14, average='macro').to(device)

print('--- Starting training & validating ---')
for epoch in range(50):
    total_loss = 0

    model.train()
    for X_batch, y_batch in train_dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    total_loss /= len(train_dataloader)

    model.eval()
    with torch.inference_mode():
        for X_batch, y_batch in val_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            probs = torch.sigmoid(y_pred)

            accuracy_fn.update(probs, y_batch)
            f1_fn.update(probs, y_batch)
            AUROC_fn.update(probs, y_batch)

        accuracy = accuracy_fn.compute()
        f1 = f1_fn.compute()
        AUROC = AUROC_fn.compute()

        accuracy_fn.reset()
        f1_fn.reset()
        AUROC_fn.reset()

    print(f'Epoch {epoch} | Accuracy: {accuracy:.2%} | Loss: {total_loss:.4f} | F1: {f1:.4f} | AUROC: {AUROC:.4f}')

print('--- Starting testing ---')
with torch.inference_mode():
    for X_batch, y_batch in test_dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)
        probs = torch.sigmoid(y_pred)

        accuracy_fn.update(probs, y_batch)
        f1_fn.update(probs, y_batch)
        AUROC_fn.update(probs, y_batch)

    accuracy = accuracy_fn.compute()
    f1 = f1_fn.compute()
    AUROC = AUROC_fn.compute()

    accuracy_fn.reset()
    f1_fn.reset()
    AUROC_fn.reset()

print(f'Accuracy: {accuracy:.2%} | F1: {f1:.4f} | AUROC: {AUROC:.4f}')

torch.save(model.state_dict(), get_model() / 'resnet18_weights.pth')