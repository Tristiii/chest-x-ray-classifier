from model import ChestDenseNet121

from torchvision.transforms import ToTensor

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchmetrics.classification import MultilabelF1Score

from medmnist import ChestMNIST

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

train_mnist = ChestMNIST(
    split='train',
    size=28,
    transform=ToTensor()
)

val_mnist = ChestMNIST(
    split='val',
    size=28,
    transform=ToTensor()
)

train_dataloader = DataLoader(train_mnist, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_mnist, batch_size=32, shuffle=False)

labels = train_mnist.labels

positive = labels.sum(axis=0)
negative = len(labels) - positive

pos_weight = torch.tensor(
    negative / positive,
    dtype=torch.float32,
    device=device
)

torch.manual_seed(42)

model = ChestDenseNet121().to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
f1_fn = MultilabelF1Score(num_labels=14, average='macro').to(device)

for epoch in range(30):
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
            f1_fn.update(probs, y_batch)

        f1 = f1_fn.compute()
        f1_fn.reset()

    print(f'Epoch {epoch} | Loss: {total_loss:.4f} | F1: {f1:.4f}')
