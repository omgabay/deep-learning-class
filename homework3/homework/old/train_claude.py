import argparse
from pathlib import Path

import torch
import torch.nn as nn

from ..datasets.classification_dataset import load_data
from ..metrics import AccuracyMetric
from ..models import Classifier, save_model


def train(
    exp_dir: str = ".",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model = Classifier(**kwargs).to(device)

    train_data = load_data(
        "classification_data/train",
        transform_pipeline="aug",
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_data = load_data(
        "classification_data/val",
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        model.train()
        train_metric = AccuracyMetric()
        train_metric.reset()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logits = model(img)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metric.add(logits.argmax(dim=1), label)

        model.eval()
        val_metric = AccuracyMetric()
        val_metric.reset()

        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                preds = model.predict(img)
                val_metric.add(preds, label)

        train_acc = train_metric.compute()["accuracy"]
        val_acc = val_metric.compute()["accuracy"]
        print(f"Epoch {epoch + 1:3d}/{num_epoch} | train acc: {train_acc:.4f} | val acc: {val_acc:.4f}")

    save_model(model)
    print("Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    train(**vars(args))
