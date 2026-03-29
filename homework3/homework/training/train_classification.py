import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric
from homework.models import Classifier, save_model


def train(
    exp_dir: str = "logs",
    num_epoch: int = 15,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    torch.manual_seed(seed)

    # Timestamped log directory
    log_dir = Path(exp_dir) / f"classifier_{datetime.now().strftime('%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    model = Classifier().to(device)

    train_loader = load_data(
        "classification_data/train",
        transform_pipeline="aug",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )
    val_loader = load_data(
        "classification_data/val",
        transform_pipeline="default",
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(num_epoch):
        # --- Training ---
        model.train()
        train_metric = AccuracyMetric()

        for img, label in train_loader:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()

            train_metric.add(logits.argmax(dim=1), label)

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_metric = AccuracyMetric()

        with torch.inference_mode():
            for img, label in val_loader:
                img = img.to(device)
                pred = model.predict(img)
                val_metric.add(pred, label)

        train_acc = train_metric.compute()["accuracy"]
        val_acc = val_metric.compute()["accuracy"]

        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)

        # Save model only when val accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model)
            print(f"Epoch {epoch + 1:3d}/{num_epoch}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  *saved*")
        else:
            print(f"Epoch {epoch + 1:3d}/{num_epoch}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    writer.close()
    print(f"Best val accuracy: {best_val_acc:.3f} — model saved to homework/classifier.th")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))
