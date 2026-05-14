import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from homework.datasets.road_dataset import load_data
from homework.metrics import ConfusionMatrix
from homework.models import Detector, save_model


def train(
    exp_dir: str = "logs",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
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

    log_dir = Path(exp_dir) / f"detector_{datetime.now().strftime('%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    model = Detector().to(device)

    train_loader = load_data(
        "drive_data/train",
        transform_pipeline="default",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )
    val_loader = load_data(
        "drive_data/val",
        transform_pipeline="default",
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    loss_fn = nn.CrossEntropyLoss()
    depth_estimation_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_iou = 0.0

    for epoch in range(num_epoch):
        # --- Training ---
        model.train()
        train_metric = ConfusionMatrix(num_classes=3)

        for batch in train_loader:
            img = batch["image"].to(device)
            track = batch["track"].to(device)

            optimizer.zero_grad()
            logits, depth = model(img)
            loss = loss_fn(logits, track)
            depth_loss = depth_estimation_loss_fn()
            loss.backward()
            optimizer.step()

            train_metric.add(logits.argmax(dim=1), track)

        # --- Validation ---
        model.eval()
        val_metric = ConfusionMatrix(num_classes=3)

        with torch.inference_mode():
            for batch in val_loader:
                img = batch["image"].to(device)
                track = batch["track"].to(device)

                logits, _ = model(img)
                val_metric.add(logits.argmax(dim=1), track)

        train_stats = train_metric.compute()
        val_stats = val_metric.compute()

        writer.add_scalar("train/iou", train_stats["iou"], epoch)
        writer.add_scalar("train/accuracy", train_stats["accuracy"], epoch)
        writer.add_scalar("val/iou", val_stats["iou"], epoch)
        writer.add_scalar("val/accuracy", val_stats["accuracy"], epoch)

        val_iou = val_stats["iou"]
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_model(model)
            print(f"Epoch {epoch + 1:3d}/{num_epoch}  train_iou={train_stats['iou']:.3f}  val_iou={val_iou:.3f}  *saved*")
        else:
            print(f"Epoch {epoch + 1:3d}/{num_epoch}  train_iou={train_stats['iou']:.3f}  val_iou={val_iou:.3f}")

    writer.close()
    print(f"Best val IoU: {best_val_iou:.3f} — model saved to homework/detector.th")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))
