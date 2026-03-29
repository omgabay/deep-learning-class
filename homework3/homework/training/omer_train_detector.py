import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from homework.datasets.road_dataset import load_data
from homework.models import Detector, save_model
from homework.metrics import DetectionMetric

def train(
    num_epoch: int,
    lr: float,
    batch_size: int,
    seed: int = 2026,
    logs_path = "logs",
    **kwargs,
):
    
    # Check which device is available for training.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    torch.manual_seed(seed)
    
     # Timestamped log directory
    log_dir = Path(logs_path) / f"detector_{datetime.now().strftime('%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    model = Detector().to(device)
    
    train_road_ds = load_data('drive_data/train',
                        transform_pipeline="default",
                        shuffle=True,
                        batch_size=batch_size,
                        num_workers=2,
                    )
    
    val_road_ds = load_data('drive_data/val',
                        transform_pipeline="default",
                        shuffle=False,
                        batch_size=batch_size,
                        num_workers=2,
                    )
    
    segmentation_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.MSELoss()
    depth_loss_scaler = 5

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # We save the model parameters each time we get a new best score in iou validation set.
    best_val_iou = 0.0
    
    
    for epoch in range(num_epoch):
        
        model.train()
        loss_segmentation, loss_depth = 0.0, 0.0 
        train_metric = DetectionMetric()
        
        for sample in train_road_ds:
            img, segmentation_label = sample["image"].to(device), sample["track"].to(device) 
            depth = sample["depth"].to(device)
            optimizer.zero_grad()
            
            logits, pred_depth = model(img)
            seg_loss = segmentation_loss_fn(logits, segmentation_label)
            depth_loss = depth_loss_fn(pred_depth, depth)
            
            total_loss = seg_loss + depth_loss_scaler * depth_loss
            total_loss.backward()
                        
            optimizer.step()
            loss_segmentation += seg_loss.item()
            loss_depth += depth_loss.item()
            
            segmentation_pred = logits.argmax(dim=1)
            train_metric.add(segmentation_pred, segmentation_label, pred_depth, depth)
            
        loss_segmentation /= len(train_road_ds)
        loss_depth /= len(train_road_ds)
        scheduler.step()
        scaled_ratio = loss_segmentation / (depth_loss_scaler * loss_depth) 
        print(f"Epoch {epoch + 1:3d}/{num_epoch} {loss_segmentation=:.3f} {loss_depth=:.3f} {scaled_ratio=:.3f}")
        
        
        # Starting evaluation with validation set. 
        model.eval()        
        with torch.inference_mode():
            val_metric = DetectionMetric()
            
            for sample in val_road_ds:
                img = sample['image'].to(device)
                segmentation_label = sample['track'].to(device)
                depth_label = sample['depth'].to(device)

                pred_segmentation, pred_depth = model.predict(img)
                val_metric.add(pred_segmentation, segmentation_label, pred_depth, depth_label)
        
        train_stats = train_metric.compute()
        train_acc = train_stats["accuracy"]
        train_iou = train_stats["iou"]
        
        # Write training stats
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("train/iou", train_iou, epoch)
        
        
        val_stats = val_metric.compute()
        val_acc = val_stats["accuracy"] 
        val_iou = val_stats["iou"]
        
        # Write validation stats        
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("val/iou", val_iou, epoch)
        
        # Save model only when val accuracy improves
        train_depth_err = train_stats["abs_depth_error"]
        val_depth_err = val_stats["abs_depth_error"]

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_model(model)
            print(f"Epoch {epoch + 1:3d}/{num_epoch}  train_iou={train_iou:.3f}  val_iou={val_iou:.3f}  train_depth_err={train_depth_err:.3f}  val_depth_err={val_depth_err:.3f}  *saved*")
        else:
            print(f"Epoch {epoch + 1:3d}/{num_epoch}  train_iou={train_iou:.3f}  val_iou={val_iou:.3f}  train_depth_err={train_depth_err:.3f}  val_depth_err={val_depth_err:.3f}")
             
    
    writer.close()
    print(f"Best val accuracy: {best_val_iou:.3f} — model saved to homework/detector.th")
                  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=2026)

    train(**vars(parser.parse_args()))