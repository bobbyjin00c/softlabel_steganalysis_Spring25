import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from tqdm import tqdm
import logging
import traceback
import matplotlib.pyplot as plt
from model.soft_label_utils import embed_rate_to_softlabel
from model.multitask_loss import MultiTaskLoss
from model.net import StegoNet
from dataset.dataset import StegoDataset
from config import Config

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from collections import defaultdict
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMPERATURE = 1.5

def train_one_epoch(model, loader, optimizer, device, scaler, loss_combiner, loss_fn_kl, loss_fn_reg):
    model.train()
    total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
    all_cls_preds, all_cls_labels = [], []
    all_reg_preds, all_reg_labels = [], []

    loop = tqdm(loader, desc="Training", leave=False)
    for imgs, labels in loop:
        try:
            imgs = imgs.to(device, non_blocking=True)
            reg_labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                cls_logits, reg_pred = model(imgs)
                with torch.no_grad():
                    soft_cls_labels = embed_rate_to_softlabel(reg_pred.detach().squeeze()).to(device)

                log_probs = F.log_softmax(cls_logits / TEMPERATURE, dim=1)
                loss_cls = loss_fn_kl(log_probs, soft_cls_labels)
                loss_reg = loss_fn_reg(reg_pred.squeeze(), reg_labels.squeeze())
                loss = loss_combiner(loss_cls, loss_reg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            total_cls_loss += loss_cls.item() * imgs.size(0)
            total_reg_loss += loss_reg.item() * imgs.size(0)

            # cls_probs = F.softmax(cls_logits, dim=1)
            # cls_preds = (cls_probs[:, 1] > 0.5).long().detach().cpu()
            # cls_labels = (reg_labels > 0).long().cpu()
            cls_preds = (torch.argmax(cls_logits, dim=1) != 0).long()  # ≠ 0 判定为“嵌入”
            cls_labels = (reg_labels > 0).long()  # ground truth：α > 0 为嵌入

            all_cls_preds.append(cls_preds)
            all_cls_labels.append(cls_labels)
            all_reg_preds.append(reg_pred.detach().cpu())
            all_reg_labels.append(reg_labels.cpu())

            loop.set_postfix(Loss=f"{loss.item():.4f}", ClsLoss=f"{loss_cls.item():.4f}", RegLoss=f"{loss_reg.item():.4f}")

        except Exception as e:
            logger.error(f"Error in train_one_epoch: {e}\n{traceback.format_exc()}")
            raise

    all_cls_preds = torch.cat(all_cls_preds)
    all_cls_labels = torch.cat(all_cls_labels)
    all_reg_preds = torch.cat(all_reg_preds).squeeze()
    all_reg_labels = torch.cat(all_reg_labels).squeeze()

    # acc = accuracy_score(all_cls_labels, all_cls_preds)
    acc = accuracy_score(all_cls_labels.cpu().numpy(), all_cls_preds.cpu().numpy())


    reg_preds_np = all_reg_preds.numpy()
    reg_labels_np = all_reg_labels.numpy()
    mae = mean_absolute_error(reg_labels_np, reg_preds_np)
    mse = mean_squared_error(reg_labels_np, reg_preds_np)
    rmse = np.sqrt(mse)
    avg_alpha = reg_preds_np.mean()
    std_alpha = reg_preds_np.std()

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc, mae, mse, rmse, avg_alpha, std_alpha


def evaluate(model, loader, device, loss_combiner, loss_fn_kl, loss_fn_reg, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    all_cls_logits = []
    all_reg_preds, all_reg_labels = [], []

    with torch.no_grad():
        loop = tqdm(loader, desc=desc, leave=False)
        for imgs, labels in loop:
            imgs = imgs.to(device, non_blocking=True)
            reg_labels = labels.to(device)

            cls_logits, reg_pred = model(imgs)
            soft_cls_labels = embed_rate_to_softlabel(reg_pred.detach().squeeze()).to(device)
            log_probs = F.log_softmax(cls_logits / TEMPERATURE, dim=1)
            loss_cls = loss_fn_kl(log_probs, soft_cls_labels)
            loss_reg = loss_fn_reg(reg_pred.squeeze(), reg_labels.squeeze())
            loss = loss_combiner(loss_cls, loss_reg)

            total_loss += loss.item() * imgs.size(0)
            all_cls_logits.append(cls_logits.cpu())
            all_reg_preds.append(reg_pred.cpu())
            all_reg_labels.append(reg_labels.cpu())

    all_cls_logits = torch.cat(all_cls_logits)
    all_reg_preds = torch.cat(all_reg_preds).squeeze()
    all_reg_labels = torch.cat(all_reg_labels).squeeze()

    # binary_preds = (F.softmax(all_cls_logits, dim=1)[:, 1] > 0.5).long()
    # binary_labels = (all_reg_labels > 0).long()
    binary_preds = (torch.argmax(all_cls_logits, dim=1) != 0).long()
    binary_labels = (all_reg_labels > 0).long()

    # acc = accuracy_score(binary_labels, binary_preds)
    acc = accuracy_score(binary_labels.cpu().numpy(), binary_preds.cpu().numpy())


    reg_preds_np = all_reg_preds.numpy()
    reg_labels_np = all_reg_labels.numpy()

    bin_stats = defaultdict(list)
    for pred, label in zip(reg_preds_np, reg_labels_np):
        key = round(label, 1)
        bin_stats[key].append(pred)

    logger.info("📊 α分档预测表现（按真实标签分组）:")
    for k in sorted(bin_stats.keys()):
        group_preds = bin_stats[k]
        mean_pred = np.mean(group_preds)
        std_pred = np.std(group_preds)
        logger.info(f"  α={k:.1f} → 预测均值={mean_pred:.3f}, σ=±{std_pred:.3f}")

    mae = mean_absolute_error(reg_labels_np, reg_preds_np)
    mse = mean_squared_error(reg_labels_np, reg_preds_np)
    rmse = np.sqrt(mse)
    avg_alpha = reg_preds_np.mean()
    std_alpha = reg_preds_np.std()

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc, mae, mse, rmse, avg_alpha, std_alpha


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), Config.checkpoint_path)
            logger.info(f"Saved checkpoint: {Config.checkpoint_path}")
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def plot_and_save_curves(metrics, out_dir="visualization"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(metrics["train_loss"]) + 1))

    def save_plot(y_train, y_val, ylabel, filename):
        plt.figure()
        plt.plot(epochs, y_train, label='Train')
        plt.plot(epochs, y_val, label='Val')
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

    save_plot(metrics["train_loss"], metrics["val_loss"], "Loss", "loss_curve.png")
    save_plot(metrics["train_acc"],  metrics["val_acc"],  "Accuracy", "acc_curve.png")
    save_plot(metrics["train_mae"],  metrics["val_mae"],  "MAE", "mae_curve.png")
    save_plot(metrics["train_mse"], metrics["val_mse"], "MSE", "mse_curve.png")


def main():
    device = Config.device
    scaler = GradScaler()

    train_ds = StegoDataset(Config.train_data_dir)
    val_ds = StegoDataset(Config.val_data_dir)
    test_ds = StegoDataset(Config.test_data_dir)

    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = StegoNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)

    loss_fn_kl = nn.KLDivLoss(reduction='batchmean')
    loss_fn_reg = nn.MSELoss()
    loss_combiner = MultiTaskLoss().to(device)
    early_stopping = EarlyStopping(patience=3)

    logger.info("Sanity checks passed ✅")
    logger.info("Starting training")
    logger.info(f"Device: {device}, Batch size: {Config.batch_size}, LR: {Config.lr}")
    logger.info(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    metrics = {
        "train_loss": [], "train_acc": [], "train_mae": [],
        "train_mse": [], "train_rmse": [], "train_alpha": [], "train_sigma": [],
        "val_loss": [], "val_acc": [], "val_mae": [],
        "val_mse": [], "val_rmse": [], "val_alpha": [], "val_sigma": []
    }



    for epoch in range(1, Config.num_epochs + 1):
        tr_loss, tr_acc, tr_mae, tr_mse, tr_rmse, tr_alpha, tr_sigma = train_one_epoch(
            model, train_loader, optimizer, device, scaler, loss_combiner, loss_fn_kl, loss_fn_reg
        )

        val_loss, val_acc, val_mae, val_mse, val_rmse, val_alpha, val_sigma = evaluate(
            model, val_loader, device, loss_combiner, loss_fn_kl, loss_fn_reg, desc="Validation"
        )

        logger.info(f"Epoch {epoch:02d} → Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}, MAE: {tr_mae:.3f}")
        logger.info(f"              → Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                    f"MAE: {val_mae:.3f}, MSE: {val_mse:.6f}, α: {val_alpha:.3f}±{val_sigma:.3f}")

        metrics["train_loss"].append(tr_loss)
        metrics["train_acc"].append(tr_acc)
        metrics["train_mae"].append(tr_mae)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["val_mae"].append(val_mae)
        metrics["train_mse"].append(tr_mse)
        metrics["train_rmse"].append(tr_rmse)
        metrics["train_alpha"].append(tr_alpha)
        metrics["train_sigma"].append(tr_sigma)
        metrics["val_mse"].append(val_mse)
        metrics["val_rmse"].append(val_rmse)
        metrics["val_alpha"].append(val_alpha)
        metrics["val_sigma"].append(val_sigma)



        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

        scheduler.step()

    model.load_state_dict(torch.load(Config.checkpoint_path))
    test_loss, test_acc, test_mae, test_mse, test_rmse, test_alpha, test_sigma = evaluate(
        model, test_loader, device, loss_combiner, loss_fn_kl, loss_fn_reg, desc="Final Test"
    )
    logger.info(f"Final Test → Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
                f"MAE: {test_mae:.3f}, MSE: {test_mse:.6f}, α: {test_alpha:.3f}±{test_sigma:.3f}")

    plot_and_save_curves(metrics)

if __name__ == '__main__':
    main()
