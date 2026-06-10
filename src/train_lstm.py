import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE     = get_device()
INPUT_DIR  = Path("saida_full")
OUTPUT_DIR = Path("output/lstm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED       = 42
BATCH_SIZE = 64

LSTM_CFG = dict(
    hidden      = 256,
    layers      = 2,
    drop_rnn    = 0.3,
    drop_reg    = 0.2,
    epochs      = 250,
    patience    = 30,
    lr          = 5e-4,
    wd          = 1e-5,
    scheduler   = "plateau",
)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data():
    X_train = np.load(INPUT_DIR / "X_train.npz")["X"].astype(np.float32)
    X_val   = np.load(INPUT_DIR / "X_val.npz")["X"].astype(np.float32)
    y_train = np.load(INPUT_DIR / "y_train.npy").astype(np.float32)
    y_val   = np.load(INPUT_DIR / "y_val.npy").astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return train_loader, val_loader, y_val


class PulseFiLSTM(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        H = cfg["hidden"]
        self.lstm = nn.LSTM(
            input_dim, H, cfg["layers"],
            batch_first=True,
            dropout=cfg["drop_rnn"] if cfg["layers"] > 1 else 0,
            bidirectional=True,
        )
        h_out = H * 2
        self.regressor = nn.Sequential(
            nn.Linear(h_out, 128),
            nn.ReLU(),
            nn.Dropout(cfg["drop_reg"]),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx    = out[:, -1, :]
        return self.regressor(ctx).squeeze(-1)


def train_model(train_loader, val_loader, y_val_true, input_dim):
    cfg = LSTM_CFG
    print(f"\n{'='*60}\nTREINANDO: LSTM\n{'='*60}")

    set_seed(SEED)
    model = PulseFiLSTM(input_dim, cfg).to(DEVICE)

    total = sum(p.numel() for p in model.parameters())
    print(f"  Device: {DEVICE} | Params: {total:,}")

    criterion = nn.HuberLoss(delta=3.0)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)

    history          = {"train_loss": [], "val_mae": []}
    best_val_mae     = float("inf")
    patience_counter = 0
    checkpoint_path  = OUTPUT_DIR / "best_model_lstm.pt"
    t_start          = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss_sum = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item()

        model.eval()
        val_mae_sum = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_mae_sum += nn.L1Loss()(model(bx), by).item()

        train_loss = train_loss_sum / len(train_loader)
        val_mae    = val_mae_sum   / len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae     = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:03d} | Huber: {train_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= cfg["patience"]:
            print(f"  Early stopping ép. {epoch+1} | Melhor Val MAE: {best_val_mae:.4f}")
            break

    train_time = time.time() - t_start

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()

    preds_all = []
    with torch.no_grad():
        for bx, _ in val_loader:
            preds_all.append(model(bx.to(DEVICE)).cpu().numpy())
    y_val_pred = np.concatenate(preds_all)

    mae_v  = float(mean_absolute_error(y_val_true, y_val_pred))
    rmse_v = float(np.sqrt(np.mean((y_val_true - y_val_pred) ** 2)))
    r2_v   = float(r2_score(y_val_true, y_val_pred))

    print(f"\n  MAE: {mae_v:.4f} | RMSE: {rmse_v:.4f} | R²: {r2_v:.4f} | "
          f"Treino: {train_time:.0f}s")

    history_path = OUTPUT_DIR / "history_lstm.json"
    with open(history_path, "w") as f:
        json.dump({**history, "input_dim": input_dim}, f)
    print(f"Histórico salvo: {history_path}")

    return history, y_val_true, y_val_pred


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"LSTM config: hidden={LSTM_CFG['hidden']} layers={LSTM_CFG['layers']} "
          f"lr={LSTM_CFG['lr']} epochs={LSTM_CFG['epochs']}")

    train_loader, val_loader, y_val_true = load_data()
    input_dim = next(iter(train_loader))[0].shape[2]
    print(f"Input dim: {input_dim} | Val samples: {len(y_val_true):,}\n")

    train_model(train_loader, val_loader, y_val_true, input_dim)


if __name__ == "__main__":
    main()
