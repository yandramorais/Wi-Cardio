import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 250
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 256
NUM_LAYERS = 2
PATIENCE = 30
SEED = 42
INPUT_DIR  = Path("saida_full")
OUTPUT_DIR = Path("output/gru")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    X_train = np.load(INPUT_DIR / "X_train.npz")["X"].astype(np.float32)
    X_val   = np.load(INPUT_DIR / "X_val.npz")["X"].astype(np.float32)
    y_train = np.load(INPUT_DIR / "y_train.npy").astype(np.float32)
    y_val   = np.load(INPUT_DIR / "y_val.npy").astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds,   batch_size=BATCH_SIZE),
        y_val,
    )


class PulseFiModelGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(PulseFiModelGRU, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True,
                          dropout=0.3, bidirectional=True)
        h_out = hidden_dim * 2
        self.norm = nn.LayerNorm(h_out)
        self.attn = nn.Linear(h_out, 1)
        self.regressor = nn.Sequential(
            nn.Linear(h_out, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)
        gru_out = self.norm(gru_out)
        weights = torch.softmax(self.attn(gru_out), dim=1)
        context = (gru_out * weights).sum(dim=1)
        return self.regressor(context).squeeze(-1)


def train():
    set_seed()
    train_loader, val_loader, y_val_true = load_data()
    example_x, _ = next(iter(train_loader))
    input_dim = example_x.shape[2]

    model = PulseFiModelGRU(input_dim, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

    criterion = nn.HuberLoss(delta=3.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    history = {"train_loss": [], "val_mae": []}
    best_val_mae = float("inf")
    patience_counter = 0
    checkpoint_path = OUTPUT_DIR / "best_model_gru.pt"

    print(f"Iniciando Treino com GRU no dispositivo: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                v_loss += nn.L1Loss()(model(bx), by).item()

        train_mae = t_loss / len(train_loader)
        val_mae   = v_loss / len(val_loader)
        history["train_loss"].append(train_mae)
        history["val_mae"].append(val_mae)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping na época {epoch + 1} (melhor val MAE: {best_val_mae:.2f})")
                break

        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1} | Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | LR: {lr_now:.2e}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()
    preds_all = []
    with torch.no_grad():
        for bx, _ in val_loader:
            preds_all.append(model(bx.to(DEVICE)).cpu().numpy())
    y_val_pred = np.concatenate(preds_all)

    print("\n--- Métricas Finais (GRU) ---")
    print(f"MAE Final: {mean_absolute_error(y_val_true, y_val_pred):.2f} BPM")
    print(f"R² Score:  {r2_score(y_val_true, y_val_pred):.4f}")

    history_path = OUTPUT_DIR / "history_gru.json"
    with open(history_path, "w") as f:
        json.dump({**history, "input_dim": input_dim}, f)
    print(f"Histórico salvo: {history_path}")


if __name__ == "__main__":
    train()
