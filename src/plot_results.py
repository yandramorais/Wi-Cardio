import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import norm as sp_norm


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE     = get_device()
INPUT_DIR  = Path("saida_full")
BATCH_SIZE = 64


# ── Model definitions ──────────────────────────────────────────────────────────

class PulseFiModelGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=2):
        super().__init__()
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
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)
        gru_out = self.norm(gru_out)
        weights = torch.softmax(self.attn(gru_out), dim=1)
        context = (gru_out * weights).sum(dim=1)
        return self.regressor(context).squeeze(-1)


class PulseFiLSTM(nn.Module):
    def __init__(self, input_dim, hidden=256, layers=2, drop_rnn=0.3, drop_reg=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, layers,
            batch_first=True,
            dropout=drop_rnn if layers > 1 else 0,
            bidirectional=True,
        )
        h_out = hidden * 2
        self.norm      = nn.LayerNorm(h_out)
        self.attn      = nn.Linear(h_out, 1)
        self.regressor = nn.Sequential(
            nn.Linear(h_out, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(drop_reg),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.norm(out)
        w      = torch.softmax(self.attn(out), dim=1)
        ctx    = (out * w).sum(dim=1)
        return self.regressor(ctx).squeeze(-1)


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_split(split):
    X  = np.load(INPUT_DIR / f"X_{split}.npz")["X"].astype(np.float32)
    y  = np.load(INPUT_DIR / f"y_{split}.npy").astype(np.float32)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False), y


def load_metadata(split):
    subj = np.load(INPUT_DIR / f"subject_{split}.npy")
    pos  = np.load(INPUT_DIR / f"positions_{split}.npy")
    return subj, pos


def run_inference(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_metrics(history, y_true, y_pred, model_name, color, hdr_clr, output_dir):
    ACCENT = "#A23B72"
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    erro     = y_true - y_pred
    abs_erro = np.abs(erro)
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    rmse  = float(np.sqrt(np.mean(erro ** 2)))
    bias  = float(np.mean(erro))
    sigma = float(np.std(erro))
    loa_u = bias + 1.96 * sigma
    loa_l = bias - 1.96 * sigma

    val_key = "val_mae" if "val_mae" in history else "val_loss"

    rc = {"font.family": "DejaVu Sans",
          "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 10}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(19, 11))
        fig.patch.set_facecolor("#F7F9FC")
        fig.suptitle(f"Avaliação do Modelo  —  {model_name}",
                     fontsize=17, fontweight="bold", y=1.01, color="#1A1A2E")
        BG = "#FFFFFF"

        ax = axes[0, 0]; ax.set_facecolor(BG)
        eps = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(eps, history["train_loss"], color="#1976D2", lw=2, label="Treino")
        ax.plot(eps, history[val_key],      color="#E53935", lw=2, ls="--", label="Validação (MAE)")
        best_ep  = int(np.argmin(history[val_key])) + 1
        best_val = history[val_key][best_ep - 1]
        ax.axvline(best_ep, color="#BDBDBD", ls=":", lw=1.4)
        offset = max(3, len(eps) // 12)
        ax.annotate(
            f"Melhor val\nMAE={best_val:.2f}\n(ép. {best_ep})",
            xy=(best_ep, best_val),
            xytext=(best_ep + offset, best_val + (max(history[val_key]) - best_val) * 0.25 + 0.2),
            fontsize=8.5, color="#424242",
            arrowprops=dict(arrowstyle="->", color="#9E9E9E", lw=1),
        )
        ax.set_title("Curva de Aprendizado", fontsize=13, fontweight="bold")
        ax.set_xlabel("Época", fontsize=11); ax.set_ylabel("MAE (BPM)", fontsize=11)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.35, ls="--", color="#CFD8DC")

        ax = axes[0, 1]; ax.set_facecolor(BG)
        lo = min(y_true.min(), y_pred.min()) - 3
        hi = max(y_true.max(), y_pred.max()) + 3
        ax.scatter(y_true, y_pred, alpha=0.30, s=14, color=color, edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.6, label="Ideal (y=x)")
        m_fit, b_fit = np.polyfit(y_true, y_pred, 1)
        xs = np.linspace(lo, hi, 300)
        ax.plot(xs, m_fit * xs + b_fit, color=ACCENT, lw=1.8, alpha=0.9,
                label=f"Regressão (m={m_fit:.2f})")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.text(0.04, 0.97,
                f"MAE  = {mae:.2f} BPM\nRMSE = {rmse:.2f} BPM\nR²    = {r2:.3f}",
                transform=ax.transAxes, fontsize=9.5, va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#CFD8DC", alpha=0.95))
        ax.set_title("Real vs Predito", fontsize=13, fontweight="bold")
        ax.set_xlabel("Ground Truth — Smartwatch (BPM)", fontsize=11)
        ax.set_ylabel(f"Predição — {model_name} (BPM)", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.30, ls="--", color="#CFD8DC")

        ax = axes[0, 2]; ax.set_facecolor(BG)
        ax.hist(erro, bins=40, color=color, edgecolor="white", alpha=0.82, density=True, label="Resíduos")
        xr = np.linspace(erro.min() - 2, erro.max() + 2, 400)
        ax.plot(xr, sp_norm.pdf(xr, bias, sigma), color="#E53935", lw=2,
                label=f"Normal(μ={bias:.2f}, σ={sigma:.2f})")
        ax.axvline(0,    color="#212121", ls="--", lw=1.5, label="Zero")
        ax.axvline(bias, color="#E53935", ls=":",  lw=1.5, alpha=0.85)
        ax.set_title("Distribuição dos Resíduos", fontsize=13, fontweight="bold")
        ax.set_xlabel("Erro  =  GT − Predição (BPM)", fontsize=11)
        ax.set_ylabel("Densidade", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.35, ls="--", color="#CFD8DC")

        ax = axes[1, 0]; ax.set_facecolor(BG)
        mean_ba = (y_true + y_pred) / 2.0
        xlo, xhi = float(mean_ba.min()) - 2, float(mean_ba.max()) + 2
        ax.scatter(mean_ba, erro, alpha=0.30, s=14, color=color, edgecolors="none", rasterized=True)
        ax.axhline(bias,  color="#E53935", lw=2.0, label=f"Viés = {bias:.2f}")
        ax.axhline(loa_u, color="#757575", lw=1.5, ls="--", label=f"+1.96σ = {loa_u:.2f}")
        ax.axhline(loa_l, color="#757575", lw=1.5, ls="--", label=f"−1.96σ = {loa_l:.2f}")
        ax.fill_between([xlo, xhi], loa_l, loa_u, alpha=0.08, color="#9E9E9E")
        ax.set_xlim(xlo, xhi)
        ax.set_title("Bland-Altman — Concordância", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Média (Smartwatch + {model_name}) / 2 (BPM)", fontsize=11)
        ax.set_ylabel("Diferença (GT − Predição) (BPM)", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.30, ls="--", color="#CFD8DC")

        ax = axes[1, 1]; ax.set_facecolor(BG)
        sorted_ae = np.sort(abs_erro)
        cdf_vals  = np.arange(1, len(sorted_ae) + 1) / len(sorted_ae) * 100
        ax.plot(sorted_ae, cdf_vals, color=color, lw=2.5)
        ax.fill_between(sorted_ae, 0, cdf_vals, alpha=0.13, color=color)
        for thr, ls_style, ytxt in [(5, "--", 10), (10, ":", 22), (15, "-.", 34)]:
            pct = float(np.mean(abs_erro <= thr) * 100)
            ax.axvline(thr, color="#9E9E9E", ls=ls_style, lw=1.3)
            ax.text(thr + 0.25, ytxt, f"{pct:.0f}%\n≤{thr} BPM", fontsize=8.5, color="#555", va="bottom")
        ax.set_title("CDF do Erro Absoluto", fontsize=13, fontweight="bold")
        ax.set_xlabel("|Erro| (BPM)", fontsize=11); ax.set_ylabel("Amostras acumuladas (%)", fontsize=11)
        ax.set_ylim(0, 103); ax.grid(True, alpha=0.35, ls="--", color="#CFD8DC")

        ax = axes[1, 2]; ax.set_facecolor("#F7F9FC"); ax.axis("off")
        rows = [
            ["MAE",             f"{mae:.3f} BPM"],
            ["RMSE",            f"{rmse:.3f} BPM"],
            ["R²",              f"{r2:.4f}"],
            ["Viés (μ erro)",   f"{bias:.3f} BPM"],
            ["Desvio (σ erro)", f"{sigma:.3f} BPM"],
            ["LoA superior",    f"+{loa_u:.3f} BPM"],
            ["LoA inferior",    f"{loa_l:.3f} BPM"],
            ["% ≤ 5 BPM",       f"{np.mean(abs_erro <= 5)*100:.1f}%"],
            ["% ≤ 10 BPM",      f"{np.mean(abs_erro <= 10)*100:.1f}%"],
            ["N amostras",      f"{len(y_true):,}"],
        ]
        tbl = ax.table(cellText=rows, colLabels=["Métrica", "Valor"],
                       loc="center", cellLoc="left")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.15, 1.72)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#E0E0E0")
            if r == 0:
                cell.set_facecolor(hdr_clr)
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#EEF4FB")
            else:
                cell.set_facecolor("white")
        ax.set_title(f"Resumo — {model_name}", fontsize=13, fontweight="bold", pad=14)

        plt.tight_layout()
        out = output_dir / f"resultado_{model_name.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Figura salva: {out}")
        plt.show()


def plot_extra(y_true_val, y_pred_val, subjects, positions, y_true_test, y_pred_test,
               model_name, color, output_dir):
    ACCENT  = "#A23B72"
    abs_err = np.abs(y_true_val - y_pred_val)

    subj_ids = np.unique(subjects)
    subj_mae = [np.mean(np.abs(y_true_val[subjects == s] - y_pred_val[subjects == s])) for s in subj_ids]

    pos_ids = np.unique(positions)
    pos_mae = [np.mean(np.abs(y_true_val[positions == p] - y_pred_val[positions == p])) for p in pos_ids]

    bins  = [40, 60, 80, 100, 120, 200]
    blbls = ["40–60", "60–80", "80–100", "100–120", "120+"]
    range_mae, range_n = [], []
    for i in range(len(bins) - 1):
        mask = (y_true_val >= bins[i]) & (y_true_val < bins[i + 1])
        if mask.sum() > 0:
            range_mae.append(float(np.mean(abs_err[mask])))
            range_n.append(int(mask.sum()))

    mae_t  = float(mean_absolute_error(y_true_test, y_pred_test))
    rmse_t = float(np.sqrt(np.mean((y_true_test - y_pred_test) ** 2)))
    r2_t   = float(r2_score(y_true_test, y_pred_test))

    rc = {"font.family": "DejaVu Sans",
          "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 10}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.patch.set_facecolor("#F7F9FC")
        fig.suptitle(f"Análise Estendida — {model_name}",
                     fontsize=17, fontweight="bold", y=1.01, color="#1A1A2E")
        BG = "#FFFFFF"

        ax = axes[0, 0]; ax.set_facecolor(BG)
        xp = np.arange(len(subj_ids))
        ax.bar(xp, subj_mae, color=color, alpha=0.82, edgecolor="white")
        ax.axhline(float(np.mean(subj_mae)), color="#E53935", ls="--", lw=1.5,
                   label=f"Média = {np.mean(subj_mae):.2f}")
        ax.set_xticks(xp)
        ax.set_xticklabels([str(s) for s in subj_ids], fontsize=8, rotation=45, ha="right")
        ax.set_title("MAE por Sujeito", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sujeito", fontsize=11); ax.set_ylabel("MAE (BPM)", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.35, ls="--", color="#CFD8DC", axis="y")

        ax = axes[0, 1]; ax.set_facecolor(BG)
        xp2 = np.arange(len(pos_ids))
        ax.bar(xp2, pos_mae, color=ACCENT, alpha=0.82, edgecolor="white")
        ax.set_xticks(xp2)
        ax.set_xticklabels([str(p) for p in pos_ids], fontsize=10)
        ax.set_title("MAE por Posição", fontsize=13, fontweight="bold")
        ax.set_xlabel("Posição", fontsize=11); ax.set_ylabel("MAE (BPM)", fontsize=11)
        ax.grid(True, alpha=0.35, ls="--", color="#CFD8DC", axis="y")

        ax = axes[1, 0]; ax.set_facecolor(BG)
        xp3 = np.arange(len(range_mae))
        bars3 = ax.bar(xp3, range_mae, color="#43A047", alpha=0.82, edgecolor="white")
        for rect, n in zip(bars3, range_n):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.1,
                    f"n={n}", ha="center", fontsize=8.5, color="#555")
        ax.set_xticks(xp3)
        ax.set_xticklabels(blbls[:len(range_mae)], fontsize=10)
        ax.set_title("MAE por Faixa de BPM", fontsize=13, fontweight="bold")
        ax.set_xlabel("Faixa (BPM)", fontsize=11); ax.set_ylabel("MAE (BPM)", fontsize=11)
        ax.grid(True, alpha=0.35, ls="--", color="#CFD8DC", axis="y")

        ax = axes[1, 1]; ax.set_facecolor(BG)
        lo = min(y_true_test.min(), y_pred_test.min()) - 3
        hi = max(y_true_test.max(), y_pred_test.max()) + 3
        ax.scatter(y_true_test, y_pred_test, alpha=0.30, s=14, color="#FF7043",
                   edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.6, label="Ideal (y=x)")
        m_fit, b_fit = np.polyfit(y_true_test, y_pred_test, 1)
        xs = np.linspace(lo, hi, 300)
        ax.plot(xs, m_fit * xs + b_fit, color=ACCENT, lw=1.8, alpha=0.9,
                label=f"Regressão (m={m_fit:.2f})")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.text(0.04, 0.97,
                f"MAE  = {mae_t:.2f} BPM\nRMSE = {rmse_t:.2f} BPM\nR²    = {r2_t:.3f}",
                transform=ax.transAxes, fontsize=9.5, va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#CFD8DC", alpha=0.95))
        ax.set_title("Conjunto de Teste — Real vs Predito", fontsize=13, fontweight="bold")
        ax.set_xlabel("Ground Truth — Smartwatch (BPM)", fontsize=11)
        ax.set_ylabel(f"Predição — {model_name} (BPM)", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.30, ls="--", color="#CFD8DC")

        plt.tight_layout()
        out = output_dir / f"resultado_{model_name.lower()}_extra.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Figura salva: {out}")
        plt.show()


# ── Per-model runners ──────────────────────────────────────────────────────────

def run_gru():
    output_dir   = Path("output/gru")
    history_path = output_dir / "history_gru.json"
    ckpt_path    = output_dir / "best_model_gru.pt"

    if not history_path.exists() or not ckpt_path.exists():
        print("GRU: arquivos não encontrados. Execute train_gru.py primeiro.")
        return

    with open(history_path) as f:
        history = json.load(f)
    input_dim = history.pop("input_dim")

    val_loader,  y_val_true  = load_split("val")
    test_loader, y_test_true = load_split("test")
    subj_val, pos_val        = load_metadata("val")

    model = PulseFiModelGRU(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

    y_val_pred  = run_inference(model, val_loader)
    y_test_pred = run_inference(model, test_loader)

    plot_metrics(history, y_val_true, y_val_pred,
                 model_name="GRU", color="#2E86AB", hdr_clr="#1565C0", output_dir=output_dir)
    plot_extra(y_val_true, y_val_pred, subj_val, pos_val,
               y_test_true, y_test_pred,
               model_name="GRU", color="#2E86AB", output_dir=output_dir)


def run_lstm():
    output_dir   = Path("output/lstm")
    history_path = output_dir / "history_lstm.json"
    ckpt_path    = output_dir / "best_model_lstm.pt"

    if not history_path.exists() or not ckpt_path.exists():
        print("LSTM: arquivos não encontrados. Execute train_lstm.py primeiro.")
        return

    with open(history_path) as f:
        history = json.load(f)
    input_dim = history.pop("input_dim")

    val_loader,  y_val_true  = load_split("val")
    test_loader, y_test_true = load_split("test")
    subj_val, pos_val        = load_metadata("val")

    model = PulseFiLSTM(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

    y_val_pred  = run_inference(model, val_loader)
    y_test_pred = run_inference(model, test_loader)

    plot_metrics(history, y_val_true, y_val_pred,
                 model_name="LSTM", color="#E07A5F", hdr_clr="#BF360C", output_dir=output_dir)
    plot_extra(y_val_true, y_val_pred, subj_val, pos_val,
               y_test_true, y_test_pred,
               model_name="LSTM", color="#E07A5F", output_dir=output_dir)


if __name__ == "__main__":
    # Uso: python src/plot_results.py          → plota GRU e LSTM
    #      python src/plot_results.py gru      → só GRU
    #      python src/plot_results.py lstm     → só LSTM
    models = sys.argv[1:] or ["gru", "lstm"]
    for m in models:
        if m.lower() == "gru":
            run_gru()
        elif m.lower() == "lstm":
            run_lstm()
        else:
            print(f"Modelo desconhecido: '{m}'. Use 'gru' ou 'lstm'.")
