# make_figures.py
# ------------------------------------------------------------
# Generates:
#   figures/avg_accuracy_by_method.(png|pdf)
#   figures/fault_3_accuracy.(png|pdf)
#   figures/fault_9_accuracy.(png|pdf)
#   figures/fault_15_accuracy.(png|pdf)
#   figures/avg_f1_by_method.(png|pdf)          [if F1 provided]
#   figures/avg_fdr_by_method.(png|pdf)         [if FDR provided]
#   figures/time_series_detection_example.(png|pdf)
# ------------------------------------------------------------

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ====== DATA (from your manuscript’s accuracy table) =========================
faults = list(range(0, 21))

SVM   = [19.27, 88.43, 86.04, 15.72, 58.02, 64.79, 71.875, 88.13, 45.10, 12.92, 27.08, 14.89, 52.34, 35.10, 62.19, 22.19, 16.78, 53.65, 30.94, 51.25, 44.90]
PCA   = [19.69, 88.50, 89.06, 21.25, 81.35, 87.81, 89.58, 88.75, 83.96, 22.60, 76.97, 70.20, 87.08, 69.58, 88.22, 26.98, 73.65, 75.94, 73.54, 85.72, 79.69]
RNN   = [7.00,  100.0, 100.0, 76.00, 100.0, 100.0, 0.00,  100.0, 73.00, 52.00, 59.00, 98.00, 84.00, 73.00, 99.00, 65.00, 35.20, 98.00, 100.0, 100.0, 90.00]
SLMLP = [97.60, 96.37, 97.62, 20.62, 82.75, 96.00, 100.0, 100.0, 96.87, 12.12, 88.25, 73.50, 93.62, 72.25, 95.87, 21.12, 78.12, 80.25, 86.37, 96.12, 86.75]
SGHT  = [88.14, 97.54, 99.80, 99.05, 98.70, 98.40, 99.01, 98.35, 99.82, 80.87, 99.75, 98.67, 97.24, 95.80, 99.61, 64.50, 99.21, 99.84, 98.42, 98.75, 99.04]

methods = ["SVM", "PCA", "RNN", "SLMLP", "SGHT"]
df = pd.DataFrame({
    "fault": faults,
    "SVM": SVM, "PCA": PCA, "RNN": RNN, "SLMLP": SLMLP, "SGHT": SGHT
}).astype({"fault": "int64"})

# ====== 1) Average Accuracy by Method =======================================
avg_series = df[methods].mean(skipna=True).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
avg_series.plot(kind="bar")
plt.ylabel("Average Accuracy (%)")
plt.title("Average Fault Classification Accuracy by Method")
plt.tight_layout()
plt.savefig(OUT / "avg_accuracy_by_method.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT / "avg_accuracy_by_method.pdf", bbox_inches="tight")
plt.close()

# ====== 2) Separate plots for faults 3, 9, 15 ===============================
for fid in [3, 9, 15]:
    row = df.loc[df["fault"] == fid, methods].iloc[0].astype(float)
    ymax = float(np.nanmax(row.values))

    plt.figure(figsize=(6, 4))
    row.plot(kind="bar")
    plt.title(f"Fault {fid}")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, max(100.0, math.ceil(ymax / 5.0) * 5 + 5))
    plt.tight_layout()

    plt.savefig(OUT / f"fault_{fid}_accuracy.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / f"fault_{fid}_accuracy.pdf", bbox_inches="tight")
    plt.close()

# ====== 3) OPTIONAL: F1 and FDR (fill arrays below, or leave as None) =======
# Example placeholders; replace with your real values if you have them.
F1 = None   # dict like {"SVM":[...21...], "PCA":[...], "RNN":[...], "SLMLP":[...], "SGHT":[...]}
FDR = None  # same structure

def plot_metric_average(metric_dict, metric_name, out_stub):
    if metric_dict is None:
        return
    dfm = pd.DataFrame({"fault": faults})
    for m in methods:
        if m in metric_dict and metric_dict[m] is not None:
            dfm[m] = metric_dict[m]
    avg = dfm[methods].mean(skipna=True).dropna().sort_values(ascending=False)
    if avg.empty:
        return
    plt.figure(figsize=(8, 5))
    avg.plot(kind="bar")
    plt.ylabel(metric_name)
    plt.title(f"Average {metric_name} by Method")
    plt.tight_layout()
    plt.savefig(OUT / f"{out_stub}.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / f"{out_stub}.pdf", bbox_inches="tight")
    plt.close()

plot_metric_average(F1,  "F1-score",             "avg_f1_by_method")
plot_metric_average(FDR, "False Detection Rate", "avg_fdr_by_method")

# ====== 4) Time-series early-detection example ==============================
T = 500
t = np.arange(T)
x = np.sin(0.02 * t) + 0.1 * np.random.randn(T)  # replace with your real signal

# Replace with your *real* model anomaly scores/probabilities (length T)
scores = {
    "SVM":  np.clip(np.linspace(0, 1, T) + 0.1*np.random.randn(T), 0, 1),
    "RNN":  np.clip(np.linspace(0, 1, T)**1.5 + 0.1*np.random.randn(T), 0, 1),
    "SGHT": np.clip(np.linspace(0, 1, T)**2.0 + 0.1*np.random.randn(T), 0, 1),
}
threshold = 0.6

plt.figure(figsize=(10, 4))
plt.plot(t, x, linewidth=1.0)
plt.xlabel("Time")
plt.ylabel("Signal")
plt.title("Process Signal with First Detection Times")

ytop = float(np.nanmax(x)); ymin = float(np.nanmin(x))
ypos = ytop - 0.05 * (ytop - ymin)

for name, s in scores.items():
    idxs = np.where(s >= threshold)[0]
    if idxs.size > 0:
        idx = int(idxs[0])
        plt.axvline(idx, linestyle="--", linewidth=0.8)
        plt.text(idx, ypos, f"{name} @ {idx}", rotation=90, va="top", ha="right")

plt.tight_layout()
plt.savefig(OUT / "time_series_detection_example.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT / "time_series_detection_example.pdf", bbox_inches="tight")
plt.close()

print("Saved figures to:", OUT.resolve())
