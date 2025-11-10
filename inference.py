import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from data import load_sampled_data
from utils import print_classification_metrics
from models.selfgated_hierarchial_transformer import (
    SelfGatedHierarchicalTransformerEncoder,
)

# we only reuse the matrix helper from train.py
from train import gates_to_sensor_segment_matrix, plot_gating_heatmap


# ---------------- MODEL HELPERS ----------------

def build_model(
    input_dim: int,
    num_classes: int = 21,
    d_model: int = 128,
    nhead: int = 4,
    num_layers_low: int = 2,
    num_layers_high: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.05,
    pool_output_size: int = 10,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    model = SelfGatedHierarchicalTransformerEncoder(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers_low=num_layers_low,
        num_layers_high=num_layers_high,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pool_output_size=pool_output_size,
        num_classes=num_classes,
    )
    return model.to(device)


def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> torch.nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded weights from: {ckpt_path}")
    return model


# ---------------- PLOTTING: TOP-K Δ GATES ----------------

def save_topk_delta_barplot(
    Md: np.ndarray,
    sensor_names: list[str],
    fault_id: int,
    k: int = 10,
    out_png: str | None = None,
):
    """
    Md: (F, S) delta matrix = M_fault - M_baseline
    sensor_names: list of length F
    """
    F, S = Md.shape

    # mean Δ per sensor across segments
    mean_delta = Md.mean(axis=1)  # shape (F,)

    # rank by absolute change so big + or − both count
    idx = np.argsort(-np.abs(mean_delta))[:k]
    vals = mean_delta[idx]
    names = [sensor_names[i] for i in idx]

    # reverse so biggest at top of plot
    vals = vals[::-1]
    names = names[::-1]

    plt.figure(figsize=(10, max(4, 0.4 * k + 2)))
    plt.barh(np.arange(len(vals)), vals)
    plt.yticks(np.arange(len(vals)), names)
    plt.xlabel("Mean Δ Gate Weight (fault − baseline)")
    plt.title(f"Top Sensors by Δ Gate – Fault {fault_id}")
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------- INFERENCE ----------------

def run_inference(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    make_plots: bool = True,
    batch_size: int = 256,
) -> None:
    """
    Run inference on the test set in batches, print metrics,
    and optionally generate gating plots.
    """

    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

    model.eval()
    preds = []

    # -------- BATCHED INFERENCE (for metrics) --------
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_np = X_test[i : i + batch_size]
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=device)
            out = model(batch_tensor)
            preds.append(out.argmax(dim=1).cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = y_test_tensor.cpu().numpy()

    print("[INFO] Test set classification metrics:")
    print_classification_metrics(y_true, y_pred)

    print("\n[INFO] Per-fault accuracy (%):")
    faults = np.unique(y_true)

    # collect per-fault accuracies for macro (mean per-fault) accuracy
    acc_list = []

    for f in faults:
        mask = (y_true == f)
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean() * 100.0
            acc_list.append(acc)
            print(f"  Fault {int(f):2d}: {acc:.2f}%")
        else:
            print(f"  Fault {int(f):2d}: N/A (no samples)")

    # macro (mean per-fault) accuracy
    if len(acc_list) > 0:
        macro_acc = float(np.mean(acc_list))
        print(f"\n[INFO] Mean per-fault accuracy: {macro_acc:.2f}%")

    if not make_plots:
        return

    os.makedirs("figures", exist_ok=True)

    # -------- 1) GLOBAL GATING HEATMAP --------
    n_plot = min(2000, X_test.shape[0])
    slice_idx = slice(0, n_plot)

    with torch.no_grad():
        X_plot_tensor = torch.tensor(
            X_test[slice_idx], dtype=torch.float32, device=device
        )
        logits, extras = model(X_plot_tensor, return_gates=True)

    M_global = gates_to_sensor_segment_matrix(extras, reduce="mean")
    sensor_names = [f"Var{i+1}" for i in range(M_global.shape[0])]

    plot_gating_heatmap(
        M_global,
        sensor_names,
        fault_id=None,
        out_png="figures/gating_heatmap_test_inference.png",
        title_prefix="Gating Weights (Sensors × Segments) [Inference]",
    )
    print("[INFO] Saved global gating plots (inference) to figures/")

    # -------- 2) PER-FAULT Δ GATE PLOTS --------
    fault_ids = [3, 9, 15]      # change this list if you want more faults
    max_per_fault = 4096

    y_np = y_test

    # baseline fault 0
    baseline_idx = np.where(y_np == 0)[0][:max_per_fault]
    if len(baseline_idx) == 0:
        print("[WARN] No baseline fault 0 windows found; per-fault plots will be skipped.")
        return

    with torch.no_grad():
        X_base = torch.tensor(
            X_test[baseline_idx], dtype=torch.float32, device=device
        )
        logits0, extras0 = model(X_base, return_gates=True)

    M0 = gates_to_sensor_segment_matrix(extras0, reduce="mean")  # (F, S)
    print(f"[BASELINE] fault=0, windows used={len(baseline_idx)}")

    for fid in fault_ids:
        fault_idx = np.where(y_np == fid)[0][:max_per_fault]
        if len(fault_idx) == 0:
            print(f"[WARN] No test windows found for fault {fid}; skipping.")
            continue

        print(f"[FAULT] fid={fid}, windows used={len(fault_idx)}")

        with torch.no_grad():
            X_fault = torch.tensor(
                X_test[fault_idx], dtype=torch.float32, device=device
            )
            logits_f, extras_f = model(X_fault, return_gates=True)

        Mf = gates_to_sensor_segment_matrix(extras_f, reduce="mean")  # (F, S)
        Md = Mf - M0  # Δ vs baseline

        # DEBUG print to confirm numbers used in plot
        mean_delta = Md.mean(axis=1)
        idx_debug = np.argsort(-np.abs(mean_delta))[:10]
        names_debug = [sensor_names[i] for i in idx_debug]
        vals_debug = mean_delta[idx_debug]
        print(f"[DEBUG] Fault {fid} Δ top sensors (used for plot):",
              list(zip(names_debug, vals_debug)))

        # save bar plot of top-k Δ sensors for this fault
        out_png = f"figures/topk_delta_fault_{fid}.png"
        save_topk_delta_barplot(
            Md,
            sensor_names,
            fault_id=fid,
            k=10,
            out_png=out_png,
        )
        print(f"[INFO] Saved Δ top-k bar plot for fault {fid} to {out_png}")


# ---------------- CLI ----------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for Self-Gated Hierarchical Transformer on TEP."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="transformer_model_1.pth",
        help="Path to the .pth checkpoint file.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Window size used during training/data loading.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Stride used during training/data loading.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="If set, skip generating gating plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using device: {device} (CUDA enabled for inference)")
    else:
        device = torch.device("cpu")
        print(f"[INFO] Using device: {device} (CUDA disabled for inference)")

    X_train, X_test, y_train, y_test = load_sampled_data(
        window_size=args.window_size,
        stride=args.stride,
        type_model="supervised",
        fault_free_path="/workspace/TEP_FaultFree_Testing.RData",
        faulty_path="/workspace/TEP_Faulty_Testing.RData",
        train_end=1000,
        test_start=10000,
        test_end=20000,
        train_run_start=5,
        train_run_end=6,
        test_run_start=1,
        test_run_end=80,
    )

    print("Training Data Shape:", X_train.shape, y_train.shape)
    print("Test Data Shape:", X_test.shape, y_test.shape)
    print("Unique classes in Training Set:", np.unique(y_train))
    print("Unique classes in Test Set:", np.unique(y_test))

    num_classes = 21
    input_dim = X_train.shape[2]

    model = build_model(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=128,
        nhead=4,
        num_layers_low=2,
        num_layers_high=2,
        dim_feedforward=128,
        dropout=0.05,
        pool_output_size=10,
        device=device,
    )
    model = load_checkpoint(model, args.ckpt, device)

    run_inference(
        model=model,
        X_test=X_test,
        y_test=y_test,
        device=device,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
