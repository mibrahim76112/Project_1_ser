import os
import argparse
import numpy as np
import torch

from data import load_sampled_data
from utils import print_classification_metrics
from models.selfgated_hierarchial_transformer import (
    SelfGatedHierarchicalTransformerEncoder,
)

from train import (
    gates_to_sensor_segment_matrix,
    plot_gating_heatmap,
    make_fault_gating_plots_with_delta,
)


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
    """
    Recreate the SelfGatedHierarchicalTransformerEncoder with the same
    hyperparameters used during training.
    """
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
    """
    Load model weights from a .pth checkpoint file.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded weights from: {ckpt_path}")
    return model


def run_inference(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    make_plots: bool = True,
    batch_size: int = 256,   # you can tune this
) -> None:
    """
    Run inference on the test set in batches, print classification metrics,
    per-fault accuracies, and optionally generate gating plots.
    """

    # labels on device
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

    model.eval()
    preds = []

    # -------- BATCHEd INFERENCE --------
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_np = X_test[i : i + batch_size]
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=device)
            out = model(batch_tensor)
            preds.append(out.argmax(dim=1).cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = y_test_tensor.cpu().numpy()

    # ---- overall metrics ----
    print("[INFO] Test set classification metrics:")
    print_classification_metrics(y_true, y_pred)

    # ---- per-fault accuracy (percent only) ----
    print("\n[INFO] Per-fault accuracy (%):")
    faults = np.unique(y_true)
    for f in faults:
        mask = (y_true == f)
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean() * 100.0
            print(f"  Fault {int(f):2d}: {acc:.2f}%")
        else:
            print(f"  Fault {int(f):2d}: N/A (no samples)")

    if not make_plots:
        return

    # -------- GATING VISUALIZATIONS --------
    os.makedirs("figures", exist_ok=True)

    # 1) GLOBAL GATING HEATMAP ON A SLICE (for speed)
    n_plot = min(2000, X_test.shape[0])   # was 128; you can tune this
    slice_idx = slice(0, n_plot)

    with torch.no_grad():
        X_plot_tensor = torch.tensor(
            X_test[slice_idx], dtype=torch.float32, device=device
        )
        logits, extras = model(X_plot_tensor, return_gates=True)

    M_global = gates_to_sensor_segment_matrix(extras, reduce="max")
    sensor_names = [f"Var{i+1}" for i in range(M_global.shape[0])]

    plot_gating_heatmap(
        M_global,
        sensor_names,
        fault_id=None,
        out_png="figures/gating_heatmap_test_inference.png",
        title_prefix="Gating Weights (Sensors × Segments) [Inference]",
    )
    print("[INFO] Saved global gating plots (inference) to figures/")

    # 2) PER-FAULT GATING PLOTS: explicitly pick windows for faults 0,3,9,15
    fault_ids = [3, 9, 15]
    max_per_fault = 128

    y_np = y_test  # full test labels as numpy array
    indices = []

    # baseline windows (fault 0)
    baseline_idx = np.where(y_np == 0)[0][:max_per_fault]
    if len(baseline_idx) == 0:
        print("[WARN] No baseline fault 0 windows found; per-fault plots may be unreliable.")
    indices.extend(baseline_idx)

    # windows for each fault of interest
    for f in fault_ids:
        f_idx = np.where(y_np == f)[0][:max_per_fault]
        if len(f_idx) == 0:
            print(f"[WARN] No test windows found for fault {f}; skipping in gating plots.")
        indices.extend(f_idx)

    indices = np.unique(indices)

    if len(indices) == 0:
        print("[WARN] No windows found for requested faults; skipping gating plots.")
        return

    with torch.no_grad():
        X_fault_tensor = torch.tensor(
            X_test[indices], dtype=torch.float32, device=device
        )
        y_fault_tensor = torch.tensor(
            y_np[indices], dtype=torch.long, device=device
        )

        make_fault_gating_plots_with_delta(
            model=model,
            X_test_tensor=X_fault_tensor,
            y_test_tensor=y_fault_tensor,
            fault_ids=fault_ids,
            baseline_fault=0,
            k_top=10,
            n_windows=len(indices),
            reduce="max",
        )

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
        test_end=15000,
        train_run_start=5,
        train_run_end=6,
        test_run_start=1,
        test_run_end=40,
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
