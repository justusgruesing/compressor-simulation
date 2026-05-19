# scripts/plotting_scripts/cross_validation_heatmap.py
#
# Two modes:
#
# 1. cross_validation (default): 3×3 heatmap of params_oil (rows) × validation_oil (cols)
#    for ONE model. Shows how well parameters fitted on one oil generalize to others.
#
# 2. model_comparison: 3×3 heatmap of model (rows) × validation_oil (cols) for ONE
#    params_oil. Shows which model generalizes best across data sets.
#
# Examples:
#   # Cross-validation: MAE for modified model
#   python scripts/plotting_scripts/cross_validation_heatmap.py \
#       --summary_dir results/validation/modified/summary --metric mae
#
#   # Model comparison: all 3 models, params fitted on LPG68
#   python scripts/plotting_scripts/cross_validation_heatmap.py \
#       --mode model_comparison --params_oil LPG68 --metric mae
#
#   # Cross-validation: for original model
#   python scripts/plotting_scripts/cross_validation_heatmap.py --summary_dir results/validation/original/summary --metric rmse --selection_mode train_only
#
#   # Model comparison with combined metric
#   python scripts/plotting_scripts/cross_validation_heatmap.py \
#       --mode model_comparison --params_oil LPG68 --metric mae_combined \
#       --base_dir results/validation

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.style.use("ebc.paper.mplstyle")


# =========================================================
# Constants
# =========================================================
OIL_ORDER = ["LPG68", "LPG100", "all"]
OIL_DISPLAY = {"LPG68": "PAG 68", "LPG100": "PAG 100", "all": "beide"}

MODEL_ORDER = ["original", "modified", "oil_path"]
MODEL_DISPLAY = {
    "original": "Basis",
    "modified": "Stufe I",
    "oil_path": "Stufe II",
}
MODEL_SUBDIRS = {
    "original": "original",
    "modified": "modified",
    "oil_path": "oil_path",
}


# =========================================================
# Helpers
# =========================================================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _norm_oil(s: str) -> str:
    s = str(s).strip()
    low = s.lower().replace(" ", "")
    if low in ("lpg68", "lpg 68"):
        return "LPG68"
    if low in ("lpg100", "lpg 100"):
        return "LPG100"
    if low == "all":
        return "all"
    return s


# =========================================================
# Data loading
# =========================================================
def load_summaries(summary_dir: Path, model_filter: str | None = None,
                   selection_mode_filter: str | None = None) -> pd.DataFrame:
    """
    Load all validation_summary_*.csv files in the directory and concatenate them.
    """
    csv_files = sorted(summary_dir.glob("validation_summary_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No validation_summary_*.csv files found in {summary_dir}"
        )

    print(f"  Found {len(csv_files)} summary files")

    dfs = []
    # Pattern: ..._{selection_mode}_{timestamp}.csv
    _MODE_PATTERN = re.compile(r"_(train_only|validation_only|all)_\d{4}-\d{2}-\d{2}_\d{6}$")

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["_source_file"] = f.name

            # Extract selection_mode from filename
            match = _MODE_PATTERN.search(f.stem)
            df["selection_mode"] = match.group(1) if match else "unknown"

            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Could not read {f.name}: {e}")

    if not dfs:
        raise ValueError("No summary files could be loaded.")

    combined = pd.concat(dfs, ignore_index=True)

    # Normalize oil columns
    combined["params_oil_norm"] = combined["params_oil"].apply(_norm_oil)
    combined["validation_oil_norm"] = combined["validation_oil"].apply(_norm_oil)

    # Filter by model
    if model_filter is not None:
        before = len(combined)
        combined = combined[combined["model"].astype(str).str.lower() == model_filter.lower()].copy()
        after = len(combined)
        if after < before:
            print(f"  Filtered to model '{model_filter}': {after}/{before} rows")

    # Filter by selection_mode
    if selection_mode_filter is not None and "selection_mode" in combined.columns:
        before = len(combined)
        combined = combined[
            combined["selection_mode"].astype(str).str.lower().str.strip()
            == selection_mode_filter.lower()
        ].copy()
        print(f"  Filtered to selection_mode='{selection_mode_filter}': {len(combined)}/{before} rows")

    return combined


def build_matrix(df: pd.DataFrame, value_col: str) -> np.ndarray:
    """
    Build a 3×3 matrix from the summary DataFrame.
    Rows = params_oil, Columns = validation_oil (order: LPG68, LPG100, all).
    Missing cells are NaN.
    """
    matrix = np.full((3, 3), np.nan)

    for i, params_oil in enumerate(OIL_ORDER):
        for j, val_oil in enumerate(OIL_ORDER):
            mask = (
                (df["params_oil_norm"] == params_oil) &
                (df["validation_oil_norm"] == val_oil)
            )
            sub = df[mask]

            if len(sub) == 0:
                continue
            if len(sub) > 1:
                print(f"  [WARN] Multiple entries for params={params_oil}, val={val_oil} — using last.")
                sub = sub.tail(1)

            val = sub[value_col].iloc[0]
            if pd.notna(val):
                matrix[i, j] = float(val)

    return matrix


# =========================================================
# Model comparison mode: load from multiple model subdirs
# =========================================================
def load_summaries_multi_model(
    base_dir: Path,
    params_oil_filter: str | None = None,
    selection_mode_filter: str | None = None,
) -> pd.DataFrame:
    """
    Load validation_summary_*.csv from all model subdirectories under base_dir.
    Expected structure: base_dir/<model>/summary/validation_summary_*.csv
    """
    dfs = []
    n_files = 0

    for model_key, subdir in MODEL_SUBDIRS.items():
        summary_dir = base_dir / subdir / "summary"
        if not summary_dir.exists():
            print(f"  [WARN] Missing directory: {summary_dir}")
            continue

        csv_files = sorted(summary_dir.glob("validation_summary_*.csv"))
        _MODE_PAT = re.compile(r"_(train_only|validation_only|all)_\d{4}-\d{2}-\d{2}_\d{6}$")
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                df["_source_file"] = f.name
                df["_source_model_dir"] = model_key

                match = _MODE_PAT.search(f.stem)
                df["selection_mode"] = match.group(1) if match else "unknown"

                dfs.append(df)
                n_files += 1
            except Exception as e:
                print(f"  [WARN] Could not read {f.name}: {e}")

    if not dfs:
        raise ValueError(f"No summary files found under {base_dir}")

    print(f"  Found {n_files} summary files across model directories")

    combined = pd.concat(dfs, ignore_index=True)

    # Normalize
    combined["params_oil_norm"] = combined["params_oil"].apply(_norm_oil)
    combined["validation_oil_norm"] = combined["validation_oil"].apply(_norm_oil)
    combined["model_norm"] = (
        combined["model"].fillna(combined["_source_model_dir"])
        .astype(str).str.lower().str.strip()
        .replace({"oilpath": "oil_path"})
    )

    # Filter by params_oil
    if params_oil_filter is not None:
        p_norm = _norm_oil(params_oil_filter)
        before = len(combined)
        combined = combined[combined["params_oil_norm"] == p_norm].copy()
        print(f"  Filtered to params_oil='{p_norm}': {len(combined)}/{before} rows")

    # Filter by selection_mode if present
    if selection_mode_filter is not None and "selection_mode" in combined.columns:
        before = len(combined)
        combined = combined[
            combined["selection_mode"].astype(str).str.lower().str.strip()
            == selection_mode_filter.lower()
        ].copy()
        print(f"  Filtered to selection_mode='{selection_mode_filter}': {len(combined)}/{before} rows")

    return combined


def build_matrix_model_comparison(df: pd.DataFrame, value_col: str) -> np.ndarray:
    """
    Build a 3×3 matrix for model comparison.
    Rows = model (original, modified, oil_path).
    Columns = validation_oil (LPG68, LPG100, all).
    """
    matrix = np.full((3, 3), np.nan)

    for i, model_key in enumerate(MODEL_ORDER):
        for j, val_oil in enumerate(OIL_ORDER):
            mask = (
                (df["model_norm"] == model_key) &
                (df["validation_oil_norm"] == val_oil)
            )
            sub = df[mask]

            if len(sub) == 0:
                continue
            if len(sub) > 1:
                print(f"  [WARN] Multiple entries for model={model_key}, val={val_oil} — using last.")
                sub = sub.tail(1)

            val = sub[value_col].iloc[0]
            if pd.notna(val):
                matrix[i, j] = float(val)

    return matrix


def add_combined_metrics(df: pd.DataFrame, Tdis_norm_K: float = 50.0) -> pd.DataFrame:
    """
    Add combined error metrics that aggregate over m_dot, P_el and T_dis.
    T_dis is normalized by Tdis_norm_K (default 50 K, like James et al.) so it
    becomes dimensionless and comparable to the relative errors.

    Adds columns:
      _combined_mae:  mean of (mae_m_rel, mae_P_rel, mae_T_dis/Tdis_norm)
      _combined_rmse: mean of (rmse_m_rel, rmse_P_rel, rmse_T_dis/Tdis_norm)
    """
    if "Tdis_norm_K" in df.columns:
        T_norm = pd.to_numeric(df["Tdis_norm_K"], errors="coerce").fillna(Tdis_norm_K)
    else:
        T_norm = pd.Series([Tdis_norm_K] * len(df), index=df.index)

    df = df.copy()
    df["_combined_mae"] = (
        df["mae_e_m_rel"] + df["mae_e_P_rel"] + df["mae_e_T_dis_K"] / T_norm
    ) / 3.0

    df["_combined_rmse"] = (
        df["rmse_e_m_rel"] + df["rmse_e_P_rel"] + df["rmse_e_T_dis_K"] / T_norm
    ) / 3.0

    return df


# =========================================================
# Metric configuration
# =========================================================
def get_metric_config(metric: str) -> list[dict]:
    """
    Returns a list of three dicts, one per subplot (m_dot, P_el, T_dis).
    Each dict contains: column name in summary CSV, plot title, colorbar label, unit scale.
    """
    m = metric.lower().strip()

    if m == "mae":
        return [
            {
                "col": "mae_e_m_rel",
                "title": "MAE Massenstrom",
                "cbar_label": "MAE $\\dot{m}$ in %",
                "scale": 100.0,  # convert fraction → percent
                "fmt": "{:.2f}",
            },
            {
                "col": "mae_e_P_rel",
                "title": "MAE elektrische Leistung",
                "cbar_label": "MAE $P_{el}$ in %",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
            {
                "col": "mae_e_T_dis_K",
                "title": "MAE Austrittstemperatur",
                "cbar_label": "MAE $T_{dis}$ in K",
                "scale": 1.0,
                "fmt": "{:.2f}",
            },
        ]

    if m == "rmse":
        return [
            {
                "col": "rmse_e_m_rel",
                "title": "RMSE Massenstrom",
                "cbar_label": "RMSE $\\dot{m}$ in %",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
            {
                "col": "rmse_e_P_rel",
                "title": "RMSE elektrische Leistung",
                "cbar_label": "RMSE $P_{el}$ in %",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
            {
                "col": "rmse_e_T_dis_K",
                "title": "RMSE Austrittstemperatur",
                "cbar_label": "RMSE $T_{dis}$ in K",
                "scale": 1.0,
                "fmt": "{:.2f}",
            },
        ]

    if m in ("mae_combined", "combined_mae"):
        return [
            {
                "col": "_combined_mae",
                "title": "Aggregierter MAE\n(Ø über $\\dot{m}$, $P_{el}$, $T_{dis}/T_{norm}$)",
                "cbar_label": "Ø MAE in %",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
        ]

    if m in ("rmse_combined", "combined_rmse"):
        return [
            {
                "col": "_combined_rmse",
                "title": "Aggregierter RMSE\n(Ø über $\\dot{m}$, $P_{el}$, $T_{dis}/T_{norm}$)",
                "cbar_label": "Ø RMSE in %",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
        ]

    raise ValueError(
        f"Unknown metric: {metric}. "
        f"Use 'mae', 'rmse', 'mae_combined' or 'rmse_combined'."
    )


# =========================================================
# Plot
# =========================================================
def plot_heatmap_cell(
    ax,
    matrix: np.ndarray,
    title: str,
    cbar_label: str,
    fmt: str = "{:.2f}",
    cmap: str = "viridis_r",
    fig=None,
):
    """
    Plot a single 3×3 heatmap on the given axes with value annotations.
    """
    masked = np.ma.masked_invalid(matrix)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#E0E0E0")  # light gray for missing cells

    finite_vals = matrix[np.isfinite(matrix)]
    if len(finite_vals) == 0:
        ax.set_title(f"{title}\n(keine Daten)")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))

    if vmin == vmax:
        vmin *= 0.95
        vmax *= 1.05

    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect="equal", origin="upper")

    # Annotate each cell with value
    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            if not np.isfinite(val):
                ax.text(
                    j, i, "—",
                    ha="center", va="center",
                    color="dimgray", fontsize=11,
                )
                continue

            # Choose text color based on background luminance
            rgba = cmap_obj(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.5 else "black"

            # Highlight diagonal (native fit) with a box
            if i == j:
                ax.text(
                    j, i, fmt.format(val),
                    ha="center", va="center",
                    color=text_color,
                    fontsize=13,
                    fontweight="bold",
                )
            else:
                ax.text(
                    j, i, fmt.format(val),
                    ha="center", va="center",
                    color=text_color,
                    fontsize=12,
                )

    # Axes
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([OIL_DISPLAY[o] for o in OIL_ORDER])
    ax.set_yticklabels([OIL_DISPLAY[o] for o in OIL_ORDER])

    ax.set_xlabel("Validierungs auf")
    ax.set_ylabel("Kalibrierung auf")

    ax.set_title(title, fontsize=13)

    # Draw cell borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # Colorbar
    if fig is not None:
        cbar = fig.colorbar(im, ax=ax, pad=0.03, shrink=0.85)
        cbar.set_label(cbar_label, fontsize=11)


def plot_cross_validation_matrix(
    df: pd.DataFrame,
    metric: str,
    model_name: str,
    out_path: Path,
):
    """
    Create a figure with one or three 3×3 heatmaps side by side, depending
    on the metric type (combined → 1 plot, per-target → 3 plots).
    """
    metric_cfgs = get_metric_config(metric)
    n_plots = len(metric_cfgs)

    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(19, 6.5))

    for idx, cfg in enumerate(metric_cfgs):
        matrix = build_matrix(df, cfg["col"])
        matrix_scaled = matrix * cfg["scale"]

        plot_heatmap_cell(
            ax=axes[idx],
            matrix=matrix_scaled,
            title=cfg["title"],
            cbar_label=cfg["cbar_label"],
            fmt=cfg["fmt"],
            cmap="viridis_r",  # reversed: dark = low error = good
            fig=fig,
        )

    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [OK] Saved: {out_path}")


def plot_heatmap_cell_model_comparison(
    ax,
    matrix: np.ndarray,
    title: str,
    cbar_label: str,
    fmt: str = "{:.2f}",
    cmap: str = "viridis_r",
    fig=None,
):
    """
    Plot a 3×3 heatmap for model comparison mode.
    Rows = models, Columns = validation_oil.
    """
    masked = np.ma.masked_invalid(matrix)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#E0E0E0")

    finite_vals = matrix[np.isfinite(matrix)]
    if len(finite_vals) == 0:
        ax.set_title(f"{title}\n(keine Daten)")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    if vmin == vmax:
        vmin *= 0.95
        vmax *= 1.05

    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect="equal", origin="upper")

    # Annotate cells
    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "—", ha="center", va="center",
                        color="dimgray", fontsize=11)
                continue

            rgba = cmap_obj(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.5 else "black"

            ax.text(
                j, i, fmt.format(val),
                ha="center", va="center",
                color=text_color, fontsize=12,
                fontweight="bold" if i == j else "normal",
            )

    # Axes — models on y, oils on x
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([OIL_DISPLAY[o] for o in OIL_ORDER])
    ax.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER])

    ax.set_xlabel("Validierungs-Daten")
    ax.set_ylabel("Modell")

    ax.set_title(title, fontsize=13)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    if fig is not None:
        cbar = fig.colorbar(im, ax=ax, pad=0.03, shrink=0.85)
        cbar.set_label(cbar_label, fontsize=11)


def plot_model_comparison_matrix(
    df: pd.DataFrame,
    metric: str,
    params_oil: str,
    out_path: Path,
):
    """
    Create figure with 1 or 3 heatmaps for model comparison mode.
    Rows = models, Columns = validation_oil.
    """
    metric_cfgs = get_metric_config(metric)
    n_plots = len(metric_cfgs)

    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(19, 6.5))

    for idx, cfg in enumerate(metric_cfgs):
        matrix = build_matrix_model_comparison(df, cfg["col"])
        matrix_scaled = matrix * cfg["scale"]

        plot_heatmap_cell_model_comparison(
            ax=axes[idx],
            matrix=matrix_scaled,
            title=cfg["title"],
            cbar_label=cfg["cbar_label"],
            fmt=cfg["fmt"],
            cmap="viridis_r",
            fig=fig,
        )

    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [OK] Saved: {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="3×3 cross-validation heatmap matrix for model error comparison."
    )

    # Mode selection
    ap.add_argument("--mode", choices=["cross_validation", "model_comparison"],
                    default="cross_validation",
                    help="cross_validation: params_oil × val_oil for one model (default). "
                         "model_comparison: model × val_oil for one params_oil.")

    # Input: either a single summary_dir or a base_dir for model comparison
    ap.add_argument("--summary_dir", type=Path, default=None,
                    help="Directory with validation_summary_*.csv (for cross_validation mode)")
    ap.add_argument("--base_dir", type=Path, default=None,
                    help="Base dir containing original/modified/oil_path subdirs "
                         "(for model_comparison mode, default: results/validation)")

    ap.add_argument("--metric", required=True,
                    choices=["mae", "rmse", "mae_combined", "rmse_combined"],
                    help="Which error metric to plot.")

    # Filters
    ap.add_argument("--model", default=None,
                    help="Filter by model (cross_validation mode only)")
    ap.add_argument("--params_oil", default=None,
                    help="Filter by params_oil (required for model_comparison mode)")
    ap.add_argument("--selection_mode", default=None,
                    help="Filter by selection_mode, e.g. 'validation_only' (optional)")

    ap.add_argument("--Tdis_norm_K", type=float, default=50.0,
                    help="Normalization for T_dis in combined metrics (default 50 K)")
    ap.add_argument("--out_dir", default="results/cross_validation_heatmap",
                    help="Output directory")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = _ts()

    # =========================================================
    # Mode: cross_validation (original behavior)
    # =========================================================
    if args.mode == "cross_validation":
        if args.summary_dir is None:
            raise ValueError("--summary_dir is required for cross_validation mode.")
        if not args.summary_dir.exists():
            raise FileNotFoundError(args.summary_dir)

        df = load_summaries(args.summary_dir, model_filter=args.model,
                            selection_mode_filter=args.selection_mode)

        # Compute combined metrics if needed
        if args.metric.lower() in ("mae_combined", "rmse_combined"):
            df = add_combined_metrics(df, Tdis_norm_K=args.Tdis_norm_K)
            print(f"  Tdis_norm_K (combined): {args.Tdis_norm_K} K")

        # Detect model name(s)
        unique_models = df["model"].dropna().unique()
        if len(unique_models) == 1:
            model_name = str(unique_models[0])
        elif len(unique_models) > 1:
            print(f"  [WARN] Multiple models found: {unique_models}. Use --model to filter.")
            model_name = "mixed"
        else:
            model_name = "unknown"

        print(f"  Model: {model_name}")

        # Report coverage
        n_cells = sum(
            1 for p in OIL_ORDER for v in OIL_ORDER
            if ((df["params_oil_norm"] == p) & (df["validation_oil_norm"] == v)).any()
        )
        print(f"  Matrix coverage: {n_cells}/9 cells filled")

        out_path = out_dir / f"cross_validation_{args.metric}_{model_name}_{stamp}.{args.out_format}"
        plot_cross_validation_matrix(
            df=df, metric=args.metric,
            model_name=model_name, out_path=out_path,
        )

    # =========================================================
    # Mode: model_comparison
    # =========================================================
    elif args.mode == "model_comparison":
        if args.params_oil is None:
            raise ValueError("--params_oil is required for model_comparison mode.")

        base_dir = args.base_dir or Path("results/validation")
        if not base_dir.exists():
            raise FileNotFoundError(f"Base dir not found: {base_dir}")

        df = load_summaries_multi_model(
            base_dir=base_dir,
            params_oil_filter=args.params_oil,
            selection_mode_filter=args.selection_mode,
        )

        # Compute combined metrics if needed
        if args.metric.lower() in ("mae_combined", "rmse_combined"):
            df = add_combined_metrics(df, Tdis_norm_K=args.Tdis_norm_K)
            print(f"  Tdis_norm_K (combined): {args.Tdis_norm_K} K")

        # Report coverage
        n_cells = sum(
            1 for m in MODEL_ORDER for v in OIL_ORDER
            if ((df["model_norm"] == m) & (df["validation_oil_norm"] == v)).any()
        )
        print(f"  Matrix coverage: {n_cells}/9 cells filled")

        p_tag = _norm_oil(args.params_oil).lower()
        out_path = out_dir / f"model_comparison_{args.metric}_params_{p_tag}_{stamp}.{args.out_format}"
        plot_model_comparison_matrix(
            df=df, metric=args.metric,
            params_oil=args.params_oil, out_path=out_path,
        )

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
