# scripts/plotting_scripts/cross_validation_heatmap.py
#
# Creates a 3×3 heatmap matrix comparing model errors across all combinations
# of params_oil (rows) × validation_oil (columns).
#
# For each call, produces three heatmaps side by side:
#   1. Mass flow error
#   2. Electrical power error
#   3. Discharge temperature error
#
# The metric (MAE or RMSE) is selectable.
#
# Input: directory containing validation_summary_*.csv files.
# The script auto-detects cells from the 'params_oil' and 'validation_oil'
# columns inside each summary CSV.
#
# Examples:
#   # MAE for modified model:
#   python scripts/plotting_scripts/cross_validation_heatmap.py --summary_dir results/final_results/validation_summary/summary_Modified --metric mae
#
#   # RMSE for original model:
#   python scripts/plotting_scripts/cross_validation_heatmap.py --summary_dir results/final_results/validation_summary/summary_Modified --metric rmse

from __future__ import annotations

import argparse
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
OIL_DISPLAY = {"LPG68": "LPG 68", "LPG100": "LPG 100", "all": "beide"}


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
def load_summaries(summary_dir: Path, model_filter: str | None = None) -> pd.DataFrame:
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
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["_source_file"] = f.name
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
                "cbar_label": "MAE $\\dot{m}$ [%]",
                "scale": 100.0,  # convert fraction → percent
                "fmt": "{:.2f}",
            },
            {
                "col": "mae_e_P_rel",
                "title": "MAE elektrische Leistung",
                "cbar_label": "MAE $P_{el}$ [%]",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
            {
                "col": "mae_e_T_dis_K",
                "title": "MAE Austrittstemperatur",
                "cbar_label": "MAE $T_{dis}$ [K]",
                "scale": 1.0,
                "fmt": "{:.2f}",
            },
        ]

    if m == "rmse":
        return [
            {
                "col": "rmse_e_m_rel",
                "title": "RMSE Massenstrom",
                "cbar_label": "RMSE $\\dot{m}$ [%]",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
            {
                "col": "rmse_e_P_rel",
                "title": "RMSE elektrische Leistung",
                "cbar_label": "RMSE $P_{el}$ [%]",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
            {
                "col": "rmse_e_T_dis_K",
                "title": "RMSE Austrittstemperatur",
                "cbar_label": "RMSE $T_{dis}$ [K]",
                "scale": 1.0,
                "fmt": "{:.2f}",
            },
        ]

    raise ValueError(f"Unknown metric: {metric}. Use 'mae' or 'rmse'.")


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

    ax.set_xlabel("Validierungs-Daten")
    ax.set_ylabel("Parameter gefittet auf")

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
    Create a figure with three 3×3 heatmaps side by side.
    """
    metric_cfgs = get_metric_config(metric)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))

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

    metric_title = {"mae": "MAE", "rmse": "RMSE"}[metric.lower()]
    fig.suptitle(
        f"Cross-Validation ({metric_title}) — {model_name.capitalize()} Modell",
        fontsize=15, y=1.02,
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
    ap.add_argument("--summary_dir", required=True, type=Path,
                    help="Directory containing validation_summary_*.csv files")
    ap.add_argument("--metric", required=True, choices=["mae", "rmse"],
                    help="Which error metric to plot")
    ap.add_argument("--model", default=None,
                    help="Filter by model (original | modified). If omitted, use all rows.")
    ap.add_argument("--out_dir", default="results/cross_validation_heatmap",
                    help="Output directory")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    if not args.summary_dir.exists():
        raise FileNotFoundError(args.summary_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    df = load_summaries(args.summary_dir, model_filter=args.model)

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

    # Report matrix coverage
    n_cells = 0
    for params_oil in OIL_ORDER:
        for val_oil in OIL_ORDER:
            mask = (
                (df["params_oil_norm"] == params_oil) &
                (df["validation_oil_norm"] == val_oil)
            )
            if mask.any():
                n_cells += 1
    print(f"  Matrix coverage: {n_cells}/9 cells filled")

    # -------------------------
    # Plot
    # -------------------------
    stamp = _ts()
    out_path = out_dir / f"cross_validation_{args.metric}_{model_name}_{stamp}.{args.out_format}"

    plot_cross_validation_matrix(
        df=df,
        metric=args.metric,
        model_name=model_name,
        out_path=out_path,
    )

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
