# scripts/plotting_scripts/train_vs_validation_scatter.py
#
# Scatter plot: error on training data (x-axis) vs. error on validation data
# (y-axis). Each point is one configuration (model, params_oil, validation_oil).
# The diagonal line marks "train error = validation error" — points on it
# generalize perfectly, points above it show overfitting.
#
# Produces 3 subplots per call: m_dot, P_el, T_dis.
#
# INPUT REQUIREMENT:
#   Run the validation script TWICE per configuration — once with
#     --selection_mode train_only
#   and once with
#     --selection_mode validation_only
#   Put all resulting validation_summary_*.csv files into one directory.
#
# Examples:
#   # MAE scatter:
#   python scripts/plotting_scripts/train_vs_validation_scatter.py \
#       --summary_dir results/validation_summaries_train_val \
#       --metric mae
#
#   # RMSE scatter, only modified model:
#   python scripts/plotting_scripts/train_vs_validation_scatter.py --summary_dir data/train_vs_validation --metric mae
#
#   # Exclude validation_oil=all configurations:
#   python scripts/plotting_scripts/train_vs_validation_scatter.py \
#       --summary_dir results/validation_summaries_train_val \
#       --metric mae --exclude_all_oil

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use("ebc.paper.mplstyle")


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
# Visual configuration
# =========================================================
MODEL_COLORS = {
    "original": "#EC635C",   # red
    "modified": "#4B81C4",   # blue
}

PARAMS_OIL_MARKERS = {
    "LPG68": "o",     # circle
    "LPG100": "s",    # square
    "all": "^",       # triangle
}

OIL_DISPLAY = {"LPG68": "LPG 68", "LPG100": "LPG 100", "all": "beide"}


# =========================================================
# Data loading and pairing
# =========================================================
def load_summaries(summary_dir: Path) -> pd.DataFrame:
    """
    Load all validation_summary_*.csv files.
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
    combined["params_oil_norm"] = combined["params_oil"].apply(_norm_oil)
    combined["validation_oil_norm"] = combined["validation_oil"].apply(_norm_oil)
    combined["model_norm"] = combined["model"].astype(str).str.lower().str.strip()
    combined["selection_mode_norm"] = combined["selection_mode"].astype(str).str.lower().str.strip()

    return combined


def pair_train_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each configuration (model, params_oil, validation_oil), find the
    train_only and validation_only rows and pair them into a single row
    with train_* and val_* columns.
    """
    train_df = df[df["selection_mode_norm"] == "train_only"].copy()
    val_df = df[df["selection_mode_norm"] == "validation_only"].copy()

    print(f"  Training rows:   {len(train_df)}")
    print(f"  Validation rows: {len(val_df)}")

    if len(train_df) == 0:
        raise ValueError(
            "No rows with selection_mode='train_only' found. "
            "Run the validation script with --selection_mode train_only first."
        )
    if len(val_df) == 0:
        raise ValueError(
            "No rows with selection_mode='validation_only' found. "
            "Run the validation script with --selection_mode validation_only first."
        )

    key_cols = ["model_norm", "params_oil_norm", "validation_oil_norm"]
    metric_cols = [
        "mae_e_m_rel", "rmse_e_m_rel",
        "mae_e_P_rel", "rmse_e_P_rel",
        "mae_e_T_dis_K", "rmse_e_T_dis_K",
    ]

    train_cols = key_cols + [c for c in metric_cols if c in train_df.columns]
    val_cols = key_cols + [c for c in metric_cols if c in val_df.columns]

    train_sub = train_df[train_cols].copy()
    val_sub = val_df[val_cols].copy()

    # Rename metrics
    train_sub = train_sub.rename(columns={c: f"train_{c}" for c in metric_cols if c in train_sub.columns})
    val_sub = val_sub.rename(columns={c: f"val_{c}" for c in metric_cols if c in val_sub.columns})

    # Inner merge on configuration keys
    paired = train_sub.merge(val_sub, on=key_cols, how="inner")

    n_train = len(train_sub)
    n_val = len(val_sub)
    n_paired = len(paired)

    print(f"  Paired configurations: {n_paired}")
    if n_paired < min(n_train, n_val):
        print(f"  [WARN] Some configurations could not be paired "
              f"({n_train - n_paired} train, {n_val - n_paired} val unmatched)")

    if n_paired == 0:
        raise ValueError(
            "No matching train/validation pairs found. "
            "Ensure each configuration has both train_only and validation_only runs."
        )

    return paired


def add_combined_metrics(paired: pd.DataFrame, Tdis_norm_K: float = 50.0) -> pd.DataFrame:
    """
    Add combined error metrics to the paired dataframe.
    Combined = mean of (m_rel, P_rel, T_dis/Tdis_norm) for both train and val.
    """
    paired = paired.copy()

    # MAE combined
    if all(c in paired.columns for c in
           ["train_mae_e_m_rel", "train_mae_e_P_rel", "train_mae_e_T_dis_K"]):
        paired["train_mae_combined"] = (
            paired["train_mae_e_m_rel"]
            + paired["train_mae_e_P_rel"]
            + paired["train_mae_e_T_dis_K"] / Tdis_norm_K
        ) / 3.0

    if all(c in paired.columns for c in
           ["val_mae_e_m_rel", "val_mae_e_P_rel", "val_mae_e_T_dis_K"]):
        paired["val_mae_combined"] = (
            paired["val_mae_e_m_rel"]
            + paired["val_mae_e_P_rel"]
            + paired["val_mae_e_T_dis_K"] / Tdis_norm_K
        ) / 3.0

    # RMSE combined
    if all(c in paired.columns for c in
           ["train_rmse_e_m_rel", "train_rmse_e_P_rel", "train_rmse_e_T_dis_K"]):
        paired["train_rmse_combined"] = (
            paired["train_rmse_e_m_rel"]
            + paired["train_rmse_e_P_rel"]
            + paired["train_rmse_e_T_dis_K"] / Tdis_norm_K
        ) / 3.0

    if all(c in paired.columns for c in
           ["val_rmse_e_m_rel", "val_rmse_e_P_rel", "val_rmse_e_T_dis_K"]):
        paired["val_rmse_combined"] = (
            paired["val_rmse_e_m_rel"]
            + paired["val_rmse_e_P_rel"]
            + paired["val_rmse_e_T_dis_K"] / Tdis_norm_K
        ) / 3.0

    return paired


# =========================================================
# Metric configuration
# =========================================================
def get_metric_config(metric: str) -> list[dict]:
    m = metric.lower().strip()

    if m == "mae":
        return [
            {
                "train_col": "train_mae_e_m_rel",
                "val_col": "val_mae_e_m_rel",
                "title": "MAE Massenstrom",
                "axis_unit": "MAE $\\dot{m}$ [%]",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
            {
                "train_col": "train_mae_e_P_rel",
                "val_col": "val_mae_e_P_rel",
                "title": "MAE elektrische Leistung",
                "axis_unit": "MAE $P_{el}$ [%]",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
            {
                "train_col": "train_mae_e_T_dis_K",
                "val_col": "val_mae_e_T_dis_K",
                "title": "MAE Austrittstemperatur",
                "axis_unit": "MAE $T_{dis}$ [K]",
                "scale": 1.0,
                "fmt": "{:.2f}",
            },
        ]

    if m == "rmse":
        return [
            {
                "train_col": "train_rmse_e_m_rel",
                "val_col": "val_rmse_e_m_rel",
                "title": "RMSE Massenstrom",
                "axis_unit": "RMSE $\\dot{m}$ [%]",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
            {
                "train_col": "train_rmse_e_P_rel",
                "val_col": "val_rmse_e_P_rel",
                "title": "RMSE elektrische Leistung",
                "axis_unit": "RMSE $P_{el}$ [%]",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
            {
                "train_col": "train_rmse_e_T_dis_K",
                "val_col": "val_rmse_e_T_dis_K",
                "title": "RMSE Austrittstemperatur",
                "axis_unit": "RMSE $T_{dis}$ [K]",
                "scale": 1.0,
                "fmt": "{:.2f}",
            },
        ]

    if m in ("mae_combined", "combined_mae"):
        return [
            {
                "train_col": "train_mae_combined",
                "val_col": "val_mae_combined",
                "title": "Aggregierter MAE\n(Ø über $\\dot{m}$, $P_{el}$, $T_{dis}/T_{norm}$)",
                "axis_unit": "Ø MAE [%]",
                "scale": 100.0,
                "fmt": "{:.2f}",
            },
        ]

    if m in ("rmse_combined", "combined_rmse"):
        return [
            {
                "train_col": "train_rmse_combined",
                "val_col": "val_rmse_combined",
                "title": "Aggregierter RMSE\n(Ø über $\\dot{m}$, $P_{el}$, $T_{dis}/T_{norm}$)",
                "axis_unit": "Ø RMSE [%]",
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
def plot_scatter_cell(
    ax,
    paired: pd.DataFrame,
    train_col: str,
    val_col: str,
    title: str,
    axis_unit: str,
    scale: float,
):
    """
    Scatter plot on one subplot: train error vs. validation error.
    """
    if train_col not in paired.columns or val_col not in paired.columns:
        ax.set_title(f"{title}\n(keine Daten)")
        return

    # Extract data
    x = paired[train_col].to_numpy(dtype=float) * scale
    y = paired[val_col].to_numpy(dtype=float) * scale
    finite_mask = np.isfinite(x) & np.isfinite(y)

    if not np.any(finite_mask):
        ax.set_title(f"{title}\n(keine gültigen Daten)")
        return

    # Axis limits: include origin, leave headroom
    all_vals = np.concatenate([x[finite_mask], y[finite_mask]])
    max_val = float(np.max(all_vals))
    lo = 0.0
    hi = max_val * 1.15 if max_val > 0 else 1.0

    # Diagonal reference
    diag = np.linspace(lo, hi, 100)
    ax.plot(diag, diag, color="black", linewidth=1.2, linestyle="--",
            label="_nolegend_", zorder=1)

    # Plot each point individually so we can control color and marker
    for _, row in paired.iterrows():
        tr = row[train_col] * scale
        vl = row[val_col] * scale
        if not (np.isfinite(tr) and np.isfinite(vl)):
            continue

        model = str(row["model_norm"])
        params_oil = str(row["params_oil_norm"])

        color = MODEL_COLORS.get(model, "gray")
        marker = PARAMS_OIL_MARKERS.get(params_oil, "D")

        ax.scatter(
            tr, vl,
            s=90,
            color=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            zorder=3,
        )

    # Formatting
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f"Training: {axis_unit}")
    ax.set_ylabel(f"Validation: {axis_unit}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.6, alpha=0.35)

    # Annotation: "Overfitting" region (above diagonal)
    ax.text(
        0.98, 0.02,
        "unterhalb: Val. besser",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9, color="gray", style="italic",
    )
    ax.text(
        0.02, 0.98,
        "oberhalb: Overfitting",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9, color="gray", style="italic",
    )


def build_legend(fig, paired: pd.DataFrame):
    """
    Build combined legend showing models (colors) and params_oils (markers).
    """
    legend_elements = []

    # Models (colors)
    unique_models = sorted(paired["model_norm"].unique())
    for model in unique_models:
        if model in MODEL_COLORS:
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="white",
                    markerfacecolor=MODEL_COLORS[model],
                    markeredgecolor="black",
                    markersize=10,
                    label=f"{model.capitalize()}",
                )
            )

    # Spacer
    legend_elements.append(Line2D([0], [0], color="none", label=""))

    # Params oils (markers, shown in gray)
    unique_params_oils = [o for o in ["LPG68", "LPG100", "all"]
                          if o in paired["params_oil_norm"].unique()]
    for oil in unique_params_oils:
        marker = PARAMS_OIL_MARKERS.get(oil, "D")
        legend_elements.append(
            Line2D(
                [0], [0],
                marker=marker,
                color="white",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=10,
                label=f"Params: {OIL_DISPLAY.get(oil, oil)}",
            )
        )

    # Spacer + diagonal
    legend_elements.append(Line2D([0], [0], color="none", label=""))
    legend_elements.append(
        Line2D(
            [0], [0],
            color="black", linestyle="--", linewidth=1.2,
            label="Training = Validation",
        )
    )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(legend_elements),
        frameon=True,
        fontsize=10,
    )


def plot_train_vs_validation(
    paired: pd.DataFrame,
    metric: str,
    out_path: Path,
):
    """
    Main plot function: creates figure with 1 or 3 subplots depending on metric.
    """
    metric_cfgs = get_metric_config(metric)
    n_plots = len(metric_cfgs)

    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(19, 7))

    for idx, cfg in enumerate(metric_cfgs):
        plot_scatter_cell(
            ax=axes[idx],
            paired=paired,
            train_col=cfg["train_col"],
            val_col=cfg["val_col"],
            title=cfg["title"],
            axis_unit=cfg["axis_unit"],
            scale=cfg["scale"],
        )

    build_legend(fig, paired)

    metric_title_map = {
        "mae": "MAE pro Zielgröße",
        "rmse": "RMSE pro Zielgröße",
        "mae_combined": "Aggregierter MAE",
        "combined_mae": "Aggregierter MAE",
        "rmse_combined": "Aggregierter RMSE",
        "combined_rmse": "Aggregierter RMSE",
    }
    metric_title = metric_title_map.get(metric.lower(), metric.upper())
    fig.suptitle(
        f"Training vs. Validation ({metric_title})",
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
        description="Scatter plot of training vs validation errors across model configurations."
    )
    ap.add_argument("--summary_dir", required=True, type=Path,
                    help="Directory containing validation_summary_*.csv files (with train_only and validation_only runs)")
    ap.add_argument("--metric", required=True,
                    choices=["mae", "rmse", "mae_combined", "rmse_combined"],
                    help="Which error metric to plot. "
                         "'mae'/'rmse' = 3 scatter plots (one per target). "
                         "'mae_combined'/'rmse_combined' = 1 scatter plot aggregating all three targets.")
    ap.add_argument("--model", default=None,
                    help="Filter: only include this model (original | modified | oil_path)")
    ap.add_argument("--Tdis_norm_K", type=float, default=50.0,
                    help="Normalization for T_dis in combined metrics (default 50 K)")
    ap.add_argument("--exclude_all_oil", action="store_true",
                    help="Exclude configurations where validation_oil=all")
    ap.add_argument("--out_dir", default="results/train_vs_validation")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    if not args.summary_dir.exists():
        raise FileNotFoundError(args.summary_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load and pair
    # -------------------------
    df = load_summaries(args.summary_dir)

    # Filter by model if requested
    if args.model is not None:
        before = len(df)
        df = df[df["model_norm"] == args.model.lower()].copy()
        print(f"  Filtered to model '{args.model}': {len(df)}/{before} rows")

    # Pair train and validation rows
    paired = pair_train_validation(df)

    # Filter validation_oil=all if requested
    if args.exclude_all_oil:
        before = len(paired)
        paired = paired[paired["validation_oil_norm"] != "all"].copy()
        print(f"  Excluded validation_oil=all: {len(paired)}/{before} configurations")

    if paired.empty:
        raise ValueError("No configurations remaining after filtering.")

    # Compute combined metrics if needed
    if args.metric.lower() in ("mae_combined", "rmse_combined", "combined_mae", "combined_rmse"):
        paired = add_combined_metrics(paired, Tdis_norm_K=args.Tdis_norm_K)
        print(f"  Tdis_norm_K (combined): {args.Tdis_norm_K} K")

    # Report what's plotted
    print(f"  Plotting {len(paired)} configurations:")
    for _, row in paired.iterrows():
        print(f"    {row['model_norm']:10s} | "
              f"Params: {row['params_oil_norm']:8s} | "
              f"Data: {row['validation_oil_norm']}")

    # -------------------------
    # Plot
    # -------------------------
    stamp = _ts()
    model_tag = f"_{args.model}" if args.model else ""
    excl_tag = "_no_all" if args.exclude_all_oil else ""
    out_path = out_dir / f"train_vs_val_{args.metric}{model_tag}{excl_tag}_{stamp}.{args.out_format}"

    plot_train_vs_validation(
        paired=paired,
        metric=args.metric,
        out_path=out_path,
    )

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
