# scripts/plotting_scripts/model_comparison_radar.py
#
# Radar plot comparing multiple compressor models across several error metrics.
# Each model is represented as a polygon; "outside = better" convention.
#
# Two configurations:
#   summary  (4 axes): James-Score, mean MAE_combined, mean RMSE_combined,
#                       share within ±3% / ±3K (all three metrics combined)
#   detailed (6 axes): MAE m_dot, RMSE m_dot, MAE P_el, RMSE P_el,
#                       MAE T_dis, RMSE T_dis
#
# Input: validation_summary_*.csv files from one or more models.
# The script auto-detects the model from each summary CSV's 'model' column
# and groups them. Multiple summaries per model (e.g. different validation
# oils) can be averaged with --aggregate mean.
#
# Examples:
#   # Compare three models (one summary CSV per model):
#   python scripts/plotting_scripts/model_comparison_radar.py --csv data/train_vs_validation/validation_summary_params_lpg68_val_lpg68_original_2026-04-13_143206.csv data/train_vs_validation/validation_summary_params_lpg68_val_lpg68_modified_2026-04-13_135214.csv results/validation/validation_summary_params_lpg68_val_lpg68_oil_path_2026-04-13_145435.csv --config summary
#
#   # Compare models with detailed axes:
#   python scripts/plotting_scripts/model_comparison_radar.py --csv data/train_vs_validation/validation_summary_params_lpg68_val_lpg68_original_2026-04-13_143206.csv data/train_vs_validation/validation_summary_params_lpg68_val_lpg68_modified_2026-04-13_135214.csv results/validation/validation_summary_params_lpg68_val_lpg68_oil_path_2026-04-13_145435.csv --config detailed
#
#   # Average over multiple validation cases per model:
#   python scripts/plotting_scripts/model_comparison_radar.py --csv data/train_vs_validation/validation_summary_params_lpg68_val_lpg68_original_2026-04-13_143206.csv data/train_vs_validation/validation_summary_params_lpg68_val_lpg68_modified_2026-04-13_135214.csv results/validation/validation_summary_params_lpg68_val_lpg68_oil_path_2026-04-13_145435.csv --aggregate mean

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ebc.paper.mplstyle")


# =========================================================
# Helpers
# =========================================================
def _ts():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _norm_oil(s):
    s = str(s).strip()
    low = s.lower().replace(" ", "")
    if low in ("lpg68", "lpg 68"): return "LPG68"
    if low in ("lpg100", "lpg 100"): return "LPG100"
    if low == "all": return "all"
    return s


# =========================================================
# Axis configurations
# =========================================================
# Each axis: (column_name_in_csv, display_label, scale_factor, is_higher_better)
# scale_factor converts raw value to display unit (e.g., 100 for fraction → percent)

AXES_SUMMARY = [
    ("james_error_mean", "James-Score",                          1.0,   False),
    ("_combined_mae",    "Ø MAE (kombiniert) [%]",               100.0, False),
    ("_combined_rmse",   "Ø RMSE (kombiniert) [%]",              100.0, False),
    ("_share_all_3",     "Anteil 'gut' (±3% / ±3K) [%]",         100.0, True),
]

AXES_DETAILED = [
    ("mae_e_m_rel",      "MAE $\\dot{m}$ [%]",                   100.0, False),
    ("rmse_e_m_rel",     "RMSE $\\dot{m}$ [%]",                  100.0, False),
    ("mae_e_P_rel",      "MAE $P_{el}$ [%]",                     100.0, False),
    ("rmse_e_P_rel",     "RMSE $P_{el}$ [%]",                    100.0, False),
    ("mae_e_T_dis_K",    "MAE $T_{dis}$ [K]",                    1.0,   False),
    ("rmse_e_T_dis_K",   "RMSE $T_{dis}$ [K]",                   1.0,   False),
]


# =========================================================
# Data loading
# =========================================================
def load_summaries(csv_paths: list[Path]) -> pd.DataFrame:
    """Load one or more validation_summary CSVs into a single DataFrame."""
    dfs = []
    for p in csv_paths:
        if not p.exists():
            print(f"  [WARN] Skipping missing file: {p}")
            continue
        try:
            df = pd.read_csv(p)
            df["_source_file"] = p.name
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Could not read {p.name}: {e}")

    if not dfs:
        raise ValueError("No summary CSVs could be loaded.")

    combined = pd.concat(dfs, ignore_index=True)
    combined["model_norm"] = combined["model"].astype(str).str.lower().str.strip()
    if "params_oil" in combined.columns:
        combined["params_oil_norm"] = combined["params_oil"].apply(_norm_oil)
    if "validation_oil" in combined.columns:
        combined["validation_oil_norm"] = combined["validation_oil"].apply(_norm_oil)

    return combined


def load_summaries_from_dir(summary_dir: Path) -> pd.DataFrame:
    """Load all validation_summary_*.csv files from a directory."""
    csv_files = sorted(summary_dir.glob("validation_summary_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No validation_summary_*.csv files in {summary_dir}")
    print(f"  Found {len(csv_files)} summary files in {summary_dir}")
    return load_summaries(csv_files)


# =========================================================
# Combined metrics computation
# =========================================================
def add_combined_metrics(df: pd.DataFrame, Tdis_norm_K: float = 50.0) -> pd.DataFrame:
    """
    Add combined metrics to the DataFrame:
      _combined_mae:  mean of (MAE_m_rel, MAE_P_rel, MAE_T_dis/Tdis_norm)
      _combined_rmse: mean of (RMSE_m_rel, RMSE_P_rel, RMSE_T_dis/Tdis_norm)
      _share_all_3:   share of points where m_dot, P_el AND T_dis are all
                      within their respective tolerance bands (3% / 3K)
    """
    # Use Tdis_norm_K from CSV if available
    if "Tdis_norm_K" in df.columns:
        df["_Tdis_norm"] = pd.to_numeric(df["Tdis_norm_K"], errors="coerce").fillna(Tdis_norm_K)
    else:
        df["_Tdis_norm"] = Tdis_norm_K

    # Combined MAE (averaged, T_dis normalized to fraction equivalent)
    df["_combined_mae"] = (
        df["mae_e_m_rel"]
        + df["mae_e_P_rel"]
        + df["mae_e_T_dis_K"] / df["_Tdis_norm"]
    ) / 3.0

    df["_combined_rmse"] = (
        df["rmse_e_m_rel"]
        + df["rmse_e_P_rel"]
        + df["rmse_e_T_dis_K"] / df["_Tdis_norm"]
    ) / 3.0

    # Share of points "good in all three" — approximate via the minimum of the
    # three individual shares (an upper bound on the joint share).
    # If you have detail CSVs you could compute the exact joint share; here we
    # estimate it as the minimum of the three individual shares.
    if all(c in df.columns for c in ["share_m_within_3pct", "share_P_within_3pct", "share_Tdis_within_3K"]):
        df["_share_all_3"] = df[
            ["share_m_within_3pct", "share_P_within_3pct", "share_Tdis_within_3K"]
        ].min(axis=1)
    else:
        df["_share_all_3"] = np.nan

    return df


# =========================================================
# Aggregation
# =========================================================
def aggregate_per_model(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Aggregate multiple rows per model into a single row.
    mode: 'mean' | 'median' | 'first'
    """
    metric_cols = [c for c in df.columns if c.startswith(("mae_", "rmse_", "mean_", "share_", "james_", "_combined_", "_share_"))]

    if mode == "first":
        agg = df.groupby("model_norm").first().reset_index()
    elif mode == "median":
        agg = df.groupby("model_norm")[metric_cols].median().reset_index()
    else:  # default: mean
        agg = df.groupby("model_norm")[metric_cols].mean().reset_index()

    return agg


# =========================================================
# Radar plot
# =========================================================
MODEL_DISPLAY = {
    "original": "Original",
    "modified": "Modified",
    "oil_path": "Oil Path",
    "oilpath":  "Oil Path",
}

MODEL_COLORS = {
    "original": "#EC635C",
    "modified": "#4B81C4",
    "oil_path": "#F49961",
    "oilpath":  "#F49961",
}


def plot_radar(
    agg_df: pd.DataFrame,
    axes_config: list,
    title: str,
    out_path: Path,
):
    """
    Create radar plot. agg_df must have one row per model.
    axes_config: list of (col, label, scale, is_higher_better)
    """
    n_axes = len(axes_config)
    if n_axes < 3:
        raise ValueError("Radar plot needs at least 3 axes.")

    # Extract values per model
    model_values = {}
    for _, row in agg_df.iterrows():
        model = str(row["model_norm"])
        values = []
        for col, _label, scale, _ in axes_config:
            if col not in agg_df.columns:
                values.append(np.nan)
                continue
            v = row.get(col)
            if pd.isna(v):
                values.append(np.nan)
            else:
                values.append(float(v) * scale)
        model_values[model] = values

    if not model_values:
        raise ValueError("No model data to plot.")

    # Per-axis normalization to [0, 1] with "outside = better" convention
    # For "lower is better" axes: invert
    # For "higher is better" axes: keep as-is, but normalize
    all_values = np.array(list(model_values.values()))  # shape (n_models, n_axes)

    normalized = np.zeros_like(all_values)
    axis_min = np.zeros(n_axes)
    axis_max = np.zeros(n_axes)

    for i, (_col, _label, _scale, is_higher_better) in enumerate(axes_config):
        col_vals = all_values[:, i]
        finite = col_vals[np.isfinite(col_vals)]

        if len(finite) == 0:
            axis_min[i] = 0.0
            axis_max[i] = 1.0
            normalized[:, i] = 0.0
            continue

        v_min = float(np.min(finite))
        v_max = float(np.max(finite))
        axis_min[i] = v_min
        axis_max[i] = v_max

        if v_max == v_min:
            normalized[:, i] = 0.5  # all equal → middle
            continue

        if is_higher_better:
            # Higher raw value → larger radial position
            normalized[:, i] = (col_vals - v_min) / (v_max - v_min)
        else:
            # Lower raw value → larger radial position (invert)
            normalized[:, i] = (v_max - col_vals) / (v_max - v_min)

    # Replace NaN with 0 for plotting
    normalized = np.nan_to_num(normalized, nan=0.0)

    # Angles
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    # Plot
    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(polar=True))

    # Draw axis grid (radial)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Axis labels
    axis_labels = []
    for i, (_col, label, _scale, _) in enumerate(axes_config):
        v_min = axis_min[i]
        v_max = axis_max[i]
        # Annotate axis with min/max on the periphery
        axis_labels.append(f"{label}\n[{v_min:.3g} ... {v_max:.3g}]")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=11)

    # Radial ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="gray")
    ax.set_ylim(0, 1.05)

    # Reference grid
    ax.grid(True, linewidth=0.6, alpha=0.4)

    # Plot each model
    for model, raw_vals in model_values.items():
        idx = list(agg_df["model_norm"]).index(model)
        norm_vals = normalized[idx, :].tolist()
        norm_vals += norm_vals[:1]  # close loop

        color = MODEL_COLORS.get(model, "gray")
        display_name = MODEL_DISPLAY.get(model, model.capitalize())

        ax.plot(angles, norm_vals, color=color, linewidth=2.5,
                marker="o", markersize=8, label=display_name, zorder=3)
        ax.fill(angles, norm_vals, color=color, alpha=0.20, zorder=2)

    ax.set_title(title, fontsize=14, pad=30)

    # Legend below the plot
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=len(model_values),
        frameon=True,
        fontsize=12,
    )

    # Subtitle / note
    fig.text(
        0.5, 0.02,
        "außen = besser  |  jede Achse einzeln normalisiert auf [Min, Max] über die verglichenen Modelle",
        ha="center", va="bottom",
        fontsize=10, style="italic", color="gray",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Radar plot comparing models across multiple error metrics."
    )

    # Input: either explicit list of CSVs or a directory
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, nargs="+",
                     help="One or more validation_summary_*.csv files")
    src.add_argument("--summary_dir", type=Path,
                     help="Directory containing validation_summary_*.csv files")

    ap.add_argument(
        "--config", default="summary",
        choices=["summary", "detailed"],
        help="Axis configuration: summary (4 axes) or detailed (6 axes)",
    )

    ap.add_argument(
        "--aggregate", default="mean",
        choices=["mean", "median", "first"],
        help="How to aggregate multiple rows per model (default: mean)",
    )

    ap.add_argument(
        "--selection_mode_filter", default=None,
        help="Optional: filter to selection_mode (e.g. 'validation_only', 'train_only')",
    )

    ap.add_argument("--Tdis_norm_K", type=float, default=50.0,
                    help="Normalization for T_dis in combined MAE/RMSE (default 50 K)")

    ap.add_argument("--out_dir", default="results/model_comparison_radar")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    if args.csv:
        df = load_summaries(args.csv)
    else:
        df = load_summaries_from_dir(args.summary_dir)

    # Optional filter
    if args.selection_mode_filter is not None:
        if "selection_mode" in df.columns:
            before = len(df)
            df = df[
                df["selection_mode"].astype(str).str.lower().str.strip()
                == args.selection_mode_filter.lower()
            ].copy()
            print(f"  Filtered to selection_mode='{args.selection_mode_filter}': "
                  f"{len(df)}/{before} rows")
        else:
            print("  [WARN] No 'selection_mode' column — filter ignored.")

    if df.empty:
        raise ValueError("No data after filtering.")

    # Compute combined metrics
    df = add_combined_metrics(df, Tdis_norm_K=args.Tdis_norm_K)

    # Report what's there
    unique_models = sorted(df["model_norm"].unique())
    print(f"  Models found: {unique_models}")
    for model in unique_models:
        n = (df["model_norm"] == model).sum()
        print(f"    {model}: {n} row(s)")

    if len(unique_models) < 2:
        print("  [WARN] Only one model found — radar plot needs at least 2 models for comparison.")

    # Aggregate
    agg_df = aggregate_per_model(df, mode=args.aggregate)
    print(f"  Aggregated using: {args.aggregate}")

    # -------------------------
    # Select axis config
    # -------------------------
    axes_config = AXES_SUMMARY if args.config == "summary" else AXES_DETAILED
    print(f"  Config: {args.config} ({len(axes_config)} axes)")

    # Check that all required columns exist
    missing = []
    for col, label, _, _ in axes_config:
        if col not in agg_df.columns:
            missing.append((col, label))

    if missing:
        print(f"  [WARN] Missing columns for some axes:")
        for col, label in missing:
            print(f"    - {label} (column: {col})")

    # -------------------------
    # Build title
    # -------------------------
    title = f"Modellvergleich — Radar ({args.config.capitalize()})"

    # Build a brief context line based on what's filtered
    context_parts = []
    if "validation_oil_norm" in df.columns:
        val_oils = sorted(df["validation_oil_norm"].dropna().unique())
        if len(val_oils) == 1:
            context_parts.append(f"Daten: {val_oils[0]}")
        elif len(val_oils) > 1:
            context_parts.append(f"Daten: {', '.join(val_oils)}")
    if "params_oil_norm" in df.columns:
        params_oils = sorted(df["params_oil_norm"].dropna().unique())
        if len(params_oils) == 1:
            context_parts.append(f"Params: {params_oils[0]}")
        elif len(params_oils) > 1:
            context_parts.append(f"Params: gemittelt über {', '.join(params_oils)}")
    if args.selection_mode_filter:
        context_parts.append(f"Mode: {args.selection_mode_filter}")
    if args.aggregate != "first":
        context_parts.append(f"Aggregation: {args.aggregate}")

    if context_parts:
        title += "\n" + " | ".join(context_parts)

    # -------------------------
    # Plot
    # -------------------------
    stamp = _ts()
    out_path = out_dir / f"radar_{args.config}_{stamp}.{args.out_format}"

    plot_radar(
        agg_df=agg_df,
        axes_config=axes_config,
        title=title,
        out_path=out_path,
    )

    # Save aggregated data
    data_csv = out_path.with_suffix(".csv")
    agg_df.to_csv(data_csv, index=False)
    print(f"  [OK] Aggregated data saved: {data_csv}")

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
