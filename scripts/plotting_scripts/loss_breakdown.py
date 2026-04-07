# scripts/plotting_scripts/loss_breakdown.py
#
# Stacked bar chart showing the breakdown of compressor loss terms per
# operating point, sortable by different operating parameters.
#
# Input: validation detail CSV from validation.py (must contain loss columns)
#
# Examples:
#   # Sort by pressure ratio (no op_rows needed):
#   python scripts/plotting_scripts/loss_breakdown.py \
#       --csv results/validation/validation_detail_params_lpg68_val_lpg68_modified_2026-03-19.csv \
#       --sort_by pressure_ratio
#
#   # Sort by evaporation temperature (requires op_rows for T_evap):
#   python scripts/plotting_scripts/loss_breakdown.py \
#       --csv results/validation/validation_detail_params_lpg68_val_lpg68_modified_2026-03-19.csv \
#       --op_rows_csv results/split_template/operating_points_rows.csv \
#       --sort_by T_evap
#
#   # Sort by condensation temperature, normalized, only validation points:
#    python scripts/plotting_scripts/loss_breakdown.py --csv results/final_results/Modified_All/Validation/validation_detail_params_all_val_lpg100_modified_2026-04-02_103908.csv --op_rows_csv results/split_template/operating_points_rows_2026-03-12_112331.csv --sort_by pressure_ratio --selection_mode validation_only --normalize

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ebc.paper.mplstyle")


# =========================================================
# Helpers
# =========================================================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_split_role(df: pd.DataFrame) -> np.ndarray | None:
    if "split_role" in df.columns:
        return df["split_role"].fillna("").astype(str).to_numpy()
    if "is_train" in df.columns:
        return np.where(df["is_train"].fillna(False).astype(bool), "train", "validation")
    return None


def _auto_subtitle(df: pd.DataFrame) -> str:
    parts = []
    if "model" in df.columns:
        vals = df["model"].dropna().unique()
        if len(vals) == 1:
            parts.append(str(vals[0]).capitalize())

    if "params_oil" in df.columns:
        vals = df["params_oil"].dropna().unique()
        if len(vals) == 1:
            params_oil = str(vals[0])
        else:
            params_oil = None
    else:
        params_oil = None

    if "oil" in df.columns:
        vals = df["oil"].dropna().unique()
        if len(vals) == 1:
            val_oil = str(vals[0])
        elif len(vals) > 1:
            val_oil = "all"
        else:
            val_oil = None
    else:
        val_oil = None

    if params_oil and val_oil:
        parts.append(f"Params: {params_oil} → Data: {val_oil}")
    elif params_oil:
        parts.append(f"Params: {params_oil}")
    elif val_oil:
        parts.append(f"Öl: {val_oil}")

    return " | ".join(parts) if parts else ""


# =========================================================
# Sort column resolution
# =========================================================
SORT_CONFIG = {
    "pressure_ratio": {
        "candidates": ["pressure_ratio"],
        "label": "Druckverhältnis $p_{aus}/p_{ein}$",
        "compute": None,
    },
    "superheat": {
        "candidates": ["superheat_C", "T1_SH"],
        "label": "Überhitzung [K]",
        "compute": None,
    },
    "T_evap": {
        "candidates": ["T_evap", "T_evap_set_C"],
        "label": "Verdampfungstemperatur [°C]",
        "compute": None,
    },
    "T_cond": {
        "candidates": ["T_cond", "T_cond_set_C"],
        "label": "Kondensationstemperatur [°C]",
        "compute": None,
    },
    "speed": {
        "candidates": ["N_rpm", "N_rpm_in", "N"],
        "label": "Drehzahl [rpm]",
        "compute": None,
    },
    "f_hz": {
        "candidates": ["f_oper_hz"],
        "label": "Drehfrequenz [Hz]",
        "compute": None,
    },
}


def _resolve_sort_column(df: pd.DataFrame, sort_by: str) -> tuple[str, str]:
    """
    Returns (column_name, axis_label) for the chosen sort parameter.
    If the column doesn't exist, tries to compute it.
    """
    cfg = SORT_CONFIG.get(sort_by)
    if cfg is None:
        # Try direct column name
        if sort_by in df.columns:
            return sort_by, sort_by
        raise ValueError(
            f"Unknown --sort_by '{sort_by}'. "
            f"Available: {', '.join(SORT_CONFIG.keys())} or any column name in CSV."
        )

    col = _find_col(df, cfg["candidates"])
    if col is not None:
        return col, cfg["label"]

    # Try to compute
    if sort_by == "pressure_ratio":
        p_out = _find_col(df, ["p_out_bar", "p_out_bar_in"])
        p_suc = _find_col(df, ["p_suc_bar", "p_suc_bar_in"])
        if p_out and p_suc:
            df["pressure_ratio"] = df[p_out] / df[p_suc]
            return "pressure_ratio", cfg["label"]

    if sort_by == "T_evap":
        # Approximate from saturation temperature at suction pressure
        # (only if T_sat_suc_C exists)
        t_sat = _find_col(df, ["T_sat_suc_C"])
        if t_sat:
            return t_sat, "Sättigungstemperatur Saugseite [°C]"

    if sort_by == "speed":
        f_col = _find_col(df, ["f_oper_hz"])
        if f_col:
            df["N_rpm_computed"] = df[f_col] * 60.0
            return "N_rpm_computed", cfg["label"]

    raise ValueError(
        f"Cannot resolve sort column for '{sort_by}'. "
        f"Tried: {cfg['candidates']}. None found in CSV columns."
        + (
            " Use --op_rows_csv to provide T_evap/T_cond from operating_points_rows.csv."
            if sort_by in {"T_evap", "T_cond"}
            else ""
        )
    )


# =========================================================
# Loss column detection
# =========================================================
def _detect_loss_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Returns list of (column_name, display_label) for available loss terms.
    Order: load-dependent, speed-dependent, friction (if modified model).
    """
    losses = []

    if "W_dot_loss_load_W" in df.columns:
        losses.append(("W_dot_loss_load_W", "Lastabhängig ($\\alpha_{loss} \\cdot \\dot{W}_{int}$)"))

    if "W_dot_loss_ref_term_W" in df.columns:
        losses.append(("W_dot_loss_ref_term_W", "Drehzahlabhängig ($\\dot{W}_{loss,ref} \\cdot (f/f_{ref})^2$)"))

    if "W_dot_loss_fric_W" in df.columns:
        # Check if any finite values exist (only for modified model)
        vals = pd.to_numeric(df["W_dot_loss_fric_W"], errors="coerce")
        if vals.notna().any() and (vals > 0).any():
            losses.append(("W_dot_loss_fric_W", "Viskositätsreibung ($\\alpha_{fric} \\cdot \\mu_{mix} \\cdot V_h \\cdot \\omega^2$)"))

    if not losses:
        raise ValueError(
            "No loss term columns found in CSV. "
            "Expected: W_dot_loss_load_W, W_dot_loss_ref_term_W, W_dot_loss_fric_W. "
            "Run validation.py with the updated Molinaroli model first."
        )

    return losses


# =========================================================
# Selection
# =========================================================
def _select_rows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    roles = _detect_split_role(df)

    if mode == "all":
        return df.copy()
    elif mode in {"train_only", "train"}:
        if roles is None:
            raise ValueError("No split info found for selection_mode='train_only'.")
        return df[roles == "train"].copy()
    elif mode in {"validation_only", "validation"}:
        if roles is None:
            raise ValueError("No split info found for selection_mode='validation_only'.")
        return df[roles == "validation"].copy()
    else:
        raise ValueError(f"Unknown selection_mode: {mode}")


# =========================================================
# Stacked bar plot
# =========================================================
def plot_loss_breakdown(
    df: pd.DataFrame,
    loss_cols: list[tuple[str, str]],
    sort_col: str,
    sort_label: str,
    title: str,
    out_path: Path,
    normalize: bool = False,
    max_points: int | None = None,
):
    """
    Create stacked bar chart of loss terms, sorted by sort_col.
    """
    # Prepare data
    plot_df = df.copy()

    # Ensure loss columns are numeric
    for col, _ in loss_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0)

    # Ensure sort column is numeric
    plot_df[sort_col] = pd.to_numeric(plot_df[sort_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[sort_col]).copy()

    # Sort
    plot_df = plot_df.sort_values(sort_col).reset_index(drop=True)

    # Limit number of points if too many
    if max_points is not None and len(plot_df) > max_points:
        # Subsample evenly
        idx = np.round(np.linspace(0, len(plot_df) - 1, max_points)).astype(int)
        plot_df = plot_df.iloc[idx].reset_index(drop=True)
        print(f"  [INFO] Subsampled to {max_points} points for readability.")

    n_bars = len(plot_df)
    if n_bars == 0:
        print("  [SKIP] No data to plot.")
        return

    # Extract loss values
    loss_data = {}
    for col, label in loss_cols:
        loss_data[label] = plot_df[col].to_numpy(dtype=float)

    # Normalize to 100% if requested
    if normalize:
        total = np.zeros(n_bars)
        for label in loss_data:
            total += np.abs(loss_data[label])
        total = np.where(total > 0, total, 1.0)  # avoid div by zero
        for label in loss_data:
            loss_data[label] = np.abs(loss_data[label]) / total * 100.0

    # Sort values for x-axis labels
    sort_values = plot_df[sort_col].to_numpy(dtype=float)

    # op_id for bar labels (append oil tag if duplicate op_ids exist)
    has_op_id = "op_id" in plot_df.columns
    if has_op_id:
        op_ids_raw = plot_df["op_id"].astype(str).to_numpy()
        has_duplicates = len(set(op_ids_raw)) < len(op_ids_raw)

        if has_duplicates and "oil_norm" in plot_df.columns:
            oil_tags = plot_df["oil_norm"].astype(str).to_numpy()
            op_ids = np.array([f"{oid}\n({oil})" for oid, oil in zip(op_ids_raw, oil_tags)])
        elif has_duplicates and "oil" in plot_df.columns:
            oil_tags = plot_df["oil"].astype(str).to_numpy()
            op_ids = np.array([f"{oid}\n({oil})" for oid, oil in zip(op_ids_raw, oil_tags)])
        else:
            op_ids = op_ids_raw
    else:
        op_ids = None

    # Plot
    fig_width = max(10, min(24, n_bars * 0.35 + 4))
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    x = np.arange(n_bars)
    bar_width = 0.8

    bottom = np.zeros(n_bars)
    colors = ["#EC635C", "#4B81C4", "#F49961"]  # from EBC style: red, blue, orange

    for i, (label, values) in enumerate(loss_data.items()):
        color = colors[i % len(colors)]
        ax.bar(
            x, values, bar_width,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.3,
        )
        bottom += values

    # Add op_id labels inside each bar (vertical, centered)
    if op_ids is not None:
        # Adaptive font size based on number of bars
        if n_bars <= 20:
            label_fontsize = 8
        elif n_bars <= 40:
            label_fontsize = 6
        else:
            label_fontsize = 5

        for j in range(n_bars):
            bar_height = bottom[j]
            if bar_height > 0:
                ax.text(
                    x[j], bar_height * 0.5,
                    op_ids[j],
                    ha="center", va="center",
                    rotation=90,
                    fontsize=label_fontsize,
                    color="white",
                    fontweight="bold",
                    clip_on=True,
                )

    # X-axis: show sort values as labels
    if n_bars <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}" for v in sort_values],
            rotation=45, ha="right", fontsize=10,
        )
        ax.set_xlabel(sort_label)
    else:
        # Too many bars for individual labels → use secondary x-axis annotation
        ax.set_xticks([])
        ax.set_xlabel(f"Betriebspunkte sortiert nach {sort_label}")

        # Add sort value as color-coded line below
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([])

        # Annotate min/max
        ax.text(
            0.0, -0.08, f"{sort_values[0]:.1f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
        )
        ax.text(
            1.0, -0.08, f"{sort_values[-1]:.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
        )
        ax.text(
            0.5, -0.08, f"← {sort_label} →",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
        )

    if normalize:
        ax.set_ylabel("Anteil an Gesamtverlust [%]")
        ax.set_ylim(0, 105)
    else:
        ax.set_ylabel("Verlustleistung [W]")

    ax.set_title(title)

    # Legend below the plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(loss_cols),
        frameon=True,
        fontsize=11,
    )

    # Info text
    if not normalize:
        total_loss = bottom
        mean_total = float(np.mean(total_loss))
        info = f"Mittlere Gesamtverlustleistung: {mean_total:.1f} W  |  n={n_bars} Punkte"
        ax.text(
            0.98, 0.98, info,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.7"),
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
        description="Stacked bar chart of compressor loss term breakdown, sortable by operating parameters."
    )
    ap.add_argument("--csv", required=True, type=Path, help="Validation detail CSV")
    ap.add_argument("--op_rows_csv", type=Path, default=None,
                    help="Optional: operating_points_rows.csv to get T_evap, T_cond columns")
    ap.add_argument("--out_dir", default="results/loss_breakdown", help="Output directory")

    ap.add_argument(
        "--sort_by",
        default="pressure_ratio",
        help="Sort bars by: pressure_ratio, superheat, T_evap, T_cond, speed, f_hz, or any CSV column name",
    )

    ap.add_argument(
        "--selection_mode",
        default="all",
        choices=["all", "train_only", "validation_only"],
        help="Which points to include",
    )

    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize loss terms to 100%% (show relative contribution)",
    )

    ap.add_argument("--max_points", type=int, default=None,
                    help="Limit number of bars for readability (evenly subsampled)")

    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    if args.op_rows_csv is not None and not args.op_rows_csv.exists():
        raise FileNotFoundError(args.op_rows_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # --- Merge T_evap / T_cond from op_rows if provided ---
    if args.op_rows_csv is not None and "op_id" in df.columns:
        op_df = pd.read_csv(args.op_rows_csv)

        # Identify columns to merge (T_evap, T_cond, and related set-point columns)
        merge_candidates = ["T_evap", "T_evap_set_C", "T_cond", "T_cond_set_C", "T1_SH", "Drehzahl"]
        available_merge_cols = [c for c in merge_candidates if c in op_df.columns and c not in df.columns]

        if available_merge_cols and "op_id" in op_df.columns:
            # Deduplicate op_rows (may have multiple rows per op_id for different oils)
            op_unique = op_df.drop_duplicates(subset=["op_id"])[["op_id"] + available_merge_cols].copy()
            df = df.merge(op_unique, on="op_id", how="left")
            print(f"  Merged from op_rows: {available_merge_cols}")
        else:
            print("  [INFO] No additional columns to merge from op_rows_csv.")

    # Filter successful points
    if "success" in df.columns:
        df = df[df["success"] == True].copy()

    # Selection
    df = _select_rows(df, args.selection_mode)
    print(f"  Selected {len(df)} points (mode: {args.selection_mode})")

    # Detect loss columns
    loss_cols = _detect_loss_columns(df)
    print(f"  Loss terms found: {[label for _, label in loss_cols]}")

    # Resolve sort column
    sort_col, sort_label = _resolve_sort_column(df, args.sort_by)
    print(f"  Sorting by: {sort_col} ({sort_label})")

    # Title
    subtitle = _auto_subtitle(df)
    norm_tag = " (normiert)" if args.normalize else ""
    title = f"Verlustaufteilung{norm_tag} — sortiert nach {sort_label}"
    if subtitle:
        title += f"\n{subtitle}"

    # Plot
    stamp = _ts()
    sort_tag = args.sort_by.replace(" ", "_").lower()
    norm_suffix = "_norm" if args.normalize else ""
    out_path = out_dir / f"loss_breakdown_{sort_tag}{norm_suffix}_{stamp}.{args.out_format}"

    plot_loss_breakdown(
        df=df,
        loss_cols=loss_cols,
        sort_col=sort_col,
        sort_label=sort_label,
        title=title,
        out_path=out_path,
        normalize=args.normalize,
        max_points=args.max_points,
    )

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
