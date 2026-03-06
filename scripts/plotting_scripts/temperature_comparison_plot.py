# scripts/temperature_compare_plots.py
#
# Creates parity-style comparison plots between temperatures from run_batch output CSV:
#   1) T_dis_C vs T_wall_C
#   2) T_dis_C vs T_oil_sump_C_meas
#   3) T_wall_C vs T_oil_sump_C_meas
#   4) T_wall_C vs T_mean_in_out_C  where T_mean_in_out_C = 0.5*(T_suc_C_in + T_dis_C)
#
# Each plot:
#   - scatter points
#   - 1:1 line
#   - linear regression line (least squares)
#   - small info box: n, slope, intercept, R^2, RMSE
#
# Uses matplotlib style: ebc.paper.mplstyle (no manual overriding of style-defined values).
#
# Example:
#   python scripts/temperature_comparison_plot.py --csv results/batch_lpg68_original_20260225_152734.csv

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ebc.paper.mplstyle")


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _finite_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def _linear_regression(x: np.ndarray, y: np.ndarray):
    """
    Returns slope, intercept, r2, rmse for y ~ slope*x + intercept
    """
    # polyfit degree 1 is least squares
    slope, intercept = np.polyfit(x, y, 1)

    y_hat = slope * x + intercept
    resid = y - y_hat

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid**2))) if len(x) else float("nan")
    return float(slope), float(intercept), r2, rmse


def parity_compare_plot(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
):
    # filter finite
    m = _finite_mask(x, y)
    x = x[m]
    y = y[m]

    if len(x) == 0:
        return {"n": 0, "slope": np.nan, "intercept": np.nan, "r2": np.nan, "rmse": np.nan}

    slope, intercept, r2, rmse = _linear_regression(x, y)

    # square limits with padding
    xy_min = float(min(np.min(x), np.min(y)))
    xy_max = float(max(np.max(x), np.max(y)))
    if xy_min == xy_max:
        xy_min -= 1.0
        xy_max += 1.0
    pad = 0.05 * (xy_max - xy_min)
    lo = xy_min - pad
    hi = xy_max + pad

    fig, ax = plt.subplots()

    # Scatter (style controls default colors/markers; we don't set explicit colors)
    ax.scatter(x, y, alpha=0.85)

    # 1:1 line
    xx = np.linspace(lo, hi, 200)
    ax.plot(xx, xx, linestyle="--", label="1:1")

    # Regression line
    ax.plot(xx, slope * xx + intercept, label="Regression")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.35)

    info = (
        f"n = {len(x)}\n"
        f"slope = {slope:.4f}\n"
        f"intercept = {intercept:.2f} °C\n"
        f"R² = {r2:.4f}\n"
        f"RMSE = {rmse:.2f} °C"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )

    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return {"n": int(len(x)), "slope": slope, "intercept": intercept, "r2": r2, "rmse": rmse}


def main():
    ap = argparse.ArgumentParser(description="Compare temperatures from run_batch output via parity-style plots.")
    ap.add_argument("--csv", required=True, type=Path, help="run_batch output CSV")
    ap.add_argument("--out_dir", default="results/temp_compare_plots", type=Path, help="Output directory")
    ap.add_argument("--only_success", action="store_true", help="Use only rows with success==True if column exists")
    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    _ensure_out_dir(args.out_dir)

    df = pd.read_csv(args.csv)

    if args.only_success and "success" in df.columns:
        df = df[df["success"] == True].copy()  # noqa: E712

    # Required columns (some plots need subsets)
    # From your run_batch output:
    #   T_dis_C, T_wall_C, T_suc_C_in, T_oil_sump_C_meas
    missing_base = [c for c in ["T_dis_C", "T_wall_C", "T_suc_C_in"] if c not in df.columns]
    if missing_base:
        raise ValueError(f"CSV missing required columns: {missing_base}")

    # Derived mean temperature
    df["T_mean_in_out_C"] = 0.5 * (df["T_suc_C_in"].astype(float) + df["T_dis_C"].astype(float))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_name = args.csv.name

    summary = []

    # 1) T_dis vs T_wall
    stats = parity_compare_plot(
        x=df["T_dis_C"].to_numpy(float),
        y=df["T_wall_C"].to_numpy(float),
        x_label="T_dis [°C]",
        y_label="T_wall [°C]",
        title="Temperaturvergleich: Austritt vs Wand",
        out_path=args.out_dir / f"parity_Tdis_vs_Twall_{ts}.png",
    )
    stats.update({"plot": "T_dis_vs_T_wall", "x_col": "T_dis_C", "y_col": "T_wall_C", "source_csv": pred_name})
    summary.append(stats)

    # Oil temperature plots only if column exists
    if "T_oil_sump_C_meas" in df.columns:
        # 2) T_dis vs T_oil
        stats = parity_compare_plot(
            x=df["T_dis_C"].to_numpy(float),
            y=df["T_oil_sump_C_meas"].to_numpy(float),
            x_label="T_dis [°C]",
            y_label="T_oil,sump,meas [°C]",
            title="Temperaturvergleich: Austritt vs Öltemperatur (Sumpf, gemessen)",
            out_path=args.out_dir / f"parity_Tdis_vs_Toil_{ts}.png",
        )
        stats.update({"plot": "T_dis_vs_T_oil", "x_col": "T_dis_C", "y_col": "T_oil_sump_C_meas", "source_csv": pred_name})
        summary.append(stats)

        # 3) T_wall vs T_oil
        stats = parity_compare_plot(
            x=df["T_wall_C"].to_numpy(float),
            y=df["T_oil_sump_C_meas"].to_numpy(float),
            x_label="T_wall [°C]",
            y_label="T_oil,sump,meas [°C]",
            title="Temperaturvergleich: Wand vs Öltemperatur (Sumpf, gemessen)",
            out_path=args.out_dir / f"parity_Twall_vs_Toil_{ts}.png",
        )
        stats.update({"plot": "T_wall_vs_T_oil", "x_col": "T_wall_C", "y_col": "T_oil_sump_C_meas", "source_csv": pred_name})
        summary.append(stats)
    else:
        print("[INFO] Column 'T_oil_sump_C_meas' not found -> skipping oil temperature plots.")

    # 4) T_wall vs mean(in,out)
    stats = parity_compare_plot(
        x=df["T_wall_C"].to_numpy(float),
        y=df["T_mean_in_out_C"].to_numpy(float),
        x_label="T_wall [°C]",
        y_label="0.5·(T_suc + T_dis) [°C]",
        title="Temperaturvergleich: Wand vs Mittelwert aus Ein- und Austritt",
        out_path=args.out_dir / f"parity_Twall_vs_TmeanInOut_{ts}.png",
    )
    stats.update({"plot": "T_wall_vs_T_mean_in_out", "x_col": "T_wall_C", "y_col": "T_mean_in_out_C", "source_csv": pred_name})
    summary.append(stats)

    # Write summary
    summary_df = pd.DataFrame(summary)
    summary_path = args.out_dir / f"temp_compare_summary_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)

    print("Done.")
    print("Output dir:", args.out_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()