# scripts/parity_plot.py
#
# Creates Molinaroli-style parity plots from a fit_predictions CSV.
# - Plots measured vs. calculated for m_dot, Pel, and (optionally) T_dis
# - Shows 1:1 line and ±5% band (NO ±10% line)
# - Colors points outside the ±5% band differently
# - Legend bottom-right (no 1:1 legend entry)
# - Out-of-band count as text in upper-left
# - Larger axis labels

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _finite_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def parity_plot(
    x_meas: np.ndarray,
    y_calc: np.ndarray,
    band: float,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    point_size: int = 28,
    axis_label_size: int = 13,
    tick_label_size: int = 11,
):
    # Filter finite + positive measured (avoid div by 0 in relative error)
    m = _finite_mask(x_meas, y_calc) & (x_meas > 0)
    x = x_meas[m]
    y = y_calc[m]
    if len(x) == 0:
        return {"n_total": 0, "n_outside": 0, "frac_outside": np.nan}

    rel_err = (y / x) - 1.0
    outside = np.abs(rel_err) > band
    n_total = int(len(x))
    n_out = int(np.sum(outside))
    frac_out = float(n_out / n_total) if n_total else np.nan

    # Limits (square + padding)
    xy_min = float(min(np.min(x), np.min(y)))
    xy_max = float(max(np.max(x), np.max(y)))
    if xy_min == xy_max:
        xy_min *= 0.95
        xy_max *= 1.05
    pad = 0.05 * (xy_max - xy_min)
    lo = xy_min - pad
    hi = xy_max + pad

    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    # Points: inside vs outside band
    ax.scatter(
        x[~outside],
        y[~outside],
        s=point_size,
        alpha=0.85,
        label=f"innerhalb ±{int(band*100)}%",
    )
    ax.scatter(
        x[outside],
        y[outside],
        s=point_size,
        alpha=0.95,
        label=f"außerhalb ±{int(band*100)}% (n={n_out})",
    )

    # Reference lines
    xx = np.linspace(lo, hi, 200)

    # 1:1 (keep, but NO legend entry)
    ax.plot(xx, xx, linewidth=1.4, label="_nolegend_")

    # ±band (same color for + and -)
    band_color = "0.5"  # gray
    ax.plot(xx, (1.0 + band) * xx, linestyle="--", linewidth=1.2, color=band_color, label="_nolegend_")
    ax.plot(xx, (1.0 - band) * xx, linestyle="--", linewidth=1.2, color=band_color, label=f"±{int(band*100)}%")

    # Title (no quantification here anymore)
    ax.set_title(title)

    # Quantification text top-left (inside axes)
    info_txt = f"Außerhalb ±{int(band*100)}%: {n_out} / {n_total} ({frac_out*100:.1f}%)"
    ax.text(
        0.02, 0.98, info_txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )

    # Axis labels larger
    ax.set_xlabel(x_label, fontsize=axis_label_size)
    ax.set_ylabel(y_label, fontsize=axis_label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.6, alpha=0.35)

    # Legend bottom-right (only points + ±band)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"n_total": n_total, "n_outside": n_out, "frac_outside": frac_out}


def main():
    ap = argparse.ArgumentParser(description="Create parity plots from fit_predictions CSV (±5% band, outliers highlighted).")
    ap.add_argument("--pred_csv", required=True, help="Path to fit_predictions_*.csv")
    ap.add_argument("--out_dir", default="results/parity_plots", help="Output directory for PNGs and summary CSV")
    ap.add_argument("--band", type=float, default=0.05, help="Relative error band (default 0.05 = ±5%)")
    ap.add_argument("--point_size", type=int, default=28, help="Scatter point size")
    ap.add_argument("--axis_label_size", type=int, default=13, help="Fontsize for axis labels")
    ap.add_argument("--tick_label_size", type=int, default=11, help="Fontsize for tick labels")
    args = ap.parse_args()

    pred_path = Path(args.pred_csv)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)

    df = pd.read_csv(pred_path)

    summary = []

    # m_dot
    if {"m_meas_gps", "m_calc_gps"}.issubset(df.columns):
        stats = parity_plot(
            x_meas=df["m_meas_gps"].to_numpy(dtype=float),
            y_calc=df["m_calc_gps"].to_numpy(dtype=float),
            band=args.band,
            title="Parity Plot: Massenstrom",
            x_label="ṁ_meas [g/s]",
            y_label="ṁ_calc [g/s]",
            out_path=out_dir / "parity_m_dot.png",
            point_size=args.point_size,
            axis_label_size=args.axis_label_size,
            tick_label_size=args.tick_label_size,
        )
        stats.update({"metric": "m_dot", "x_col": "m_meas_gps", "y_col": "m_calc_gps"})
        summary.append(stats)

    # Pel
    if {"P_meas_W", "P_calc_W"}.issubset(df.columns):
        stats = parity_plot(
            x_meas=df["P_meas_W"].to_numpy(dtype=float),
            y_calc=df["P_calc_W"].to_numpy(dtype=float),
            band=args.band,
            title="Parity Plot: Elektrische Leistung",
            x_label="P_el,meas [W]",
            y_label="P_el,calc [W]",
            out_path=out_dir / "parity_P_el.png",
            point_size=args.point_size,
            axis_label_size=args.axis_label_size,
            tick_label_size=args.tick_label_size,
        )
        stats.update({"metric": "P_el", "x_col": "P_meas_W", "y_col": "P_calc_W"})
        summary.append(stats)

    # T_dis (optional)
    if {"T_dis_meas_C", "T_dis_calc_C"}.issubset(df.columns):
        stats = parity_plot(
            x_meas=df["T_dis_meas_C"].to_numpy(dtype=float),
            y_calc=df["T_dis_calc_C"].to_numpy(dtype=float),
            band=args.band,
            title="Parity Plot: Austrittstemperatur",
            x_label="T_dis,meas [°C]",
            y_label="T_dis,calc [°C]",
            out_path=out_dir / "parity_T_dis.png",
            point_size=args.point_size,
            axis_label_size=args.axis_label_size,
            tick_label_size=args.tick_label_size,
        )
        stats.update({"metric": "T_dis", "x_col": "T_dis_meas_C", "y_col": "T_dis_calc_C"})
        summary.append(stats)

    # Write summary
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_csv = out_dir / "parity_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print("Saved:", summary_csv)

    print("Done. Output dir:", out_dir)


if __name__ == "__main__":
    main()
