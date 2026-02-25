# scripts/parity_plot.py
#
# Creates Molinaroli-style parity plots from a fit_predictions CSV.
# - Plots measured vs. calculated for m_dot, Pel, and (optionally) T_dis
# - Uses style defaults from 'ebc.paper.mplstyle' (no manual overrides of sizes, dpi, grid, legend)
# - For m_dot and Pel: 1:1 line and ±5% band (relative)
# - For T_dis: 1:1 line and ±3 K absolute band
# - Colors points outside the band differently
# - Out-of-band count + error span as text inside the axes
#
# NEW:
# - Output filenames include timestamp
# - Summary includes the predictions filename/path

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ebc.paper.mplstyle")


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
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    band_value: float,
    band_mode: str = "rel",  # "rel" => ±band_value relative; "abs" => ±band_value in same units as x/y
):
    band_mode = str(band_mode).lower().strip()
    if band_mode not in ("rel", "abs"):
        raise ValueError("band_mode must be 'rel' or 'abs'")

    # Filter finite + measured > 0 for rel-mode (avoid div by 0)
    m = _finite_mask(x_meas, y_calc)
    if band_mode == "rel":
        m &= (x_meas > 0)

    x = x_meas[m].astype(float)
    y = y_calc[m].astype(float)

    if len(x) == 0:
        return {"n_total": 0, "n_outside": 0, "frac_outside": np.nan}

    # Determine outside points + error statistics
    if band_mode == "rel":
        rel_err = (y / x) - 1.0
        outside = np.abs(rel_err) > float(band_value)
        err_min = float(np.min(rel_err) * 100.0)
        err_max = float(np.max(rel_err) * 100.0)
        band_label = f"±{int(round(float(band_value) * 100))}%"
        info_span = f"Fehlerspanne: {err_min:.2f}% bis {err_max:.2f}%"
    else:
        abs_err = (y - x)
        outside = np.abs(abs_err) > float(band_value)
        err_min = float(np.min(abs_err))
        err_max = float(np.max(abs_err))
        band_label = f"±{float(band_value):g} K"
        info_span = f"Fehlerspanne: {err_min:.2f} K bis {err_max:.2f} K"

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

    fig, ax = plt.subplots()

    # Points: inside vs outside band
    ax.scatter(
        x[~outside],
        y[~outside],
        alpha=0.85,
        label=f"innerhalb {band_label}",
    )
    ax.scatter(
        x[outside],
        y[outside],
        alpha=0.95,
        label=f"außerhalb {band_label} (n={n_out})",
    )

    # Reference lines
    xx = np.linspace(lo, hi, 200)

    # 1:1 (no legend entry)
    ax.plot(xx, xx, label="_nolegend_")

    # Band lines
    band_color = "0.5"  # neutral gray
    if band_mode == "rel":
        ax.plot(xx, (1.0 + float(band_value)) * xx, linestyle="--", color=band_color, label="_nolegend_")
        ax.plot(xx, (1.0 - float(band_value)) * xx, linestyle="--", color=band_color, label=band_label)
    else:
        ax.plot(xx, xx + float(band_value), linestyle="--", color=band_color, label="_nolegend_")
        ax.plot(xx, xx - float(band_value), linestyle="--", color=band_color, label=band_label)

    # Title + labels (no fontsize overrides; style controls sizes)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Quantification text
    info_txt = (
        f"Außerhalb {band_label}: {n_out} / {n_total} ({frac_out*100:.1f}%)\n"
        f"{info_span}"
    )
    ax.text(
        0.02, 0.98, info_txt,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # No explicit styling here; rely on mplstyle defaults
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return {"n_total": n_total, "n_outside": n_out, "frac_outside": frac_out}


def main():
    ap = argparse.ArgumentParser(description="Create parity plots from fit_predictions CSV using ebc.paper.mplstyle.")
    ap.add_argument("--pred_csv", required=True, help="Path to fit_predictions_*.csv")
    ap.add_argument("--out_dir", default="results/parity_plots", help="Output directory for PNGs and summary CSV")

    # Bands
    ap.add_argument("--band_rel", type=float, default=0.05, help="Relative error band for m_dot and Pel (default 0.05 = ±5%)")
    ap.add_argument("--band_Tdis_K", type=float, default=3.0, help="Absolute band for T_dis in K/°C (default ±3 K)")

    args = ap.parse_args()

    pred_path = Path(args.pred_csv)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)

    df = pd.read_csv(pred_path)

    # Timestamp for filenames
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    summary = []
    pred_name = pred_path.name
    pred_path_str = str(pred_path)

    # m_dot: relative band
    if {"m_meas_gps", "m_calc_gps"}.issubset(df.columns):
        out_png = out_dir / f"parity_m_dot_{ts}.png"
        stats = parity_plot(
            x_meas=df["m_meas_gps"].to_numpy(dtype=float),
            y_calc=df["m_calc_gps"].to_numpy(dtype=float),
            band_value=float(args.band_rel),
            band_mode="rel",
            title="Parity Plot: Massenstrom",
            x_label="ṁ_meas [g/s]",
            y_label="ṁ_calc [g/s]",
            out_path=out_png,
        )
        stats.update({
            "metric": "m_dot",
            "x_col": "m_meas_gps",
            "y_col": "m_calc_gps",
            "band_mode": "rel",
            "band_value": float(args.band_rel),
            "predictions_file": pred_name,
            "predictions_path": pred_path_str,
            "out_png": str(out_png),
            "timestamp": ts,
        })
        summary.append(stats)

    # Pel: relative band
    if {"P_meas_W", "P_calc_W"}.issubset(df.columns):
        out_png = out_dir / f"parity_P_el_{ts}.png"
        stats = parity_plot(
            x_meas=df["P_meas_W"].to_numpy(dtype=float),
            y_calc=df["P_calc_W"].to_numpy(dtype=float),
            band_value=float(args.band_rel),
            band_mode="rel",
            title="Parity Plot: Elektrische Leistung",
            x_label="P_el,meas [W]",
            y_label="P_el,calc [W]",
            out_path=out_png,
        )
        stats.update({
            "metric": "P_el",
            "x_col": "P_meas_W",
            "y_col": "P_calc_W",
            "band_mode": "rel",
            "band_value": float(args.band_rel),
            "predictions_file": pred_name,
            "predictions_path": pred_path_str,
            "out_png": str(out_png),
            "timestamp": ts,
        })
        summary.append(stats)

    # T_dis: absolute band ±3 K (°C differences same magnitude)
    if {"T_dis_meas_C", "T_dis_calc_C"}.issubset(df.columns):
        out_png = out_dir / f"parity_T_dis_{ts}.png"
        stats = parity_plot(
            x_meas=df["T_dis_meas_C"].to_numpy(dtype=float),
            y_calc=df["T_dis_calc_C"].to_numpy(dtype=float),
            band_value=float(args.band_Tdis_K),
            band_mode="abs",
            title="Parity Plot: Austrittstemperatur",
            x_label="T_dis,meas [°C]",
            y_label="T_dis,calc [°C]",
            out_path=out_png,
        )
        stats.update({
            "metric": "T_dis",
            "x_col": "T_dis_meas_C",
            "y_col": "T_dis_calc_C",
            "band_mode": "abs",
            "band_value": float(args.band_Tdis_K),
            "predictions_file": pred_name,
            "predictions_path": pred_path_str,
            "out_png": str(out_png),
            "timestamp": ts,
        })
        summary.append(stats)

    # Write summary
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_csv = out_dir / f"parity_summary_{ts}.csv"
        summary_df.to_csv(summary_csv, index=False)
        print("Saved:", summary_csv)

    print("Done. Output dir:", out_dir)


if __name__ == "__main__":
    main()