# scripts/parity_plot.py
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


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def parity_plot_rel_band(
    x_meas: np.ndarray,
    y_calc: np.ndarray,
    band: float,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    *,
    color_values: np.ndarray | None = None,
    color_label: str = "Überhitzung [°C]",
    cmap: str = "viridis",
    cmin: float | None = None,
    cmax: float | None = None,
    point_size: int | None = None,
):
    # Filter finite + positive measured (avoid div by 0 in relative error)
    if color_values is None:
        m = _finite_mask(x_meas, y_calc) & (x_meas > 0)
        c = None
    else:
        m = _finite_mask(x_meas, y_calc, color_values) & (x_meas > 0)
        c = color_values[m]

    x = x_meas[m]
    y = y_calc[m]

    if len(x) == 0:
        return {"n_total": 0, "n_outside": 0, "frac_outside": np.nan}

    rel_err = (y / x) - 1.0
    outside = np.abs(rel_err) > band

    n_total = int(len(x))
    n_out = int(np.sum(outside))
    frac_out = float(n_out / n_total) if n_total else np.nan

    err_min_pct = float(np.min(rel_err) * 100.0)
    err_max_pct = float(np.max(rel_err) * 100.0)

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

    # Reference lines
    xx = np.linspace(lo, hi, 200)
    ax.plot(xx, xx, linewidth=1.4, label="_nolegend_")  # 1:1 no legend
    band_color = "0.5"
    ax.plot(xx, (1.0 + band) * xx, linestyle="--", linewidth=1.2, color=band_color, label="_nolegend_")
    ax.plot(xx, (1.0 - band) * xx, linestyle="--", linewidth=1.2, color=band_color, label=f"±{int(band*100)}%")

    s = point_size

    if c is None:
        ax.scatter(x[~outside], y[~outside], s=s, alpha=0.85, label=f"innerhalb ±{int(band*100)}%")
        ax.scatter(x[outside], y[outside], s=s, alpha=0.95, label=f"außerhalb ±{int(band*100)}% (n={n_out})")
    else:
        vmin = float(np.nanmin(c)) if cmin is None else float(cmin)
        vmax = float(np.nanmax(c)) if cmax is None else float(cmax)

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            ax.scatter(x[~outside], y[~outside], s=s, alpha=0.85, label=f"innerhalb ±{int(band*100)}%")
            ax.scatter(x[outside], y[outside], s=s, alpha=0.95, label=f"außerhalb ±{int(band*100)}% (n={n_out})")
        else:
            sc_in = ax.scatter(
                x[~outside], y[~outside],
                c=c[~outside], cmap=cmap, vmin=vmin, vmax=vmax,
                s=s, alpha=0.90, edgecolors="none",
                label=f"innerhalb ±{int(band*100)}%",
            )
            ax.scatter(
                x[outside], y[outside],
                c=c[outside], cmap=cmap, vmin=vmin, vmax=vmax,
                s=s, alpha=0.98, edgecolors="black", linewidths=0.6,
                label=f"außerhalb ±{int(band*100)}% (n={n_out})",
            )
            cbar = fig.colorbar(sc_in, ax=ax, pad=0.02)
            cbar.set_label(color_label)

    ax.set_title(title)

    info_txt = (
        f"Außerhalb ±{int(band*100)}%: {n_out} / {n_total} ({frac_out*100:.1f}%)\n"
        f"Fehlerspanne: {err_min_pct:.2f}% bis {err_max_pct:.2f}%"
    )
    ax.text(
        0.02, 0.98, info_txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, linewidth=0.6, alpha=0.35)

    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"n_total": n_total, "n_outside": n_out, "frac_outside": frac_out}


def parity_plot_abs_band(
    x_meas: np.ndarray,
    y_calc: np.ndarray,
    band_abs: float,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    *,
    color_values: np.ndarray | None = None,
    color_label: str = "Überhitzung [°C]",
    cmap: str = "viridis",
    cmin: float | None = None,
    cmax: float | None = None,
    point_size: int | None = None,
):
    if color_values is None:
        m = _finite_mask(x_meas, y_calc)
        c = None
    else:
        m = _finite_mask(x_meas, y_calc, color_values)
        c = color_values[m]

    x = x_meas[m]
    y = y_calc[m]
    if len(x) == 0:
        return {"n_total": 0, "n_outside": 0, "frac_outside": np.nan}

    diff = y - x
    outside = np.abs(diff) > band_abs

    n_total = int(len(x))
    n_out = int(np.sum(outside))
    frac_out = float(n_out / n_total) if n_total else np.nan

    # Limits
    xy_min = float(min(np.min(x), np.min(y)))
    xy_max = float(max(np.max(x), np.max(y)))
    if xy_min == xy_max:
        xy_min *= 0.95
        xy_max *= 1.05
    pad = 0.05 * (xy_max - xy_min)
    lo = xy_min - pad
    hi = xy_max + pad

    fig, ax = plt.subplots()

    xx = np.linspace(lo, hi, 200)
    ax.plot(xx, xx, linewidth=1.4, label="_nolegend_")
    band_color = "0.5"
    ax.plot(xx, xx + band_abs, linestyle="--", linewidth=1.2, color=band_color, label="_nolegend_")
    ax.plot(xx, xx - band_abs, linestyle="--", linewidth=1.2, color=band_color, label=f"±{band_abs:.0f}°C")

    s = point_size

    if c is None:
        ax.scatter(x[~outside], y[~outside], s=s, alpha=0.85, label=f"innerhalb ±{band_abs:.0f}°C")
        ax.scatter(x[outside], y[outside], s=s, alpha=0.95, label=f"außerhalb ±{band_abs:.0f}°C (n={n_out})")
    else:
        vmin = float(np.nanmin(c)) if cmin is None else float(cmin)
        vmax = float(np.nanmax(c)) if cmax is None else float(cmax)

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            ax.scatter(x[~outside], y[~outside], s=s, alpha=0.85, label=f"innerhalb ±{band_abs:.0f}°C")
            ax.scatter(x[outside], y[outside], s=s, alpha=0.95, label=f"außerhalb ±{band_abs:.0f}°C (n={n_out})")
        else:
            sc_in = ax.scatter(
                x[~outside], y[~outside],
                c=c[~outside], cmap=cmap, vmin=vmin, vmax=vmax,
                s=s, alpha=0.90, edgecolors="none",
                label=f"innerhalb ±{band_abs:.0f}°C",
            )
            ax.scatter(
                x[outside], y[outside],
                c=c[outside], cmap=cmap, vmin=vmin, vmax=vmax,
                s=s, alpha=0.98, edgecolors="black", linewidths=0.6,
                label=f"außerhalb ±{band_abs:.0f}°C (n={n_out})",
            )
            cbar = fig.colorbar(sc_in, ax=ax, pad=0.02)
            cbar.set_label(color_label)

    ax.set_title(title)

    info_txt = (
        f"Außerhalb ±{band_abs:.0f}°C: {n_out} / {n_total} ({frac_out*100:.1f}%)\n"
        f"Fehlerspanne: {float(np.min(diff)):.2f}°C bis {float(np.max(diff)):.2f}°C"
    )
    ax.text(
        0.02, 0.98, info_txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"n_total": n_total, "n_outside": n_out, "frac_outside": frac_out}


def _pick_pair(df: pd.DataFrame, candidates: list[tuple[str, str]]):
    """Return first (meas, calc) pair that exists, else None."""
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    return None


def main():
    ap = argparse.ArgumentParser(description="Create parity plots from predictions OR run_batch CSV.")
    ap.add_argument("--pred_csv", required=True, help="Path to CSV (predictions or run_batch output)")
    ap.add_argument("--out_dir", default="results/parity_plots", help="Output directory for PNGs and summary CSV")

    ap.add_argument("--band", type=float, default=0.05, help="Relative error band (default 0.05 = ±5%)")
    ap.add_argument("--band_T_dis_abs", type=float, default=3.0, help="Absolute band for T_dis in °C (default ±3°C)")

    ap.add_argument("--color_by_superheat", action="store_true",
                    help="Color points by column 'superheat_C' and show a colorbar on the right.")
    ap.add_argument("--cmin", type=float, default=None, help="Optional: fixed min for color scale (superheat_C)")
    ap.add_argument("--cmax", type=float, default=None, help="Optional: fixed max for color scale (superheat_C)")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap name (default: viridis)")
    ap.add_argument("--point_size", type=int, default=None, help="Optional: scatter point size (overrides style)")

    args = ap.parse_args()

    csv_path = Path(args.pred_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)

    df = pd.read_csv(csv_path)
    stamp = _ts()
    src_name = csv_path.name

    # Color values (optional)
    use_color = bool(args.color_by_superheat) and ("superheat_C" in df.columns)
    color_vals = df["superheat_C"].to_numpy(dtype=float) if use_color else None

    # --- Column mapping: predictions vs run_batch ---
    # m_dot candidates:
    m_pair = _pick_pair(df, [
        ("m_meas_gps", "m_calc_gps"),   # GA predictions format
        ("m_meas_g_s", "m_flow_g_s"),   # run_batch format
    ])

    # Pel candidates:
    p_pair = _pick_pair(df, [
        ("P_meas_W", "P_calc_W"),       # GA predictions format
        ("P_meas_W", "P_el_W"),         # run_batch format
    ])

    # T_dis candidates (absolute band ±3°C):
    t_pair = _pick_pair(df, [
        ("T_dis_meas_C", "T_dis_calc_C"),  # predictions format
        ("T_dis_meas_C", "T_dis_C"),       # possible run_batch extension
    ])

    summary = []
    generated_any = False

    if m_pair is not None:
        meas, calc = m_pair
        stats = parity_plot_rel_band(
            x_meas=df[meas].to_numpy(dtype=float),
            y_calc=df[calc].to_numpy(dtype=float),
            band=args.band,
            title="Parity Plot: Massenstrom",
            x_label=f"{meas} [g/s]",
            y_label=f"{calc} [g/s]",
            out_path=out_dir / f"parity_m_dot_{stamp}.png",
            color_values=color_vals,
            color_label="Überhitzung [°C]",
            cmap=args.cmap,
            cmin=args.cmin,
            cmax=args.cmax,
            point_size=args.point_size,
        )
        stats.update({"metric": "m_dot", "x_col": meas, "y_col": calc, "source_file": src_name})
        summary.append(stats)
        generated_any = True
        print(f"[OK] m_dot plot: {meas} vs {calc}")
    else:
        print("[SKIP] m_dot plot: keine passenden Spalten gefunden.")

    if p_pair is not None:
        meas, calc = p_pair
        stats = parity_plot_rel_band(
            x_meas=df[meas].to_numpy(dtype=float),
            y_calc=df[calc].to_numpy(dtype=float),
            band=args.band,
            title="Parity Plot: Elektrische Leistung",
            x_label=f"{meas} [W]",
            y_label=f"{calc} [W]",
            out_path=out_dir / f"parity_P_el_{stamp}.png",
            color_values=color_vals,
            color_label="Überhitzung [°C]",
            cmap=args.cmap,
            cmin=args.cmin,
            cmax=args.cmax,
            point_size=args.point_size,
        )
        stats.update({"metric": "P_el", "x_col": meas, "y_col": calc, "source_file": src_name})
        summary.append(stats)
        generated_any = True
        print(f"[OK] P_el plot: {meas} vs {calc}")
    else:
        print("[SKIP] P_el plot: keine passenden Spalten gefunden.")

    if t_pair is not None:
        meas, calc = t_pair
        stats = parity_plot_abs_band(
            x_meas=df[meas].to_numpy(dtype=float),
            y_calc=df[calc].to_numpy(dtype=float),
            band_abs=args.band_T_dis_abs,
            title="Parity Plot: Austrittstemperatur",
            x_label=f"{meas} [°C]",
            y_label=f"{calc} [°C]",
            out_path=out_dir / f"parity_T_dis_{stamp}.png",
            color_values=color_vals,
            color_label="Überhitzung [°C]",
            cmap=args.cmap,
            cmin=args.cmin,
            cmax=args.cmax,
            point_size=args.point_size,
        )
        stats.update({"metric": "T_dis", "x_col": meas, "y_col": calc, "source_file": src_name})
        summary.append(stats)
        generated_any = True
        print(f"[OK] T_dis plot: {meas} vs {calc}")
    else:
        print("[SKIP] T_dis plot: keine passenden Spalten gefunden (keine Mess-/Calc-Paarung).")

    if summary:
        summary_df = pd.DataFrame(summary)
        summary_csv = out_dir / f"parity_summary_{stamp}.csv"
        summary_df.to_csv(summary_csv, index=False)
        print("[OK] Saved summary:", summary_csv)

    if not generated_any:
        print("\n[ERROR] Es wurden keine Plots erzeugt, weil keine erwarteten Spalten gefunden wurden.")
        print("        Gefundene Spalten im CSV sind z.B.:")
        print("        ", ", ".join(list(df.columns)[:25]), ("..." if len(df.columns) > 25 else ""))

    print("Done. Output dir:", out_dir)


if __name__ == "__main__":
    main()