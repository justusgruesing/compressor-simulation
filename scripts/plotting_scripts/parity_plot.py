# scripts/plotting_scripts/parity_plot.py
#
# Examples:
#   # Validation CSV, color by superheat:
#   python scripts/plotting_scripts/parity_plot.py --pred_csv results\validation\validation_detail_params_lpg68_val_lpg68_original_2026-03-19_142319.csv --color_by superheat
#
#   # Validation CSV, color by pressure ratio:
#   python scripts/plotting_scripts/parity_plot.py --pred_csv results/validation/validation_detail_params_lpg68_val_lpg100_original_2026-03-19_130107.csv --color_by pressure_ratio
#
#   # GA predictions CSV, no color:
#   python scripts/plotting_scripts/parity_plot.py --pred_csv results/validation/validation_detail_params_lpg68_val_lpg68_modified_2026-03-19_130709.csv --color_by none
#
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
def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _finite_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _detect_split_role(df: pd.DataFrame) -> np.ndarray | None:
    """
    Detect train/validation role per row.
    Returns array of strings ('train', 'validation', '') or None if no info available.
    """
    if "split_role" in df.columns:
        return df["split_role"].fillna("").astype(str).to_numpy()
    if "is_train" in df.columns:
        roles = np.where(
            df["is_train"].fillna(False).astype(bool), "train", "validation"
        )
        return roles
    return None


def _auto_title(df: pd.DataFrame, metric_label: str) -> str:
    """Generate a title from metadata columns if available."""
    parts = [f"Parity Plot: {metric_label}"]

    model = None
    if "model" in df.columns:
        vals = df["model"].dropna().unique()
        if len(vals) == 1:
            model = str(vals[0]).capitalize()

    params_oil = None
    if "params_oil" in df.columns:
        vals = df["params_oil"].dropna().unique()
        if len(vals) == 1:
            params_oil = str(vals[0])

    val_oil = None
    if "oil" in df.columns:
        vals = df["oil"].dropna().unique()
        if len(vals) == 1:
            val_oil = str(vals[0])
        elif len(vals) > 1:
            val_oil = "all"

    subtitle_parts = []
    if model:
        subtitle_parts.append(model)
    if params_oil and val_oil:
        subtitle_parts.append(f"Params: {params_oil} → Data: {val_oil}")
    elif params_oil:
        subtitle_parts.append(f"Params: {params_oil}")
    elif val_oil:
        subtitle_parts.append(f"Öl: {val_oil}")

    if subtitle_parts:
        parts.append(" | ".join(subtitle_parts))

    return "\n".join(parts)


# =========================================================
# Color column mapping
# =========================================================
COLOR_CONFIG = {
    "superheat": {
        "col": "superheat_C",
        "label": "Überhitzung in K",
        "cmap_default": "viridis",
    },
    "pressure_ratio": {
        "col": "pressure_ratio",
        "label": "Druckverhältnis",
        "cmap_default": "viridis",
    },
}


def _resolve_color(df: pd.DataFrame, color_by: str, cmap_override: str | None) -> tuple:
    """
    Returns (color_values_array_or_None, color_label, cmap).
    """
    if color_by == "none" or color_by is None:
        return None, "", "viridis"

    cfg = COLOR_CONFIG.get(color_by)
    if cfg is None:
        print(f"[WARN] Unknown --color_by '{color_by}', falling back to none.")
        return None, "", "viridis"

    col = cfg["col"]
    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found in CSV, falling back to no color.")
        return None, "", "viridis"

    cmap = cmap_override if cmap_override else cfg["cmap_default"]
    return df[col].to_numpy(dtype=float), cfg["label"], cmap


# =========================================================
# Core plot functions
# =========================================================
def _scatter_split(
    ax, x, y, roles, outside, band_label,
    color_values=None, cmap="viridis", vmin=None, vmax=None,
    point_size=None, fig=None, color_label="",
):
    """
    Scatter with train (hollow) / validation (filled) distinction.
    Returns n_out for the legend text.
    """
    s = point_size
    n_out = int(np.sum(outside))

    has_split = roles is not None
    has_color = (color_values is not None and
                 np.any(np.isfinite(color_values)) and
                 vmin is not None and vmax is not None and
                 vmin != vmax)

    if has_split:
        is_train = (roles == "train")
        is_val = ~is_train  # validation + unknown → treated as validation
    else:
        is_train = np.zeros(len(x), dtype=bool)
        is_val = np.ones(len(x), dtype=bool)

    sc_ref = None  # reference scatter for colorbar

    if has_color:
        c = color_values

        # --- Validation points (filled) ---
        mask_val_in = is_val & ~outside
        mask_val_out = is_val & outside

        if np.any(mask_val_in):
            sc_ref = ax.scatter(
                x[mask_val_in], y[mask_val_in],
                c=c[mask_val_in], cmap=cmap, vmin=vmin, vmax=vmax,
                s=s, alpha=0.90, marker="o", edgecolors="none",
                label=f"Validation innerhalb {band_label}",
            )
        if np.any(mask_val_out):
            sc = ax.scatter(
                x[mask_val_out], y[mask_val_out],
                c=c[mask_val_out], cmap=cmap, vmin=vmin, vmax=vmax,
                s=s, alpha=0.95, marker="s", edgecolors="none",
                label=f"Validation außerhalb {band_label} (n={int(mask_val_out.sum())})",
            )
            if sc_ref is None:
                sc_ref = sc

        # --- Training points (hollow) ---
        mask_tr_in = is_train & ~outside
        mask_tr_out = is_train & outside

        # For hollow markers with color: use edgecolors mapped to color values
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        if np.any(mask_tr_in):
            edge_colors = cmap_obj(norm(c[mask_tr_in]))
            ax.scatter(
                x[mask_tr_in], y[mask_tr_in],
                s=s, alpha=0.80, marker="o",
                facecolors="none", edgecolors=edge_colors, linewidths=1.2,
                label=f"Training innerhalb {band_label}",
            )
        if np.any(mask_tr_out):
            edge_colors = cmap_obj(norm(c[mask_tr_out]))
            ax.scatter(
                x[mask_tr_out], y[mask_tr_out],
                s=s, alpha=0.90, marker="s",
                facecolors="none", edgecolors=edge_colors, linewidths=1.2,
                label=f"Training außerhalb {band_label} (n={int(mask_tr_out.sum())})",
            )

        # Colorbar
        if sc_ref is not None and fig is not None:
            cbar = fig.colorbar(sc_ref, ax=ax, pad=0.02)
            cbar.set_label(color_label)

    else:
        # No color variable → use default colors

        # --- Validation (filled) ---
        mask_val_in = is_val & ~outside
        mask_val_out = is_val & outside

        if np.any(mask_val_in):
            ax.scatter(
                x[mask_val_in], y[mask_val_in],
                s=s, alpha=0.85, marker="o",
                label=f"Validation innerhalb {band_label}",
            )
        if np.any(mask_val_out):
            ax.scatter(
                x[mask_val_out], y[mask_val_out],
                s=s, alpha=0.95, marker="s", linewidths=0.9,
                label=f"Validation außerhalb {band_label} (n={int(mask_val_out.sum())})",
            )

        # --- Training (hollow) ---
        mask_tr_in = is_train & ~outside
        mask_tr_out = is_train & outside

        if np.any(mask_tr_in):
            ax.scatter(
                x[mask_tr_in], y[mask_tr_in],
                s=s, alpha=0.75, marker="o",
                facecolors="none", edgecolors="C0", linewidths=1.2,
                label=f"Training innerhalb {band_label}",
            )
        if np.any(mask_tr_out):
            ax.scatter(
                x[mask_tr_out], y[mask_tr_out],
                s=s, alpha=0.85, marker="s",
                facecolors="none", edgecolors="C1", linewidths=1.2,
                label=f"Training außerhalb {band_label} (n={int(mask_tr_out.sum())})",
            )

    return n_out


def parity_plot_rel_band(
    x_meas: np.ndarray,
    y_calc: np.ndarray,
    band: float,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    *,
    roles: np.ndarray | None = None,
    color_values: np.ndarray | None = None,
    color_label: str = "",
    cmap: str = "viridis",
    cmin: float | None = None,
    cmax: float | None = None,
    point_size: int | None = None,
):
    # Build valid mask
    arrays = [x_meas, y_calc]
    if color_values is not None:
        arrays.append(color_values)
    m = _finite_mask(*arrays) & (x_meas > 0)

    x = x_meas[m]
    y = y_calc[m]
    c = color_values[m] if color_values is not None else None
    r = roles[m] if roles is not None else None

    if len(x) == 0:
        return {"n_total": 0, "n_outside": 0, "frac_outside": np.nan}

    rel_err = (y / x) - 1.0
    outside = np.abs(rel_err) > band

    n_total = int(len(x))
    n_out = int(np.sum(outside))
    frac_out = float(n_out / n_total) if n_total else np.nan

    err_min_pct = float(np.min(rel_err) * 100.0)
    err_max_pct = float(np.max(rel_err) * 100.0)

    # Limits
    xy_min = float(min(np.min(x), np.min(y)))
    xy_max = float(max(np.max(x), np.max(y)))
    if xy_min == xy_max:
        xy_min *= 0.95
        xy_max *= 1.05
    pad = 0.05 * (xy_max - xy_min)
    lo = xy_min - pad
    hi = xy_max + pad

    fig, ax = plt.subplots(figsize=(8, 8))

    # Reference lines
    xx = np.linspace(lo, hi, 200)
    ax.plot(xx, xx, linewidth=1.4, label="_nolegend_")
    band_color = "0.5"
    ax.plot(xx, (1.0 + band) * xx, linestyle="--", linewidth=1.2, color=band_color, label="_nolegend_")
    ax.plot(xx, (1.0 - band) * xx, linestyle="--", linewidth=1.2, color=band_color, label=f"±{int(band*100)}%")

    # Color range
    vmin = float(np.nanmin(c)) if (c is not None and cmin is None) else cmin
    vmax = float(np.nanmax(c)) if (c is not None and cmax is None) else cmax

    band_label = f"±{int(band*100)}%"
    _scatter_split(
        ax, x, y, r, outside, band_label,
        color_values=c, cmap=cmap, vmin=vmin, vmax=vmax,
        point_size=point_size, fig=fig, color_label=color_label,
    )

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
    fig.savefig(out_path, format=out_path.suffix.lstrip("."))
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
    roles: np.ndarray | None = None,
    color_values: np.ndarray | None = None,
    color_label: str = "",
    cmap: str = "viridis",
    cmin: float | None = None,
    cmax: float | None = None,
    point_size: int | None = None,
):
    arrays = [x_meas, y_calc]
    if color_values is not None:
        arrays.append(color_values)
    m = _finite_mask(*arrays)

    x = x_meas[m]
    y = y_calc[m]
    c = color_values[m] if color_values is not None else None
    r = roles[m] if roles is not None else None

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

    fig, ax = plt.subplots(figsize=(8, 8))

    xx = np.linspace(lo, hi, 200)
    ax.plot(xx, xx, linewidth=1.4, label="_nolegend_")
    band_color = "0.5"
    ax.plot(xx, xx + band_abs, linestyle="--", linewidth=1.2, color=band_color, label="_nolegend_")
    ax.plot(xx, xx - band_abs, linestyle="--", linewidth=1.2, color=band_color, label=f"±{band_abs:.0f} K")

    vmin = float(np.nanmin(c)) if (c is not None and cmin is None) else cmin
    vmax = float(np.nanmax(c)) if (c is not None and cmax is None) else cmax

    band_label = f"±{band_abs:.0f} K"
    _scatter_split(
        ax, x, y, r, outside, band_label,
        color_values=c, cmap=cmap, vmin=vmin, vmax=vmax,
        point_size=point_size, fig=fig, color_label=color_label,
    )

    ax.set_title(title)

    info_txt = (
        f"Außerhalb ±{band_abs:.0f} K: {n_out} / {n_total} ({frac_out*100:.1f}%)\n"
        f"Fehlerspanne: {float(np.min(diff)):.2f} K bis {float(np.max(diff)):.2f} K"
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
    fig.savefig(out_path, format=out_path.suffix.lstrip("."))
    plt.close(fig)

    return {"n_total": n_total, "n_outside": n_out, "frac_outside": frac_out}


# =========================================================
# Column detection
# =========================================================
def _pick_pair(df: pd.DataFrame, candidates: list[tuple[str, str]]):
    """Return first (meas, calc) pair that exists, else None."""
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    return None


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Create parity plots from predictions, validation, or run_batch CSV.")
    ap.add_argument("--pred_csv", required=True, help="Path to CSV (predictions, validation, or run_batch output)")
    ap.add_argument("--out_dir", default="results/parity_plots", help="Output directory")

    ap.add_argument("--band", type=float, default=0.05, help="Relative error band (default 0.05 = ±5%)")
    ap.add_argument("--band_T_dis_abs", type=float, default=3.0, help="Absolute band for T_dis in K (default ±3)")

    ap.add_argument(
        "--color_by",
        choices=["superheat", "pressure_ratio", "none"],
        default="none",
        help="Color points by: superheat (superheat_C), pressure_ratio, or none",
    )
    ap.add_argument("--cmin", type=float, default=None, help="Fixed min for color scale")
    ap.add_argument("--cmax", type=float, default=None, help="Fixed max for color scale")
    ap.add_argument("--cmap", default=None, help="Override colormap (default depends on --color_by)")
    ap.add_argument("--point_size", type=int, default=None, help="Scatter point size")

    ap.add_argument(
        "--out_format",
        choices=["png", "svg"],
        default="png",
        help="Output format for plots",
    )

    args = ap.parse_args()

    csv_path = Path(args.pred_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)

    df = pd.read_csv(csv_path)
    stamp = _ts()
    src_name = csv_path.name

    # --- Detect split roles ---
    roles = _detect_split_role(df)
    if roles is not None:
        n_train = int(np.sum(roles == "train"))
        n_val = int(np.sum(roles == "validation"))
        print(f"  Split info detected: {n_train} training, {n_val} validation points")
    else:
        print("  No split info detected — all points treated equally.")

    # --- Resolve color ---
    color_vals, color_label, cmap = _resolve_color(df, args.color_by, args.cmap)
    if color_vals is not None:
        print(f"  Coloring by: {args.color_by} ({color_label})")

    # --- Column mapping ---
    m_pair = _pick_pair(df, [
        ("m_meas_gps", "m_calc_gps"),
        ("m_meas_g_s", "m_flow_g_s"),
    ])

    p_pair = _pick_pair(df, [
        ("P_meas_W", "P_calc_W"),
        ("P_meas_W", "P_el_W"),
    ])

    t_pair = _pick_pair(df, [
        ("T_dis_meas_C", "T_dis_calc_C"),
        ("T_dis_meas_C", "T_dis_C"),
    ])

    summary = []
    generated_any = False

    # --- Massenstrom ---
    if m_pair is not None:
        meas, calc = m_pair
        title = _auto_title(df, "Massenstrom")
        stats = parity_plot_rel_band(
            x_meas=df[meas].to_numpy(dtype=float),
            y_calc=df[calc].to_numpy(dtype=float),
            band=args.band,
            title=title,
            x_label="gemessener Massenstrom in g/s",
            y_label="berechneter Massenstrom in g/s",
            out_path=out_dir / f"parity_m_dot_{stamp}.{args.out_format}",
            roles=roles,
            color_values=color_vals,
            color_label=color_label,
            cmap=cmap,
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

    # --- Elektrische Leistung ---
    if p_pair is not None:
        meas, calc = p_pair
        title = _auto_title(df, "Elektrische Antriebsleistung")
        stats = parity_plot_rel_band(
            x_meas=df[meas].to_numpy(dtype=float),
            y_calc=df[calc].to_numpy(dtype=float),
            band=args.band,
            title=title,
            x_label="gemessene Antriebsleistung in W",
            y_label="berechnete Antriebsleistung in W",
            out_path=out_dir / f"parity_P_el_{stamp}.{args.out_format}",
            roles=roles,
            color_values=color_vals,
            color_label=color_label,
            cmap=cmap,
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

    # --- Austrittstemperatur ---
    if t_pair is not None:
        meas, calc = t_pair
        title = _auto_title(df, "Austrittstemperatur")
        stats = parity_plot_abs_band(
            x_meas=df[meas].to_numpy(dtype=float),
            y_calc=df[calc].to_numpy(dtype=float),
            band_abs=args.band_T_dis_abs,
            title=title,
            x_label="gemessene Austrittstemperatur in °C",
            y_label="berechnete Austrittstemperatur in °C",
            out_path=out_dir / f"parity_T_dis_{stamp}.{args.out_format}",
            roles=roles,
            color_values=color_vals,
            color_label=color_label,
            cmap=cmap,
            cmin=args.cmin,
            cmax=args.cmax,
            point_size=args.point_size,
        )
        stats.update({"metric": "T_dis", "x_col": meas, "y_col": calc, "source_file": src_name})
        summary.append(stats)
        generated_any = True
        print(f"[OK] T_dis plot: {meas} vs {calc}")
    else:
        print("[SKIP] T_dis plot: keine passenden Spalten gefunden.")

    # --- Summary ---
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_csv = out_dir / f"parity_summary_{stamp}.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[OK] Saved summary: {summary_csv}")

    if not generated_any:
        print("\n[ERROR] Keine Plots erzeugt. Gefundene Spalten:")
        print("        ", ", ".join(list(df.columns)[:25]), ("..." if len(df.columns) > 25 else ""))

    print("Done. Output dir:", out_dir)


if __name__ == "__main__":
    main()
