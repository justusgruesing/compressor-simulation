# scripts/plotting_scripts/t_dis_error_analysis.py
#
# Plots the discharge temperature error (T_dis_calc - T_dis_meas) against:
#   1. Pressure ratio (p_out / p_suc)
#   2. Rotational speed (N in rpm or f in Hz)
#   3. Oil mixture viscosity (mu_mix_eff in Pa*s or mu_oil in mPa*s)
#
# Input: validation detail CSV from validation.py
#
# Examples:
#   python scripts/plotting_scripts/t_dis_error_analysis.py --csv results/validation/validation_detail_params_lpg68_val_lpg68_modified_2026-03-19_173731.csv --band 5.0
#
#   # With custom band and output format:
#   python scripts/plotting_scripts/t_dis_error_analysis.py --csv results/validation/validation_detail_params_lpg68_val_all_modified_2026-03-19.csv --band 5.0 --out_format svg

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


def _finite_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _detect_split_role(df: pd.DataFrame) -> np.ndarray | None:
    if "split_role" in df.columns:
        return df["split_role"].fillna("").astype(str).to_numpy()
    if "is_train" in df.columns:
        return np.where(df["is_train"].fillna(False).astype(bool), "train", "validation")
    return None


def _auto_title_base(df: pd.DataFrame) -> str:
    """Generate subtitle from metadata columns."""
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
# Core plot function
# =========================================================
def plot_error_vs_x(
    x: np.ndarray,
    e_T: np.ndarray,
    roles: np.ndarray | None,
    band: float,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    point_size: int | None = None,
):
    """
    Scatter plot of T_dis error vs. an x-variable.
    Training points are hollow, validation points are filled.
    Horizontal band at ±band K shown as reference.
    """
    m = _finite_mask(x, e_T)
    x_plot = x[m]
    e_plot = e_T[m]
    r_plot = roles[m] if roles is not None else None

    if len(x_plot) == 0:
        print(f"  [SKIP] No finite data for: {x_label}")
        return

    has_split = r_plot is not None
    if has_split:
        is_train = (r_plot == "train")
        is_val = ~is_train
    else:
        is_train = np.zeros(len(x_plot), dtype=bool)
        is_val = np.ones(len(x_plot), dtype=bool)

    outside = np.abs(e_plot) > band
    n_total = len(x_plot)
    n_out = int(np.sum(outside))
    frac_out = n_out / n_total if n_total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))

    s = point_size

    # ±band reference
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(+band, color="0.5", linestyle="--", linewidth=1.0, label=f"±{band:.0f} K")
    ax.axhline(-band, color="0.5", linestyle="--", linewidth=1.0, label="_nolegend_")
    ax.axhspan(-band, +band, color="0.85", alpha=0.3, label="_nolegend_")

    # --- Validation (filled) ---
    mask_val_in = is_val & ~outside
    mask_val_out = is_val & outside

    if np.any(mask_val_in):
        ax.scatter(
            x_plot[mask_val_in], e_plot[mask_val_in],
            s=s, alpha=0.85, marker="o",
            label=f"Validation innerhalb ±{band:.0f} K",
        )
    if np.any(mask_val_out):
        ax.scatter(
            x_plot[mask_val_out], e_plot[mask_val_out],
            s=s, alpha=0.95, marker="s", linewidths=0.9,
            label=f"Validation außerhalb ±{band:.0f} K (n={int(mask_val_out.sum())})",
        )

    # --- Training (hollow) ---
    mask_tr_in = is_train & ~outside
    mask_tr_out = is_train & outside

    if np.any(mask_tr_in):
        ax.scatter(
            x_plot[mask_tr_in], e_plot[mask_tr_in],
            s=s, alpha=0.75, marker="o",
            facecolors="none", edgecolors="C0", linewidths=1.2,
            label=f"Training innerhalb ±{band:.0f} K",
        )
    if np.any(mask_tr_out):
        ax.scatter(
            x_plot[mask_tr_out], e_plot[mask_tr_out],
            s=s, alpha=0.85, marker="s",
            facecolors="none", edgecolors="C1", linewidths=1.2,
            label=f"Training außerhalb ±{band:.0f} K (n={int(mask_tr_out.sum())})",
        )

    # Info text
    mae = float(np.mean(np.abs(e_plot)))
    rmse = float(np.sqrt(np.mean(e_plot ** 2)))
    info_txt = (
        f"MAE: {mae:.2f} K  |  RMSE: {rmse:.2f} K\n"
        f"Außerhalb ±{band:.0f} K: {n_out}/{n_total} ({frac_out*100:.1f}%)"
    )
    ax.text(
        0.02, 0.98, info_txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."))
    plt.close(fig)

    print(f"  [OK] Saved: {out_path}")


# =========================================================
# Column detection
# =========================================================
def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Plot discharge temperature error against pressure ratio, speed, and viscosity."
    )
    ap.add_argument("--csv", required=True, type=Path, help="Validation detail CSV")
    ap.add_argument("--out_dir", default="results/t_dis_error_plots", help="Output directory")
    ap.add_argument("--band", type=float, default=3.0, help="Error band in K (default ±3)")
    ap.add_argument("--point_size", type=int, default=None, help="Scatter point size")
    ap.add_argument(
        "--out_format", choices=["png", "svg"], default="png", help="Output format"
    )

    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    stamp = _ts()

    # Filter to successful points
    if "success" in df.columns:
        df = df[df["success"] == True].copy()

    # --- Detect T_dis error column ---
    e_T_col = _find_col(df, ["e_T_dis_K"])
    if e_T_col is None:
        # Try to compute from available columns
        t_meas_col = _find_col(df, ["T_dis_meas_C"])
        t_calc_col = _find_col(df, ["T_dis_calc_C", "T_dis_C"])
        if t_meas_col and t_calc_col:
            df["e_T_dis_K"] = df[t_calc_col] - df[t_meas_col]
            e_T_col = "e_T_dis_K"
        else:
            raise ValueError(
                "Cannot find or compute T_dis error. "
                "Need 'e_T_dis_K' or both 'T_dis_meas_C' and 'T_dis_calc_C'."
            )

    e_T = df[e_T_col].to_numpy(dtype=float)

    # --- Split roles ---
    roles = _detect_split_role(df)
    if roles is not None:
        n_tr = int(np.sum(roles == "train"))
        n_val = int(np.sum(roles == "validation"))
        print(f"  Split info: {n_tr} training, {n_val} validation")

    # --- Auto title base ---
    subtitle = _auto_title_base(df)
    y_label = "Fehler Austrittstemperatur in K"

    # =============================================
    # 1. Error vs Pressure Ratio
    # =============================================
    pr_col = _find_col(df, ["pressure_ratio"])
    if pr_col is None:
        # Try to compute
        p_out_col = _find_col(df, ["p_out_bar", "p_out_bar_in"])
        p_suc_col = _find_col(df, ["p_suc_bar", "p_suc_bar_in"])
        if p_out_col and p_suc_col:
            df["pressure_ratio"] = df[p_out_col] / df[p_suc_col]
            pr_col = "pressure_ratio"

    if pr_col is not None:
        title = "Fehler Austrittstemperatur vs. Druckverhältnis"
        if subtitle:
            title += f"\n{subtitle}"

        plot_error_vs_x(
            x=df[pr_col].to_numpy(dtype=float),
            e_T=e_T,
            roles=roles,
            band=args.band,
            title=title,
            x_label="Druckverhältnis $p_{aus} / p_{ein}$ [-]",
            y_label=y_label,
            out_path=out_dir / f"e_Tdis_vs_pressure_ratio_{stamp}.{args.out_format}",
            point_size=args.point_size,
        )
    else:
        print("  [SKIP] Druckverhältnis: keine passenden Spalten gefunden.")

    # =============================================
    # 2. Error vs Speed
    # =============================================
    speed_col = _find_col(df, ["N_rpm", "N_rpm_in"])
    speed_label = "Drehzahl in rpm"

    if speed_col is None:
        speed_col = _find_col(df, ["f_oper_hz"])
        speed_label = "Drehfrequenz in Hz"

    if speed_col is not None:
        title = f"Fehler Austrittstemperatur vs. Drehzahl"
        if subtitle:
            title += f"\n{subtitle}"

        plot_error_vs_x(
            x=df[speed_col].to_numpy(dtype=float),
            e_T=e_T,
            roles=roles,
            band=args.band,
            title=title,
            x_label=speed_label,
            y_label=y_label,
            out_path=out_dir / f"e_Tdis_vs_speed_{stamp}.{args.out_format}",
            point_size=args.point_size,
        )
    else:
        print("  [SKIP] Drehzahl: keine passenden Spalten gefunden.")

    # =============================================
    # 3. Error vs Viscosity
    # =============================================
    # Prefer mu_mix_eff (Pa*s, from modified model), fallback to mu_oil (mPa*s)
    visc_col = _find_col(df, ["mu_mix_eff_Pas"])
    visc_label = "Effektive Mischungsviskosität $\\mu_{mix,eff}$ in Pa·s"

    if visc_col is None:
        visc_col = _find_col(df, ["mu_oil_mPas"])
        visc_label = "Dynamische Viskosität Öl $\\mu_{oil}$ in mPa·s"

    if visc_col is not None:
        # Check that column has finite values
        n_finite = int(np.sum(np.isfinite(df[visc_col].to_numpy(dtype=float))))
        if n_finite == 0:
            print(f"  [SKIP] Viskosität ({visc_col}): keine finiten Werte vorhanden.")
        else:
            title = "Fehler Austrittstemperatur vs. Viskosität"
            if subtitle:
                title += f"\n{subtitle}"

            plot_error_vs_x(
                x=df[visc_col].to_numpy(dtype=float),
                e_T=e_T,
                roles=roles,
                band=args.band,
                title=title,
                x_label=visc_label,
                y_label=y_label,
                out_path=out_dir / f"e_Tdis_vs_viscosity_{stamp}.{args.out_format}",
                point_size=args.point_size,
            )
    else:
        print("  [SKIP] Viskosität: keine passenden Spalten gefunden (nur für modified Modell verfügbar).")

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
