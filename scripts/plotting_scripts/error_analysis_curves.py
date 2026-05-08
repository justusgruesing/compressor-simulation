# scripts/plotting_scripts/error_analysis_curves.py
#
# Error analysis plots in the style of Giuffrida (2016) Fig. 6.
#
# For each of the three target quantities (m_dot, P_el, T_dis) one separate
# plot is created showing the prediction error vs. an operating parameter.
# The default x-axes follow Giuffrida's convention:
#   - m_dot error  vs. evaporation temperature
#   - P_el error   vs. pressure ratio
#   - T_dis error  vs. pressure ratio
# Each x-axis is independently configurable via CLI.
#
# Features:
#   - Tolerance bands (±3 % / ±3 K) drawn as horizontal dashed lines
#   - Train (hollow) / Validation (filled) marker distinction
#   - Optional coloring by a third variable (superheat, speed, etc.)
#   - Statistics box per plot (MAE, RMSE, max error, share within band)
#
# Input: validation_detail_*.csv produced by validation.py
#
# Activate REFPROP only if needed (--color_by T_cond):
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Default Giuffrida-style:
#   python scripts/plotting_scripts/error_analysis_curves.py --pred_csv results/validation/oil_path/detail/validation_detail_params_all_val_all_oil_path_validation_only_2026-05-07_112413.csv --x_m_dot pressure_ratio --color_by superheat
#
#   # Override x-axes:
#   python scripts/plotting_scripts/error_analysis_curves.py \
#       --pred_csv ... --x_m_dot pressure_ratio --x_P_el T_evap --x_T_dis T_cond
#
#   # Color by superheat:
#   python scripts/plotting_scripts/error_analysis_curves.py \
#       --pred_csv ... --color_by superheat

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


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _finite_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _detect_split_role(df: pd.DataFrame):
    """Detect train/validation role per row."""
    if "split_role" in df.columns:
        return df["split_role"].fillna("").astype(str).to_numpy()
    if "is_train" in df.columns:
        return np.where(
            df["is_train"].fillna(False).astype(bool), "train", "validation"
        )
    return None


def _model_display_name(model_str) -> str:
    """Map model strings to thesis display names."""
    s = str(model_str).strip().lower()
    mapping = {
        "original": "Basismodell", "orig": "Basismodell",
        "modified": "Modellausbaustufe I", "mod": "Modellausbaustufe I",
        "oil_path": "Modellausbaustufe II", "oilpath": "Modellausbaustufe II",
    }
    return mapping.get(s, str(model_str).capitalize())


def _oil_display_name(oil_str) -> str:
    """Map oil strings to thesis display names."""
    s = str(oil_str).strip().lower().replace(" ", "")
    mapping = {
        "lpg68": "PAG68", "lpg 68": "PAG68",
        "lpg100": "PAG100", "lpg 100": "PAG100",
        "all": "beide",
    }
    return mapping.get(s, str(oil_str))


TARGET_DISPLAY = {
    "m_dot": "Massenstromfehler",
    "P_el": "Leistungsfehler",
    "T_dis": "Austrittstemperaturfehler",
}


def _auto_title(df: pd.DataFrame, target_key: str) -> str:
    """
    Generate a two-line title in thesis format:
    Line 1: <Zielgröße> <Modell>
    Line 2: Kalibrierung: <params_oil> → Validierung: <val_oil>
    """
    target_disp = TARGET_DISPLAY.get(target_key, target_key)

    model = ""
    if "model" in df.columns:
        vals = df["model"].dropna().unique()
        if len(vals) == 1:
            model = _model_display_name(vals[0])

    params_oil = ""
    if "params_oil" in df.columns:
        vals = df["params_oil"].dropna().unique()
        if len(vals) == 1:
            params_oil = _oil_display_name(str(vals[0]))

    val_oil = ""
    if "oil" in df.columns:
        vals = df["oil"].dropna().unique()
        if len(vals) == 1:
            val_oil = _oil_display_name(str(vals[0]))
        elif len(vals) > 1:
            val_oil = "beide"
    elif "oil_norm" in df.columns:
        vals = df["oil_norm"].dropna().unique()
        if len(vals) == 1:
            val_oil = _oil_display_name(str(vals[0]))
        elif len(vals) > 1:
            val_oil = "beide"

    line1 = target_disp
    if model:
        line1 += f" {model}"

    line2 = ""
    if params_oil and val_oil:
        line2 = f"Kalibrierung: {params_oil} \u2192 Validierung: {val_oil}"
    elif params_oil:
        line2 = f"Kalibrierung: {params_oil}"
    elif val_oil:
        line2 = f"Validierung: {val_oil}"

    if line2:
        return f"{line1}\n{line2}"
    return line1


# =========================================================
# T_cond computation from p_out via RefProp
# =========================================================
_REFPROP_INSTANCES = {}


def _get_refprop(fluid_name: str):
    if fluid_name in _REFPROP_INSTANCES:
        return _REFPROP_INSTANCES[fluid_name]
    try:
        from vclibpy.media import RefProp
        med = RefProp(fluid_name=fluid_name)
        _REFPROP_INSTANCES[fluid_name] = med
        return med
    except Exception as e:
        print(f"  [WARN] Could not load RefProp for '{fluid_name}': {e}")
        _REFPROP_INSTANCES[fluid_name] = None
        return None


def _compute_T_cond_from_p_out(df: pd.DataFrame) -> np.ndarray:
    """Compute T_cond [°C] from p_out_bar using RefProp saturation."""
    if "p_out_bar" not in df.columns:
        return np.full(len(df), np.nan)

    refrigerant = "PROPANE"
    if "refrigerant" in df.columns:
        vals = df["refrigerant"].dropna().unique()
        if len(vals) == 1:
            refrigerant = str(vals[0])

    med = _get_refprop(refrigerant)
    if med is None:
        return np.full(len(df), np.nan)

    p_out_bar = df["p_out_bar"].to_numpy(dtype=float)
    T_cond_C = np.full(len(p_out_bar), np.nan)

    print(f"  Computing T_cond from p_out using RefProp ({refrigerant}) ...")
    for i, p_bar in enumerate(p_out_bar):
        if not np.isfinite(p_bar) or p_bar <= 0:
            continue
        try:
            state_sat = med.calc_state("PQ", float(p_bar) * 1e5, 0.0)
            T_cond_C[i] = float(state_sat.T) - 273.15
        except Exception:
            pass
    return T_cond_C


# =========================================================
# X-axis configuration
# =========================================================
X_AXIS_CONFIG = {
    "T_evap": {
        "col": "T_sat_suc_C",
        "label": "Verdampfungstemperatur in \u00b0C",
        "computed": False,
    },
    "T_cond": {
        "col": "_T_cond_computed",
        "label": "Kondensationstemperatur in \u00b0C",
        "computed": True,
    },
    "speed": {
        "col": "N_rpm",
        "label": "Drehzahl in rpm",
        "computed": False,
    },
    "superheat": {
        "col": "superheat_C",
        "label": "\u00dcberhitzung in K",
        "computed": False,
    },
    "pressure_ratio": {
        "col": "pressure_ratio",
        "label": "Druckverh\u00e4ltnis $p_{aus}/p_{ein}$",
        "computed": False,
    },
    "P_el": {
        "col": "P_meas_W",
        "label": "Gemessene elektrische Leistung $P_{el}$ in W",
        "computed": False,
    },
}


def _resolve_x_axis(df: pd.DataFrame, axis_name: str, T_cond_cache):
    """Return (x_array, label) or (None, None) if not available."""
    cfg = X_AXIS_CONFIG.get(axis_name)
    if cfg is None:
        print(f"  [WARN] Unknown x-axis '{axis_name}'.")
        return None, None

    if cfg["computed"] and axis_name == "T_cond":
        if T_cond_cache[0] is None:
            T_cond_cache[0] = _compute_T_cond_from_p_out(df)
        if not np.any(np.isfinite(T_cond_cache[0])):
            return None, None
        return T_cond_cache[0], cfg["label"]

    col = cfg["col"]
    if col not in df.columns:
        print(f"  [WARN] Column '{col}' for x-axis '{axis_name}' not found.")
        return None, None
    return df[col].to_numpy(dtype=float), cfg["label"]


# =========================================================
# Color column mapping (same as parity_plot)
# =========================================================
COLOR_CONFIG = {
    "superheat": {
        "col": "superheat_C",
        "label": "\u00dcberhitzung in K",
        "cmap_default": "viridis",
        "computed": False,
    },
    "pressure_ratio": {
        "col": "pressure_ratio",
        "label": "Druckverh\u00e4ltnis",
        "cmap_default": "viridis",
        "computed": False,
    },
    "T_evap": {
        "col": "T_sat_suc_C",
        "label": "Verdampfungstemperatur in \u00b0C",
        "cmap_default": "viridis",
        "computed": False,
    },
    "T_cond": {
        "col": "_T_cond_computed",
        "label": "Kondensationstemperatur in \u00b0C",
        "cmap_default": "viridis",
        "computed": True,
    },
    "speed": {
        "col": "N_rpm",
        "label": "Drehzahl in rpm",
        "cmap_default": "viridis",
        "computed": False,
    },
}


def _resolve_color(df: pd.DataFrame, color_by: str, cmap_override, T_cond_cache):
    if color_by == "none" or color_by is None:
        return None, "", "viridis"

    cfg = COLOR_CONFIG.get(color_by)
    if cfg is None:
        print(f"  [WARN] Unknown --color_by '{color_by}', falling back to none.")
        return None, "", "viridis"

    if cfg["computed"] and color_by == "T_cond":
        if T_cond_cache[0] is None:
            T_cond_cache[0] = _compute_T_cond_from_p_out(df)
        if not np.any(np.isfinite(T_cond_cache[0])):
            return None, "", "viridis"
        cmap = cmap_override if cmap_override else cfg["cmap_default"]
        return T_cond_cache[0], cfg["label"], cmap

    col = cfg["col"]
    if col not in df.columns:
        print(f"  [WARN] Column '{col}' not found for color, falling back to none.")
        return None, "", "viridis"

    cmap = cmap_override if cmap_override else cfg["cmap_default"]
    return df[col].to_numpy(dtype=float), cfg["label"], cmap


# =========================================================
# Categorical color modes (discrete groups)
# =========================================================
CATEGORICAL_MODES = {"oil"}

CATEGORICAL_COLORS = {
    "oil": {
        "groups": {
            "PAG68": "#EC635C",
            "PAG100": "#4B81C4",
        },
    },
}


def _resolve_categorical_groups(df, color_by):
    """Return (group_labels, group_config) or (None, None) if not categorical."""
    if color_by not in CATEGORICAL_MODES:
        return None, None

    if color_by == "oil":
        oil_col = None
        for c in ("oil_norm", "oil"):
            if c in df.columns:
                oil_col = c
                break
        if oil_col is None:
            print("  [WARN] No oil column found for oil coloring.")
            return None, None
        raw = df[oil_col].fillna("").astype(str)
        labels = raw.apply(lambda s: "PAG68" if "68" in s else ("PAG100" if "100" in s else s))
        return labels.to_numpy(), CATEGORICAL_COLORS["oil"]

    return None, None


# =========================================================
# Core plot function
# =========================================================
def plot_error_curve(
    x: np.ndarray,
    y_err: np.ndarray,
    band_abs: float,
    title: str,
    x_label: str,
    y_label: str,
    band_label: str,
    out_path: Path,
    *,
    roles=None,
    color_values=None,
    color_label: str = "",
    cmap: str = "viridis",
    cmin=None,
    cmax=None,
    point_size=None,
    err_unit: str = "%",
    cat_labels=None,
    cat_config=None,
):
    """
    Generic error vs. x scatter with tolerance band.
    Supports continuous coloring, categorical coloring (oil), and train/val split.
    """
    arrays = [x, y_err]
    if color_values is not None:
        arrays.append(color_values)
    m = _finite_mask(*arrays)

    x = x[m]
    y = y_err[m]
    c = color_values[m] if color_values is not None else None
    r = roles[m] if roles is not None else None
    cl = cat_labels[m] if cat_labels is not None else None

    if len(x) == 0:
        print(f"  [SKIP] {out_path.name}: no valid data.")
        return

    outside = np.abs(y) > band_abs

    # Plot ranges
    x_pad = 0.03 * (np.max(x) - np.min(x) + 1e-9)
    x_lo = float(np.min(x) - x_pad)
    x_hi = float(np.max(x) + x_pad)

    y_max_abs = max(float(np.max(np.abs(y))), band_abs * 1.5)
    y_lo = -y_max_abs * 1.1
    y_hi = y_max_abs * 1.1

    fig, ax = plt.subplots(figsize=(9, 6))

    # Reference lines
    ax.axhline(0.0, color="black", linewidth=1.2, zorder=2)
    ax.axhline(+band_abs, color="0.5", linestyle="--", linewidth=1.0,
               zorder=2, label=f"\u00b1{band_label}")
    ax.axhline(-band_abs, color="0.5", linestyle="--", linewidth=1.0, zorder=2)
    ax.axhspan(-band_abs, +band_abs, color="0.5", alpha=0.10, zorder=1)

    # === Categorical coloring (e.g. oil) ===
    if cl is not None and cat_config is not None:
        groups = cat_config["groups"]
        for group_name, color in groups.items():
            mask_group = (cl == group_name)
            if np.any(mask_group):
                ax.scatter(
                    x[mask_group], y[mask_group],
                    s=point_size, alpha=0.85, marker="o",
                    color=color, edgecolors="none",
                    zorder=3, label=group_name,
                )

    # === Continuous coloring ===
    else:
        has_split = r is not None
        has_color = (c is not None and np.any(np.isfinite(c)))

        if has_color:
            vmin = float(np.nanmin(c)) if cmin is None else cmin
            vmax = float(np.nanmax(c)) if cmax is None else cmax
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
        else:
            vmin = vmax = None

        if has_split:
            is_train = (r == "train")
            is_val = ~is_train
        else:
            is_train = np.zeros(len(x), dtype=bool)
            is_val = np.ones(len(x), dtype=bool)

        n_train = int(is_train.sum())
        n_val = int(is_val.sum())
        train_only = (n_train > 0 and n_val == 0)

        sc_ref = None

        if has_color:
            cmap_obj = plt.get_cmap(cmap)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            if train_only:
                sc_ref = ax.scatter(
                    x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=point_size, alpha=0.90, marker="o", edgecolors="none",
                    zorder=3,
                )
            else:
                if np.any(is_val):
                    sc_ref = ax.scatter(
                        x[is_val], y[is_val],
                        c=c[is_val], cmap=cmap, vmin=vmin, vmax=vmax,
                        s=point_size, alpha=0.90, marker="o", edgecolors="none",
                        zorder=3, label="Validierung",
                    )
                if np.any(is_train):
                    edge_colors = cmap_obj(norm(c[is_train]))
                    ax.scatter(
                        x[is_train], y[is_train],
                        s=point_size, alpha=0.85, marker="o",
                        facecolors="none", edgecolors=edge_colors, linewidths=1.2,
                        zorder=3, label="Training",
                    )

            # Colorbar
            if sc_ref is not None:
                cbar = fig.colorbar(sc_ref, ax=ax, pad=0.02)
                cbar.set_label(color_label)
            else:
                sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label(color_label)
        else:
            if train_only:
                ax.scatter(
                    x, y, s=point_size, alpha=0.85, marker="o",
                    color="#4B81C4", edgecolors="none", zorder=3,
                )
            else:
                if np.any(is_val):
                    ax.scatter(
                        x[is_val], y[is_val],
                        s=point_size, alpha=0.85, marker="o",
                        color="#4B81C4", edgecolors="none",
                        zorder=3, label="Validierung",
                    )
                if np.any(is_train):
                    ax.scatter(
                        x[is_train], y[is_train],
                        s=point_size, alpha=0.80, marker="o",
                        facecolors="none", edgecolors="#EC635C", linewidths=1.2,
                        zorder=3, label="Training",
                    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(title)
    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Error analysis plots vs. operating parameters (Giuffrida 2016 style)."
    )
    ap.add_argument("--pred_csv", required=True, type=Path,
                    help="validation_detail_*.csv from validation.py")
    ap.add_argument("--out_dir", default="results/error_analysis", type=Path)
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    # Tolerance bands
    ap.add_argument("--band_m", type=float, default=3.0,
                    help="Tolerance band for m_dot error in %% (default 3)")
    ap.add_argument("--band_P", type=float, default=3.0,
                    help="Tolerance band for P_el error in %% (default 3)")
    ap.add_argument("--band_T", type=float, default=3.0,
                    help="Tolerance band for T_dis error in K (default 3)")

    # X-axis selection per metric
    x_choices = ["T_evap", "T_cond", "speed", "superheat", "pressure_ratio", "P_el"]
    ap.add_argument("--x_m_dot", choices=x_choices, default="T_evap",
                    help="X-axis for m_dot error plot (default: T_evap, like Giuffrida)")
    ap.add_argument("--x_P_el", choices=x_choices, default="pressure_ratio",
                    help="X-axis for P_el error plot (default: pressure_ratio)")
    ap.add_argument("--x_T_dis", choices=x_choices, default="pressure_ratio",
                    help="X-axis for T_dis error plot (default: pressure_ratio)")

    # Color
    ap.add_argument(
        "--color_by",
        choices=["superheat", "pressure_ratio", "T_evap", "T_cond", "speed", "oil", "none"],
        default="none",
        help="Color points by additional variable. "
             "'oil' uses categorical colors for PAG68/PAG100.",
    )
    ap.add_argument("--cmin", type=float, default=None)
    ap.add_argument("--cmax", type=float, default=None)
    ap.add_argument("--cmap", default=None,
                    help="Override colormap (default depends on --color_by)")
    ap.add_argument("--point_size", type=int, default=None)

    args = ap.parse_args()

    if not args.pred_csv.exists():
        raise FileNotFoundError(args.pred_csv)

    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)

    df = pd.read_csv(args.pred_csv)
    stamp = _ts()

    # Detect roles
    roles = _detect_split_role(df)
    if roles is not None:
        n_train = int(np.sum(roles == "train"))
        n_val = int(np.sum(roles == "validation"))
        print(f"  Split: {n_train} training, {n_val} validation points")
    else:
        print("  No split info detected.")

    # T_cond cache (computed lazily, at most once)
    T_cond_cache = [None]

    # Resolve color — categorical or continuous
    cat_labels, cat_config = _resolve_categorical_groups(df, args.color_by)
    is_categorical = cat_labels is not None

    if is_categorical:
        color_vals, color_label, cmap = None, "", "viridis"
        unique_groups = np.unique(cat_labels)
        print(f"  Coloring by: {args.color_by} (categorical: {', '.join(unique_groups)})")
    else:
        color_vals, color_label, cmap = _resolve_color(
            df, args.color_by, args.cmap, T_cond_cache,
        )
        if color_vals is not None:
            print(f"  Coloring by: {args.color_by} ({color_label})")

    # Build per-metric configurations
    plot_specs = []

    # --- m_dot error ---
    if "e_m_rel" in df.columns:
        x_arr, x_lbl = _resolve_x_axis(df, args.x_m_dot, T_cond_cache)
        if x_arr is not None:
            y = df["e_m_rel"].to_numpy(dtype=float) * 100.0  # to percent
            title = _auto_title(df, "m_dot")
            plot_specs.append({
                "x": x_arr,
                "y": y,
                "band": args.band_m,
                "x_label": x_lbl,
                "y_label": "$e_{\\dot{m}}$ in %",
                "band_label": f"{args.band_m:.0f}%",
                "title": title,
                "out_path": out_dir / f"err_m_dot_vs_{args.x_m_dot}_{stamp}.{args.out_format}",
                "err_unit": "%",
            })
    else:
        print("  [SKIP] m_dot: column 'e_m_rel' not found.")

    # --- P_el error ---
    if "e_P_rel" in df.columns:
        x_arr, x_lbl = _resolve_x_axis(df, args.x_P_el, T_cond_cache)
        if x_arr is not None:
            y = df["e_P_rel"].to_numpy(dtype=float) * 100.0
            title = _auto_title(df, "P_el")
            plot_specs.append({
                "x": x_arr,
                "y": y,
                "band": args.band_P,
                "x_label": x_lbl,
                "y_label": "$e_{P_{el}}$ in %",
                "band_label": f"{args.band_P:.0f}%",
                "title": title,
                "out_path": out_dir / f"err_P_el_vs_{args.x_P_el}_{stamp}.{args.out_format}",
                "err_unit": "%",
            })
    else:
        print("  [SKIP] P_el: column 'e_P_rel' not found.")

    # --- T_dis error ---
    if "e_T_dis_K" in df.columns:
        x_arr, x_lbl = _resolve_x_axis(df, args.x_T_dis, T_cond_cache)
        if x_arr is not None:
            y = df["e_T_dis_K"].to_numpy(dtype=float)  # already in K
            title = _auto_title(df, "T_dis")
            plot_specs.append({
                "x": x_arr,
                "y": y,
                "band": args.band_T,
                "x_label": x_lbl,
                "y_label": "$\\Delta T_{dis}$ in K",
                "band_label": f"{args.band_T:.0f} K",
                "title": title,
                "out_path": out_dir / f"err_T_dis_vs_{args.x_T_dis}_{stamp}.{args.out_format}",
                "err_unit": "K",
            })
    else:
        print("  [SKIP] T_dis: column 'e_T_dis_K' not found.")

    if not plot_specs:
        raise RuntimeError("No plots could be generated. Check input columns.")

    # Generate plots
    for spec in plot_specs:
        plot_error_curve(
            x=spec["x"],
            y_err=spec["y"],
            band_abs=spec["band"],
            title=spec["title"],
            x_label=spec["x_label"],
            y_label=spec["y_label"],
            band_label=spec["band_label"],
            out_path=spec["out_path"],
            roles=roles if not is_categorical else None,
            color_values=color_vals,
            color_label=color_label,
            cmap=cmap,
            cmin=args.cmin,
            cmax=args.cmax,
            point_size=args.point_size,
            err_unit=spec["err_unit"],
            cat_labels=cat_labels,
            cat_config=cat_config,
        )

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
