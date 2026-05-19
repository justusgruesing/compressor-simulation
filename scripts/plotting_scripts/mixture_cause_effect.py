# scripts/plotting_scripts/mixture_cause_effect.py
#
# Two-panel cause-effect figure for the defense slide on
# refrigerant-lubricant interaction (replaces the Daniel diagram).
#
#   Left:   Solubility w_KM as function of pressure p, isotherms at T = const
#           for ONE oil.  Take-home:  "Druck steigt -> Loeslichkeit steigt."
#   Right:  Dynamic viscosity mu as function of w_KM at a fixed reference
#           temperature, for BOTH oils.  Take-home: "Loeslichkeit steigt
#           -> Viskositaet sinkt."
#
# The right panel sweeps w_KM directly via the viscosity correlation
# (without solving the solubility equilibrium), because the message is
# the structural mu(w_KM) relationship, not the (T,p) -> w_KM coupling.
#
# Activate REFPROP first, then:
#   python scripts/plotting_scripts/mixture_cause_effect.py
#
# Examples:
#   # Default (PAG 68 isotherms left, PAG 68 + PAG 100 on right at 50 C):
#   python scripts/plotting_scripts/mixture_cause_effect.py
#
#   # Custom isotherms and viscosity reference temperature:
#   python scripts/plotting_scripts/mixture_cause_effect.py \
#       --T_isotherms 20 50 80 --T_visc 60
#
#   # Linear viscosity axis and SVG output:
#   python scripts/plotting_scripts/mixture_cause_effect.py \
#       --linear_visc --out_format svg

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from vclibpy.media import RefProp

# Import lubricant fitting (same fallback chain as the reference script)
try:
    from vclibpy.media.lubricant_fitting_shared_refprop import (
        LubricantFitting as SharedLubricantFitting,
    )
    LUBRICANT_AVAILABLE = True
except ImportError:
    try:
        from vclibpy.media.lubricant_fitting import LubricantFitting as SharedLubricantFitting
        LUBRICANT_AVAILABLE = True
    except ImportError:
        LUBRICANT_AVAILABLE = False
        SharedLubricantFitting = None

# EBC paper style if available; otherwise fall back to defaults
try:
    plt.style.use("ebc.paper.mplstyle")
except OSError:
    pass


# =========================================================
# Constants
# =========================================================
# Isotherm colors (left panel) — reuse EBC palette from the existing script
ISOTHERM_COLORS = [
    "#EC635C",  # red
    "#4B81C4",  # blue
    "#6EBB96",  # green
    "#8768B4",  # purple
    "#F49961",  # orange
]

# Oil colors (right panel) — consistent with the presentation palette
OIL_COLORS = {
    "LPG 68":  "#1B3A4B",  # deep petrol (matches slide deck primary)
    "LPG 100": "#E07A1F",  # warm orange (matches slide deck accent)
}

OIL_DISPLAY = {
    "LPG68": "PAG 68", "LPG100": "PAG 100",
    "lpg68": "PAG 68", "lpg100": "PAG 100",
    "LPG 68": "PAG 68", "LPG 100": "PAG 100",
}
REFRIGERANT_DISPLAY = {
    "propane": "Propan (R-290)", "PROPANE": "Propan (R-290)",
}


# =========================================================
# Helpers
# =========================================================
def _ts():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def c_to_k(t):
    return float(t) + 273.15


def map_oil(name):
    s = str(name).strip().lower().replace(" ", "")
    if s == "lpg68":
        return "LPG 68"
    if s == "lpg100":
        return "LPG 100"
    raise ValueError(f"Unsupported oil: {name}")


def map_refrigerant(name):
    s = str(name).strip().upper()
    return "propane" if s in {"PROPANE", "R290", "PROPAN"} else str(name).strip()


def build_lubricant(fluid_name, lub_name, med=None):
    """Build and return a LubricantFitting model with RefProp injected."""
    if not LUBRICANT_AVAILABLE:
        raise ImportError("LubricantFitting not available in this vclibpy installation.")
    if med is None:
        med = RefProp(fluid_name=fluid_name)
    return SharedLubricantFitting(
        fluid_name=fluid_name, lub_name=lub_name, shared_refprop=med
    )


# =========================================================
# Compute: solubility isotherms on the pressure axis
# =========================================================
def compute_solubility_isotherms(lubricant, T_C_list, p_bar_array, debug=False):
    """
    For each isotherm T (in T_C_list), sweep over pressure and compute
    equilibrium w_KM(p).

    Returns:
        dict {T_C: {"p_bar": np.ndarray, "w_KM": np.ndarray}}
    """
    results = {}
    for T_C in T_C_list:
        T_K = c_to_k(T_C)
        p_arr, w_arr = [], []
        n_err, last_err = 0, None
        for p_bar in p_bar_array:
            p_Pa = float(p_bar) * 1e5
            try:
                w = lubricant.solve_w_KM(T_K, p_Pa)
                if w is not None and 0.0 <= w <= 1.0:
                    p_arr.append(float(p_bar))
                    w_arr.append(float(w))
                elif debug and n_err < 3:
                    print(f"    [DEBUG] solve_w_KM(T={T_C:.1f} C, p={p_bar:.1f} bar) "
                          f"returned {w}")
            except Exception as e:
                n_err += 1
                last_err = e
                if debug and n_err <= 3:
                    print(f"    [DEBUG] solve_w_KM(T={T_C:.1f} C, p={p_bar:.1f} bar) "
                          f"raised {type(e).__name__}: {e}")
        if n_err > 0 and debug:
            print(f"    [DEBUG] T={T_C:.1f} C: {n_err} exceptions total, "
                  f"last: {type(last_err).__name__}: {last_err}")
        results[float(T_C)] = {
            "p_bar": np.array(p_arr),
            "w_KM":  np.array(w_arr),
        }
    return results


# =========================================================
# Compute: viscosity vs w_KM at fixed temperature
# =========================================================
def calc_dyn_vis_at_w(lubricant, T_K, w_KM):
    """
    Dynamic viscosity [mPa*s] directly from the lubricant's correlation
    at given T and w_KM, BYPASSING solve_w_KM.

    Replicates the math used in LubricantFitting.calc_transport_properties
    for phase='liquid' exactly.
    """
    logT = math.log10(T_K)
    y = (
        (lubricant.dyn_visc_a + lubricant.dyn_visc_b * logT
         + lubricant.dyn_visc_c * logT ** 2)
        + w_KM * (lubricant.dyn_visc_d + lubricant.dyn_visc_e * logT
                  + lubricant.dyn_visc_f * logT ** 2)
        + w_KM ** 2 * (lubricant.dyn_visc_g + lubricant.dyn_visc_h * logT
                       + lubricant.dyn_visc_i * logT ** 2)
    )
    return math.pow(10.0, math.pow(10.0, y)) - 0.7  # mPa*s


def compute_viscosity_vs_w(lubricant, T_C, w_array, debug=False):
    """
    Compute mu(w_KM) at a single temperature, sweeping w_KM directly.

    Returns:
        dict with arrays "w_KM" (-) and "mu_mPas" (mPa*s).
    """
    T_K = c_to_k(T_C)
    mu_arr = np.full_like(w_array, np.nan, dtype=float)
    for i, w in enumerate(w_array):
        try:
            mu = calc_dyn_vis_at_w(lubricant, T_K, float(w))
            if np.isfinite(mu) and mu > 0:
                mu_arr[i] = mu
            elif debug:
                print(f"    [DEBUG] mu(T={T_C} C, w={w:.3f}) = {mu}")
        except Exception as e:
            if debug:
                print(f"    [DEBUG] mu(T={T_C} C, w={w:.3f}) raised "
                      f"{type(e).__name__}: {e}")
    return {"w_KM": np.asarray(w_array, dtype=float), "mu_mPas": mu_arr}


# =========================================================
# Plot: two-panel cause-effect figure
# =========================================================
def plot_cause_effect(
    sol_data, vis_data, oil_for_sol, T_visc_C,
    out_paths, log_visc=True, show_titles=False, figsize=(14, 5.6),
):
    """
    Two-panel cause-effect figure.

    Args:
        sol_data: dict {T_C: {"p_bar", "w_KM"}}  — left panel (isotherms)
        vis_data: dict {lub_name: {"w_KM", "mu_mPas"}}  — right panel (two oils)
        oil_for_sol: lubricant name used for solubility panel (for subtitle)
        T_visc_C: temperature [C] used for viscosity panel (for subtitle)
        out_paths: pathlib.Path OR list of Paths. The figure is rendered once
            and saved to every path; the file format is inferred from each
            path's suffix (.png / .svg / .pdf).
        log_visc: True -> log y-axis for viscosity (recommended)
        show_titles: include panel subtitles (off by default — slide gives context)
        figsize: figure size in inches (default fits well next to text on a 16:9 slide)
    """
    # Normalize to list so callers can pass a single Path or a list of Paths.
    if isinstance(out_paths, (str, Path)):
        out_paths = [Path(out_paths)]
    else:
        out_paths = [Path(p) for p in out_paths]
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"wspace": 0.30}
    )

    # ---------------- Left panel: p(w_KM), isotherms ----------------
    # Convention follows Sun et al.: w_KM on x-axis, p on y-axis.
    # The saturation asymptote (curve approaches p_sat at high w_KM) now
    # appears as a clean horizontal plateau at the top, instead of a
    # near-vertical line at the right edge.
    for i, T_C in enumerate(sorted(sol_data.keys())):
        curve = sol_data[T_C]
        color = ISOTHERM_COLORS[i % len(ISOTHERM_COLORS)]
        if len(curve["p_bar"]) > 0:
            ax_l.plot(
                curve["w_KM"] * 100.0, curve["p_bar"],
                color=color, linewidth=2.4,
                label=f"$T$ = {T_C:.0f} °C",
            )

    ax_l.set_xlabel("Massenanteil $w_{\\mathrm{KM}}$ in %")
    ax_l.set_ylabel("Druck $p$ in bar")
    ax_l.grid(True, linewidth=0.5, alpha=0.4)
    ax_l.set_xlim(left=0)
    ax_l.set_ylim(bottom=0)
    # Legend in upper-left (low w, high p region is empty for all isotherms)
    ax_l.legend(loc="upper left", fontsize=10, frameon=False)

    if show_titles:
        ax_l.set_title(f"Propan / {OIL_DISPLAY.get(oil_for_sol, oil_for_sol)}", pad=10)

    # Take-home annotation: BOTH cause-effect relationships in one box,
    # placed in the lower-right (high w, low p region is empty after the swap).
    # Monospace font + padding align the two "↑" trigger arrows vertically:
    # "Druck     " is padded to 10 chars to match "Temperatur".
    ax_l.text(
        0.97, 0.05,
        "Druck      ↑  →  Löslichkeit ↑\n"
        "Temperatur ↑  →  Löslichkeit ↓",
        transform=ax_l.transAxes, ha="right", va="bottom",
        fontsize=10, style="italic", color="#1B3A4B",
        family="monospace",
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F7FA",
                  edgecolor="#D5DCE2", linewidth=0.6),
    )

    # ---------------- Right panel: mu(w_KM), both oils ----------------
    for lub_key, data in vis_data.items():
        color = OIL_COLORS.get(lub_key, "#333333")
        # Mask non-finite values for clean plotting
        m = np.isfinite(data["mu_mPas"])
        ax_r.plot(
            data["w_KM"][m] * 100.0, data["mu_mPas"][m],
            color=color, linewidth=2.6,
            label=OIL_DISPLAY.get(lub_key, lub_key),
        )

    ax_r.set_xlabel("Massenanteil $w_{\\mathrm{KM}}$ in %")
    ax_r.set_ylabel("Dynamische Viskosität $\\mu$ in mPa·s")
    ax_r.grid(True, which="both", linewidth=0.5, alpha=0.4)
    ax_r.set_xlim(left=0)
    ax_r.legend(loc="upper right", fontsize=11, frameon=False)

    if log_visc:
        ax_r.set_yscale("log")
        # Major ticks at 1, 2, 5 of each decade → dense, conventional log labels.
        # Minor ticks at remaining integer multiples (3, 4, 6, 7, 8, 9) → the
        # uneven spacing makes it visually obvious that the scale is logarithmic.
        ax_r.yaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12)
        )
        ax_r.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs="auto", numticks=20)
        )
        ax_r.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_r.yaxis.set_minor_formatter(mticker.NullFormatter())

    if show_titles:
        ax_r.set_title(f"$T$ = {T_visc_C:.0f} °C", pad=10)

    # Take-home annotation — bottom-left (curves decrease from upper-left,
    # so lower-left is the empty quadrant; legend stays in upper-right)
    ax_r.text(
        0.03, 0.05, "Löslichkeit ↑   →   Viskosität ↓",
        transform=ax_r.transAxes, ha="left", va="bottom",
        fontsize=11, style="italic", color="#1B3A4B",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F7FA",
                  edgecolor="#D5DCE2", linewidth=0.6),
    )

    fig.tight_layout()
    for out_path in out_paths:
        fig.savefig(
            out_path, dpi=300, bbox_inches="tight",
            format=out_path.suffix.lstrip("."),
        )
        print(f"  [OK] Saved: {out_path}")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Two-panel cause-effect figure for the slide on "
            "refrigerant-lubricant interaction."
        )
    )

    # Lubricants / refrigerant
    ap.add_argument("--oil_sol", default="LPG68",
                    help="Oil for the solubility panel (LPG68 | LPG100). "
                         "Default: LPG68")
    ap.add_argument("--oils_visc", nargs="+", default=["LPG68", "LPG100"],
                    help="Oils for the viscosity panel (one or two). "
                         "Default: LPG68 LPG100")
    ap.add_argument("--refrigerant", default="propane")

    # Solubility panel
    ap.add_argument("--T_isotherms", type=float, nargs="+",
                    default=[20.0, 50.0, 80.0],
                    help="Isotherm temperatures [°C]. Default: 20 50 80")
    ap.add_argument("--p_min", type=float, default=1.0,
                    help="Min pressure [bar]. Default: 1")
    ap.add_argument("--p_max", type=float, default=30.0,
                    help="Max pressure [bar]. Default: 30")
    ap.add_argument("--n_p", type=int, default=120,
                    help="Number of pressure points. Default: 120")

    # Viscosity panel
    ap.add_argument("--T_visc", type=float, default=50.0,
                    help="Temperature [°C] for the viscosity panel. Default: 50")
    ap.add_argument("--w_max", type=float, default=0.30,
                    help="Max refrigerant mass fraction in the sweep. Default: 0.30")
    ap.add_argument("--n_w", type=int, default=150,
                    help="Number of w_KM points. Default: 150")
    ap.add_argument("--linear_visc", action="store_true",
                    help="Use linear y-axis on viscosity panel (default: log)")

    # Plot options
    ap.add_argument("--show_titles", action="store_true",
                    help="Show subtitles on each panel (off by default).")
    ap.add_argument("--figsize", type=float, nargs=2, default=[14.0, 5.6],
                    help="Figure size in inches. Default: 14 5.6")

    # Output
    ap.add_argument("--out_dir", default="results/mixture_cause_effect",
                    type=Path)
    ap.add_argument("--out_format", nargs="+", choices=["png", "svg", "pdf"],
                    default=["png", "svg"],
                    help="One or more output formats. Default: png svg "
                         "(figure is rendered once and saved to all formats).")
    ap.add_argument("--debug", action="store_true",
                    help="Print debug info when solubility evaluation fails")

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fluid_name = map_refrigerant(args.refrigerant)
    p_array = np.linspace(args.p_min, args.p_max, args.n_p)
    w_array = np.linspace(0.0, args.w_max, args.n_w)

    # --- Build RefProp + lubricant model(s) ---
    med = RefProp(fluid_name=fluid_name)
    oil_sol_name = map_oil(args.oil_sol)
    lub_sol = build_lubricant(fluid_name, oil_sol_name, med=med)

    oils_visc_names = [map_oil(o) for o in args.oils_visc]
    lubs_visc = {
        name: build_lubricant(fluid_name, name, med=med) for name in oils_visc_names
    }

    # Sanity check + quick API test (mirrors reference script)
    print(f"\n  Refrigerant: {fluid_name}")
    print(f"  Solubility panel oil:  {oil_sol_name}")
    print(f"  Viscosity panel oils:  {', '.join(oils_visc_names)}")
    print("  Quick API test ...")
    try:
        w_test = lub_sol.solve_w_KM(c_to_k(50.0), 10.0 * 1e5)
        mu_test_pure = calc_dyn_vis_at_w(lub_sol, c_to_k(50.0), 0.001)
        mu_test_10pct = calc_dyn_vis_at_w(lub_sol, c_to_k(50.0), 0.10)
        print(f"    solve_w_KM(T=50 °C, p=10 bar)   = {w_test}")
        print(f"    mu(T=50 °C, w=0.001) [near pure] = {mu_test_pure:.3f} mPa·s")
        print(f"    mu(T=50 °C, w=0.10)              = {mu_test_10pct:.3f} mPa·s")
    except Exception as e:
        print(f"    API test failed: {type(e).__name__}: {e}")

    # --- Compute solubility isotherms ---
    print("\n  Computing solubility isotherms ...")
    sol_data = compute_solubility_isotherms(
        lub_sol, args.T_isotherms, p_array, debug=args.debug
    )
    for T_C in sorted(sol_data.keys()):
        n = len(sol_data[T_C]["p_bar"])
        print(f"    T={T_C:5.1f} °C: {n} points computed")

    # --- Compute viscosity sweeps ---
    print("\n  Computing viscosity sweeps ...")
    vis_data = {}
    for lub_name, lub in lubs_visc.items():
        vis_data[lub_name] = compute_viscosity_vs_w(
            lub, args.T_visc, w_array, debug=args.debug
        )
        n = int(np.isfinite(vis_data[lub_name]["mu_mPas"]).sum())
        print(f"    {OIL_DISPLAY.get(lub_name, lub_name):8s} at T={args.T_visc:.0f} °C: "
              f"{n} points computed")

    # --- Plot ---
    stamp = _ts()
    suffix = f"sol-{args.oil_sol}_visc-{'-'.join(args.oils_visc)}"
    out_paths = [
        args.out_dir / f"cause_effect_{suffix}_{stamp}.{fmt}"
        for fmt in args.out_format
    ]

    plot_cause_effect(
        sol_data=sol_data,
        vis_data=vis_data,
        oil_for_sol=oil_sol_name,
        T_visc_C=args.T_visc,
        out_paths=out_paths,
        log_visc=not args.linear_visc,
        show_titles=args.show_titles,
        figsize=tuple(args.figsize),
    )

    print(f"\nDone. Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
