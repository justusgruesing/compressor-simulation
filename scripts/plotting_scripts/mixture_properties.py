# scripts/plotting_scripts/mixture_properties.py
#
# Plots two figures for the thesis chapter on mixture properties:
#   1. Solubility: equilibrium refrigerant mass fraction w_KM as a function
#      of temperature at various pressures.
#   2. Viscosity: kinematic viscosity of the oil-refrigerant mixture as a
#      function of temperature at various pressures.
#
# Both plots show the indirect pressure effect via the pressure-dependent
# solubility: at higher pressure, more refrigerant dissolves in the oil,
# which lowers the viscosity.
#
# Activate REFPROP first:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Both plots for LPG68:
#   python scripts/plotting_scripts/mixture_properties.py --oil LPG68
#
#   # Only solubility, custom pressures:
#   python scripts/plotting_scripts/mixture_properties.py --oil LPG68 --plot solubility --pressures 2 5 10 15 20
#
#   # Both oils side by side:
#   python scripts/plotting_scripts/mixture_properties.py --oil LPG68 --oil2 LPG100

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from vclibpy.media import RefProp, ThermodynamicState

# Import lubricant fitting
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

plt.style.use("ebc.paper.mplstyle")


# =========================================================
# Constants
# =========================================================
# EBC color palette for pressure isobars
ISOBAR_COLORS = [
    "#EC635C",  # red
    "#4B81C4",  # blue
    "#F49961",  # orange
    "#6EBB96",  # green
    "#8768B4",  # purple
    "#B45955",  # dark red
    "#CB74F4",  # violet
]

OIL_DISPLAY = {"LPG68": "PAG 68", "LPG100": "PAG 100", "lpg68": "PAG 68", "lpg100": "PAG 100"}
REFRIGERANT_DISPLAY = {"propane": "Propan (R-290)", "PROPANE": "Propan (R-290)"}


# =========================================================
# Helpers
# =========================================================
def _ts():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def c_to_k(t):
    return float(t) + 273.15

def k_to_c(t):
    return float(t) - 273.15

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
    lub = SharedLubricantFitting(fluid_name=fluid_name, lub_name=lub_name,
                                 shared_refprop=med)
    return lub


# =========================================================
# Compute solubility curves
# =========================================================
def compute_solubility(lubricant, T_range_C, pressures_bar, debug=False):
    """
    Compute equilibrium w_KM(T) at each pressure.
    Returns dict: {p_bar: {"T_C": [...], "w_KM": [...]}}
    """
    results = {}
    for p_bar in pressures_bar:
        p_Pa = p_bar * 1e5
        T_arr = []
        w_arr = []
        n_err = 0
        last_err = None
        for T_C in T_range_C:
            T_K = c_to_k(T_C)
            try:
                w = lubricant.solve_w_KM(T_K, p_Pa)
                if w is not None and 0.0 <= w <= 1.0:
                    T_arr.append(T_C)
                    w_arr.append(w)
                elif debug and n_err < 3:
                    print(f"    [DEBUG] solve_w_KM({T_C:.1f}°C, {p_bar:.0f} bar) "
                          f"returned {w}")
            except Exception as e:
                n_err += 1
                last_err = e
                if debug and n_err <= 3:
                    print(f"    [DEBUG] solve_w_KM({T_C:.1f}°C, {p_bar:.0f} bar) "
                          f"raised {type(e).__name__}: {e}")
        if n_err > 0 and debug:
            print(f"    [DEBUG] p={p_bar:.0f} bar: {n_err} exceptions total, "
                  f"last: {type(last_err).__name__}: {last_err}")
        results[p_bar] = {"T_C": np.array(T_arr), "w_KM": np.array(w_arr)}
    return results


# =========================================================
# Compute viscosity curves
# =========================================================
def compute_viscosity(lubricant, T_range_C, pressures_bar, debug=False):
    """
    Compute kinematic viscosity of the mixture at each (T, p).
    The equilibrium w_KM is determined internally via solubility.
    Returns dict: {p_bar: {"T_C": [...], "nu_mm2s": [...], "mu_mPas": [...], "rho_kgm3": [...]}}
    """
    results = {}
    for p_bar in pressures_bar:
        p_Pa = p_bar * 1e5
        T_arr = []
        nu_arr = []
        mu_arr = []
        rho_arr = []
        n_err = 0
        last_err = None
        for T_C in T_range_C:
            T_K = c_to_k(T_C)
            try:
                # Get equilibrium w_KM
                w = lubricant.solve_w_KM(T_K, p_Pa)
                if w is None or w < 0.0 or w > 1.0:
                    continue

                # Dynamic viscosity via transport properties
                state = ThermodynamicState(p=p_Pa, T=T_K)
                tp = lubricant.calc_transport_properties(state=state, phase="liquid")
                if tp is None or tp.dyn_vis is None:
                    if debug and n_err < 3:
                        print(f"    [DEBUG] viscosity: tp={tp}, dyn_vis={getattr(tp, 'dyn_vis', '?')} "
                              f"at T={T_C:.1f}°C, p={p_bar:.0f} bar")
                    continue
                mu = float(tp.dyn_vis)  # mPa·s (from lubricant fitting)
                if not np.isfinite(mu) or mu <= 0:
                    continue

                # Density
                rho = lubricant.calc_rho_mix(T_K, w)
                if rho is None or rho <= 0:
                    continue
                rho = float(rho)

                # Kinematic viscosity: nu = mu / rho
                # mu is in mPa·s = 1e-3 Pa·s, rho in kg/m³
                # nu [m²/s] = mu [Pa·s] / rho [kg/m³]
                # nu [mm²/s] = nu [m²/s] * 1e6
                nu = (mu * 1e-3) / rho * 1e6  # mm²/s (= cSt)

                T_arr.append(T_C)
                nu_arr.append(nu)
                mu_arr.append(mu)
                rho_arr.append(rho)
            except Exception as e:
                n_err += 1
                last_err = e
                if debug and n_err <= 3:
                    print(f"    [DEBUG] viscosity at T={T_C:.1f}°C, p={p_bar:.0f} bar: "
                          f"{type(e).__name__}: {e}")
        if n_err > 0 and debug:
            print(f"    [DEBUG] p={p_bar:.0f} bar: {n_err} viscosity exceptions total")
        results[p_bar] = {
            "T_C": np.array(T_arr),
            "nu_mm2s": np.array(nu_arr),
            "mu_mPas": np.array(mu_arr),
            "rho_kgm3": np.array(rho_arr),
        }
    return results


# =========================================================
# Plotting: Solubility
# =========================================================
def plot_solubility(sol_data, oil_name, refrigerant_name, out_path,
                    sol_data2=None, oil_name2=None):
    """
    Plot w_KM vs T at various pressures.
    If sol_data2 is provided, a second oil is shown side by side.
    """
    two_oils = sol_data2 is not None

    if two_oils:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        axes_data = [(ax1, sol_data, oil_name), (ax2, sol_data2, oil_name2)]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 7))
        axes_data = [(ax1, sol_data, oil_name)]

    for ax, data, oname in axes_data:
        for i, (p_bar, curves) in enumerate(sorted(data.items())):
            color = ISOBAR_COLORS[i % len(ISOBAR_COLORS)]
            if len(curves["T_C"]) > 0:
                ax.plot(curves["T_C"], curves["w_KM"] * 100,
                        color=color, linewidth=2.0,
                        label=f"$p$ = {p_bar:.0f} bar")

        ax.set_xlabel("Temperatur $T$ in °C")
        ax.set_ylabel("gelöster Kältemittelmassenanteil $w_{\\mathrm{KM}}$ in %")
        oil_disp = OIL_DISPLAY.get(oname, oname)
        ref_disp = REFRIGERANT_DISPLAY.get(refrigerant_name, refrigerant_name)
        ax.set_title(f"PROPAN / {oil_disp}")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                format=out_path.suffix.lstrip("."))
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# =========================================================
# Plotting: Viscosity
# =========================================================
def plot_viscosity(vis_data, oil_name, refrigerant_name, out_path,
                   vis_data2=None, oil_name2=None, log_scale=True):
    """
    Plot dynamic viscosity vs T at various pressures.
    If vis_data2 is provided, a second oil is shown side by side.
    """
    two_oils = vis_data2 is not None

    if two_oils:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        axes_data = [(ax1, vis_data, oil_name), (ax2, vis_data2, oil_name2)]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 7))
        axes_data = [(ax1, vis_data, oil_name)]

    for ax, data, oname in axes_data:
        for i, (p_bar, curves) in enumerate(sorted(data.items())):
            color = ISOBAR_COLORS[i % len(ISOBAR_COLORS)]
            if len(curves["T_C"]) > 0:
                ax.plot(curves["T_C"], curves["mu_mPas"],
                        color=color, linewidth=2.0,
                        label=f"$p$ = {p_bar:.0f} bar")

        ax.set_xlabel("Temperatur $T$ in °C")
        ax.set_ylabel("Dynamische Viskosität $\\mu$ in mPa$\\cdot$s")
        oil_disp = OIL_DISPLAY.get(oname, oname)
        ref_disp = REFRIGERANT_DISPLAY.get(refrigerant_name, refrigerant_name)
        ax.set_title(f"PROPAN / {oil_disp}")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, which="both", linewidth=0.5, alpha=0.3)

        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                format=out_path.suffix.lstrip("."))
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Plot solubility and viscosity diagrams for oil-refrigerant mixtures."
    )
    ap.add_argument("--oil", required=True, help="Oil name (LPG68 | LPG100)")
    ap.add_argument("--oil2", default=None,
                    help="Second oil for side-by-side comparison (optional)")
    ap.add_argument("--refrigerant", default="propane")

    ap.add_argument("--plot", choices=["solubility", "viscosity", "both"],
                    default="both", help="Which plot(s) to create")

    # Temperature range
    ap.add_argument("--T_min", type=float, default=-40.0,
                    help="Minimum temperature [°C] (default -10)")
    ap.add_argument("--T_max", type=float, default=150.0,
                    help="Maximum temperature [°C] (default 95, near T_crit of propane)")
    ap.add_argument("--n_T", type=int, default=800,
                    help="Number of temperature points")

    # Pressure range
    ap.add_argument("--pressures", type=float, nargs="+",
                    default=[2, 5, 10, 15, 20],
                    help="Pressures [bar] for isobars (default: 2 5 10 15 20)")

    # Viscosity options
    ap.add_argument("--linear_viscosity", action="store_true",
                    help="Use linear y-axis for viscosity (default: log)")

    # Output
    ap.add_argument("--out_dir", default="results/mixture_properties", type=Path)
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")
    ap.add_argument("--debug", action="store_true",
                    help="Print debug info when solubility/viscosity fails")

    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fluid_name = map_refrigerant(args.refrigerant)
    T_range = np.linspace(args.T_min, args.T_max, args.n_T)
    pressures = sorted(args.pressures)

    # --- Build lubricant model(s) ---
    # --- Build RefProp and lubricant model(s) ---
    med = RefProp(fluid_name=fluid_name)

    print(f"  Oil 1: {args.oil}")
    lub1 = build_lubricant(fluid_name=fluid_name, lub_name=map_oil(args.oil), med=med)

    lub2 = None
    if args.oil2:
        print(f"  Oil 2: {args.oil2}")
        lub2 = build_lubricant(fluid_name=fluid_name, lub_name=map_oil(args.oil2), med=med)

    # Quick diagnostic: test one solve_w_KM call to verify the API works
    print("\n  Quick API test ...")
    T_test_K = c_to_k(50.0)
    p_test_Pa = 10.0 * 1e5
    try:
        w_test = lub1.solve_w_KM(T_test_K, p_test_Pa)
        print(f"    solve_w_KM(T=50°C, p=10 bar) = {w_test}")
    except Exception as e:
        print(f"    solve_w_KM(T=50°C, p=10 bar) raised {type(e).__name__}: {e}")

    # Also check if the method signature expects different args
    import inspect
    sig = inspect.signature(lub1.solve_w_KM)
    print(f"    solve_w_KM signature: {sig}")

    # Check available methods
    methods = [m for m in dir(lub1) if not m.startswith("_") and callable(getattr(lub1, m, None))]
    print(f"    Available methods: {', '.join(methods[:20])}")

    stamp = _ts()

    # --- Solubility ---
    if args.plot in ("solubility", "both"):
        print("\n  Computing solubility curves ...")
        sol1 = compute_solubility(lub1, T_range, pressures, debug=args.debug)
        sol2 = None
        if lub2 is not None:
            sol2 = compute_solubility(lub2, T_range, pressures, debug=args.debug)

        for p_bar in pressures:
            n = len(sol1[p_bar]["T_C"])
            print(f"    p={p_bar:5.0f} bar: {n} points computed")

        suffix = f"{args.oil}"
        if args.oil2:
            suffix += f"_vs_{args.oil2}"
        out_path = out_dir / f"solubility_{suffix}_{stamp}.{args.out_format}"

        plot_solubility(
            sol1, args.oil, fluid_name, out_path,
            sol_data2=sol2, oil_name2=args.oil2,
        )

    # --- Viscosity ---
    if args.plot in ("viscosity", "both"):
        print("\n  Computing viscosity curves ...")
        vis1 = compute_viscosity(lub1, T_range, pressures, debug=args.debug)
        vis2 = None
        if lub2 is not None:
            vis2 = compute_viscosity(lub2, T_range, pressures, debug=args.debug)

        for p_bar in pressures:
            n = len(vis1[p_bar]["T_C"])
            print(f"    p={p_bar:5.0f} bar: {n} points computed")

        suffix = f"{args.oil}"
        if args.oil2:
            suffix += f"_vs_{args.oil2}"
        out_path = out_dir / f"dyn_viscosity_{suffix}_{stamp}.{args.out_format}"

        plot_viscosity(
            vis1, args.oil, fluid_name, out_path,
            vis_data2=vis2, oil_name2=args.oil2,
            log_scale=not args.linear_viscosity,
        )

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
