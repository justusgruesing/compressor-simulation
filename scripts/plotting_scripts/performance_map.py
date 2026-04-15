# scripts/plotting_scripts/performance_map.py
#
# Creates 2D performance maps (heatmaps) of isentropic efficiency and
# volumetric efficiency over a T_evap × T_cond grid, with fixed speed and
# superheat.
#
# Each invocation produces TWO plots:
#   - eta_is heatmap
#   - lambda_h heatmap
#
# Invalid regions (T_cond ≤ T_evap, simulation failures) are shown as white.
#
# Activate REFPROP:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Default grid (40x40), N=3600 rpm, SH=10 K, LPG68:
#   python scripts/plotting_scripts/performance_map.py --params_csv results/final_results/Modified_LPG100/Fitting/fitted_params_lpg100_modified_ga_2026-03-28_092941.csv --oil LPG100 --metric all
#
#   # Custom grid and operating conditions:
#   python scripts/plotting_scripts/performance_map.py \
#       --params_csv results/ga_fit/fitted_params_lpg100_modified_ga_2026-03-19.csv \
#       --oil LPG100 --N_rpm 5400 --SH_K 15 \
#       --T_evap_min -10 --T_evap_max 25 \
#       --T_cond_min 25 --T_cond_max 70 \
#       --n_grid 50

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from vclibpy.media import RefProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import Molinaroli_2017_Compressor
from vclibpy.components.compressors.rolling_piston_Molinaroli_2017_modified import (
    Molinaroli_2017_Compressor_Modified,
)

# Optional: oil_path model
# Note: the class in vclibpy is named Molinaroli_2017_Compressor_Oil_Path
# (with underscore between Oil and Path). We try multiple naming variants
# for robustness across vclibpy versions.
try:
    from vclibpy.components.compressors.rolling_piston_Molinaroli_oil_path import (
        Molinaroli_2017_Compressor_Oil_Path as Molinaroli_2017_Compressor_OilPath,
    )
    OIL_PATH_AVAILABLE = True
except ImportError:
    try:
        from vclibpy.components.compressors.rolling_piston_Molinaroli_oil_path import (
            Molinaroli_2017_Compressor_OilPath,
        )
        OIL_PATH_AVAILABLE = True
    except ImportError:
        try:
            from vclibpy.components.compressors.rolling_piston_Molinaroli_oil_path import (
                Molinaroli_OilPath_Compressor as Molinaroli_2017_Compressor_OilPath,
            )
            OIL_PATH_AVAILABLE = True
        except ImportError:
            OIL_PATH_AVAILABLE = False
            Molinaroli_2017_Compressor_OilPath = None

plt.style.use("ebc.paper.mplstyle")


# =========================================================
# Constants
# =========================================================
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0


# =========================================================
# Parameter definitions
# =========================================================
PARAM_NAMES_ORIGINAL = [
    "Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
    "V_IC", "alpha_loss", "W_dot_loss_ref",
]

PARAM_NAMES_MODIFIED = [
    "Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
    "V_IC", "alpha_loss", "W_dot_loss_ref", "alpha_fric_tot",
]

DEFAULT_PARAMS_ORIGINAL = {
    "Ua_suc_ref": 16.05, "Ua_dis_ref": 13.96, "Ua_amb": 0.36,
    "A_tot": 9.47e-9, "A_dis": 86.1e-6, "V_IC": 30.7e-6,
    "alpha_loss": 0.16, "W_dot_loss_ref": 83.0,
    "m_dot_ref": None, "f_ref": F_REF,
}

DEFAULT_PARAMS_MODIFIED = {
    "Ua_suc_ref": 16.05, "Ua_dis_ref": 13.96, "Ua_amb": 0.36,
    "A_tot": 9.47e-9, "A_dis": 86.1e-6, "V_IC": 30.7e-6,
    "alpha_loss": 0.16, "W_dot_loss_ref": 10.0, "alpha_fric_tot": 120.0,
    "m_dot_ref": None, "f_ref": F_REF,
}

PARAM_NAMES_OIL_PATH = [
    "Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
    "V_IC", "alpha_loss", "W_dot_loss_ref", "alpha_fric_tot",
    "m_dot_oil_ref", "Ua_suc_oil_ref",
]

DEFAULT_PARAMS_OIL_PATH = {
    "Ua_suc_ref": 16.05, "Ua_dis_ref": 13.96, "Ua_amb": 0.36,
    "A_tot": 9.47e-9, "A_dis": 86.1e-6, "V_IC": 30.7e-6,
    "alpha_loss": 0.16, "W_dot_loss_ref": 10.0, "alpha_fric_tot": 120.0,
    "m_dot_oil_ref": 0.005, "Ua_suc_oil_ref": 5.0,
    "m_dot_ref": None, "f_ref": F_REF,
}


# =========================================================
# Helpers
# =========================================================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _finite(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def c_to_k(t):
    return float(t) + 273.15


def k_to_c(t):
    return float(t) - 273.15


def rpm_to_hz(n):
    return float(n) / 60.0


# =========================================================
# Model helpers
# =========================================================
def map_refrigerant_for_modified_model(name: str) -> str:
    s = str(name).strip().upper()
    if s in {"PROPANE", "R290", "PROPAN"}:
        return "propane"
    return str(name).strip()


def map_oil_for_modified_model(name: str) -> str:
    s = str(name).strip().lower().replace(" ", "")
    if s == "lpg68":
        return "LPG 68"
    if s == "lpg100":
        return "LPG 100"
    raise ValueError(f"Unsupported oil: {name}")


def get_param_names(model: str) -> list[str]:
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return list(PARAM_NAMES_ORIGINAL)
    if m in ("mod", "modified"):
        return list(PARAM_NAMES_MODIFIED)
    if m in ("oil_path", "oilpath"):
        return list(PARAM_NAMES_OIL_PATH)
    raise ValueError("Unknown model. Use original | modified | oil_path")


def get_default_params(model: str) -> dict:
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return dict(DEFAULT_PARAMS_ORIGINAL)
    if m in ("mod", "modified"):
        return dict(DEFAULT_PARAMS_MODIFIED)
    if m in ("oil_path", "oilpath"):
        return dict(DEFAULT_PARAMS_OIL_PATH)
    raise ValueError("Unknown model. Use original | modified | oil_path")


def make_compressor(model, N_max_hz, V_h_m3, params, refrigerant_name, oil_name=None):
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    if m in ("mod", "modified"):
        if oil_name is None:
            raise ValueError("Modified model requires oil_name.")
        return Molinaroli_2017_Compressor_Modified(
            N_max=N_max_hz, V_h=V_h_m3,
            fluid_name=map_refrigerant_for_modified_model(refrigerant_name),
            lub_name=map_oil_for_modified_model(oil_name),
            parameters=params,
        )
    if m in ("oil_path", "oilpath"):
        if not OIL_PATH_AVAILABLE:
            raise ImportError("oil_path model not available in this vclibpy installation.")
        if oil_name is None:
            raise ValueError("oil_path model requires oil_name.")
        return Molinaroli_2017_Compressor_OilPath(
            N_max=N_max_hz, V_h=V_h_m3,
            fluid_name=map_refrigerant_for_modified_model(refrigerant_name),
            lub_name=map_oil_for_modified_model(oil_name),
            parameters=params,
        )
    raise ValueError("Unknown model. Use original | modified | oil_path")


def compute_m_dot_ref(med, V_h_m3: float) -> float:
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * F_REF


# =========================================================
# Inputs wrapper
# =========================================================
@dataclass
class Control:
    n: float


@dataclass
class SimpleInputs:
    control: Control
    T_amb: float
    lsq_max_nfev: int = 50000
    lsq_ftol: float = 1e-12
    lsq_xtol: float = 1e-12


# =========================================================
# Parameter loading
# =========================================================
def load_params_csv(path: Path, model: str) -> tuple[dict, dict]:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError("Params CSV must contain exactly one row.")
    row = df.iloc[0].to_dict()

    param_names = get_param_names(model)
    default_params = get_default_params(model)

    params = dict(default_params)
    for k in param_names:
        if k in row and pd.notna(row[k]):
            params[k] = float(row[k])
    if "f_ref" in row and pd.notna(row["f_ref"]):
        params["f_ref"] = float(row["f_ref"])

    meta = {}
    for key in ["oil", "refrigerant", "model"]:
        if key in row:
            meta[key] = row[key]

    return params, meta


# =========================================================
# State helpers
# =========================================================
def compute_suction_state(med, T_evap_C: float, SH_K: float):
    T_evap_K = c_to_k(T_evap_C)
    state_sat = med.calc_state("TQ", T_evap_K, Q_REF)
    p_suc = float(state_sat.p)
    T_suc_K = T_evap_K + float(SH_K)
    return p_suc, T_suc_K


def compute_discharge_pressure(med, T_cond_C: float):
    T_cond_K = c_to_k(T_cond_C)
    state_sat = med.calc_state("TQ", T_cond_K, 0.0)
    return float(state_sat.p)


# =========================================================
# Efficiency calculation
# =========================================================
def compute_efficiency_point(
    comp, med, p_suc_pa, T_suc_K, p_out_pa, n_rel, T_amb_K, V_h_m3,
    lsq_ftol=1e-12, lsq_xtol=1e-12, lsq_max_nfev=50000,
):
    """
    Returns (eta_is, lambda_h, zeta_gl) or (nan, nan, nan) on failure.
    """
    inputs = SimpleInputs(
        control=Control(n=max(1e-9, min(1.0, n_rel))),
        T_amb=float(T_amb_K),
        lsq_ftol=lsq_ftol, lsq_xtol=lsq_xtol, lsq_max_nfev=lsq_max_nfev,
    )
    fs_state = FlowsheetState()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            comp.state_inlet = med.calc_state("PT", float(p_suc_pa), float(T_suc_K))
            comp.calc_state_outlet(p_outlet=float(p_out_pa), inputs=inputs, fs_state=fs_state)

            h_suc = float(comp.state_inlet.h)
            s_suc = float(comp.state_inlet.s)
            h_dis_actual = float(comp.state_outlet.h)

            if h_dis_actual <= h_suc:
                return float("nan"), float("nan"), float("nan")

            state_dis_isen = med.calc_state("PS", float(p_out_pa), s_suc)
            h_dis_isen = float(state_dis_isen.h)

            if h_dis_isen <= h_suc:
                return float("nan"), float("nan"), float("nan")

            w_is = h_dis_isen - h_suc
            w_actual = h_dis_actual - h_suc

            if w_actual <= 0:
                return float("nan"), float("nan"), float("nan")

            eta_is = w_is / w_actual

            m_flow = float(comp.m_flow)
            P_el = float(comp.P_el)
            rho_suc = float(comp.state_inlet.d)
            f_hz = comp.get_n_absolute(inputs.control.n)
            m_dot_theoretical = rho_suc * V_h_m3 * f_hz

            if m_dot_theoretical <= 0 or P_el <= 0:
                return float("nan"), float("nan"), float("nan")

            lambda_h = m_flow / m_dot_theoretical

            # Global goodness: reversible power / electrical power
            P_rev = m_flow * w_is
            zeta_gl = P_rev / P_el

        # Sanity check ranges
        if not (0 < eta_is < 1.5):
            eta_is = float("nan")
        if not (0 < lambda_h < 1.5):
            lambda_h = float("nan")
        if not (0 < zeta_gl < 1.5):
            zeta_gl = float("nan")

        return eta_is, lambda_h, zeta_gl

    except Exception:
        return float("nan"), float("nan"), float("nan")


def compute_grid(
    med, model, refrigerant_name, oil_name,
    params, N_max_hz, V_h_m3,
    T_evap_grid, T_cond_grid,
    N_rpm, SH_K, T_amb_C,
    lsq_ftol=1e-12, lsq_xtol=1e-12, lsq_max_nfev=50000,
):
    """
    Compute eta_is, lambda_h and zeta_gl over a 2D T_evap × T_cond grid.
    Returns (eta_grid, lambda_grid, zeta_grid) as 2D arrays of shape (n_T_cond, n_T_evap).
    """
    comp = make_compressor(
        model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
        params=params, refrigerant_name=refrigerant_name, oil_name=oil_name,
    )
    comp.med_prop = med
    if hasattr(comp, "debug_enabled"):
        comp.debug_enabled = False

    n_evap = len(T_evap_grid)
    n_cond = len(T_cond_grid)

    eta_grid = np.full((n_cond, n_evap), np.nan)
    lambda_grid = np.full((n_cond, n_evap), np.nan)
    zeta_grid = np.full((n_cond, n_evap), np.nan)

    f_hz = rpm_to_hz(N_rpm)
    n_rel = f_hz / N_max_hz
    T_amb_K = c_to_k(T_amb_C)

    if n_rel <= 0 or n_rel > 1.0:
        raise ValueError(f"n_rel={n_rel} out of range. Check N_rpm and N_max_rpm.")

    n_total = n_evap * n_cond
    n_done = 0
    n_valid = 0

    for j, T_cond_C in enumerate(T_cond_grid):
        for i, T_evap_C in enumerate(T_evap_grid):
            n_done += 1

            # Skip invalid: T_cond must be > T_evap
            if T_cond_C <= T_evap_C:
                continue

            try:
                p_suc, T_suc_K = compute_suction_state(med, T_evap_C, SH_K)
                p_out = compute_discharge_pressure(med, T_cond_C)
            except Exception:
                continue

            if p_out <= p_suc:
                continue

            eta_is, lambda_h, zeta_gl = compute_efficiency_point(
                comp, med, p_suc, T_suc_K, p_out, n_rel, T_amb_K, V_h_m3,
                lsq_ftol=lsq_ftol, lsq_xtol=lsq_xtol, lsq_max_nfev=lsq_max_nfev,
            )

            eta_grid[j, i] = eta_is
            lambda_grid[j, i] = lambda_h
            zeta_grid[j, i] = zeta_gl

            if np.isfinite(eta_is) and np.isfinite(lambda_h):
                n_valid += 1

        # Progress update per row
        progress = n_done / n_total * 100
        print(f"\r  Computing grid: {progress:.0f}% ({n_valid} valid points)", end="", flush=True)

    print(f"\r  Computing grid: 100% — {n_valid}/{n_total} valid points          ")
    return eta_grid, lambda_grid, zeta_grid


# =========================================================
# Plot
# =========================================================
def plot_heatmap(
    grid: np.ndarray,
    T_evap_grid: np.ndarray,
    T_cond_grid: np.ndarray,
    metric_label: str,
    cbar_label: str,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    contour_levels: int = 10,
):
    """
    Plot a single 2D heatmap with contour lines on top.
    NaN values are rendered as white.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    # Mask NaN for white background
    masked = np.ma.masked_invalid(grid)

    # Get colormap and force NaN to white
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    # Color limits from finite data
    finite_vals = grid[np.isfinite(grid)]
    if len(finite_vals) == 0:
        print(f"  [WARN] No valid data for {metric_label}")
        plt.close(fig)
        return

    vmin = float(np.nanpercentile(finite_vals, 2))
    vmax = float(np.nanpercentile(finite_vals, 98))

    if vmin == vmax:
        vmin -= 0.01
        vmax += 0.01

    norm = Normalize(vmin=vmin, vmax=vmax)

    # pcolormesh with grid edges
    dT_e = (T_evap_grid[1] - T_evap_grid[0]) if len(T_evap_grid) > 1 else 1.0
    dT_c = (T_cond_grid[1] - T_cond_grid[0]) if len(T_cond_grid) > 1 else 1.0
    T_evap_edges = np.concatenate([
        T_evap_grid - dT_e / 2,
        [T_evap_grid[-1] + dT_e / 2],
    ])
    T_cond_edges = np.concatenate([
        T_cond_grid - dT_c / 2,
        [T_cond_grid[-1] + dT_c / 2],
    ])

    mesh = ax.pcolormesh(
        T_evap_edges, T_cond_edges, masked,
        cmap=cmap_obj, norm=norm,
        shading="flat",
    )

    # Contour lines on top (only over finite region)
    try:
        E, C = np.meshgrid(T_evap_grid, T_cond_grid)
        cs = ax.contour(
            E, C, grid,
            levels=contour_levels,
            colors="black",
            linewidths=0.7,
            alpha=0.6,
        )
        ax.clabel(cs, inline=True, fontsize=9, fmt="%.2f")
    except Exception:
        pass

    # Colorbar
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    ax.set_xlabel("Verdampfungstemperatur $T_{evap}$ [°C]")
    ax.set_ylabel("Kondensationstemperatur $T_{cond}$ [°C]")
    ax.set_title(title)

    # Set limits to grid extent (no padding)
    ax.set_xlim(T_evap_edges[0], T_evap_edges[-1])
    ax.set_ylim(T_cond_edges[0], T_cond_edges[-1])

    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [OK] Saved: {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Performance map heatmaps: eta_is, lambda_h, zeta_gl over T_evap × T_cond grid."
    )

    ap.add_argument("--params_csv", required=True, type=Path, help="Fitted parameter CSV")
    ap.add_argument("--oil", required=True, help="Oil name (LPG68 | LPG100)")

    # Model / fluid
    ap.add_argument("--model", default="auto", help="original | modified | oil_path | auto")
    ap.add_argument("--refrigerant", default="auto", help="RefProp fluid or auto")

    # Metric selection
    ap.add_argument(
        "--metric", nargs="+", default=["eta_is", "lambda_h"],
        help="Which metric(s) to plot: eta_is | lambda_h | zeta_gl | all (default: eta_is lambda_h)",
    )

    # Compressor geometry
    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)
    ap.add_argument("--T_amb_C", type=float, default=25.0)

    # Fixed conditions
    ap.add_argument("--N_rpm", type=float, default=3600.0, help="Fixed speed [rpm]")
    ap.add_argument("--SH_K", type=float, default=10.0, help="Fixed superheat [K]")

    # Grid ranges
    ap.add_argument("--T_evap_min", type=float, default=-10.0)
    ap.add_argument("--T_evap_max", type=float, default=25.0)
    ap.add_argument("--T_cond_min", type=float, default=25.0)
    ap.add_argument("--T_cond_max", type=float, default=70.0)
    ap.add_argument("--n_grid", type=int, default=40,
                    help="Grid points per axis (default 40 → 40×40 = 1600 simulations)")

    # Solver tolerances
    ap.add_argument("--lsq_ftol", type=float, default=1e-12,
                    help="Solver function tolerance (default 1e-12)")
    ap.add_argument("--lsq_xtol", type=float, default=1e-12,
                    help="Solver variable tolerance (default 1e-12)")
    ap.add_argument("--lsq_max_nfev", type=int, default=50000,
                    help="Solver max function evaluations (default 50000)")

    # Plot options
    ap.add_argument("--cmap_eta", default="viridis", help="Colormap for eta_is")
    ap.add_argument("--cmap_lambda", default="viridis", help="Colormap for lambda_h")
    ap.add_argument("--cmap_zeta", default="viridis", help="Colormap for zeta_gl")
    ap.add_argument("--contour_levels", type=int, default=10)

    ap.add_argument("--out_dir", default="results/performance_map")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    if not args.params_csv.exists():
        raise FileNotFoundError(args.params_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Resolve auto values
    # -------------------------
    params_peek = pd.read_csv(args.params_csv).iloc[0].to_dict()

    if args.model == "auto":
        args.model = str(params_peek.get("model", "modified"))
    if args.refrigerant == "auto":
        args.refrigerant = str(params_peek.get("refrigerant", "PROPANE"))

    params, params_meta = load_params_csv(args.params_csv, args.model)

    # -------------------------
    # RefProp
    # -------------------------
    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6

    med = RefProp(fluid_name=args.refrigerant)
    params["f_ref"] = F_REF
    params["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

    print(f"  Model:           {args.model}")
    print(f"  Refrigerant:     {args.refrigerant}")
    print(f"  Oil:             {args.oil}  (params from: {params_meta.get('oil', '?')})")
    print(f"  Fixed N:         {args.N_rpm:.0f} rpm")
    print(f"  Fixed SH:        {args.SH_K:.1f} K")
    print(f"  T_evap range:    {args.T_evap_min} → {args.T_evap_max} °C")
    print(f"  T_cond range:    {args.T_cond_min} → {args.T_cond_max} °C")
    print(f"  Grid:            {args.n_grid}×{args.n_grid} = {args.n_grid**2} points")
    print(f"  Solver tol:      ftol={args.lsq_ftol:.0e}, xtol={args.lsq_xtol:.0e}, max_nfev={args.lsq_max_nfev}")

    # -------------------------
    # Metric selection
    # -------------------------
    all_metrics = ["eta_is", "lambda_h", "zeta_gl"]
    requested = list(args.metric)
    if "all" in requested:
        metrics_to_plot = list(all_metrics)
    else:
        invalid = [m for m in requested if m not in all_metrics]
        if invalid:
            raise ValueError(
                f"Unknown metric(s): {invalid}. Valid: {all_metrics + ['all']}"
            )
        seen = set()
        metrics_to_plot = [m for m in requested if not (m in seen or seen.add(m))]

    print(f"  Metric(s):       {', '.join(metrics_to_plot)}")

    # -------------------------
    # Build grid
    # -------------------------
    T_evap_grid = np.linspace(args.T_evap_min, args.T_evap_max, args.n_grid)
    T_cond_grid = np.linspace(args.T_cond_min, args.T_cond_max, args.n_grid)

    # Oil name needed for modified and oil_path models
    m = str(args.model).lower().strip()
    oil_name = args.oil if m in ("mod", "modified", "oil_path", "oilpath") else None

    # -------------------------
    # Compute grids
    # -------------------------
    eta_grid, lambda_grid, zeta_grid = compute_grid(
        med=med, model=args.model,
        refrigerant_name=args.refrigerant, oil_name=oil_name,
        params=params, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
        T_evap_grid=T_evap_grid, T_cond_grid=T_cond_grid,
        N_rpm=args.N_rpm, SH_K=args.SH_K, T_amb_C=args.T_amb_C,
        lsq_ftol=args.lsq_ftol, lsq_xtol=args.lsq_xtol,
        lsq_max_nfev=args.lsq_max_nfev,
    )

    # -------------------------
    # Title base
    # -------------------------
    subtitle = (
        f"{args.model.capitalize()} | Öl: {args.oil} | {args.refrigerant} | "
        f"N={args.N_rpm:.0f} rpm | SH={args.SH_K:.0f} K"
    )

    stamp = _ts()
    oil_tag = args.oil.lower()

    # Metric → (grid, filename_tag, cbar_label, title_label, cmap)
    metric_info = {
        "eta_is": {
            "grid": eta_grid,
            "cbar_label": "Isentroper Wirkungsgrad $\\eta_{is}$",
            "title_label": "Isentroper Wirkungsgrad",
            "cmap": args.cmap_eta,
        },
        "lambda_h": {
            "grid": lambda_grid,
            "cbar_label": "Liefergrad $\\lambda_h$",
            "title_label": "Liefergrad",
            "cmap": args.cmap_lambda,
        },
        "zeta_gl": {
            "grid": zeta_grid,
            "cbar_label": "Globaler Gütegrad $\\zeta_{gl}$",
            "title_label": "Globaler Gütegrad",
            "cmap": args.cmap_zeta,
        },
    }

    # -------------------------
    # Plot each requested metric
    # -------------------------
    for metric_name in metrics_to_plot:
        info = metric_info[metric_name]
        out_path = (
            out_dir
            / f"perfmap_{metric_name}_{oil_tag}_N{int(args.N_rpm)}_SH{int(args.SH_K)}_{stamp}.{args.out_format}"
        )
        plot_heatmap(
            grid=info["grid"],
            T_evap_grid=T_evap_grid,
            T_cond_grid=T_cond_grid,
            metric_label=metric_name,
            cbar_label=info["cbar_label"],
            title=f"{info['title_label']}\n{subtitle}",
            out_path=out_path,
            cmap=info["cmap"],
            contour_levels=args.contour_levels,
        )

    # -------------------------
    # Save data as CSV (always includes all three metrics)
    # -------------------------
    records = []
    for j, T_cond_C in enumerate(T_cond_grid):
        for i, T_evap_C in enumerate(T_evap_grid):
            records.append({
                "T_evap_C": T_evap_C,
                "T_cond_C": T_cond_C,
                "eta_is": eta_grid[j, i],
                "lambda_h": lambda_grid[j, i],
                "zeta_gl": zeta_grid[j, i],
                "N_rpm": args.N_rpm,
                "SH_K": args.SH_K,
                "oil": args.oil,
                "params_oil": params_meta.get("oil", "?"),
                "model": args.model,
            })

    data_csv = out_dir / f"perfmap_data_{oil_tag}_N{int(args.N_rpm)}_SH{int(args.SH_K)}_{stamp}.csv"
    pd.DataFrame.from_records(records).to_csv(data_csv, index=False)
    print(f"  [OK] Data saved: {data_csv}")

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
