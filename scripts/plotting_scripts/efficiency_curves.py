# scripts/plotting_scripts/efficiency_curves.py
#
# Plots a selectable target quantity vs. one varying operating parameter,
# with two curves per subplot (one per oil).
#
# Supported metrics (--metric):
#   eta_is    – Isentroper Wirkungsgrad [-]
#   lambda_h  – Liefergrad [-]
#   zeta_gl   – Globaler Gütegrad [-]
#   m_flow    – Massenstrom [g/s]
#   P_el      – Elektrische Leistung [W]
#   T_dis     – Austrittstemperatur [°C]
#
# Each oil can use its own parameter set, enabling cross-validation:
#   - Same params for both oils → cross-validation
#   - Each oil with its own fitted params → native comparison
#
# Activate REFPROP:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Isentropic efficiency:
#   python scripts/plotting_scripts/efficiency_curves.py --params_csv_oil1 results/final_results/Modified_LPG68/Fitting/fitted_params_lpg68_modified_ga_2026-03-22_185546.csv --params_csv_oil2 results/final_results/Modified_LPG100/Fitting/fitted_params_lpg100_modified_ga_2026-03-28_092941.csv --oil1 LPG68 --oil2 LPG100 --metric all --vary T_cond --T_evap 10 --N_rpm 3600 --SH_K 10 20 30
#   python scripts/plotting_scripts/efficiency_curves.py --params_csv_oil1 results/final_results/Modified_All/Fitting/fitted_params_all_modified_ga_2026-03-26_110247.csv --params_csv_oil2 results/final_results/Modified_All/Fitting/fitted_params_all_modified_ga_2026-03-26_110247.csv --oil1 LPG68 --oil2 LPG100 --metric all --vary T_cond --T_evap 0 10 20 --N_rpm 3600 --SH_K 10
#
#   # Global efficiency (zeta_gl):
#   python scripts/plotting_scripts/efficiency_curves.py \
#       --params_csv_oil1 results/ga_fit/fitted_params_lpg68_modified_ga_2026-03-19.csv \
#       --params_csv_oil2 results/ga_fit/fitted_params_lpg100_modified_ga_2026-03-19.csv \
#       --oil1 LPG68 --oil2 LPG100 \
#       --metric zeta_gl --vary T_cond --T_evap 10 --N_rpm 3600 --SH_K 10
#
#   # Mass flow:
#   python scripts/plotting_scripts/efficiency_curves.py \
#       --params_csv_oil1 results/ga_fit/fitted_params_lpg68_modified_ga_2026-03-19.csv \
#       --params_csv_oil2 results/ga_fit/fitted_params_lpg100_modified_ga_2026-03-19.csv \
#       --oil1 LPG68 --oil2 LPG100 \
#       --metric m_flow --vary speed --T_evap 10 --T_cond 50 --SH_K 10

from __future__ import annotations

import argparse
import itertools
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from vclibpy.media import RefProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import Molinaroli_2017_Compressor
from vclibpy.components.compressors.rolling_piston_Molinaroli_2017_modified import (
    Molinaroli_2017_Compressor_Modified,
)

# Optional: oil_path model
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
# Metric configuration
# =========================================================
METRIC_CONFIG = {
    "eta_is": {
        "label": "Isentroper Wirkungsgrad $\\eta_{is}$ [-]",
        "column": "eta_is",
        "title_short": "Isentroper Wirkungsgrad",
    },
    "lambda_h": {
        "label": "Liefergrad $\\lambda_h$ [-]",
        "column": "lambda_h",
        "title_short": "Liefergrad",
    },
    "zeta_gl": {
        "label": "Globaler Gütegrad $\\zeta_{gl}$ [-]",
        "column": "zeta_gl",
        "title_short": "Globaler Gütegrad",
    },
    "m_flow": {
        "label": "Massenstrom $\\dot{m}$ [g/s]",
        "column": "m_flow_gps",
        "title_short": "Massenstrom",
    },
    "P_el": {
        "label": "Elektrische Leistung $P_{el}$ [W]",
        "column": "P_el",
        "title_short": "Elektrische Leistung",
    },
    "T_dis": {
        "label": "Austrittstemperatur $T_{dis}$ [°C]",
        "column": "T_dis_C",
        "title_short": "Austrittstemperatur",
    },
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
# Point simulation — computes all metrics in one pass
# =========================================================
def compute_point_metrics(
    comp, med, p_suc_pa, T_suc_K, p_out_pa, n_rel, T_amb_K, V_h_m3,
    lsq_ftol=1e-12, lsq_xtol=1e-12, lsq_max_nfev=50000,
):
    """
    Simulate one operating point and compute all supported metrics.
    Returns dict or None on failure.
    """
    inputs = SimpleInputs(
        control=Control(n=max(1e-9, min(1.0, n_rel))),
        T_amb=float(T_amb_K),
        lsq_ftol=lsq_ftol,
        lsq_xtol=lsq_xtol,
        lsq_max_nfev=lsq_max_nfev,
    )
    fs_state = FlowsheetState()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            comp.state_inlet = med.calc_state("PT", float(p_suc_pa), float(T_suc_K))
            comp.calc_state_outlet(p_outlet=float(p_out_pa), inputs=inputs, fs_state=fs_state)

            h_suc = float(comp.state_inlet.h)
            s_suc = float(comp.state_inlet.s)
            rho_suc = float(comp.state_inlet.d)
            h_dis_actual = float(comp.state_outlet.h)
            T_dis_K = float(comp.state_outlet.T)
            m_flow = float(comp.m_flow)
            P_el = float(comp.P_el)

            state_dis_isen = med.calc_state("PS", float(p_out_pa), s_suc)
            h_dis_isen = float(state_dis_isen.h)

            if h_dis_actual <= h_suc or h_dis_isen <= h_suc:
                return None
            if m_flow <= 0 or P_el <= 0:
                return None
            if not np.isfinite(m_flow) or not np.isfinite(P_el) or not np.isfinite(T_dis_K):
                return None

            # --- Isentropic efficiency ---
            w_is = h_dis_isen - h_suc
            w_actual = h_dis_actual - h_suc
            eta_is = w_is / w_actual if w_actual > 0 else float("nan")

            # --- Volumetric efficiency ---
            f_hz = comp.get_n_absolute(inputs.control.n)
            m_dot_theoretical = rho_suc * V_h_m3 * f_hz
            lambda_h = m_flow / m_dot_theoretical if m_dot_theoretical > 0 else float("nan")

            # --- Global efficiency (zeta_gl = P_rev / P_el) ---
            P_rev = m_flow * w_is  # reversible (isentropic) power
            zeta_gl = P_rev / P_el if P_el > 0 else float("nan")

        # Sanity checks
        if not (0 < eta_is < 1.5):
            eta_is = float("nan")
        if not (0 < lambda_h < 1.5):
            lambda_h = float("nan")
        if not (0 < zeta_gl < 1.5):
            zeta_gl = float("nan")

        return {
            "eta_is": eta_is,
            "lambda_h": lambda_h,
            "zeta_gl": zeta_gl,
            "m_flow_kgps": m_flow,
            "m_flow_gps": m_flow * 1e3,
            "P_el": P_el,
            "P_rev": P_rev,
            "T_dis_K": T_dis_K,
            "T_dis_C": k_to_c(T_dis_K),
        }

    except Exception:
        return None


def smooth_sweep_data(df: pd.DataFrame, metric_cols: list[str],
                      window: int = 7, polyorder: int = 2) -> pd.DataFrame:
    """
    Apply Savitzky-Golay filter to metric columns.
    Adds '_smooth' suffixed columns, preserving the raw values.

    Args:
        df: sweep DataFrame sorted by vary_value
        metric_cols: list of column names to smooth
        window: filter window length (must be odd, >= polyorder+2)
        polyorder: polynomial order for the filter
    """
    if df.empty or len(df) < window:
        # Not enough points to smooth — copy raw as smooth
        for col in metric_cols:
            if col in df.columns:
                df[f"{col}_smooth"] = df[col]
        return df

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    for col in metric_cols:
        if col not in df.columns:
            continue

        vals = df[col].to_numpy(dtype=float)
        finite_mask = np.isfinite(vals)

        smoothed = vals.copy()

        if np.sum(finite_mask) >= window:
            # Only smooth the finite values
            smoothed[finite_mask] = savgol_filter(
                vals[finite_mask], window_length=window, polyorder=polyorder,
            )

        df[f"{col}_smooth"] = smoothed

    return df


# All metric columns that should be smoothed
SMOOTHABLE_COLS = [
    "eta_is", "lambda_h", "zeta_gl",
    "m_flow_gps", "m_flow_kgps", "P_el", "P_rev",
    "T_dis_C", "T_dis_K",
]


def run_sweep(
    med, model, refrigerant_name, oil_name,
    params, N_max_hz, V_h_m3,
    vary_name, vary_values,
    T_evap_C_fixed, T_cond_C_fixed, N_rpm_fixed, SH_K_fixed,
    T_amb_C=25.0,
    lsq_ftol=1e-12, lsq_xtol=1e-12, lsq_max_nfev=50000,
):
    """
    Sweep one parameter, compute all metrics for each point.
    """
    comp = make_compressor(
        model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
        params=params, refrigerant_name=refrigerant_name, oil_name=oil_name,
    )
    comp.med_prop = med
    if hasattr(comp, "debug_enabled"):
        comp.debug_enabled = False

    T_amb_K = c_to_k(T_amb_C)
    records = []

    for val in vary_values:
        if vary_name == "T_evap":
            T_evap_C, T_cond_C, N_rpm, SH_K = float(val), T_cond_C_fixed, N_rpm_fixed, SH_K_fixed
        elif vary_name == "T_cond":
            T_evap_C, T_cond_C, N_rpm, SH_K = T_evap_C_fixed, float(val), N_rpm_fixed, SH_K_fixed
        elif vary_name == "speed":
            T_evap_C, T_cond_C, N_rpm, SH_K = T_evap_C_fixed, T_cond_C_fixed, float(val), SH_K_fixed
        elif vary_name == "superheat":
            T_evap_C, T_cond_C, N_rpm, SH_K = T_evap_C_fixed, T_cond_C_fixed, N_rpm_fixed, float(val)
        elif vary_name == "pressure_ratio":
            T_evap_C, N_rpm, SH_K = T_evap_C_fixed, N_rpm_fixed, SH_K_fixed
            T_cond_C = None
        else:
            raise ValueError(f"Unknown vary: {vary_name}")

        try:
            p_suc, T_suc_K = compute_suction_state(med, T_evap_C, SH_K)

            if vary_name == "pressure_ratio":
                pr = float(val)
                p_out = p_suc * pr
            else:
                p_out = compute_discharge_pressure(med, T_cond_C)
        except Exception:
            continue

        f_hz = rpm_to_hz(N_rpm)
        n_rel = f_hz / N_max_hz

        if p_out <= p_suc or n_rel <= 0 or n_rel > 1.0:
            continue

        result = compute_point_metrics(
            comp, med, p_suc, T_suc_K, p_out, n_rel, T_amb_K, V_h_m3,
            lsq_ftol=lsq_ftol, lsq_xtol=lsq_xtol, lsq_max_nfev=lsq_max_nfev,
        )

        if result is None:
            continue

        rec = {
            "vary_value": float(val),
            "T_evap_C": T_evap_C,
            "T_cond_C": T_cond_C if T_cond_C is not None else float("nan"),
            "N_rpm": N_rpm,
            "SH_K": SH_K,
            "p_suc_bar": p_suc / 1e5,
            "p_out_bar": p_out / 1e5,
            "pressure_ratio": p_out / p_suc,
        }
        rec.update(result)
        records.append(rec)

    return pd.DataFrame(records)


# =========================================================
# Plot
# =========================================================
def plot_curves(
    sweep_results: dict,
    metric_label: str,
    metric_col: str,
    vary_label: str,
    oil1_label: str,
    oil2_label: str,
    title: str,
    out_path: Path,
):
    series_list = list(sweep_results.items())
    n_series = len(series_list)

    if n_series == 0:
        print("  [SKIP] No series to plot.")
        return

    n_cols_layout = min(3, n_series)
    n_rows_layout = int(np.ceil(n_series / n_cols_layout))

    fig, axes = plt.subplots(
        n_rows_layout, n_cols_layout,
        figsize=(7 * n_cols_layout, 5.5 * n_rows_layout),
        squeeze=False,
    )

    color_oil1 = "#EC635C"
    color_oil2 = "#4B81C4"

    for idx, (series_label, oil_data) in enumerate(series_list):
        row_idx = idx // n_cols_layout
        col_idx = idx % n_cols_layout
        ax = axes[row_idx, col_idx]

        df1 = oil_data["oil1"]
        df2 = oil_data["oil2"]

        if df1.empty and df2.empty:
            ax.set_title(f"{series_label}\n(keine gültigen Punkte)")
            continue

        if not df1.empty and metric_col in df1.columns:
            x1 = df1["vary_value"].to_numpy(dtype=float)
            y1 = df1[metric_col].to_numpy(dtype=float)
            ax.plot(
                x1, y1,
                marker="o", markersize=4,
                color=color_oil1, linewidth=2.0,
                label=oil1_label,
            )

        if not df2.empty and metric_col in df2.columns:
            x2 = df2["vary_value"].to_numpy(dtype=float)
            y2 = df2[metric_col].to_numpy(dtype=float)
            ax.plot(
                x2, y2,
                marker="s", markersize=4,
                color=color_oil2, linewidth=2.0,
                label=oil2_label,
            )

        ax.set_title(series_label, fontsize=12)
        ax.set_xlabel(vary_label)
        ax.set_ylabel(metric_label)
        ax.grid(True, linewidth=0.6, alpha=0.35)

    for idx in range(n_series, n_rows_layout * n_cols_layout):
        axes[idx // n_cols_layout, idx % n_cols_layout].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=True,
            fontsize=12,
        )

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [OK] Saved: {out_path}  ({n_series} series)")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Efficiency and target quantity curves comparing two oils."
    )

    ap.add_argument("--params_csv_oil1", required=True, type=Path)
    ap.add_argument("--params_csv_oil2", required=True, type=Path)

    ap.add_argument("--oil1", required=True)
    ap.add_argument("--oil2", required=True)

    ap.add_argument("--model", default="auto", help="original | modified | oil_path | auto")
    ap.add_argument("--refrigerant", default="auto")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)
    ap.add_argument("--T_amb_C", type=float, default=25.0)

    ap.add_argument(
        "--metric", required=True, nargs="+",
        help=(
            "y-axis quantity (one or more): eta_is | lambda_h | zeta_gl | "
            "m_flow | P_el | T_dis | all. Use 'all' to plot every supported metric."
        ),
    )

    ap.add_argument(
        "--vary", required=True,
        choices=["T_evap", "T_cond", "speed", "superheat", "pressure_ratio"],
    )

    ap.add_argument("--T_evap_min", type=float, default=-5.0)
    ap.add_argument("--T_evap_max", type=float, default=25.0)
    ap.add_argument("--T_cond_min", type=float, default=25.0)
    ap.add_argument("--T_cond_max", type=float, default=65.0)
    ap.add_argument("--N_rpm_min", type=float, default=1800.0)
    ap.add_argument("--N_rpm_max", type=float, default=7200.0)
    ap.add_argument("--SH_K_min", type=float, default=5.0)
    ap.add_argument("--SH_K_max", type=float, default=35.0)
    ap.add_argument("--PR_min", type=float, default=2.0)
    ap.add_argument("--PR_max", type=float, default=6.0)
    ap.add_argument("--n_points", type=int, default=50)

    ap.add_argument("--T_evap", type=float, nargs="+", default=[10.0])
    ap.add_argument("--T_cond", type=float, nargs="+", default=[50.0])
    ap.add_argument("--N_rpm", type=float, nargs="+", default=[3600.0])
    ap.add_argument("--SH_K", type=float, nargs="+", default=[10.0])

    # Solver tolerances
    ap.add_argument("--lsq_ftol", type=float, default=1e-12,
                    help="Solver function tolerance (default 1e-12)")
    ap.add_argument("--lsq_xtol", type=float, default=1e-12,
                    help="Solver variable tolerance (default 1e-12)")
    ap.add_argument("--lsq_max_nfev", type=int, default=50000,
                    help="Solver max function evaluations (default 50000)")

    # Smoothing
    ap.add_argument("--smooth", action="store_true",
                    help="Apply Savitzky-Golay smoothing to curves")
    ap.add_argument("--smooth_window", type=int, default=7,
                    help="Smoothing window length (odd number, default 7)")
    ap.add_argument("--smooth_polyorder", type=int, default=2,
                    help="Smoothing polynomial order (default 2)")

    ap.add_argument("--out_dir", default="results/efficiency_curves")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    if not args.params_csv_oil1.exists():
        raise FileNotFoundError(args.params_csv_oil1)
    if not args.params_csv_oil2.exists():
        raise FileNotFoundError(args.params_csv_oil2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expand 'all' shortcut and validate
    requested_metrics = list(args.metric)
    if "all" in requested_metrics:
        metrics_to_plot = list(METRIC_CONFIG.keys())
    else:
        invalid = [m for m in requested_metrics if m not in METRIC_CONFIG]
        if invalid:
            raise ValueError(
                f"Unknown metric(s): {invalid}. "
                f"Valid options: {list(METRIC_CONFIG.keys()) + ['all']}"
            )
        # Preserve order, remove duplicates
        seen = set()
        metrics_to_plot = [m for m in requested_metrics if not (m in seen or seen.add(m))]

    params_peek = pd.read_csv(args.params_csv_oil1).iloc[0].to_dict()

    if args.model == "auto":
        args.model = str(params_peek.get("model", "modified"))
    if args.refrigerant == "auto":
        args.refrigerant = str(params_peek.get("refrigerant", "PROPANE"))

    params_oil1, meta_oil1 = load_params_csv(args.params_csv_oil1, args.model)
    params_oil2, meta_oil2 = load_params_csv(args.params_csv_oil2, args.model)

    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6

    med = RefProp(fluid_name=args.refrigerant)
    m_dot_ref = compute_m_dot_ref(med, V_h_m3)

    params_oil1["f_ref"] = F_REF
    params_oil1["m_dot_ref"] = m_dot_ref
    params_oil2["f_ref"] = F_REF
    params_oil2["m_dot_ref"] = m_dot_ref

    print(f"  Model:           {args.model}")
    print(f"  Refrigerant:     {args.refrigerant}")
    print(f"  Oil 1:           {args.oil1}  (params from: {meta_oil1.get('oil', '?')})")
    print(f"  Oil 2:           {args.oil2}  (params from: {meta_oil2.get('oil', '?')})")
    print(f"  Metric(s):       {', '.join(metrics_to_plot)}")
    print(f"  Vary:            {args.vary}")

    same_params = args.params_csv_oil1.resolve() == args.params_csv_oil2.resolve()
    if same_params:
        print(f"  → Cross-validation: both oils use the same params ({meta_oil1.get('oil', '?')})")

    print(f"  Solver tol:      ftol={args.lsq_ftol:.0e}, xtol={args.lsq_xtol:.0e}, max_nfev={args.lsq_max_nfev}")
    if args.smooth:
        print(f"  Smoothing:       ON (window={args.smooth_window}, polyorder={args.smooth_polyorder})")
    else:
        print(f"  Smoothing:       OFF")

    vary_config = {
        "T_evap": {
            "values": np.linspace(args.T_evap_min, args.T_evap_max, args.n_points),
            "label": "Verdampfungstemperatur [°C]",
            "series_params": ["T_cond", "N_rpm", "SH_K"],
        },
        "T_cond": {
            "values": np.linspace(args.T_cond_min, args.T_cond_max, args.n_points),
            "label": "Kondensationstemperatur [°C]",
            "series_params": ["T_evap", "N_rpm", "SH_K"],
        },
        "speed": {
            "values": np.linspace(args.N_rpm_min, args.N_rpm_max, args.n_points),
            "label": "Drehzahl [rpm]",
            "series_params": ["T_evap", "T_cond", "SH_K"],
        },
        "superheat": {
            "values": np.linspace(args.SH_K_min, args.SH_K_max, args.n_points),
            "label": "Überhitzung [K]",
            "series_params": ["T_evap", "T_cond", "N_rpm"],
        },
        "pressure_ratio": {
            "values": np.linspace(args.PR_min, args.PR_max, args.n_points),
            "label": "Druckverhältnis $p_{aus}/p_{ein}$ [-]",
            "series_params": ["T_evap", "N_rpm", "SH_K"],
        },
    }

    cfg = vary_config[args.vary]
    vary_values = cfg["values"]
    vary_label = cfg["label"]

    series_param_map = {
        "T_evap": args.T_evap,
        "T_cond": args.T_cond,
        "N_rpm": args.N_rpm,
        "SH_K": args.SH_K,
    }

    series_keys = cfg["series_params"]
    series_value_lists = [series_param_map[k] for k in series_keys]
    series_combinations = list(itertools.product(*series_value_lists))

    print(f"  Sweep: {len(vary_values)} points, {vary_values[0]:.1f} → {vary_values[-1]:.1f}")
    print(f"  Series: {len(series_combinations)} combinations of {series_keys}")

    m = str(args.model).lower().strip()
    needs_oil_name = m in ("mod", "modified", "oil_path", "oilpath")

    sweep_results = {}

    label_map = {
        "T_evap": ("$T_{evap}$", "°C"),
        "T_cond": ("$T_{cond}$", "°C"),
        "N_rpm": ("N", "rpm"),
        "SH_K": ("SH", "K"),
    }

    for combo in series_combinations:
        fixed = dict(zip(series_keys, combo))

        T_evap_fixed = fixed.get("T_evap", args.T_evap[0])
        T_cond_fixed = fixed.get("T_cond", args.T_cond[0])
        N_rpm_fixed = fixed.get("N_rpm", args.N_rpm[0])
        SH_K_fixed = fixed.get("SH_K", args.SH_K[0])

        label_parts = []
        for k, v in fixed.items():
            name, unit = label_map[k]
            label_parts.append(f"{name}={v:.0f} {unit}")
        series_label = ", ".join(label_parts)

        print(f"  Simulating: {series_label} ...")

        df_oil1 = run_sweep(
            med=med, model=args.model,
            refrigerant_name=args.refrigerant,
            oil_name=args.oil1 if needs_oil_name else None,
            params=params_oil1, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            vary_name=args.vary, vary_values=vary_values,
            T_evap_C_fixed=T_evap_fixed, T_cond_C_fixed=T_cond_fixed,
            N_rpm_fixed=N_rpm_fixed, SH_K_fixed=SH_K_fixed,
            T_amb_C=args.T_amb_C,
            lsq_ftol=args.lsq_ftol, lsq_xtol=args.lsq_xtol,
            lsq_max_nfev=args.lsq_max_nfev,
        )

        df_oil2 = run_sweep(
            med=med, model=args.model,
            refrigerant_name=args.refrigerant,
            oil_name=args.oil2 if needs_oil_name else None,
            params=params_oil2, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            vary_name=args.vary, vary_values=vary_values,
            T_evap_C_fixed=T_evap_fixed, T_cond_C_fixed=T_cond_fixed,
            N_rpm_fixed=N_rpm_fixed, SH_K_fixed=SH_K_fixed,
            T_amb_C=args.T_amb_C,
            lsq_ftol=args.lsq_ftol, lsq_xtol=args.lsq_xtol,
            lsq_max_nfev=args.lsq_max_nfev,
        )

        n1, n2 = len(df_oil1), len(df_oil2)
        print(f"    → {args.oil1}: {n1}/{len(vary_values)} OK, {args.oil2}: {n2}/{len(vary_values)} OK")

        sweep_results[series_label] = {"oil1": df_oil1, "oil2": df_oil2}

    # -------------------------
    # Apply smoothing (if requested)
    # -------------------------
    if args.smooth:
        print(f"  Smoothing: window={args.smooth_window}, polyorder={args.smooth_polyorder}")
        for series_label, oil_data in sweep_results.items():
            for oil_key in ("oil1", "oil2"):
                df = oil_data[oil_key]
                if not df.empty:
                    oil_data[oil_key] = smooth_sweep_data(
                        df, SMOOTHABLE_COLS,
                        window=args.smooth_window,
                        polyorder=args.smooth_polyorder,
                    )

    oil1_params_src = meta_oil1.get("oil", "?")
    oil2_params_src = meta_oil2.get("oil", "?")

    oil1_legend = f"{args.oil1} (Params: {oil1_params_src})"
    oil2_legend = f"{args.oil2} (Params: {oil2_params_src})"

    stamp = _ts()
    smooth_suffix = "_smooth" if args.smooth else ""

    # Generate one plot per requested metric
    for metric_name in metrics_to_plot:
        metric_cfg = METRIC_CONFIG[metric_name]
        metric_label = metric_cfg["label"]
        metric_col_raw = metric_cfg["column"]
        metric_short = metric_cfg["title_short"]

        # Use smoothed column if available, otherwise raw
        if args.smooth:
            metric_col = f"{metric_col_raw}_smooth"
        else:
            metric_col = metric_col_raw

        title = f"{metric_short} vs. {vary_label}"
        if args.smooth:
            title += f" (geglättet, w={args.smooth_window})"
        title += f"\n{args.model.capitalize()} | {args.refrigerant}"

        out_path = (
            out_dir
            / f"efficiency_{metric_name}_{args.vary}_{args.oil1.lower()}_vs_{args.oil2.lower()}{smooth_suffix}_{stamp}.{args.out_format}"
        )

        plot_curves(
            sweep_results=sweep_results,
            metric_label=metric_label,
            metric_col=metric_col,
            vary_label=vary_label,
            oil1_label=oil1_legend,
            oil2_label=oil2_legend,
            title=title,
            out_path=out_path,
        )

    # Save the simulation data once (contains all metrics + smoothed if applicable)
    data_stem = (
        f"efficiency_data_{args.vary}_{args.oil1.lower()}_vs_{args.oil2.lower()}{smooth_suffix}_{stamp}"
    )

    all_data = []
    for series_label, oil_data in sweep_results.items():
        for oil_key, df in oil_data.items():
            if not df.empty:
                df_out = df.copy()
                df_out["series"] = series_label
                df_out["oil_label"] = args.oil1 if oil_key == "oil1" else args.oil2
                df_out["params_source"] = oil1_params_src if oil_key == "oil1" else oil2_params_src
                all_data.append(df_out)

    if all_data:
        data_csv = out_dir / f"{data_stem}.csv"
        pd.concat(all_data, ignore_index=True).to_csv(data_csv, index=False)
        print(f"  [OK] Data saved: {data_csv}")

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
