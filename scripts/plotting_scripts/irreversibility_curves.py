# scripts/plotting_scripts/irreversibility_curves.py
#
# Plots compressor irreversibility breakdown (James et al. 2016 Fig. 10/11 style)
# vs. one varying operating parameter, with TWO oils each in their own subplots.
#
# Each oil gets its own column of subplots (one per series combination).
#
# Irreversibility categories (auto-detected per model):
#   1. Sauggas-Aufheizung    (suction heat transfer)
#   2. Leckage + Reexpansion  (leakage mixing)
#   3. Druckseiten-Wärme      (discharge heat transfer)
#   4. Lastabhängig           (W_dot_loss_load)
#   5. Drehzahlabhängig       (W_dot_loss_ref_term)
#   6. Viskositätsreibung     (W_dot_loss_fric, modified + oil_path only)
#   7. Öl-Hydraulik           (W_dot_oil_recirc, oil_path only)
#
# Activate REFPROP:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Vary T_evap, both oils with own params:
#   python scripts/plotting_scripts/irreversibility_curves.py --params_csv_oil1 results/final_results/Modified_LPG68/Fitting/fitted_params_lpg68_modified_ga_2026-03-22_185546.csv --params_csv_oil2 results/final_results/Modified_LPG100/Fitting/fitted_params_lpg100_modified_ga_2026-03-28_092941.csv --oil1 LPG68 --oil2 LPG100 --vary T_evap --T_cond 50 --N_rpm 3600 --SH_K 10 --normalize_by_mflow
#
#   # Normalize by mass flow (J/g), like James:
#   python scripts/plotting_scripts/irreversibility_curves.py \
#       --params_csv_oil1 results/final_results/Modified_LPG68/Fitting/fitted_params_lpg68_modified_ga_2026-03-22_185546.csv \
#       --params_csv_oil2 results/final_results/Modified_LPG100/Fitting/fitted_params_lpg100_modified_ga_2026-03-28_092941.csv \
#       --oil1 LPG68 --oil2 LPG100 \
#       --vary speed --T_evap 10 --T_cond 50 --SH_K 10 --normalize_by_mflow

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

from vclibpy.media import RefProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import Molinaroli_2017_Compressor
from vclibpy.components.compressors.rolling_piston_Molinaroli_2017_modified import (
    Molinaroli_2017_Compressor_Modified,
)

# Optional: oil_path model (only import if available)
try:
    from vclibpy.components.compressors.rolling_piston_Molinaroli_oil_path import (
        Molinaroli_2017_Compressor_OilPath,
    )
    OIL_PATH_AVAILABLE = True
except ImportError:
    try:
        # Alternative class name
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
T_AMB_REF_K = 298.15  # Reference temperature for exergy calculations (25°C)


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
PARAM_NAMES_OIL_PATH = [
    "Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
    "V_IC", "alpha_loss", "W_dot_loss_ref", "alpha_fric_tot",
    "m_dot_oil_ref", "Ua_suc_oil_ref",
]

DEFAULT_PARAMS_ORIGINAL = {
    "Ua_suc_ref": 16.05, "Ua_dis_ref": 13.96, "Ua_amb": 0.36,
    "A_tot": 9.47e-9, "A_dis": 86.1e-6, "V_IC": 30.7e-6,
    "alpha_loss": 0.16, "W_dot_loss_ref": 83.0,
    "m_dot_ref": None, "f_ref": F_REF,
}
DEFAULT_PARAMS_MODIFIED = {
    **DEFAULT_PARAMS_ORIGINAL,
    "W_dot_loss_ref": 10.0, "alpha_fric_tot": 120.0,
}
DEFAULT_PARAMS_OIL_PATH = {
    **DEFAULT_PARAMS_MODIFIED,
    "m_dot_oil_ref": 0.005, "Ua_suc_oil_ref": 5.0,
}


# =========================================================
# Helpers
# =========================================================
def _ts():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _finite(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def c_to_k(t): return float(t) + 273.15
def k_to_c(t): return float(t) - 273.15
def rpm_to_hz(n): return float(n) / 60.0


def map_refrigerant_for_modified_model(name):
    s = str(name).strip().upper()
    return "propane" if s in {"PROPANE", "R290", "PROPAN"} else str(name).strip()


def map_oil_for_modified_model(name):
    s = str(name).strip().lower().replace(" ", "")
    if s == "lpg68": return "LPG 68"
    if s == "lpg100": return "LPG 100"
    raise ValueError(f"Unsupported oil: {name}")


def get_param_names(model):
    m = str(model).lower().strip()
    if m in ("orig", "original"): return list(PARAM_NAMES_ORIGINAL)
    if m in ("mod", "modified"): return list(PARAM_NAMES_MODIFIED)
    if m in ("oil_path", "oilpath"): return list(PARAM_NAMES_OIL_PATH)
    raise ValueError(f"Unknown model: {model}")


def get_default_params(model):
    m = str(model).lower().strip()
    if m in ("orig", "original"): return dict(DEFAULT_PARAMS_ORIGINAL)
    if m in ("mod", "modified"): return dict(DEFAULT_PARAMS_MODIFIED)
    if m in ("oil_path", "oilpath"): return dict(DEFAULT_PARAMS_OIL_PATH)
    raise ValueError(f"Unknown model: {model}")


def make_compressor(model, N_max_hz, V_h_m3, params, refrigerant_name, oil_name=None):
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    if m in ("mod", "modified"):
        if oil_name is None: raise ValueError("Modified model requires oil_name.")
        return Molinaroli_2017_Compressor_Modified(
            N_max=N_max_hz, V_h=V_h_m3,
            fluid_name=map_refrigerant_for_modified_model(refrigerant_name),
            lub_name=map_oil_for_modified_model(oil_name),
            parameters=params,
        )
    if m in ("oil_path", "oilpath"):
        if not OIL_PATH_AVAILABLE:
            raise ImportError("oil_path model not available in this vclibpy installation.")
        if oil_name is None: raise ValueError("oil_path model requires oil_name.")
        return Molinaroli_2017_Compressor_OilPath(
            N_max=N_max_hz, V_h=V_h_m3,
            fluid_name=map_refrigerant_for_modified_model(refrigerant_name),
            lub_name=map_oil_for_modified_model(oil_name),
            parameters=params,
        )
    raise ValueError(f"Unknown model: {model}")


def compute_m_dot_ref(med, V_h_m3):
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
def load_params_csv(path, model):
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    params = get_default_params(model)
    for k in get_param_names(model):
        if k in row and pd.notna(row[k]):
            params[k] = float(row[k])
    if "f_ref" in row and pd.notna(row["f_ref"]):
        params["f_ref"] = float(row["f_ref"])
    meta = {k: row[k] for k in ("oil", "refrigerant", "model") if k in row}
    return params, meta


# =========================================================
# State helpers
# =========================================================
def compute_suction_state(med, T_evap_C, SH_K):
    T_evap_K = c_to_k(T_evap_C)
    state_sat = med.calc_state("TQ", T_evap_K, Q_REF)
    return float(state_sat.p), T_evap_K + float(SH_K)


def compute_discharge_pressure(med, T_cond_C):
    T_cond_K = c_to_k(T_cond_C)
    state_sat = med.calc_state("TQ", T_cond_K, 0.0)
    return float(state_sat.p)


# =========================================================
# Irreversibility computation
# =========================================================
def compute_irreversibilities(comp, T_o_K=T_AMB_REF_K):
    """
    Compute irreversibility contributions for a converged operating point.

    Returns dict with W of each irreversibility category (always positive).
    Some keys may be NaN if the model doesn't provide them.
    """
    result = {}

    # Mass flows
    m_dot_suc = _finite(getattr(comp, "m_flow", np.nan))

    # Internal states
    st_in = getattr(comp, "state_inlet", None)
    st_c1 = getattr(comp, "state_c_1", None)
    st_c3 = getattr(comp, "state_c_3", None)
    st_c4 = getattr(comp, "state_c_4", None)
    st_c5 = getattr(comp, "state_c_5", None)
    st_out = getattr(comp, "state_outlet", None)

    # ----- 1. Suction heat transfer (state_inlet → state_c_1) -----
    # Exergy change of refrigerant + carnot loss on heat transfer
    # Δψ = m * [(h1 - h_suc) - T_o * (s1 - s_suc)] - Q_suc * (1 - T_o/T_w)
    # Approximation: take Q_suc = m * (h1 - h_suc) (from heat exchanger)
    # Then irreversibility = Q_suc * (1 - T_o/T_w) - m * Δψ_refrigerant
    # Simpler: entropy generation × T_o
    try:
        if st_in is not None and st_c1 is not None and m_dot_suc > 0:
            T_w = _finite(getattr(comp, "T_w", np.nan))
            ds = float(st_c1.s) - float(st_in.s)
            # Heat from wall side: Q = m * cp * (T_c1 - T_in) ≈ m * (h1 - h_suc)
            Q = m_dot_suc * (float(st_c1.h) - float(st_in.h))
            # Entropy of wall
            ds_wall = -Q / T_w if (np.isfinite(T_w) and T_w > 0) else 0.0
            s_gen = m_dot_suc * ds + ds_wall
            irr = T_o_K * s_gen
            result["I_suc_heat"] = max(0.0, _finite(irr))
        else:
            result["I_suc_heat"] = float("nan")
    except Exception:
        result["I_suc_heat"] = float("nan")

    # ----- 2. Leakage + Reexpansion mixing (c1 + c4 → c3) -----
    # Entropy generation when leak from c4 mixes with suction stream c1
    # Result is c3 with mass flow m_dot_3 = m_dot_suc + m_dot_tot
    try:
        if (st_c1 is not None and st_c3 is not None and st_c4 is not None
                and m_dot_suc > 0):
            # m_dot_3 = rho_3 * V_IC * f
            f_hz = comp.get_n_absolute(comp.parameters.get("f_ref", F_REF) /
                                       comp.parameters.get("f_ref", F_REF))
            # Better: get from internal state
            V_IC = comp.parameters["V_IC"]
            # Need actual frequency -- recompute from comp.f if available
            try:
                # comp may have stored the operating frequency
                f_hz = comp.get_n_absolute(getattr(comp, "_n_rel_last", 1.0))
            except Exception:
                # Fall back: compute from state_inlet density and m_dot_suc
                # m_dot_suc = rho_suc * V_IC * f (approximation)
                rho_suc = float(st_in.d)
                f_hz = m_dot_suc / (rho_suc * V_IC) if rho_suc > 0 else F_REF

            rho_3 = float(st_c3.d)
            m_dot_3 = rho_3 * V_IC * f_hz
            m_dot_leak = max(0.0, m_dot_3 - m_dot_suc)

            # Entropy generation in mixing
            s_gen = (m_dot_3 * float(st_c3.s)
                     - m_dot_suc * float(st_c1.s)
                     - m_dot_leak * float(st_c4.s))
            irr = T_o_K * s_gen
            result["I_leak_reexp"] = max(0.0, _finite(irr))
        else:
            result["I_leak_reexp"] = float("nan")
    except Exception:
        result["I_leak_reexp"] = float("nan")

    # ----- 3. Discharge heat transfer (state_c_4 / c_5 → state_outlet) -----
    # Heat is transferred from gas to wall on the discharge side
    try:
        if st_c4 is not None and st_out is not None and m_dot_suc > 0:
            T_w = _finite(getattr(comp, "T_w", np.nan))
            # Use h_dis difference and s difference for refrigerant
            # Approximation: m_dot_discharge ≈ m_dot_suc (for original/modified)
            # For oil_path, m_dot_gas_discharge is available
            m_dot_gas = _finite(getattr(comp, "m_dot_gas_discharge", m_dot_suc))
            if not np.isfinite(m_dot_gas) or m_dot_gas <= 0:
                m_dot_gas = m_dot_suc

            # Heat lost from gas to wall
            Q_dis = m_dot_gas * (float(st_c4.h) - float(st_out.h))
            ds_gas = float(st_out.s) - float(st_c4.s)
            ds_wall = Q_dis / T_w if (np.isfinite(T_w) and T_w > 0) else 0.0
            s_gen = m_dot_gas * ds_gas + ds_wall
            irr = T_o_K * s_gen
            result["I_dis_heat"] = max(0.0, _finite(irr))
        else:
            result["I_dis_heat"] = float("nan")
    except Exception:
        result["I_dis_heat"] = float("nan")

    # ----- 4. Load-dependent loss -----
    val = _finite(getattr(comp, "W_dot_loss_load", np.nan))
    result["I_load"] = max(0.0, val) if np.isfinite(val) else float("nan")

    # ----- 5. Speed-dependent loss -----
    val = _finite(getattr(comp, "W_dot_loss_ref_term", np.nan))
    result["I_speed"] = max(0.0, val) if np.isfinite(val) else float("nan")

    # ----- 6. Viscous friction (modified + oil_path only) -----
    val = _finite(getattr(comp, "W_dot_loss_fric", np.nan))
    if np.isfinite(val) and val > 0:
        result["I_visc"] = val
    else:
        result["I_visc"] = float("nan")

    # ----- 7. Oil hydraulic (oil_path only) -----
    val = _finite(getattr(comp, "W_dot_oil_recirc", np.nan))
    if np.isfinite(val) and val > 0:
        result["I_oil_recirc"] = val
    else:
        result["I_oil_recirc"] = float("nan")

    return result


# =========================================================
# Sweep
# =========================================================
def compute_point(comp, med, p_suc_pa, T_suc_K, p_out_pa, n_rel, T_amb_K,
                  T_o_K, lsq_ftol=1e-12, lsq_xtol=1e-12, lsq_max_nfev=50000):
    """Simulate one point and compute irreversibilities + basic metrics."""
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
            # Store n_rel for irreversibility helper
            comp._n_rel_last = inputs.control.n
            comp.calc_state_outlet(p_outlet=float(p_out_pa), inputs=inputs, fs_state=fs_state)

        m_flow = _finite(getattr(comp, "m_flow", np.nan))
        P_el = _finite(getattr(comp, "P_el", np.nan))
        if not np.isfinite(m_flow) or m_flow <= 0 or not np.isfinite(P_el) or P_el <= 0:
            return None

        result = {
            "m_flow": m_flow,
            "m_flow_gps": m_flow * 1e3,
            "P_el": P_el,
        }
        result.update(compute_irreversibilities(comp, T_o_K=T_o_K))
        return result
    except Exception:
        return None


def run_sweep(med, model, refrigerant_name, oil_name, params,
              N_max_hz, V_h_m3, vary_name, vary_values,
              T_evap_C_fixed, T_cond_C_fixed, N_rpm_fixed, SH_K_fixed,
              T_amb_C=25.0, T_o_K=T_AMB_REF_K,
              lsq_ftol=1e-12, lsq_xtol=1e-12, lsq_max_nfev=50000):
    """Sweep one parameter, compute irreversibilities for each point."""
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
                p_out = p_suc * float(val)
            else:
                p_out = compute_discharge_pressure(med, T_cond_C)
        except Exception:
            continue

        f_hz = rpm_to_hz(N_rpm)
        n_rel = f_hz / N_max_hz
        if p_out <= p_suc or n_rel <= 0 or n_rel > 1.0:
            continue

        result = compute_point(comp, med, p_suc, T_suc_K, p_out, n_rel, T_amb_K, T_o_K,
                               lsq_ftol, lsq_xtol, lsq_max_nfev)
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
IRR_CONFIG = [
    ("I_suc_heat",   "Sauggas-Aufheizung",  "#EC635C"),
    ("I_leak_reexp", "Leckage + Reexpansion", "#4B81C4"),
    ("I_dis_heat",   "Druckseiten-Wärme",   "#F49961"),
    ("I_load",       "Lastabhängig",        "#8768B4"),
    ("I_speed",      "Drehzahlabhängig",    "#B45955"),
    ("I_visc",       "Viskositätsreibung",  "#CB74F4"),
    ("I_oil_recirc", "Öl-Hydraulik",        "#6EBB96"),
]


def detect_available_irr(sweep_results: dict) -> list[tuple[str, str, str]]:
    """Detect which irreversibility columns have valid data across sweeps."""
    available = []
    for col, label, color in IRR_CONFIG:
        has_data = False
        for oil_data in sweep_results.values():
            for df in oil_data.values():
                if df.empty or col not in df.columns:
                    continue
                vals = pd.to_numeric(df[col], errors="coerce")
                if vals.notna().any() and (vals > 0).any():
                    has_data = True
                    break
            if has_data:
                break
        if has_data:
            available.append((col, label, color))
    return available


def plot_irreversibilities(
    sweep_results: dict,
    irr_cols: list[tuple[str, str, str]],
    vary_label: str,
    oil_legends: dict,
    title: str,
    out_path: Path,
    normalize: bool = False,
):
    """
    Plot one column per oil, one row per series.
    Each subplot: lines for each irreversibility category.
    """
    series_list = list(sweep_results.keys())
    n_series = len(series_list)

    if n_series == 0:
        print("  [SKIP] No series.")
        return

    # 2 columns (one per oil), n_series rows
    fig, axes = plt.subplots(
        n_series, 2,
        figsize=(13, 5 * n_series),
        squeeze=False,
    )

    y_unit = "kW/(g/s)" if normalize else "W"
    y_factor = 1.0 / 1000.0 if normalize else 1.0  # W/(g/s) → kW/(g/s) only if normalize

    for row_idx, series_label in enumerate(series_list):
        for col_idx, oil_key in enumerate(("oil1", "oil2")):
            ax = axes[row_idx, col_idx]
            df = sweep_results[series_label][oil_key]

            if df.empty:
                ax.set_title(f"{oil_legends[oil_key]} — {series_label}\n(keine Daten)")
                continue

            x = df["vary_value"].to_numpy(dtype=float)

            for col, label, color in irr_cols:
                if col not in df.columns:
                    continue
                y = df[col].to_numpy(dtype=float)

                if normalize:
                    m_dot_gps = df["m_flow_gps"].to_numpy(dtype=float)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        y = np.where(m_dot_gps > 0, y / m_dot_gps, np.nan)
                    # y is now in W/(g/s) = J/g; scale to kJ/g for readability? No -- keep W/(g/s)
                    # Better: keep absolute units
                    # W/(g/s) = J/g = kJ/kg

                # Skip series that are all-NaN or all-zero
                if not np.any(np.isfinite(y) & (y > 0)):
                    continue

                ax.plot(
                    x, y,
                    marker="o", markersize=4, linewidth=2.0,
                    color=color, label=label,
                )

            if row_idx == 0:
                ax.set_title(f"{oil_legends[oil_key]}\n{series_label}", fontsize=12)
            else:
                ax.set_title(series_label, fontsize=12)

            ax.set_xlabel(vary_label)
            if normalize:
                ax.set_ylabel("Irreversibilität in W/(g/s)")
            else:
                ax.set_ylabel("Irreversibilität in W")
            ax.grid(True, linewidth=0.6, alpha=0.35)

    # Single legend below the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        # Try second oil column
        handles, labels = axes[0, 1].get_legend_handles_labels()

    if handles:
        # Determine number of legend columns
        n_legend = min(len(handles), 4)
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=n_legend,
            frameon=True, fontsize=11,
        )

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Compressor irreversibility breakdown (James et al. 2016 style)."
    )

    ap.add_argument("--params_csv_oil1", required=True, type=Path)
    ap.add_argument("--params_csv_oil2", required=True, type=Path)
    ap.add_argument("--oil1", required=True)
    ap.add_argument("--oil2", required=True)

    ap.add_argument("--model", default="auto",
                    help="original | modified | oil_path | auto")
    ap.add_argument("--refrigerant", default="auto")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)
    ap.add_argument("--T_amb_C", type=float, default=25.0)
    ap.add_argument("--T_ref_C", type=float, default=25.0,
                    help="Reference temperature for exergy [°C]")

    ap.add_argument(
        "--vary", required=True,
        choices=["T_evap", "T_cond", "speed", "superheat", "pressure_ratio"],
    )

    # Sweep ranges
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

    # Fixed parameters
    ap.add_argument("--T_evap", type=float, nargs="+", default=[10.0])
    ap.add_argument("--T_cond", type=float, nargs="+", default=[50.0])
    ap.add_argument("--N_rpm", type=float, nargs="+", default=[3600.0])
    ap.add_argument("--SH_K", type=float, nargs="+", default=[10.0])

    # Solver
    ap.add_argument("--lsq_ftol", type=float, default=1e-12)
    ap.add_argument("--lsq_xtol", type=float, default=1e-12)
    ap.add_argument("--lsq_max_nfev", type=int, default=50000)

    # Plot options
    ap.add_argument("--normalize_by_mflow", action="store_true",
                    help="Normalize irreversibilities by mass flow [W/(g/s)]")
    ap.add_argument("--out_dir", default="results/irreversibility_curves")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    if not args.params_csv_oil1.exists():
        raise FileNotFoundError(args.params_csv_oil1)
    if not args.params_csv_oil2.exists():
        raise FileNotFoundError(args.params_csv_oil2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve auto
    params_peek = pd.read_csv(args.params_csv_oil1).iloc[0].to_dict()
    if args.model == "auto":
        args.model = str(params_peek.get("model", "modified"))
    if args.refrigerant == "auto":
        args.refrigerant = str(params_peek.get("refrigerant", "PROPANE"))

    params_oil1, meta_oil1 = load_params_csv(args.params_csv_oil1, args.model)
    params_oil2, meta_oil2 = load_params_csv(args.params_csv_oil2, args.model)

    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6
    T_o_K = c_to_k(args.T_ref_C)

    med = RefProp(fluid_name=args.refrigerant)
    m_dot_ref = compute_m_dot_ref(med, V_h_m3)
    params_oil1["f_ref"] = F_REF
    params_oil1["m_dot_ref"] = m_dot_ref
    params_oil2["f_ref"] = F_REF
    params_oil2["m_dot_ref"] = m_dot_ref

    same_params = args.params_csv_oil1.resolve() == args.params_csv_oil2.resolve()

    print(f"  Model:           {args.model}")
    print(f"  Refrigerant:     {args.refrigerant}")
    print(f"  Oil 1:           {args.oil1}  (params from: {meta_oil1.get('oil', '?')})")
    print(f"  Oil 2:           {args.oil2}  (params from: {meta_oil2.get('oil', '?')})")
    print(f"  Vary:            {args.vary}")
    print(f"  T_o (Exergie):   {args.T_ref_C}°C")
    print(f"  Normalize:       {args.normalize_by_mflow}")
    if same_params:
        print(f"  → Cross-validation: both oils use the same params")

    # Sweep config
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
    series_combinations = list(itertools.product(*[series_param_map[k] for k in series_keys]))

    print(f"  Sweep: {len(vary_values)} points, {vary_values[0]:.1f} → {vary_values[-1]:.1f}")
    print(f"  Series: {len(series_combinations)} combinations of {series_keys}")

    m = str(args.model).lower().strip()
    needs_oil_name = m in ("mod", "modified", "oil_path", "oilpath")

    label_map = {
        "T_evap": ("$T_{evap}$", "°C"),
        "T_cond": ("$T_{cond}$", "°C"),
        "N_rpm": ("N", "rpm"),
        "SH_K": ("SH", "K"),
    }

    sweep_results = {}

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
            med=med, model=args.model, refrigerant_name=args.refrigerant,
            oil_name=args.oil1 if needs_oil_name else None,
            params=params_oil1, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            vary_name=args.vary, vary_values=vary_values,
            T_evap_C_fixed=T_evap_fixed, T_cond_C_fixed=T_cond_fixed,
            N_rpm_fixed=N_rpm_fixed, SH_K_fixed=SH_K_fixed,
            T_amb_C=args.T_amb_C, T_o_K=T_o_K,
            lsq_ftol=args.lsq_ftol, lsq_xtol=args.lsq_xtol,
            lsq_max_nfev=args.lsq_max_nfev,
        )
        df_oil2 = run_sweep(
            med=med, model=args.model, refrigerant_name=args.refrigerant,
            oil_name=args.oil2 if needs_oil_name else None,
            params=params_oil2, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            vary_name=args.vary, vary_values=vary_values,
            T_evap_C_fixed=T_evap_fixed, T_cond_C_fixed=T_cond_fixed,
            N_rpm_fixed=N_rpm_fixed, SH_K_fixed=SH_K_fixed,
            T_amb_C=args.T_amb_C, T_o_K=T_o_K,
            lsq_ftol=args.lsq_ftol, lsq_xtol=args.lsq_xtol,
            lsq_max_nfev=args.lsq_max_nfev,
        )
        n1, n2 = len(df_oil1), len(df_oil2)
        print(f"    → {args.oil1}: {n1}/{len(vary_values)} OK, {args.oil2}: {n2}/{len(vary_values)} OK")
        sweep_results[series_label] = {"oil1": df_oil1, "oil2": df_oil2}

    # Detect available irreversibility categories
    irr_cols = detect_available_irr(sweep_results)
    print(f"  Categories: {[label for _, label, _ in irr_cols]}")

    if not irr_cols:
        raise RuntimeError("No irreversibility categories with valid data found.")

    oil1_params_src = meta_oil1.get("oil", "?")
    oil2_params_src = meta_oil2.get("oil", "?")
    oil_legends = {
        "oil1": f"{args.oil1} (Params: {oil1_params_src})",
        "oil2": f"{args.oil2} (Params: {oil2_params_src})",
    }

    norm_tag = " (normalisiert)" if args.normalize_by_mflow else ""
    title = f"Irreversibilitäten vs. {vary_label}{norm_tag}"
    title += f"\n{args.model.capitalize()} | {args.refrigerant} | $T_o$={args.T_ref_C}°C"

    stamp = _ts()
    norm_suffix = "_norm" if args.normalize_by_mflow else ""
    out_path = (
        out_dir
        / f"irreversibility_{args.vary}_{args.oil1.lower()}_vs_{args.oil2.lower()}{norm_suffix}_{stamp}.{args.out_format}"
    )

    plot_irreversibilities(
        sweep_results=sweep_results,
        irr_cols=irr_cols,
        vary_label=vary_label,
        oil_legends=oil_legends,
        title=title,
        out_path=out_path,
        normalize=args.normalize_by_mflow,
    )

    # Save data
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
        data_csv = out_path.with_suffix(".csv")
        pd.concat(all_data, ignore_index=True).to_csv(data_csv, index=False)
        print(f"  [OK] Data saved: {data_csv}")

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
