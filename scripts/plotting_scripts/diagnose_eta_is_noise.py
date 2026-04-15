# scripts/plotting_scripts/diagnose_eta_is_noise.py
#
# Extended diagnostic script for investigating eta_is noise.
#
# Three sweeps:
#   1. FORWARD: T_cond from min to max (standard order)
#   2. REVERSE: T_cond from max to min (checks order-dependence)
#   3. FORWARD with solver telemetry: captures nfev, residual, status
#
# If forward and reverse show spikes at the SAME T_cond values:
#   → Model-inherent (regime switches, non-smooth correlations)
# If spikes shift position:
#   → Order-dependent (warm start / state transfer effect)
#
# The solver telemetry reveals whether the solver struggles at spike positions.
#
# Activate REFPROP:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Example:
#   python scripts/plotting_scripts/diagnose_eta_is_noise.py --params_csv results/final_results/Modified_LPG68/Fitting/fitted_params_lpg68_modified_ga_2026-03-22_185546.csv --oil LPG68 --T_evap 10 --T_cond_min 25 --T_cond_max 65 --n_points 80

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize

from vclibpy.media import RefProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import Molinaroli_2017_Compressor
from vclibpy.components.compressors.rolling_piston_Molinaroli_2017_modified import (
    Molinaroli_2017_Compressor_Modified,
)

plt.style.use("ebc.paper.mplstyle")


# =========================================================
# Constants
# =========================================================
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0

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


# =========================================================
# Helpers
# =========================================================
def _ts():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

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
    raise ValueError("Unknown model.")

def get_default_params(model):
    m = str(model).lower().strip()
    if m in ("orig", "original"): return dict(DEFAULT_PARAMS_ORIGINAL)
    if m in ("mod", "modified"): return dict(DEFAULT_PARAMS_MODIFIED)
    raise ValueError("Unknown model.")

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
    raise ValueError("Unknown model.")

def compute_m_dot_ref(med, V_h_m3):
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * F_REF

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


@dataclass
class Control:
    n: float

@dataclass
class SimpleInputs:
    control: Control
    T_amb: float
    lsq_max_nfev: int = 20000
    lsq_ftol: float = 1e-8
    lsq_xtol: float = 1e-8


# =========================================================
# Solver telemetry capture
# =========================================================
class SolverCapture:
    """
    Context manager that monkey-patches scipy.optimize.least_squares
    to capture the result object after each call.
    """
    def __init__(self):
        self.calls = []
        self._original_fn = None

    def __enter__(self):
        self._original_fn = scipy.optimize.least_squares

        capture = self

        def wrapped_least_squares(*args, **kwargs):
            result = capture._original_fn(*args, **kwargs)
            capture.calls.append({
                "nfev": int(result.nfev),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "status": int(result.status),
                "message": str(result.message),
                "x": result.x.copy() if hasattr(result.x, "copy") else result.x,
            })
            return result

        scipy.optimize.least_squares = wrapped_least_squares
        return self

    def __exit__(self, *args):
        scipy.optimize.least_squares = self._original_fn

    def pop_last(self):
        """Return and remove the last captured call info, or None."""
        if self.calls:
            return self.calls.pop(-1)
        return None

    def pop_all_since(self, n_before):
        """Return all calls captured since count was n_before."""
        new_calls = self.calls[n_before:]
        return new_calls

    @property
    def count(self):
        return len(self.calls)


# =========================================================
# Simulation
# =========================================================
def simulate_point(comp, med, p_suc_pa, T_suc_K, p_out_pa, n_rel, T_amb_K, V_h_m3):
    """Simulate one point, return metrics dict or None."""
    inputs = SimpleInputs(control=Control(n=max(1e-9, min(1.0, n_rel))), T_amb=float(T_amb_K))
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

            if h_dis_actual <= h_suc or h_dis_isen <= h_suc: return None
            if m_flow <= 0 or P_el <= 0: return None

            w_is = h_dis_isen - h_suc
            w_actual = h_dis_actual - h_suc
            eta_is = w_is / w_actual
            f_hz = comp.get_n_absolute(inputs.control.n)
            lambda_h = m_flow / (rho_suc * V_h_m3 * f_hz)
            zeta_gl = (m_flow * w_is) / P_el

        # Internal states for diagnostics
        T_w = float(getattr(comp, "T_w", np.nan))
        T_oil = float(getattr(comp, "T_oil_sump", np.nan))

        return {
            "eta_is": eta_is, "lambda_h": lambda_h, "zeta_gl": zeta_gl,
            "m_flow_gps": m_flow * 1e3, "P_el": P_el,
            "T_dis_C": k_to_c(T_dis_K),
            "h_suc": h_suc, "h_dis_actual": h_dis_actual, "h_dis_isen": h_dis_isen,
            "w_is": w_is, "w_actual": w_actual,
            "T_wall_C": k_to_c(T_w) if np.isfinite(T_w) else np.nan,
            "T_oil_sump_C": k_to_c(T_oil) if np.isfinite(T_oil) else np.nan,
        }
    except Exception:
        return None


def run_sweep_with_telemetry(
    med, model, refrigerant_name, oil_name, params,
    N_max_hz, V_h_m3, T_evap_C, T_cond_values, N_rpm, SH_K, T_amb_C,
    label="forward",
):
    """
    Run sweep with solver telemetry capture.
    """
    T_amb_K = c_to_k(T_amb_C)
    f_hz = rpm_to_hz(N_rpm)
    n_rel = f_hz / N_max_hz

    T_evap_K = c_to_k(T_evap_C)
    state_sat_suc = med.calc_state("TQ", T_evap_K, Q_REF)
    p_suc = float(state_sat_suc.p)
    T_suc_K = T_evap_K + SH_K

    comp = make_compressor(
        model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
        params=params, refrigerant_name=refrigerant_name, oil_name=oil_name,
    )
    comp.med_prop = med
    if hasattr(comp, "debug_enabled"):
        comp.debug_enabled = False

    records = []

    with SolverCapture() as capture:
        for T_cond_C in T_cond_values:
            T_cond_K = c_to_k(T_cond_C)
            state_sat_dis = med.calc_state("TQ", T_cond_K, 0.0)
            p_out = float(state_sat_dis.p)

            if p_out <= p_suc:
                continue

            n_calls_before = capture.count

            result = simulate_point(comp, med, p_suc, T_suc_K, p_out, n_rel, T_amb_K, V_h_m3)

            # Collect solver calls for this point
            solver_calls = capture.pop_all_since(n_calls_before)

            if result is None:
                continue

            # Aggregate solver telemetry
            if solver_calls:
                total_nfev = sum(c["nfev"] for c in solver_calls)
                max_cost = max(c["cost"] for c in solver_calls)
                max_optimality = max(c["optimality"] for c in solver_calls)
                statuses = [c["status"] for c in solver_calls]
                n_solver_calls = len(solver_calls)
                any_not_converged = any(s <= 0 for s in statuses)
            else:
                total_nfev = 0
                max_cost = np.nan
                max_optimality = np.nan
                n_solver_calls = 0
                any_not_converged = False

            rec = {
                "sweep": label,
                "T_cond_C": T_cond_C,
                "p_out_bar": p_out / 1e5,
                "pressure_ratio": p_out / p_suc,
                # Solver telemetry
                "solver_n_calls": n_solver_calls,
                "solver_total_nfev": total_nfev,
                "solver_max_cost": max_cost,
                "solver_max_optimality": max_optimality,
                "solver_any_not_converged": any_not_converged,
            }
            rec.update(result)
            records.append(rec)

    return pd.DataFrame(records)


# =========================================================
# Plots
# =========================================================
def plot_forward_vs_reverse(df_fwd, df_rev, title_base, out_path):
    """
    2x3 grid: overlay forward and reverse sweep for all six metrics.
    Spikes at the same T_cond → model-inherent.
    Spikes shift → order-dependent.
    """
    metrics = [
        ("eta_is", "$\\eta_{is}$ [-]"),
        ("lambda_h", "$\\lambda_h$ [-]"),
        ("zeta_gl", "$\\zeta_{gl}$ [-]"),
        ("m_flow_gps", "$\\dot{m}$ [g/s]"),
        ("P_el", "$P_{el}$ [W]"),
        ("T_dis_C", "$T_{dis}$ [°C]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))

    for idx, (col, ylabel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        if not df_fwd.empty and col in df_fwd.columns:
            ax.plot(df_fwd["T_cond_C"], df_fwd[col],
                    marker="o", markersize=3, color="#EC635C", linewidth=1.8,
                    label="Vorwärts (25→65°C)")
        if not df_rev.empty and col in df_rev.columns:
            ax.plot(df_rev["T_cond_C"], df_rev[col],
                    marker="s", markersize=3, color="#4B81C4", linewidth=1.8,
                    label="Rückwärts (65→25°C)")

        ax.set_xlabel("Kondensationstemperatur [°C]")
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.6, alpha=0.35)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=True, fontsize=12)

    fig.suptitle(f"Vorwärts vs. Rückwärts Sweep\n{title_base}", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


def plot_fwd_rev_difference(df_fwd, df_rev, title_base, out_path):
    """
    Difference plot (forward - reverse), sorted by T_cond.
    """
    if df_fwd.empty or df_rev.empty:
        print("  [SKIP] Cannot plot difference.")
        return

    merged = df_fwd.merge(df_rev, on="T_cond_C", suffixes=("_fwd", "_rev"))
    if merged.empty:
        print("  [SKIP] No matching T_cond points.")
        return

    metrics = [
        ("eta_is", "$\\Delta\\eta_{is}$"),
        ("lambda_h", "$\\Delta\\lambda_h$"),
        ("zeta_gl", "$\\Delta\\zeta_{gl}$"),
        ("m_flow_gps", "$\\Delta\\dot{m}$ [g/s]"),
        ("P_el", "$\\Delta P_{el}$ [W]"),
        ("T_dis_C", "$\\Delta T_{dis}$ [K]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))

    for idx, (col, ylabel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        if f"{col}_fwd" in merged.columns and f"{col}_rev" in merged.columns:
            diff = merged[f"{col}_fwd"] - merged[f"{col}_rev"]
            ax.axhline(0, color="black", linewidth=0.8)
            ax.plot(merged["T_cond_C"], diff,
                    marker="o", markersize=3, color="#8768B4", linewidth=1.8)

            max_abs = float(np.max(np.abs(diff.dropna())))
            mean_abs = float(np.mean(np.abs(diff.dropna())))
            ax.text(0.02, 0.98,
                    f"max |Δ|: {max_abs:.4g}\nmean |Δ|: {mean_abs:.4g}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.8, edgecolor="0.7"))

        ax.set_xlabel("Kondensationstemperatur [°C]")
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.6, alpha=0.35)

    fig.suptitle(f"Differenz (Vorwärts − Rückwärts)\n{title_base}", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


def plot_solver_telemetry(df_telem, title_base, out_path):
    """
    Plot solver telemetry alongside eta_is and T_dis to reveal correlations
    between solver behavior and spikes.
    """
    if df_telem.empty:
        print("  [SKIP] No telemetry data.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    x = df_telem["T_cond_C"].to_numpy()

    # Row 1: eta_is and T_dis (the noisy quantities)
    axes[0, 0].plot(x, df_telem["eta_is"], marker="o", markersize=3,
                    color="#EC635C", linewidth=1.5)
    axes[0, 0].set_ylabel("$\\eta_{is}$ [-]")
    axes[0, 0].set_title("Isentroper Wirkungsgrad")

    axes[0, 1].plot(x, df_telem["T_dis_C"], marker="o", markersize=3,
                    color="#EC635C", linewidth=1.5)
    axes[0, 1].set_ylabel("$T_{dis}$ [°C]")
    axes[0, 1].set_title("Austrittstemperatur")

    # Row 2: Solver nfev and cost
    if "solver_total_nfev" in df_telem.columns:
        axes[1, 0].bar(x, df_telem["solver_total_nfev"],
                       width=0.4, color="#4B81C4", alpha=0.8)
        axes[1, 0].set_ylabel("Solver nfev (gesamt)")
        axes[1, 0].set_title("Solver-Iterationen pro Punkt")

        # Highlight high-nfev points
        median_nfev = float(df_telem["solver_total_nfev"].median())
        axes[1, 0].axhline(median_nfev, color="gray", linestyle="--",
                           linewidth=1.0, label=f"Median: {median_nfev:.0f}")
        axes[1, 0].legend(fontsize=10)

    if "solver_max_cost" in df_telem.columns:
        cost = df_telem["solver_max_cost"].to_numpy(dtype=float)
        axes[1, 1].bar(x, cost, width=0.4, color="#F49961", alpha=0.8)
        axes[1, 1].set_ylabel("Solver max cost")
        axes[1, 1].set_title("Solver-Residuum (max cost)")
        axes[1, 1].set_yscale("log")

    # Row 3: Solver optimality and internal T_wall
    if "solver_max_optimality" in df_telem.columns:
        opt = df_telem["solver_max_optimality"].to_numpy(dtype=float)
        axes[2, 0].bar(x, opt, width=0.4, color="#8768B4", alpha=0.8)
        axes[2, 0].set_ylabel("Solver max optimality")
        axes[2, 0].set_title("Solver-Optimalität (Gradient)")
        axes[2, 0].set_yscale("log")

    if "T_wall_C" in df_telem.columns:
        T_wall = df_telem["T_wall_C"].to_numpy(dtype=float)
        axes[2, 1].plot(x, T_wall, marker="o", markersize=3,
                        color="#6EBB96", linewidth=1.5)
        axes[2, 1].set_ylabel("$T_{wall}$ [°C]")
        axes[2, 1].set_title("Interne Wandtemperatur")

    for ax in axes.flat:
        ax.set_xlabel("Kondensationstemperatur [°C]")
        ax.grid(True, linewidth=0.6, alpha=0.35)

    fig.suptitle(f"Solver-Telemetrie\n{title_base}", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


def plot_enthalpy_detail(df_telem, title_base, out_path):
    """
    Plot h_suc, h_dis_actual, h_dis_isen, w_is, w_actual to show
    which enthalpy component causes the eta_is spikes.
    """
    if df_telem.empty:
        print("  [SKIP] No data for enthalpy detail.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    x = df_telem["T_cond_C"].to_numpy()

    # h values
    axes[0, 0].plot(x, df_telem["h_suc"] / 1e3, marker=".", markersize=3,
                    linewidth=1.5, label="$h_{suc}$")
    axes[0, 0].set_ylabel("$h_{suc}$ [kJ/kg]")
    axes[0, 0].set_title("Spez. Enthalpie Eintritt")

    axes[0, 1].plot(x, df_telem["h_dis_actual"] / 1e3, marker=".", markersize=3,
                    color="#EC635C", linewidth=1.5, label="$h_{dis,actual}$")
    axes[0, 1].set_ylabel("$h_{dis,actual}$ [kJ/kg]")
    axes[0, 1].set_title("Spez. Enthalpie Austritt (tatsächlich)")

    axes[0, 2].plot(x, df_telem["h_dis_isen"] / 1e3, marker=".", markersize=3,
                    color="#4B81C4", linewidth=1.5, label="$h_{dis,isen}$")
    axes[0, 2].set_ylabel("$h_{dis,isen}$ [kJ/kg]")
    axes[0, 2].set_title("Spez. Enthalpie Austritt (isentrop)")

    # w values (numerator and denominator of eta_is)
    axes[1, 0].plot(x, df_telem["w_is"] / 1e3, marker=".", markersize=3,
                    color="#4B81C4", linewidth=1.5)
    axes[1, 0].set_ylabel("$w_{is}$ [kJ/kg]")
    axes[1, 0].set_title("Isentrope Arbeit (Zähler $\\eta_{is}$)")

    axes[1, 1].plot(x, df_telem["w_actual"] / 1e3, marker=".", markersize=3,
                    color="#EC635C", linewidth=1.5)
    axes[1, 1].set_ylabel("$w_{actual}$ [kJ/kg]")
    axes[1, 1].set_title("Tatsächliche Arbeit (Nenner $\\eta_{is}$)")

    # eta_is for reference
    axes[1, 2].plot(x, df_telem["eta_is"], marker=".", markersize=3,
                    color="#8768B4", linewidth=1.5)
    axes[1, 2].set_ylabel("$\\eta_{is}$ [-]")
    axes[1, 2].set_title("$\\eta_{is}$ = $w_{is}$ / $w_{actual}$")

    for ax in axes.flat:
        ax.set_xlabel("Kondensationstemperatur [°C]")
        ax.grid(True, linewidth=0.6, alpha=0.35)

    fig.suptitle(f"Enthalpie-Detail\n{title_base}", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Extended diagnostic: forward/reverse sweep + solver telemetry."
    )
    ap.add_argument("--params_csv", required=True, type=Path)
    ap.add_argument("--oil", required=True)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--refrigerant", default="auto")
    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)
    ap.add_argument("--T_amb_C", type=float, default=25.0)
    ap.add_argument("--T_evap", type=float, default=10.0)
    ap.add_argument("--N_rpm", type=float, default=3600.0)
    ap.add_argument("--SH_K", type=float, default=10.0)
    ap.add_argument("--T_cond_min", type=float, default=25.0)
    ap.add_argument("--T_cond_max", type=float, default=65.0)
    ap.add_argument("--n_points", type=int, default=80)
    ap.add_argument("--out_dir", default="results/diagnose")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params_peek = pd.read_csv(args.params_csv).iloc[0].to_dict()
    if args.model == "auto":
        args.model = str(params_peek.get("model", "modified"))
    if args.refrigerant == "auto":
        args.refrigerant = str(params_peek.get("refrigerant", "PROPANE"))

    params, params_meta = load_params_csv(args.params_csv, args.model)

    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6
    med = RefProp(fluid_name=args.refrigerant)
    params["f_ref"] = F_REF
    params["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

    m = str(args.model).lower().strip()
    oil_name = args.oil if m in ("mod", "modified") else None

    T_cond_fwd = np.linspace(args.T_cond_min, args.T_cond_max, args.n_points)
    T_cond_rev = T_cond_fwd[::-1].copy()

    title_base = (
        f"{args.model.capitalize()} | {args.oil} | {args.refrigerant}  |  "
        f"$T_{{evap}}$={args.T_evap}°C, N={args.N_rpm:.0f} rpm, SH={args.SH_K}K"
    )

    print(f"  Model:       {args.model}")
    print(f"  Oil:         {args.oil}")
    print(f"  T_evap:      {args.T_evap}°C")
    print(f"  N:           {args.N_rpm} rpm")
    print(f"  SH:          {args.SH_K} K")
    print(f"  T_cond:      {args.T_cond_min}→{args.T_cond_max}°C ({args.n_points} pts)")
    print()

    # --- Sweep 1: Forward with telemetry ---
    print("  [1/2] Forward sweep (25→65°C) with solver telemetry ...")
    df_fwd = run_sweep_with_telemetry(
        med=med, model=args.model, refrigerant_name=args.refrigerant,
        oil_name=oil_name, params=params, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
        T_evap_C=args.T_evap, T_cond_values=T_cond_fwd,
        N_rpm=args.N_rpm, SH_K=args.SH_K, T_amb_C=args.T_amb_C,
        label="forward",
    )
    print(f"    → {len(df_fwd)}/{len(T_cond_fwd)} points")

    # --- Sweep 2: Reverse with telemetry ---
    print("  [2/2] Reverse sweep (65→25°C) with solver telemetry ...")
    df_rev = run_sweep_with_telemetry(
        med=med, model=args.model, refrigerant_name=args.refrigerant,
        oil_name=oil_name, params=params, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
        T_evap_C=args.T_evap, T_cond_values=T_cond_rev,
        N_rpm=args.N_rpm, SH_K=args.SH_K, T_amb_C=args.T_amb_C,
        label="reverse",
    )
    # Sort reverse by T_cond ascending for plotting
    df_rev = df_rev.sort_values("T_cond_C").reset_index(drop=True)
    print(f"    → {len(df_rev)}/{len(T_cond_rev)} points")
    print()

    stamp = _ts()
    suffix = f"{args.oil.lower()}_{args.model}_Tevap{int(args.T_evap)}"

    # Plot 1: Forward vs Reverse comparison
    plot_forward_vs_reverse(
        df_fwd, df_rev, title_base,
        out_dir / f"diagnose_fwd_vs_rev_{suffix}_{stamp}.{args.out_format}",
    )

    # Plot 2: Forward-Reverse difference
    plot_fwd_rev_difference(
        df_fwd, df_rev, title_base,
        out_dir / f"diagnose_fwd_rev_diff_{suffix}_{stamp}.{args.out_format}",
    )

    # Plot 3: Solver telemetry (forward sweep)
    plot_solver_telemetry(
        df_fwd, title_base,
        out_dir / f"diagnose_solver_telemetry_{suffix}_{stamp}.{args.out_format}",
    )

    # Plot 4: Enthalpy detail (forward sweep)
    plot_enthalpy_detail(
        df_fwd, title_base,
        out_dir / f"diagnose_enthalpy_detail_{suffix}_{stamp}.{args.out_format}",
    )

    # Save all data
    all_df = pd.concat([df_fwd, df_rev], ignore_index=True)
    data_csv = out_dir / f"diagnose_data_{suffix}_{stamp}.csv"
    all_df.to_csv(data_csv, index=False)
    print(f"  [OK] Data saved: {data_csv}")

    # Summary / Interpretation
    print()
    print("  =========================================================")
    print("  INTERPRETATION:")
    print("  =========================================================")
    print()
    print("  PLOT 1 (Forward vs Reverse):")
    print("    Spikes an gleichen T_cond → modell-inhärent")
    print("    Spikes verschieben sich   → reihenfolge-abhängig")
    print()
    print("  PLOT 2 (Differenz Fwd-Rev):")
    print("    Differenz ≈ 0 überall     → identisch, kein Reihenfolge-Effekt")
    print("    Differenz ≠ 0 bei Spikes  → Solver findet verschiedene Lösungen")
    print()
    print("  PLOT 3 (Solver-Telemetrie):")
    print("    Hohe nfev bei Spike-Positionen → Solver kämpft dort")
    print("    Hohe cost/optimality          → schlechte Konvergenz")
    print("    Gleichmäßige nfev             → Solver unschuldig")
    print()
    print("  PLOT 4 (Enthalpie-Detail):")
    print("    Spikes in w_actual (Nenner)   → T_dis/h_dis_actual schwankt")
    print("    w_is (Zähler) glatt           → RefProp PS-Lookup ist stabil")
    print("    Beides rauscht                → Tieferes Problem")
    print("  =========================================================")
    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
