# scripts/plotting_scripts/state_diagram.py
#
# Plots log(p)-h and T-h diagrams for a single compressor operating point.
# The saturation dome is drawn as background; the internal thermodynamic
# states of the Molinaroli compressor model are marked as labeled points
# connected by a path showing the refrigerant's journey through the machine.
#
# States plotted:
#   IN   = state_inlet      (suction, after superheat)
#   C1   = state_c_1        (after suction heat transfer from wall)
#   C3   = state_c_3        (suction chamber, after mixing with leakage)
#   C4   = state_c_4        (after isentropic compression in cylinder)
#   OUT  = state_outlet      (discharge, after discharge heat transfer)
#   IS   = isentropic discharge reference (calc_state("PS", p_out, s_in))
#
# Activate REFPROP first:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Single operating point, original model:
#   python scripts/plotting_scripts/state_diagram.py --params_csv results/final_results/Molinaroli_LPG68/Fitting/fitted_params_lpg68_original_ga_2026-03-08_101308.csv --oil LPG68 --T_evap 0 --T_cond 50 --N_rpm 3600 --SH_K 10
#
#   # Single operating point, modified model:
#   python scripts/plotting_scripts/state_diagram.py --params_csv results/final_results/Modified_LPG68/Fitting/fitted_params_lpg68_modified_ga_2026-03-22_185546.csv --oil LPG68 --T_evap 0 --T_cond 50 --N_rpm 3600 --SH_K 10
#
#   # Single operating point, oil_path model:
#   python scripts/plotting_scripts/state_diagram.py --params_csv results/final_results/Oil_Path_LPG68/Fitting/fitted_params_lpg68_oil_path_ga_2026-04-17_113953.csv --oil LPG68 --T_evap 0 --T_cond 50 --N_rpm 3600 --SH_K 10
#
#   # Compare two models on the same diagram:
#   python scripts/plotting_scripts/state_diagram.py --params_csv results/final_results/Modified_LPG68/Fitting/fitted_params_lpg68_modified_ga_2026-03-22_185546.csv --params_csv2 results/final_results/Oil_Path_LPG68/Fitting/fitted_params_lpg68_oil_path_ga_2026-04-17_113953.csv --oil LPG68 --T_evap 0 --T_cond 50 --N_rpm 3600 --SH_K 10

from __future__ import annotations

import argparse
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

# Optional: oil_path model
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

PARAM_NAMES = {
    "original": [
        "Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
        "V_IC", "alpha_loss", "W_dot_loss_ref",
    ],
    "modified": [
        "Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
        "V_IC", "alpha_loss", "W_dot_loss_ref", "alpha_fric_tot",
    ],
    "oil_path": [
        "Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
        "V_IC", "alpha_loss", "W_dot_loss_ref", "alpha_fric_tot",
        "m_dot_oil_ref", "Ua_suc_oil_ref",
    ],
}

DEFAULT_PARAMS = {
    "original": {
        "Ua_suc_ref": 16.05, "Ua_dis_ref": 13.96, "Ua_amb": 0.36,
        "A_tot": 9.47e-9, "A_dis": 86.1e-6, "V_IC": 30.7e-6,
        "alpha_loss": 0.16, "W_dot_loss_ref": 83.0,
        "m_dot_ref": None, "f_ref": F_REF,
    },
    "modified": {
        "Ua_suc_ref": 16.05, "Ua_dis_ref": 13.96, "Ua_amb": 0.36,
        "A_tot": 9.47e-9, "A_dis": 86.1e-6, "V_IC": 30.7e-6,
        "alpha_loss": 0.16, "W_dot_loss_ref": 10.0, "alpha_fric_tot": 120.0,
        "m_dot_ref": None, "f_ref": F_REF,
    },
    "oil_path": {
        "Ua_suc_ref": 16.05, "Ua_dis_ref": 13.96, "Ua_amb": 0.36,
        "A_tot": 9.47e-9, "A_dis": 86.1e-6, "V_IC": 30.7e-6,
        "alpha_loss": 0.16, "W_dot_loss_ref": 10.0, "alpha_fric_tot": 120.0,
        "m_dot_oil_ref": 0.005, "Ua_suc_oil_ref": 5.0,
        "m_dot_ref": None, "f_ref": F_REF,
    },
}

# EBC color palette
COLORS = {
    "dome": "#AAAAAA",
    "model1": "#EC635C",
    "model2": "#4B81C4",
    "isentropic": "#6EBB96",
    "isobar_evap": "#8768B4",
    "isobar_cond": "#F49961",
}

# State labels and descriptions
STATE_INFO = {
    "IN":  {"label": "IN",  "desc": "Saugeintritt"},
    "C1":  {"label": "C1",  "desc": "Nach Saugwärme"},
    "C3":  {"label": "C3",  "desc": "Saugkammer\n(nach Leckage-Mischung)"},
    "C4":  {"label": "C4",  "desc": "Nach isentroper\nKompression"},
    "OUT": {"label": "OUT", "desc": "Druckaustritt"},
    "IS":  {"label": "IS",  "desc": "Isentrop\n(Referenz)"},
}


# =========================================================
# Helpers
# =========================================================
def _ts():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def c_to_k(t):
    return float(t) + 273.15

def k_to_c(t):
    return float(t) - 273.15

def rpm_to_hz(n):
    return float(n) / 60.0

def map_refrigerant(name):
    s = str(name).strip().upper()
    return "propane" if s in {"PROPANE", "R290", "PROPAN"} else str(name).strip()

def map_oil(name):
    s = str(name).strip().lower().replace(" ", "")
    if s == "lpg68": return "LPG 68"
    if s == "lpg100": return "LPG 100"
    raise ValueError(f"Unsupported oil: {name}")


def make_compressor(model, N_max_hz, V_h_m3, params, refrigerant_name, oil_name=None):
    m = model.lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(
            N_max=N_max_hz, V_h=V_h_m3,
            fluid_name=map_refrigerant(refrigerant_name),
            lub_name=map_oil(oil_name),
            parameters=params,
        )
    if m in ("oil_path", "oilpath"):
        if not OIL_PATH_AVAILABLE:
            raise ImportError("oil_path model not available.")
        return Molinaroli_2017_Compressor_OilPath(
            N_max=N_max_hz, V_h=V_h_m3,
            fluid_name=map_refrigerant(refrigerant_name),
            lub_name=map_oil(oil_name),
            parameters=params,
        )
    raise ValueError(f"Unknown model: {model}")


def load_params_csv(path, model):
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    m = model.lower().strip()
    params = dict(DEFAULT_PARAMS.get(m, DEFAULT_PARAMS["modified"]))
    for k in PARAM_NAMES.get(m, PARAM_NAMES["modified"]):
        if k in row and pd.notna(row[k]):
            params[k] = float(row[k])
    if "f_ref" in row and pd.notna(row["f_ref"]):
        params["f_ref"] = float(row["f_ref"])
    meta = {k: row[k] for k in ("oil", "refrigerant", "model") if k in row}
    return params, meta


def compute_m_dot_ref(med, V_h_m3):
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * F_REF


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
# Saturation dome
# =========================================================
def compute_saturation_dome(med, T_min_C=-40.0, T_max_C=95.0, n_points=200):
    """
    Compute bubble and dew lines for the saturation dome.
    Returns dict with arrays: h_bubble, h_dew, p_bubble, p_dew, T_array.
    """
    T_arr = np.linspace(c_to_k(T_min_C), c_to_k(T_max_C), n_points)

    h_bubble, h_dew = [], []
    p_bubble, p_dew = [], []
    T_out = []

    for T_K in T_arr:
        try:
            st_bub = med.calc_state("TQ", float(T_K), 0.0)  # bubble
            st_dew = med.calc_state("TQ", float(T_K), 1.0)  # dew
            h_bubble.append(float(st_bub.h) / 1e3)  # kJ/kg
            h_dew.append(float(st_dew.h) / 1e3)
            p_bubble.append(float(st_bub.p) / 1e5)  # bar
            p_dew.append(float(st_dew.p) / 1e5)
            T_out.append(k_to_c(T_K))
        except Exception:
            pass

    return {
        "h_bubble": np.array(h_bubble),
        "h_dew": np.array(h_dew),
        "p_bubble": np.array(p_bubble),
        "p_dew": np.array(p_dew),
        "T": np.array(T_out),
    }


# =========================================================
# Isobars for background context
# =========================================================
def compute_isobar(med, p_bar, h_min_kJkg, h_max_kJkg, n_points=100):
    """Compute T(h) along an isobar. Returns arrays of h [kJ/kg] and T [°C]."""
    h_arr = np.linspace(h_min_kJkg * 1e3, h_max_kJkg * 1e3, n_points)  # J/kg
    T_arr = []
    h_out = []
    for h in h_arr:
        try:
            st = med.calc_state("PH", float(p_bar * 1e5), float(h))
            T_arr.append(k_to_c(float(st.T)))
            h_out.append(h / 1e3)
        except Exception:
            pass
    return np.array(h_out), np.array(T_arr)


# =========================================================
# Simulation & state extraction
# =========================================================
def run_single_point(med, model, refrigerant_name, oil_name, params,
                     N_max_hz, V_h_m3, T_evap_C, T_cond_C, N_rpm, SH_K,
                     T_amb_C=25.0):
    """
    Run one operating point and extract all internal states.
    Returns a dict of {state_name: {"p_bar", "T_C", "h_kJkg", "s_kJkgK"}} plus metadata.
    """
    f_hz = rpm_to_hz(N_rpm)
    n_rel = f_hz / N_max_hz
    T_amb_K = c_to_k(T_amb_C)

    # Suction state
    T_evap_K = c_to_k(T_evap_C)
    state_sat_suc = med.calc_state("TQ", T_evap_K, Q_REF)
    p_suc = float(state_sat_suc.p)
    T_suc_K = T_evap_K + SH_K

    # Discharge pressure
    T_cond_K = c_to_k(T_cond_C)
    state_sat_dis = med.calc_state("TQ", T_cond_K, 0.0)
    p_out = float(state_sat_dis.p)

    # Build compressor
    needs_oil = model.lower().strip() in ("mod", "modified", "oil_path", "oilpath")
    comp = make_compressor(
        model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
        params=params, refrigerant_name=refrigerant_name,
        oil_name=oil_name if needs_oil else None,
    )
    comp.med_prop = med
    if hasattr(comp, "debug_enabled"):
        comp.debug_enabled = False

    # Simulate
    inputs = SimpleInputs(control=Control(n=max(1e-9, min(1.0, n_rel))),
                          T_amb=T_amb_K)
    fs_state = FlowsheetState()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comp.state_inlet = med.calc_state("PT", p_suc, T_suc_K)
        comp.calc_state_outlet(p_outlet=p_out, inputs=inputs, fs_state=fs_state)

    # Extract states
    def _extract(st):
        if st is None:
            return None
        return {
            "p_bar": float(st.p) / 1e5,
            "T_C": k_to_c(float(st.T)),
            "h_kJkg": float(st.h) / 1e3,
            "s_kJkgK": float(st.s) / 1e3,
        }

    states = {}
    states["IN"] = _extract(comp.state_inlet)
    states["C1"] = _extract(getattr(comp, "state_c_1", None))
    states["C3"] = _extract(getattr(comp, "state_c_3", None))
    states["C4"] = _extract(getattr(comp, "state_c_4", None))
    states["OUT"] = _extract(getattr(comp, "state_outlet", None))

    # Isentropic reference
    try:
        s_in = float(comp.state_inlet.s)
        st_is = med.calc_state("PS", p_out, s_in)
        states["IS"] = _extract(st_is)
    except Exception:
        states["IS"] = None

    # Metadata
    meta = {
        "model": model,
        "T_evap_C": T_evap_C,
        "T_cond_C": T_cond_C,
        "N_rpm": N_rpm,
        "SH_K": SH_K,
        "p_suc_bar": p_suc / 1e5,
        "p_out_bar": p_out / 1e5,
        "m_flow_gps": float(comp.m_flow) * 1e3 if hasattr(comp, "m_flow") else None,
        "P_el_W": float(comp.P_el) if hasattr(comp, "P_el") else None,
        "T_w_C": k_to_c(float(comp.T_w)) if hasattr(comp, "T_w") and comp.T_w is not None else None,
    }

    return states, meta


# =========================================================
# Plotting
# =========================================================
def _state_style(state_name):
    """Return marker, size, and annotation offset for each state."""
    styles = {
        "IN":  {"marker": "o", "ms": 12, "offset": (-4, -16), "ha": "right"},
        "C1":  {"marker": "o", "ms": 11, "offset": (16, -4),   "ha": "left"},
        "C3":  {"marker": "o", "ms": 12, "offset": (-20, 16),  "ha": "right"},
        "C4":  {"marker": "o", "ms": 12, "offset": (28, 20),   "ha": "right"},
        "OUT": {"marker": "o", "ms": 13, "offset": (-26, 20),   "ha": "left"},
        "IS":  {"marker": "D", "ms": 10, "offset": (-24, 16),   "ha": "left"},
    }
    return styles.get(state_name, {"marker": "o", "ms": 10, "offset": (10, 6), "ha": "left"})


def plot_state_diagram(
    dome, states_list, meta_list, labels_list,
    colors_list, out_path, med=None,
    show_isobars=True,
):
    """
    Plot log(p)-h and T-h diagrams side by side.

    states_list:  list of state dicts (one per model run)
    meta_list:    list of meta dicts
    labels_list:  list of legend labels (e.g. ["Modified", "Original"])
    colors_list:  list of color strings
    """
    fig, (ax_ph, ax_th) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Saturation dome ---
    # log(p)-h
    ax_ph.plot(dome["h_bubble"], dome["p_bubble"],
               color=COLORS["dome"], linewidth=1.5, zorder=1)
    ax_ph.plot(dome["h_dew"], dome["p_dew"],
               color=COLORS["dome"], linewidth=1.5, zorder=1)
    ax_ph.fill_betweenx(
        dome["p_bubble"],
        dome["h_bubble"], dome["h_dew"],
        alpha=0.08, color=COLORS["dome"], zorder=0,
    )

    # T-h
    ax_th.plot(dome["h_bubble"], dome["T"],
               color=COLORS["dome"], linewidth=1.5, zorder=1)
    ax_th.plot(dome["h_dew"], dome["T"],
               color=COLORS["dome"], linewidth=1.5, zorder=1)
    ax_th.fill_betweenx(
        dome["T"],
        dome["h_bubble"], dome["h_dew"],
        alpha=0.08, color=COLORS["dome"], zorder=0,
    )

    # --- Isobars (evaporation and condensation pressure) ---
    if show_isobars and med is not None and len(meta_list) > 0:
        m0 = meta_list[0]
        # Determine h range from all states
        all_h = []
        for states in states_list:
            for st in states.values():
                if st is not None:
                    all_h.append(st["h_kJkg"])
        if all_h:
            h_min = min(all_h) - 30
            h_max = max(all_h) + 30
        else:
            h_min, h_max = 350, 750

        for p_bar, color, lbl in [
            (m0["p_suc_bar"], COLORS["isobar_evap"], f"p_evap={m0['p_suc_bar']:.1f} bar"),
            (m0["p_out_bar"], COLORS["isobar_cond"], f"p_cond={m0['p_out_bar']:.1f} bar"),
        ]:
            h_iso, T_iso = compute_isobar(med, p_bar, h_min, h_max, n_points=80)
            if len(h_iso) > 0:
                ax_th.plot(h_iso, T_iso, color=color, linewidth=1.0,
                           linestyle="--", alpha=0.5, zorder=1, label=lbl)
                # log(p)-h: horizontal lines
                ax_ph.axhline(p_bar, color=color, linewidth=1.0,
                              linestyle="--", alpha=0.5, zorder=1)

    # --- State points and paths per model ---
    # The thermodynamic path through the compressor
    path_order = ["IN", "C1", "C3", "C4", "OUT"]

    for run_idx, (states, meta, label, color) in enumerate(
        zip(states_list, meta_list, labels_list, colors_list)
    ):
        # Collect path coordinates
        path_h, path_p, path_T = [], [], []
        for sname in path_order:
            st = states.get(sname)
            if st is not None:
                path_h.append(st["h_kJkg"])
                path_p.append(st["p_bar"])
                path_T.append(st["T_C"])

        # Draw path lines
        if len(path_h) >= 2:
            ax_ph.plot(path_h, path_p, color=color, linewidth=2.0,
                       linestyle="-", alpha=0.6, zorder=2)
            ax_th.plot(path_h, path_T, color=color, linewidth=2.0,
                       linestyle="-", alpha=0.6, zorder=2)

        # Draw isentropic reference line (IN → IS)
        st_in = states.get("IN")
        st_is = states.get("IS")
        if st_in is not None and st_is is not None:
            ax_ph.plot(
                [st_in["h_kJkg"], st_is["h_kJkg"]],
                [st_in["p_bar"], st_is["p_bar"]],
                color=COLORS["isentropic"], linewidth=1.5,
                linestyle=":", alpha=0.7, zorder=2,
            )
            ax_th.plot(
                [st_in["h_kJkg"], st_is["h_kJkg"]],
                [st_in["T_C"], st_is["T_C"]],
                color=COLORS["isentropic"], linewidth=1.5,
                linestyle=":", alpha=0.7, zorder=2,
            )

        # Draw state points
        arrow_props = dict(
            arrowstyle="-",
            color="0.4",
            linewidth=0.8,
            shrinkA=0, shrinkB=5,
        )
        for sname in list(path_order) + ["IS"]:
            st = states.get(sname)
            if st is None:
                continue

            style = _state_style(sname)
            marker = style["marker"]
            ms = style["ms"]
            edge_color = COLORS["isentropic"] if sname == "IS" else color

            # log(p)-h
            ax_ph.plot(st["h_kJkg"], st["p_bar"],
                       marker=marker, markersize=ms,
                       color=edge_color, markeredgecolor="white",
                       markeredgewidth=1.5, zorder=5)

            # T-h
            ax_th.plot(st["h_kJkg"], st["T_C"],
                       marker=marker, markersize=ms,
                       color=edge_color, markeredgecolor="white",
                       markeredgewidth=1.5, zorder=5)

            # Labels with arrow (only for the first run to avoid clutter)
            if run_idx == 0:
                dx, dy = style["offset"]
                ha = style["ha"]
                lbl_text = STATE_INFO[sname]["label"]

                bbox_style = dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor=edge_color,
                    alpha=0.9,
                    linewidth=1.2,
                )

                ax_ph.annotate(
                    lbl_text,
                    (st["h_kJkg"], st["p_bar"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=10, fontweight="bold", color=edge_color,
                    ha=ha, va="center",
                    bbox=bbox_style,
                    arrowprops=arrow_props,
                    zorder=6,
                )
                ax_th.annotate(
                    lbl_text,
                    (st["h_kJkg"], st["T_C"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=10, fontweight="bold", color=edge_color,
                    ha=ha, va="center",
                    bbox=bbox_style,
                    arrowprops=arrow_props,
                    zorder=6,
                )

    # --- Auto-zoom to relevant region ---
    all_h, all_p, all_T = [], [], []
    for states in states_list:
        for st in states.values():
            if st is not None:
                all_h.append(st["h_kJkg"])
                all_p.append(st["p_bar"])
                all_T.append(st["T_C"])

    if all_h:
        h_margin = max(20.0, (max(all_h) - min(all_h)) * 0.25)
        h_lo = min(all_h) - h_margin
        h_hi = max(all_h) + h_margin

        p_lo = min(all_p) * 0.55
        p_hi = max(all_p) * 1.8

        T_margin = max(8.0, (max(all_T) - min(all_T)) * 0.25)
        T_lo = min(all_T) - T_margin
        T_hi = max(all_T) + T_margin

        ax_ph.set_xlim(h_lo, h_hi)
        ax_ph.set_ylim(p_lo, p_hi)
        ax_th.set_xlim(h_lo, h_hi)
        ax_th.set_ylim(T_lo, T_hi)

    # --- Axes setup ---
    ax_ph.set_yscale("log")
    ax_ph.set_xlabel("Spezifische Enthalpie $h$ in kJ/kg")
    ax_ph.set_ylabel("Druck log($p$) in bar]")
    ax_ph.set_title("log($p$)-$h$-Diagramm")
    ax_ph.grid(True, which="both", linewidth=0.5, alpha=0.3)

    ax_th.set_xlabel("Spezifische Enthalpie $h$ in kJ/kg")
    ax_th.set_ylabel("Temperatur $T$ in °C")
    ax_th.set_title("$T$-$h$-Diagramm")
    ax_th.grid(True, linewidth=0.5, alpha=0.3)

    # --- Legend ---
    from matplotlib.lines import Line2D
    handles = []

    # Model legend entries
    for label, color in zip(labels_list, colors_list):
        handles.append(Line2D([0], [0], color=color, linewidth=2.5,
                              marker="o", markersize=7, label=label))

    # Separator + state marker legend
    state_order = ["IN", "C1", "C3", "C4", "OUT", "IS"]
    state_colors_map = {"IS": COLORS["isentropic"]}
    default_sc = colors_list[0] if colors_list else "#333333"

    for sname in state_order:
        st_style = _state_style(sname)
        sc = state_colors_map.get(sname, default_sc)
        handles.append(Line2D(
            [0], [0], linestyle="None",
            marker=st_style["marker"], markersize=8,
            color=sc, markeredgecolor="white", markeredgewidth=1.0,
            label=f"{STATE_INFO[sname]['label']}",
        ))

    handles.append(Line2D([0], [0], color=COLORS["dome"], linewidth=1.5,
                          label="Sättigungsglocke"))

    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=4,
               frameon=True, fontsize=9, columnspacing=1.2)

    # --- Supertitle ---
    m0 = meta_list[0]
    fig.suptitle(
        f"$T_{{evap}}$={m0['T_evap_C']:.0f} °C,  "
        f"$T_{{cond}}$={m0['T_cond_C']:.0f} °C,  "
        f"$N$={m0['N_rpm']:.0f} rpm,  "
        f"$\\Delta T_{{SH}}$={m0['SH_K']:.0f} K",
        fontsize=13, y=1.01,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                format=out_path.suffix.lstrip("."))
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


def print_state_table(states, meta, label):
    """Print a summary table of all states."""
    print(f"\n  States for: {label} ({meta['model']})")
    print(f"  {'State':>5}  {'p [bar]':>10}  {'T [°C]':>10}  {'h [kJ/kg]':>12}  {'s [kJ/kgK]':>12}")
    print("  " + "-" * 62)
    for sname in ["IN", "C1", "C3", "C4", "OUT", "IS"]:
        st = states.get(sname)
        if st is None:
            print(f"  {sname:>5}  {'—':>10}  {'—':>10}  {'—':>12}  {'—':>12}")
        else:
            print(f"  {sname:>5}  {st['p_bar']:10.3f}  {st['T_C']:10.2f}  "
                  f"{st['h_kJkg']:12.3f}  {st['s_kJkgK']:12.6f}")
    if meta.get("m_flow_gps") is not None:
        print(f"\n  m_dot = {meta['m_flow_gps']:.2f} g/s,  "
              f"P_el = {meta['P_el_W']:.1f} W")
    if meta.get("T_w_C") is not None:
        print(f"  T_wall = {meta['T_w_C']:.1f} °C")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Plot log(p)-h and T-h state diagrams for a compressor operating point."
    )

    # Model 1 (required)
    ap.add_argument("--params_csv", required=True, type=Path,
                    help="Fitted params CSV for model 1")
    ap.add_argument("--oil", required=True,
                    help="Oil name (LPG68 | LPG100)")
    ap.add_argument("--model", default="auto",
                    help="original | modified | oil_path | auto")
    ap.add_argument("--refrigerant", default="auto")
    ap.add_argument("--label", default=None,
                    help="Legend label for model 1 (auto-detected if omitted)")

    # Model 2 (optional, for comparison)
    ap.add_argument("--params_csv2", type=Path, default=None,
                    help="Fitted params CSV for model 2 (optional comparison)")
    ap.add_argument("--model2", default="auto")
    ap.add_argument("--label2", default=None)

    # Operating point
    ap.add_argument("--T_evap", type=float, required=True, help="Evaporation temperature [°C]")
    ap.add_argument("--T_cond", type=float, required=True, help="Condensation temperature [°C]")
    ap.add_argument("--N_rpm", type=float, default=3600.0)
    ap.add_argument("--SH_K", type=float, default=10.0)
    ap.add_argument("--T_amb_C", type=float, default=25.0)

    # Geometry
    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)

    # Output
    ap.add_argument("--out_dir", default="results/state_diagram", type=Path)
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    # Options
    ap.add_argument("--no_isobars", action="store_true",
                    help="Hide isobar lines in T-h diagram")

    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup ---
    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6

    # Detect model 1
    peek1 = pd.read_csv(args.params_csv).iloc[0].to_dict()
    if args.model == "auto":
        args.model = str(peek1.get("model", "modified")).lower().strip()
    if args.refrigerant == "auto":
        args.refrigerant = str(peek1.get("refrigerant", "PROPANE"))

    med = RefProp(fluid_name=args.refrigerant)

    params1, meta1 = load_params_csv(args.params_csv, args.model)
    params1["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)
    params1["f_ref"] = F_REF

    label1 = args.label or f"{args.model.capitalize()} ({meta1.get('oil', args.oil)})"

    print(f"  Model 1: {args.model} | Oil: {args.oil}")
    print(f"  Operating point: T_evap={args.T_evap}°C, T_cond={args.T_cond}°C, "
          f"N={args.N_rpm} rpm, SH={args.SH_K} K")

    # --- Saturation dome ---
    print("  Computing saturation dome ...")
    dome = compute_saturation_dome(med)

    # --- Model 1 ---
    print("  Simulating model 1 ...")
    states1, run_meta1 = run_single_point(
        med=med, model=args.model, refrigerant_name=args.refrigerant,
        oil_name=args.oil, params=params1,
        N_max_hz=N_max_hz, V_h_m3=V_h_m3,
        T_evap_C=args.T_evap, T_cond_C=args.T_cond,
        N_rpm=args.N_rpm, SH_K=args.SH_K, T_amb_C=args.T_amb_C,
    )
    print_state_table(states1, run_meta1, label1)

    states_list = [states1]
    meta_list = [run_meta1]
    labels_list = [label1]
    colors_list = [COLORS["model1"]]

    # --- Model 2 (optional) ---
    if args.params_csv2 is not None:
        peek2 = pd.read_csv(args.params_csv2).iloc[0].to_dict()
        if args.model2 == "auto":
            args.model2 = str(peek2.get("model", "original")).lower().strip()

        params2, meta2 = load_params_csv(args.params_csv2, args.model2)
        params2["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)
        params2["f_ref"] = F_REF

        label2 = args.label2 or f"{args.model2.capitalize()} ({meta2.get('oil', args.oil)})"

        print(f"\n  Model 2: {args.model2}")
        print("  Simulating model 2 ...")
        states2, run_meta2 = run_single_point(
            med=med, model=args.model2, refrigerant_name=args.refrigerant,
            oil_name=args.oil, params=params2,
            N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            T_evap_C=args.T_evap, T_cond_C=args.T_cond,
            N_rpm=args.N_rpm, SH_K=args.SH_K, T_amb_C=args.T_amb_C,
        )
        print_state_table(states2, run_meta2, label2)

        states_list.append(states2)
        meta_list.append(run_meta2)
        labels_list.append(label2)
        colors_list.append(COLORS["model2"])

    # --- Plot ---
    stamp = _ts()
    model_tag = args.model
    if args.params_csv2 is not None:
        model_tag += f"_vs_{args.model2}"
    out_name = (f"state_diagram_{model_tag}_{args.oil}_"
                f"Tevap{int(args.T_evap)}_Tcond{int(args.T_cond)}_"
                f"N{int(args.N_rpm)}_{stamp}.{args.out_format}")
    out_path = out_dir / out_name

    print("\n  Plotting ...")
    plot_state_diagram(
        dome=dome,
        states_list=states_list,
        meta_list=meta_list,
        labels_list=labels_list,
        colors_list=colors_list,
        out_path=out_path,
        med=med,
        show_isobars=not args.no_isobars,
    )

    print(f"\nDone. Output: {out_path}")


if __name__ == "__main__":
    main()
