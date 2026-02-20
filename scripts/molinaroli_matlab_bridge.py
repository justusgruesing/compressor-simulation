# scripts/molinaroli_matlab_bridge.py
#
# Bridge for MATLAB Optimization Toolbox:
# - MATLAB calls init(...) once (loads dataset + caches)
# - MATLAB calls cost(x) many times (scalar objective like Molinaroli eq. 31–33)
# - MATLAB calls save_results(...) once at the end
#
# Uses RefProp backend

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from vclibpy.media import RefProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import (
    Molinaroli_2017_Compressor,
    Molinaroli_2017_Compressor_Modified,
)

# -------------------------
# CSV columns (your dataset)
# -------------------------
OIL_COL = "Ölbezeichnung"
P_SUC_COL = "P1_mean"        # bar
T_SUC_COL = "T1_mean"        # °C
P_OUT_COL = "P2_mean"        # bar
T_AMB_COL = "Tamb_mean"      # °C

M_FLOW_MEAS_COL = "suction_mf_mean"  # g/s
P_EL_MEAS_COL = "Pel_mean"           # W
SPEED_COL = "N"                      # rpm

# -------------------------
# Molinaroli references
# -------------------------
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0

# Higher penalty for failed simulations (dimensionless relative error)
FAIL_E = 1e3

PARAM_NAMES = [
    "Ua_suc_ref",
    "Ua_dis_ref",
    "Ua_amb",
    "A_tot",
    "A_dis",
    "V_IC",
    "alpha_loss",
    "W_dot_loss_ref",
]

DEFAULT_PARAMS = {
    "Ua_suc_ref": 16.05,
    "Ua_dis_ref": 13.96,
    "Ua_amb": 0.36,
    "A_tot": 9.47e-9,
    "A_dis": 86.1e-6,
    "V_IC": 16.11e-6,
    "alpha_loss": 0.16,
    "W_dot_loss_ref": 83.0,
    "m_dot_ref": None,
    "f_ref": F_REF,
}

# -------------------------
# Helpers
# -------------------------
def bar_to_pa(p_bar: float) -> float:
    return float(p_bar) * 100000.0

def c_to_k(t_c: float) -> float:
    return float(t_c) + 273.15

def rpm_to_hz(rpm: float) -> float:
    return float(rpm) / 60.0

def gs_to_kgps(g_s: float) -> float:
    return float(g_s) / 1000.0

def make_compressor(model: str, N_max_hz: float, V_h_m3: float, params: dict):
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    raise ValueError("Unknown model. Use original | modified")

def compute_m_dot_ref(med, V_h_m3: float) -> float:
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * float(F_REF)

def read_dataset_csv(path: Path) -> pd.DataFrame:
    # Your dataset has: first line = units, second line = header
    return pd.read_csv(path, sep=";", header=1, decimal=",")

def _x_to_params(x: np.ndarray) -> dict:
    p = dict(DEFAULT_PARAMS)
    for name, val in zip(PARAM_NAMES, x):
        p[name] = float(val)
    return p

@dataclass
class Control:
    n: float  # relative speed 0..1

@dataclass
class SimpleInputs:
    control: Control
    T_amb: float  # K

def _clamp01(x: float) -> float:
    return max(1e-9, min(1.0, float(x)))

def _simulate_with_comp(
    comp,
    med,
    inputs: SimpleInputs,
    p_suc_pa: float,
    T_suc_K: float,
    p_out_pa: float,
    n_rel: float,
    T_amb_K: float,
):
    # Update inputs/state for this operating point
    inputs.control.n = _clamp01(n_rel)
    inputs.T_amb = float(T_amb_K)

    comp.state_inlet = med.calc_state("PT", float(p_suc_pa), float(T_suc_K))

    fs_state = FlowsheetState()
    comp.calc_state_outlet(p_outlet=float(p_out_pa), inputs=inputs, fs_state=fs_state)

    m_flow = float(comp.m_flow)
    P_el = float(comp.P_el)

    # Basic sanity checks
    if (not np.isfinite(m_flow)) or (not np.isfinite(P_el)) or (m_flow <= 0.0) or (P_el <= 0.0):
        raise ValueError("Non-finite or non-positive outputs.")

    return m_flow, P_el

# -------------------------
# Module cache (so MATLAB doesn't reload every call)
# -------------------------
_CACHE = {
    "rows": None,
    "med": None,
    "model": None,
    "oil": None,
    "N_max_hz": None,
    "V_h_m3": None,
    "refrigerant": None,
}

def init(
    csv_path: str,
    oil: str = "all",
    model: str = "original",
    refrigerant: str = "PROPANE",
    N_max_rpm: float = 7200.0,
    V_h_cm3: float = 30.7,
    max_rows: int | None = None,
):
    csv_path = Path(str(csv_path))
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = read_dataset_csv(csv_path)

    oil_sel = str(oil).strip().lower()
    if oil_sel != "all":
        df = df[df[OIL_COL].astype(str).str.lower() == oil_sel]

    if max_rows is not None:
        df = df.head(int(max_rows))

    required = [
        OIL_COL, P_SUC_COL, T_SUC_COL, P_OUT_COL, T_AMB_COL,
        M_FLOW_MEAS_COL, P_EL_MEAS_COL, SPEED_COL
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows left after filtering / NaN removal.")

    N_max_hz = rpm_to_hz(N_max_rpm)
    V_h_m3 = float(V_h_cm3) * 1e-6

    rows = []
    for _, r in df.iterrows():
        p_suc_pa = bar_to_pa(r[P_SUC_COL])
        p_out_pa = bar_to_pa(r[P_OUT_COL])
        T_suc_K = c_to_k(r[T_SUC_COL])
        T_amb_K = c_to_k(r[T_AMB_COL])
        f_oper_hz = rpm_to_hz(r[SPEED_COL])

        m_meas = gs_to_kgps(r[M_FLOW_MEAS_COL])
        P_meas = float(r[P_EL_MEAS_COL])

        if (m_meas <= 0) or (P_meas <= 0) or (not np.isfinite(m_meas)) or (not np.isfinite(P_meas)):
            continue

        n_rel = _clamp01(f_oper_hz / N_max_hz)

        rows.append({
            "oil": r[OIL_COL],
            "p_suc_pa": float(p_suc_pa),
            "T_suc_K": float(T_suc_K),
            "p_out_pa": float(p_out_pa),
            "T_amb_K": float(T_amb_K),
            "f_oper_hz": float(f_oper_hz),
            "n_rel": float(n_rel),
            "m_meas": float(m_meas),
            "P_meas": float(P_meas),
        })

    if len(rows) == 0:
        raise ValueError("No valid rows after unit conversion/filtering.")

    # RefProp backend
    try:
        med = RefProp(fluid_name=refrigerant)
    except TypeError:
        # fallback if wrapper uses different argument name
        med = RefProp(refrigerant)

    _CACHE.update({
        "rows": rows,
        "med": med,
        "model": str(model),
        "oil": str(oil),
        "N_max_hz": float(N_max_hz),
        "V_h_m3": float(V_h_m3),
        "refrigerant": str(refrigerant),
    })

    return True

def cost(x_in) -> float:
    x = np.asarray(list(x_in), dtype=float).reshape(-1)
    if x.size != len(PARAM_NAMES):
        raise ValueError(f"x must have length {len(PARAM_NAMES)}")

    rows = _CACHE["rows"]
    med = _CACHE["med"]
    model = _CACHE["model"]
    N_max_hz = _CACHE["N_max_hz"]
    V_h_m3 = _CACHE["V_h_m3"]

    params = _x_to_params(x)
    params["f_ref"] = F_REF
    params["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

    comp = make_compressor(model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3, params=params)
    comp.med_prop = med

    inputs = SimpleInputs(control=Control(n=1e-6), T_amb=298.15)

    e_m = np.empty(len(rows), dtype=float)
    e_W = np.empty(len(rows), dtype=float)

    for i, row in enumerate(rows):
        try:
            m_calc, P_calc = _simulate_with_comp(
                comp=comp,
                med=med,
                inputs=inputs,
                p_suc_pa=row["p_suc_pa"],
                T_suc_K=row["T_suc_K"],
                p_out_pa=row["p_out_pa"],
                n_rel=row["n_rel"],
                T_amb_K=row["T_amb_K"],
            )
            e_m[i] = (m_calc / row["m_meas"]) - 1.0
            e_W[i] = (P_calc / row["P_meas"]) - 1.0
        except Exception:
            e_m[i] = FAIL_E
            e_W[i] = FAIL_E

    # Molinaroli objective (eq. 31–33):
    g = 0.5 * (float(np.mean(e_m**2)) + float(np.mean(e_W**2)))
    return float(g)

def save_results(x_in, out_params_csv: str, out_pred_csv: str, tag: str = "matlab", x0_in=None):
    """
    Save fitted parameter row and predictions.
    If x0_in is provided, also store start parameters as x0_<name> columns.
    """
    x = np.asarray(list(x_in), dtype=float).reshape(-1)
    if x.size != len(PARAM_NAMES):
        raise ValueError(f"x must have length {len(PARAM_NAMES)}")
    params = _x_to_params(x)

    # optional start parameters
    x0 = None
    if x0_in is not None:
        try:
            x0_tmp = np.asarray(list(x0_in), dtype=float).reshape(-1)
            if x0_tmp.size == len(PARAM_NAMES):
                x0 = x0_tmp
        except Exception:
            x0 = None

    rows = _CACHE["rows"]
    med = _CACHE["med"]
    model = _CACHE["model"]
    oil = _CACHE["oil"]
    N_max_hz = _CACHE["N_max_hz"]
    V_h_m3 = _CACHE["V_h_m3"]
    refrigerant = _CACHE["refrigerant"]

    params["f_ref"] = F_REF
    params["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

    out_params_csv = str(out_params_csv)
    out_pred_csv = str(out_pred_csv)
    Path(out_params_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_pred_csv).parent.mkdir(parents=True, exist_ok=True)

    comp = make_compressor(model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3, params=params)
    comp.med_prop = med
    inputs = SimpleInputs(control=Control(n=1e-6), T_amb=298.15)

    pred = []
    e_m_list = []
    e_W_list = []
    n_fail = 0

    for row in rows:
        ok = True
        try:
            m_calc, P_calc = _simulate_with_comp(
                comp=comp,
                med=med,
                inputs=inputs,
                p_suc_pa=row["p_suc_pa"],
                T_suc_K=row["T_suc_K"],
                p_out_pa=row["p_out_pa"],
                n_rel=row["n_rel"],
                T_amb_K=row["T_amb_K"],
            )
            e_m = (m_calc / row["m_meas"]) - 1.0
            e_W = (P_calc / row["P_meas"]) - 1.0
            e_m_list.append(float(e_m))
            e_W_list.append(float(e_W))

        except Exception:
            ok = False
            n_fail += 1
            m_calc = float("nan")
            P_calc = float("nan")
            e_m = float("nan")
            e_W = float("nan")
            # for objective calculation, mimic cost(): use FAIL_E
            e_m_list.append(float(FAIL_E))
            e_W_list.append(float(FAIL_E))

        pred.append({
            "oil": row["oil"],
            "f_oper_hz": row["f_oper_hz"],
            "p_suc_bar": row["p_suc_pa"] / 1e5,
            "T_suc_C": row["T_suc_K"] - 273.15,
            "p_out_bar": row["p_out_pa"] / 1e5,
            "T_amb_C": row["T_amb_K"] - 273.15,
            "m_meas_gps": row["m_meas"] * 1000.0,
            "m_calc_gps": (m_calc * 1000.0) if np.isfinite(m_calc) else float("nan"),
            "e_m_rel": e_m,
            "P_meas_W": row["P_meas"],
            "P_calc_W": P_calc,
            "e_P_rel": e_W,
            "ok": ok,
        })

    e_m_arr = np.asarray(e_m_list, dtype=float)
    e_W_arr = np.asarray(e_W_list, dtype=float)
    cost_g = 0.5 * (float(np.mean(e_m_arr**2)) + float(np.mean(e_W_arr**2)))

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    fitted_row = {k: float(params[k]) for k in PARAM_NAMES}

    if x0 is not None:
        for name, val in zip(PARAM_NAMES, x0):
            fitted_row[f"x0_{name}"] = float(val)

    fitted_row.update({
        "f_ref": F_REF,
        "T_ref": T_REF,
        "m_dot_ref_definition": "rho_sat_vapor(T=273.15K,Q=1)*V_h*f_ref",
        "oil_fit_mode": oil,
        "refrigerant": refrigerant,
        "model": model,
        "fit_mode": f"{tag}_{run_id}",
        "cost_g": float(cost_g),
        "n_points": int(len(rows)),
        "n_fail": int(n_fail),
        "fail_penalty_e": float(FAIL_E),
    })

    pd.DataFrame([fitted_row]).to_csv(out_params_csv, index=False)
    pd.DataFrame(pred).to_csv(out_pred_csv, index=False)
    return True