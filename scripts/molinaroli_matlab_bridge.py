# scripts/molinaroli_matlab_bridge.py
#
# Bridge for MATLAB Optimization Toolbox:
# - MATLAB calls init(...) once (loads dataset + caches)
# - MATLAB calls cost(x) many times (scalar objective like Molinaroli eq. 31–33)
# - MATLAB calls save_results(x, out_params_csv, out_pred_csv) once at the end
#
# Uses CoolProp backend via vclibpy.media.cool_prop.CoolProp.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from vclibpy.media.cool_prop import CoolProp
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
T_DIS_COL = "T2_mean"        # °C measured discharge temperature (optional)

M_FLOW_MEAS_COL = "suction_mf_mean"  # g/s
P_EL_MEAS_COL = "Pel_mean"           # W
SPEED_COL = "N"                      # rpm

# -------------------------
# Molinaroli references
# -------------------------
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0

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
    m = model.lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    raise ValueError("Unknown model. Use original | modified")

def compute_m_dot_ref(med: CoolProp, V_h_m3: float) -> float:
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * float(F_REF)

@dataclass
class Control:
    n: float

@dataclass
class SimpleInputs:
    control: Control
    T_amb: float

def simulate_point(
    med: CoolProp, model: str, params_base: dict,
    N_max_hz: float, V_h_m3: float,
    p_suc_pa: float, T_suc_K: float, p_out_pa: float,
    f_oper_hz: float, T_amb_K: float
):
    n_rel = f_oper_hz / N_max_hz
    n_rel = max(1e-6, min(1.0, n_rel))

    params = dict(params_base)
    params["f_ref"] = F_REF
    params["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

    comp = make_compressor(model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3, params=params)
    comp.med_prop = med
    comp.state_inlet = med.calc_state("PT", p_suc_pa, T_suc_K)

    inputs = SimpleInputs(control=Control(n=n_rel), T_amb=T_amb_K)
    fs_state = FlowsheetState()
    comp.calc_state_outlet(p_outlet=p_out_pa, inputs=inputs, fs_state=fs_state)

    T_dis_K = float(comp.state_outlet.T)
    return float(comp.m_flow), float(comp.P_el), T_dis_K

def read_dataset_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", header=1, decimal=",")

def _x_to_params(x: np.ndarray) -> dict:
    p = dict(DEFAULT_PARAMS)
    for name, val in zip(PARAM_NAMES, x):
        p[name] = float(val)
    return p

# -------------------------
# Module cache (so MATLAB doesn't reload every call)
# -------------------------
_CACHE = {
    "rows": None,
    "med": None,
    "model": None,
    "oil": None,
    "use_t_dis": None,
    "w_t_dis": None,
    "N_max_hz": None,
    "V_h_m3": None,
    "refrigerant": None,
}

def init(
    csv_path: str,
    oil: str = "all",
    model: str = "original",
    refrigerant: str = "R290",
    use_t_dis: bool = False,
    w_t_dis: float = 1.0,
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

    required = [OIL_COL, P_SUC_COL, T_SUC_COL, P_OUT_COL, T_AMB_COL, M_FLOW_MEAS_COL, P_EL_MEAS_COL, SPEED_COL]
    if use_t_dis:
        required.append(T_DIS_COL)

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
        if m_meas <= 0 or P_meas <= 0:
            continue

        row = {
            "oil": r[OIL_COL],
            "p_suc_pa": p_suc_pa,
            "T_suc_K": T_suc_K,
            "p_out_pa": p_out_pa,
            "T_amb_K": T_amb_K,
            "f_oper_hz": f_oper_hz,
            "m_meas": m_meas,
            "P_meas": P_meas,
        }

        if use_t_dis:
            T_dis_meas_K = c_to_k(r[T_DIS_COL])
            if T_dis_meas_K <= 0:
                continue
            row["T_dis_meas_K"] = T_dis_meas_K

        rows.append(row)

    if len(rows) == 0:
        raise ValueError("No valid rows after unit conversion/filtering.")

    med = CoolProp(fluid_name=refrigerant)

    _CACHE.update({
        "rows": rows,
        "med": med,
        "model": str(model),
        "oil": str(oil),
        "use_t_dis": bool(use_t_dis),
        "w_t_dis": float(w_t_dis),
        "N_max_hz": float(N_max_hz),
        "V_h_m3": float(V_h_m3),
        "refrigerant": str(refrigerant),
    })

    return True

def cost(x_in) -> float:
    # Accept MATLAB py.list or numpy-ish
    x = np.asarray(list(x_in), dtype=float).reshape(-1)
    if x.size != len(PARAM_NAMES):
        raise ValueError(f"x must have length {len(PARAM_NAMES)}")

    rows = _CACHE["rows"]
    med = _CACHE["med"]
    model = _CACHE["model"]
    N_max_hz = _CACHE["N_max_hz"]
    V_h_m3 = _CACHE["V_h_m3"]
    use_t_dis = _CACHE["use_t_dis"]
    w_t_dis = _CACHE["w_t_dis"]

    params = _x_to_params(x)

    e_m = []
    e_p = []
    e_t = []

    for row in rows:
        try:
            m_calc, P_calc, T_dis_calc = simulate_point(
                med=med, model=model, params_base=params,
                N_max_hz=N_max_hz, V_h_m3=V_h_m3,
                p_suc_pa=row["p_suc_pa"], T_suc_K=row["T_suc_K"], p_out_pa=row["p_out_pa"],
                f_oper_hz=row["f_oper_hz"], T_amb_K=row["T_amb_K"],
            )

            e_m.append((m_calc / row["m_meas"]) - 1.0)
            e_p.append((P_calc / row["P_meas"]) - 1.0)

            if use_t_dis:
                e_t.append((T_dis_calc / row["T_dis_meas_K"]) - 1.0)

        except Exception:
            e_m.append(10.0)
            e_p.append(10.0)
            if use_t_dis:
                e_t.append(10.0)

    e_m = np.asarray(e_m, dtype=float)
    e_p = np.asarray(e_p, dtype=float)

    # Molinaroli objective (eq. 31–33): RMS(m) + RMS(W)
    g = float(np.sqrt(np.mean(e_m**2)) + np.sqrt(np.mean(e_p**2)))

    # Optional: add T_dis penalty (not in Molinaroli eq 31–33, but helpful if you want)
    if use_t_dis and len(e_t) > 0:
        e_t = np.asarray(e_t, dtype=float)
        g += float(w_t_dis) * float(np.sqrt(np.mean(e_t**2)))

    return g

def save_results(x_in, out_params_csv: str, out_pred_csv: str, tag: str = "matlab"):
    x = np.asarray(list(x_in), dtype=float).reshape(-1)
    params = _x_to_params(x)

    rows = _CACHE["rows"]
    med = _CACHE["med"]
    model = _CACHE["model"]
    oil = _CACHE["oil"]
    N_max_hz = _CACHE["N_max_hz"]
    V_h_m3 = _CACHE["V_h_m3"]
    refrigerant = _CACHE["refrigerant"]
    use_t_dis = _CACHE["use_t_dis"]
    w_t_dis = _CACHE["w_t_dis"]

    # fitted params row (+ metadata)
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    fitted_row = {k: float(params[k]) for k in PARAM_NAMES}
    fitted_row.update({
        "f_ref": F_REF,
        "T_ref": T_REF,
        "m_dot_ref_definition": "rho_sat_vapor(T=273.15K,Q=1)*V_h*f_ref",
        "oil_fit_mode": oil,
        "refrigerant": refrigerant,
        "model": model,
        "fit_mode": f"{tag}_{run_id}",
        "use_t_dis": bool(use_t_dis),
        "w_T_dis": float(w_t_dis),
        "cost_g": float(cost(x)),
        "n_points": int(len(rows)),
    })

    out_params_csv = str(out_params_csv)
    out_pred_csv = str(out_pred_csv)
    Path(out_params_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_pred_csv).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([fitted_row]).to_csv(out_params_csv, index=False)

    # predictions
    pred = []
    for row in rows:
        m_calc, P_calc, T_dis_calc = simulate_point(
            med=med, model=model, params_base=params,
            N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            p_suc_pa=row["p_suc_pa"], T_suc_K=row["T_suc_K"], p_out_pa=row["p_out_pa"],
            f_oper_hz=row["f_oper_hz"], T_amb_K=row["T_amb_K"],
        )

        out = {
            "oil": row["oil"],
            "f_oper_hz": row["f_oper_hz"],
            "p_suc_bar": row["p_suc_pa"] / 1e5,
            "T_suc_C": row["T_suc_K"] - 273.15,
            "p_out_bar": row["p_out_pa"] / 1e5,
            "T_amb_C": row["T_amb_K"] - 273.15,
            "m_meas_gps": row["m_meas"] * 1000.0,
            "m_calc_gps": m_calc * 1000.0,
            "e_m_rel": (m_calc / row["m_meas"]) - 1.0,
            "P_meas_W": row["P_meas"],
            "P_calc_W": P_calc,
            "e_P_rel": (P_calc / row["P_meas"]) - 1.0,
        }
        if use_t_dis:
            out["T_dis_meas_C"] = row["T_dis_meas_K"] - 273.15
            out["T_dis_calc_C"] = T_dis_calc - 273.15
            out["e_T_dis_rel"] = (T_dis_calc / row["T_dis_meas_K"]) - 1.0

        pred.append(out)

    pd.DataFrame(pred).to_csv(out_pred_csv, index=False)
    return True
