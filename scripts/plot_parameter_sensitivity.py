# scripts/plot_parameter_sensitivity.py
# run script example:
# python scripts/plot_parameter_sensitivity.py --csv data/Datensatz_Fitting_1.csv --oil LPG68 --model original --params_csv results/ga_fit/fitted_params_lpg68_original_ga_2026-03-04_121512.csv

import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vclibpy.media import RefProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import (
    Molinaroli_2017_Compressor,
    Molinaroli_2017_Compressor_Modified,
)

plt.style.use("ebc.paper.mplstyle")

# --- CSV columns ---
OIL_COL = "Ölbezeichnung"
P_SUC_COL = "P1_mean"            # bar
T_SUC_COL = "T1_mean"            # °C
P_OUT_COL = "P2_mean"            # bar
T_AMB_COL = "Tamb_mean"          # °C
T_DIS_MEAS_COL = "T2_mean"       # °C

M_FLOW_MEAS_COL = "suction_mf_mean"   # g/s
P_EL_MEAS_COL = "Pel_mean"            # W
SPEED_COL = "N"  # rpm

# --- model constants ---
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0
T_DIS_NORM_K = 50.0  # James objective normalization (Eq. 40) :contentReference[oaicite:3]{index=3}

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

# --- units ---
def bar_to_pa(p_bar: float) -> float:
    return float(p_bar) * 100000.0

def c_to_k(t_c: float) -> float:
    return float(t_c) + 273.15

def rpm_to_hz(rpm: float) -> float:
    return float(rpm) / 60.0

def gs_to_kgps(g_s: float) -> float:
    return float(g_s) / 1000.0

@dataclass
class Control:
    n: float

@dataclass
class SimpleInputs:
    control: Control
    T_amb: float

def read_dataset_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", header=1, decimal=",")

def load_params_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError("Params CSV must contain exactly one row.")
    row = df.iloc[0].to_dict()

    params = DEFAULT_PARAMS.copy()
    for k in PARAM_NAMES:
        if k in row and pd.notna(row[k]):
            params[k] = float(row[k])
    if "f_ref" in row and pd.notna(row["f_ref"]):
        params["f_ref"] = float(row["f_ref"])
    return params

def _make_med(refrigerant: str) -> RefProp:
    r = refrigerant.strip().upper()
    alias = {"R290": "PROPANE", "PROPAN": "PROPANE"}
    fluid = alias.get(r, refrigerant)
    try:
        return RefProp(fluid_name=fluid)
    except TypeError:
        return RefProp(fluid)

def make_compressor(model: str, N_max_hz: float, V_h_m3: float, params: dict):
    m = model.lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    raise ValueError("Unknown --model. Use: original | modified")

def compute_m_dot_ref(med: RefProp, V_h_m3: float) -> float:
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * V_h_m3 * F_REF

def simulate_point(med, model, params, N_max_hz, V_h_m3, p_suc_pa, T_suc_K, p_out_pa, f_oper_hz, T_amb_K):
    n_rel = max(1e-6, min(1.0, f_oper_hz / N_max_hz))

    p = dict(params)
    p["f_ref"] = F_REF
    p["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

    comp = make_compressor(model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3, params=p)
    comp.med_prop = med
    comp.state_inlet = med.calc_state("PT", p_suc_pa, T_suc_K)

    inputs = SimpleInputs(control=Control(n=n_rel), T_amb=T_amb_K)
    fs_state = FlowsheetState()
    comp.calc_state_outlet(p_outlet=p_out_pa, inputs=inputs, fs_state=fs_state)

    if not hasattr(comp, "state_outlet") or comp.state_outlet is None or not hasattr(comp.state_outlet, "T"):
        raise RuntimeError("No valid discharge temperature available (comp.state_outlet.T missing).")

    return float(comp.m_flow), float(comp.P_el), float(comp.state_outlet.T)

def build_rows(df: pd.DataFrame):
    required = [
        OIL_COL, P_SUC_COL, T_SUC_COL, P_OUT_COL, T_AMB_COL, T_DIS_MEAS_COL,
        M_FLOW_MEAS_COL, P_EL_MEAS_COL, SPEED_COL
    ]
    df = df.dropna(subset=required).reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        m_meas = gs_to_kgps(r[M_FLOW_MEAS_COL])
        P_meas = float(r[P_EL_MEAS_COL])
        if m_meas <= 0 or P_meas <= 0:
            continue

        rows.append({
            "oil": str(r[OIL_COL]),
            "p_suc_pa": bar_to_pa(r[P_SUC_COL]),
            "T_suc_K": c_to_k(r[T_SUC_COL]),
            "p_out_pa": bar_to_pa(r[P_OUT_COL]),
            "T_amb_K": c_to_k(r[T_AMB_COL]),
            "f_oper_hz": rpm_to_hz(r[SPEED_COL]),
            "m_meas": m_meas,
            "P_meas": P_meas,
            "T_dis_meas_K": c_to_k(r[T_DIS_MEAS_COL]),
        })
    return rows

def objective_g(rows, med, model, params, N_max_hz, V_h_m3, fail_penalty=10.0) -> float:
    em2 = 0.0
    eW2 = 0.0
    eT2 = 0.0
    n = 0

    for row in rows:
        try:
            m_calc, P_calc, T_dis_calc_K = simulate_point(
                med, model, params, N_max_hz, V_h_m3,
                row["p_suc_pa"], row["T_suc_K"], row["p_out_pa"],
                row["f_oper_hz"], row["T_amb_K"],
            )
            em = (m_calc / row["m_meas"]) - 1.0
            eW = (P_calc / row["P_meas"]) - 1.0
            eT = (T_dis_calc_K - row["T_dis_meas_K"]) / T_DIS_NORM_K

            em2 += em * em
            eW2 += eW * eW
            eT2 += eT * eT
        except Exception:
            em2 += fail_penalty * fail_penalty
            eW2 += fail_penalty * fail_penalty
            eT2 += fail_penalty * fail_penalty
        n += 1

    if n == 0:
        return float("inf")
    return float((em2 + eW2 + eT2) / n)

def main():
    ap = argparse.ArgumentParser(description="Create parameter sensitivity curves (James objective) like Fig. 9.")
    ap.add_argument("--csv", required=True, help="Dataset CSV (units row + header row)")
    ap.add_argument("--params_csv", required=True, help="One-row CSV with identified params")
    ap.add_argument("--model", default="original", help="original | modified")
    ap.add_argument("--refrigerant", default="PROPANE", help="RefProp fluid (alias: R290->PROPANE)")
    ap.add_argument("--oil", required=True, help="Which oil subset to use (e.g. LPG68/LPG100)")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)

    ap.add_argument("--r_min", type=float, default=0.95, help="Min ratio (param/identified)")
    ap.add_argument("--r_max", type=float, default=1.05, help="Max ratio (param/identified)")
    ap.add_argument("--n_points", type=int, default=11, help="Number of ratio points")

    ap.add_argument("--out_dir", default="results/sensitivity", help="Output folder for png/csv")
    ap.add_argument("--fail_penalty", type=float, default=10.0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset_csv(Path(args.csv))
    df_sub = df[df[OIL_COL].astype(str).str.strip() == str(args.oil).strip()].copy()
    if df_sub.empty:
        raise ValueError(f"Keine Daten für Öl '{args.oil}' gefunden.")

    rows = build_rows(df_sub)
    if not rows:
        raise ValueError("Keine gültigen Datenzeilen nach Filter/Validierung gefunden.")

    params_base = load_params_csv(Path(args.params_csv))

    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6
    med = _make_med(args.refrigerant)

    ratios = np.linspace(args.r_min, args.r_max, args.n_points)

    # Baseline/min objective at identified parameters
    g_min = objective_g(rows, med, args.model, params_base, N_max_hz, V_h_m3, fail_penalty=args.fail_penalty)
    if not np.isfinite(g_min) or g_min <= 0:
        raise RuntimeError(f"Ungültiges g_min={g_min}. Prüfe Modellstabilität / Daten / Parameter.")

    records = []
    fig, ax = plt.subplots()

    for pname in PARAM_NAMES:
        p0 = float(params_base[pname])
        y = []
        for r in ratios:
            p = dict(params_base)
            p[pname] = p0 * float(r)
            g = objective_g(rows, med, args.model, p, N_max_hz, V_h_m3, fail_penalty=args.fail_penalty)
            y.append(g / g_min if np.isfinite(g) else np.nan)

            records.append({
                "param": pname,
                "ratio": float(r),
                "g": float(g),
                "g_norm": float(g / g_min) if np.isfinite(g) else np.nan,
            })

        ax.plot(ratios, y, marker="o", label=pname)

    ax.set_xlabel("Actual parameter / Identified parameter [-]")
    ax.set_ylabel("$g / g_{min}$ [-]")
    ax.set_title(f"Sensitivity | {args.oil} | {args.refrigerant} | {args.model}")
    ax.legend(loc="best", frameon=True)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_png = out_dir / f"sensitivity_curve_{args.oil.lower()}_{args.model.lower()}_{stamp}.png"
    out_csv = out_dir / f"sensitivity_curve_{args.oil.lower()}_{args.model.lower()}_{stamp}.csv"

    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)

    print("Saved plot:", out_png)
    print("Saved data:", out_csv)

if __name__ == "__main__":
    main()