import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vclibpy.media.cool_prop import CoolProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import (
    Molinaroli_2017_Compressor,
    Molinaroli_2017_Compressor_Modified,
)

# run script example:
# python scripts/sensitivity_analysis.py --csv data/Datensatz_Fitting_1.csv --oil LPG100 --model original --delta 0.5 --params_csv results/fitted_params_lpg100_original2.csv

# =========================
# CSV columns (your file)
# =========================
OIL_COL = "Ölbezeichnung"

P_SUC_COL = "P1_mean"            # bar
T_SUC_COL = "T1_mean"            # °C
P_OUT_COL = "P2_mean"            # bar
T_AMB_COL = "Tamb_mean"          # °C

M_FLOW_MEAS_COL = "suction_mf_mean"   # g/s
P_EL_MEAS_COL = "Pel_mean"            # W
SPEED_COL = "N"  # 1/min (rpm)

# =========================
# Reference values (paper)
# =========================
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0  # saturated vapor

# =========================
# Inputs wrapper
# =========================
@dataclass
class Control:
    n: float  # relative speed 0..1

@dataclass
class SimpleInputs:
    control: Control
    T_amb: float  # K

# =========================
# 8 parameters
# =========================
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
    "m_dot_ref": None,   # computed
    "f_ref": F_REF,
}

# =========================
# Unit conversion
# =========================
def bar_to_pa(p_bar: float) -> float:
    return float(p_bar) * 100000.0

def c_to_k(t_c: float) -> float:
    return float(t_c) + 273.15

def rpm_to_hz(rpm: float) -> float:
    return float(rpm) / 60.0

def gs_to_kgps(g_s: float) -> float:
    return float(g_s) / 1000.0

# =========================
# Helpers
# =========================
def read_dataset_csv(path: Path) -> pd.DataFrame:
    # first row units, second row header -> header=1
    return pd.read_csv(path, sep=";", header=1, decimal=",")

def load_params_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError("Params CSV must contain exactly one row.")
    row = df.iloc[0].to_dict()
    # Only take what the model expects; ignore metadata columns if present
    params = DEFAULT_PARAMS.copy()
    for k in PARAM_NAMES:
        if k in row and pd.notna(row[k]):
            params[k] = float(row[k])
    # keep f_ref if present, else default
    if "f_ref" in row and pd.notna(row["f_ref"]):
        params["f_ref"] = float(row["f_ref"])
    return params

def make_compressor(model: str, N_max_hz: float, V_h_m3: float, params: dict):
    m = model.lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    raise ValueError("Unknown --model. Use: original | modified")

def compute_m_dot_ref(med: CoolProp, V_h_m3: float) -> float:
    st = med.calc_state("TQ", T_REF, Q_REF)
    return st.d * V_h_m3 * F_REF

def simulate_point(
    med: CoolProp, model: str, params: dict,
    N_max_hz: float, V_h_m3: float,
    p_suc_pa: float, T_suc_K: float, p_out_pa: float,
    f_oper_hz: float, T_amb_K: float
):
    n_rel = f_oper_hz / N_max_hz
    n_rel = max(1e-6, min(1.0, n_rel))

    p = dict(params)
    p["f_ref"] = F_REF
    p["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

    comp = make_compressor(model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3, params=p)
    comp.med_prop = med
    comp.state_inlet = med.calc_state("PT", p_suc_pa, T_suc_K)

    inputs = SimpleInputs(control=Control(n=n_rel), T_amb=T_amb_K)
    fs_state = FlowsheetState()
    comp.calc_state_outlet(p_outlet=p_out_pa, inputs=inputs, fs_state=fs_state)

    return comp.m_flow, comp.P_el

def build_rows(df: pd.DataFrame):
    required = [OIL_COL, P_SUC_COL, T_SUC_COL, P_OUT_COL, T_AMB_COL,
                M_FLOW_MEAS_COL, P_EL_MEAS_COL, SPEED_COL]
    df = df.dropna(subset=required).reset_index(drop=True)

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

        rows.append({
            "oil": str(r[OIL_COL]),
            "p_suc_pa": p_suc_pa,
            "T_suc_K": T_suc_K,
            "p_out_pa": p_out_pa,
            "T_amb_K": T_amb_K,
            "f_oper_hz": f_oper_hz,
            "m_meas": m_meas,
            "P_meas": P_meas,
        })
    return rows

# =========================
# Metrics / Objective
# g = (0.5/n) * sum( e_m^2 + e_W^2 )
# e_m = m_calc/m_meas - 1
# e_W = W_calc/W_meas - 1
#
# Erweiterung:
#  - em_rms = sqrt( (1/n) * sum(e_m^2) )
#  - eW_rms = sqrt( (1/n) * sum(e_W^2) )
# =========================
def evaluate_metrics(rows, med, model, params, N_max_hz, V_h_m3, fail_penalty=10.0):
    em2 = 0.0
    eW2 = 0.0
    n_ok = 0
    n_fail = 0

    for row in rows:
        try:
            m_calc, P_calc = simulate_point(
                med=med, model=model, params=params,
                N_max_hz=N_max_hz, V_h_m3=V_h_m3,
                p_suc_pa=row["p_suc_pa"], T_suc_K=row["T_suc_K"], p_out_pa=row["p_out_pa"],
                f_oper_hz=row["f_oper_hz"], T_amb_K=row["T_amb_K"],
            )
            em = (m_calc / row["m_meas"]) - 1.0
            eW = (P_calc / row["P_meas"]) - 1.0
            em2 += em * em
            eW2 += eW * eW
            n_ok += 1
        except Exception:
            n_fail += 1
            # Penalize failed points (keeps metric finite + marks instability)
            em2 += fail_penalty * fail_penalty
            eW2 += fail_penalty * fail_penalty

    n_total = n_ok + n_fail
    if n_total == 0:
        return {
            "g": np.inf,
            "em_rms": np.inf,
            "eW_rms": np.inf,
            "n_fail": 0,
            "n_total": 0,
        }

    g = 0.5 * (em2 + eW2) / float(n_total)
    em_rms = np.sqrt(em2 / float(n_total))
    eW_rms = np.sqrt(eW2 / float(n_total))

    return {
        "g": float(g),
        "em_rms": float(em_rms),
        "eW_rms": float(eW_rms),
        "n_fail": int(n_fail),
        "n_total": int(n_total),
    }

def sensitivity_delta(rows, med, model, params_base, N_max_hz, V_h_m3, delta=0.05, fail_penalty=10.0):
    base = evaluate_metrics(rows, med, model, params_base, N_max_hz, V_h_m3, fail_penalty=fail_penalty)

    out = []
    for name in PARAM_NAMES:
        p0 = float(params_base[name])

        p_plus = p0 * (1.0 + delta) if p0 != 0 else 1e-12
        p_minus = p0 * (1.0 - delta) if p0 != 0 else 0.0

        pA = dict(params_base); pA[name] = p_plus
        pB = dict(params_base); pB[name] = p_minus

        plus = evaluate_metrics(rows, med, model, pA, N_max_hz, V_h_m3, fail_penalty=fail_penalty)
        minus = evaluate_metrics(rows, med, model, pB, N_max_hz, V_h_m3, fail_penalty=fail_penalty)

        g0 = base["g"]
        em0 = base["em_rms"]
        eW0 = base["eW_rms"]

        out.append({
            "param": name,
            "p_base": p0,
            "p_plus": p_plus,
            "p_minus": p_minus,

            # Overall objective (like paper)
            "g_base": base["g"],
            "g_plus": plus["g"],
            "g_minus": minus["g"],
            "g_plus_norm": plus["g"] / g0 if np.isfinite(g0) and g0 > 0 else np.nan,
            "g_minus_norm": minus["g"] / g0 if np.isfinite(g0) and g0 > 0 else np.nan,

            # Separate output influences (RMS relative errors)
            "em_rms_base": base["em_rms"],
            "em_rms_plus": plus["em_rms"],
            "em_rms_minus": minus["em_rms"],
            "em_rms_plus_norm": plus["em_rms"] / em0 if np.isfinite(em0) and em0 > 0 else np.nan,
            "em_rms_minus_norm": minus["em_rms"] / em0 if np.isfinite(em0) and em0 > 0 else np.nan,

            "eW_rms_base": base["eW_rms"],
            "eW_rms_plus": plus["eW_rms"],
            "eW_rms_minus": minus["eW_rms"],
            "eW_rms_plus_norm": plus["eW_rms"] / eW0 if np.isfinite(eW0) and eW0 > 0 else np.nan,
            "eW_rms_minus_norm": minus["eW_rms"] / eW0 if np.isfinite(eW0) and eW0 > 0 else np.nan,

            # Fail statistics
            "fail_base": base["n_fail"],
            "fail_plus": plus["n_fail"],
            "fail_minus": minus["n_fail"],
            "n_total": base["n_total"],
        ##   "fail_rate_base": base["n_fail"] / base["n_total"] if base["n_total"] > 0 else np.nan,
        ##   "fail_rate_plus": plus["n_fail"] / plus["n_total"] if plus["n_total"] > 0 else np.nan,
        ##   "fail_rate_minus": minus["n_fail"] / minus["n_total"] if minus["n_total"] > 0 else np.nan,
        })

    return base, pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser(description="Molinaroli sensitivity analysis (±delta per parameter). Outputs g + separate m_dot and Pel influence.")

    ap.add_argument("--csv", required=True, help="Dataset CSV (with units row + header row)")
    ap.add_argument("--model", default="original", help="original | modified")
    ap.add_argument("--refrigerant", default="R290")

    ap.add_argument("--oil", default="all", help="LPG100 | LPG68 | all")
    ap.add_argument("--params_csv", default=None, help="Optional fitted params CSV (one-row)")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0, help="Max speed [1/min] from datasheet")
    ap.add_argument("--V_h_cm3", type=float, default=30.7, help="Displacement [cm^3] from datasheet")

    ap.add_argument("--delta", type=float, default=0.05, help="Relative variation for sensitivity (default: 0.05 => ±5%)")
    ap.add_argument("--out", default="results/sensitivity", help="Output folder")
    ap.add_argument("--fail_penalty", type=float, default=10.0, help="Penalty (relative) used when a point fails to simulate")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset_csv(csv_path)

    # build base params
    if args.params_csv:
        params_base = load_params_csv(Path(args.params_csv))
    else:
        params_base = DEFAULT_PARAMS.copy()

    # constants from datasheet
    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6

    med = CoolProp(fluid_name=args.refrigerant)

    # Decide which oil subsets to run
    oil_arg = args.oil.strip().lower()
    oil_values = sorted(df[OIL_COL].dropna().astype(str).unique().tolist())

    if oil_arg == "all":
        subsets = [("combined", df)]
        for ov in oil_values:
            subsets.append((ov, df[df[OIL_COL].astype(str) == ov]))
    else:
        # exact selection
        sel = None
        for ov in oil_values:
            if ov.strip().lower() == oil_arg:
                sel = ov
                break
        if sel is None:
            raise ValueError(f"Oil '{args.oil}' not found in CSV. Found: {oil_values}")
        subsets = [(sel, df[df[OIL_COL].astype(str) == sel])]

    # Run analysis
    for subset_name, df_sub in subsets:
        rows = build_rows(df_sub)
        if len(rows) == 0:
            print(f"[WARN] no valid rows for subset: {subset_name}")
            continue

        base_metrics, sens_df = sensitivity_delta(
            rows=rows, med=med, model=args.model, params_base=params_base,
            N_max_hz=N_max_hz, V_h_m3=V_h_m3, delta=args.delta,
            fail_penalty=args.fail_penalty
        )

        out_csv = out_dir / f"sensitivity_{subset_name.lower()}_{args.model.lower()}.csv"
        sens_df.to_csv(out_csv, index=False)

        print(f"\n=== Sensitivity done: {subset_name} ===")
        print(f"base g     = {base_metrics['g']:.6e}")
        print(f"base em_rms= {base_metrics['em_rms']:.6e}")
        print(f"base eW_rms= {base_metrics['eW_rms']:.6e}")
        print(f"fails      = {base_metrics['n_fail']}/{base_metrics['n_total']}")
        print(f"saved: {out_csv}")

if __name__ == "__main__":
    main()
