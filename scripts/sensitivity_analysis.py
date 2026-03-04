# scripts/sensitivity_analysis.py
import argparse
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

# run script example:
# python scripts/sensitivity_analysis.py --csv data/Datensatz_Fitting_1.csv --oil LPG68 --model original --delta 0.05 --params_csv results/ga_fit/fitted_params_lpg68_original_ga_2026-03-04_121512.csv

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
SPEED_COL = "N"                       # 1/min (rpm)

# NEW: discharge temperature measurement (likely)
T_DIS_MEAS_COL_DEFAULT = "T2_mean"    # °C (adjust if needed)

# =========================
# Reference values
# =========================
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0  # saturated vapor

# James et al. objective normalizes T_dis by 50 K
T_DIS_NORM_K = 50.0

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

    params = DEFAULT_PARAMS.copy()
    for k in PARAM_NAMES:
        if k in row and pd.notna(row[k]):
            params[k] = float(row[k])
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

def compute_m_dot_ref(med: RefProp, V_h_m3: float) -> float:
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * V_h_m3 * F_REF

def simulate_point(
    med: RefProp, model: str, params: dict,
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

    # discharge temperature: prefer comp.state_outlet if present, otherwise fs_state fallback
    T_dis_K = None
    if getattr(comp, "state_outlet", None) is not None and hasattr(comp.state_outlet, "T"):
        T_dis_K = float(comp.state_outlet.T)
    elif hasattr(fs_state, "state_outlet") and fs_state.state_outlet is not None and hasattr(fs_state.state_outlet, "T"):
        T_dis_K = float(fs_state.state_outlet.T)

    if T_dis_K is None or not np.isfinite(T_dis_K):
        raise RuntimeError("No valid discharge temperature available from compressor model.")

    return float(comp.m_flow), float(comp.P_el), float(T_dis_K)

def build_rows(df: pd.DataFrame, col_t_dis: str):
    required = [
        OIL_COL, P_SUC_COL, T_SUC_COL, P_OUT_COL, T_AMB_COL,
        M_FLOW_MEAS_COL, P_EL_MEAS_COL, SPEED_COL, col_t_dis
    ]
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
        T_dis_meas_K = c_to_k(r[col_t_dis])

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
            "T_dis_meas_K": T_dis_meas_K,
        })
    return rows

# =========================
# Metrics / Objective (James et al. Eq. 40 style)
# error = sum( eW^2 + em^2 + eT^2 )
# eW = (Wcalc - Wmeas)/Wmeas
# em = (mcalc - mmeas)/mmeas
# eT = (Tdis_calc - Tdis_meas)/50K
# For convenience: g = (1/n)*sum(...)  (scale doesn’t matter for sensitivity ratios)
# =========================
def evaluate_metrics(rows, med, model, params, N_max_hz, V_h_m3, fail_penalty=10.0):
    em2 = 0.0
    eW2 = 0.0
    eT2 = 0.0
    Td2 = 0.0  # absolute squared K-error (for RMSE in K)

    n_ok = 0
    n_fail = 0

    for row in rows:
        try:
            m_calc, P_calc, T_dis_calc_K = simulate_point(
                med=med, model=model, params=params,
                N_max_hz=N_max_hz, V_h_m3=V_h_m3,
                p_suc_pa=row["p_suc_pa"], T_suc_K=row["T_suc_K"], p_out_pa=row["p_out_pa"],
                f_oper_hz=row["f_oper_hz"], T_amb_K=row["T_amb_K"],
            )

            em = (m_calc - row["m_meas"]) / row["m_meas"]
            eW = (P_calc - row["P_meas"]) / row["P_meas"]
            dT_K = (T_dis_calc_K - row["T_dis_meas_K"])
            eT = dT_K / T_DIS_NORM_K

            em2 += em * em
            eW2 += eW * eW
            eT2 += eT * eT
            Td2 += dT_K * dT_K

            n_ok += 1
        except Exception:
            n_fail += 1
            # Penalize failed points (keeps metric finite + marks instability)
            em2 += fail_penalty * fail_penalty
            eW2 += fail_penalty * fail_penalty
            eT2 += fail_penalty * fail_penalty
            Td2 += (fail_penalty * T_DIS_NORM_K) ** 2

    n_total = n_ok + n_fail
    if n_total == 0:
        return {
            "g": np.inf,
            "em_rms": np.inf,
            "eW_rms": np.inf,
            "eT_rms_norm": np.inf,
            "Tdis_rmse_K": np.inf,
            "n_fail": 0,
            "n_total": 0,
        }

    # mean of sum of squares (scaling irrelevant for sensitivity)
    g = (em2 + eW2 + eT2) / float(n_total)

    em_rms = np.sqrt(em2 / float(n_total))
    eW_rms = np.sqrt(eW2 / float(n_total))
    eT_rms_norm = np.sqrt(eT2 / float(n_total))
    Tdis_rmse_K = np.sqrt(Td2 / float(n_total))

    return {
        "g": float(g),
        "em_rms": float(em_rms),
        "eW_rms": float(eW_rms),
        "eT_rms_norm": float(eT_rms_norm),
        "Tdis_rmse_K": float(Tdis_rmse_K),
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
        eT0 = base["eT_rms_norm"]

        out.append({
            "param": name,
            "p_base": p0,
            "p_plus": p_plus,
            "p_minus": p_minus,

            # Overall objective (James et al. Eq. 40 style, averaged)
            "g_base": base["g"],
            "g_plus": plus["g"],
            "g_minus": minus["g"],
            "g_plus_norm": plus["g"] / g0 if np.isfinite(g0) and g0 > 0 else np.nan,
            "g_minus_norm": minus["g"] / g0 if np.isfinite(g0) and g0 > 0 else np.nan,

            # RMS relative errors
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

            # Discharge temperature error
            "eT_rms_norm_base": base["eT_rms_norm"],
            "eT_rms_norm_plus": plus["eT_rms_norm"],
            "eT_rms_norm_minus": minus["eT_rms_norm"],
            "eT_rms_norm_plus_norm": plus["eT_rms_norm"] / eT0 if np.isfinite(eT0) and eT0 > 0 else np.nan,
            "eT_rms_norm_minus_norm": minus["eT_rms_norm"] / eT0 if np.isfinite(eT0) and eT0 > 0 else np.nan,

            "Tdis_rmse_K_base": base["Tdis_rmse_K"],
            "Tdis_rmse_K_plus": plus["Tdis_rmse_K"],
            "Tdis_rmse_K_minus": minus["Tdis_rmse_K"],

            # Fail statistics
            "fail_base": base["n_fail"],
            "fail_plus": plus["n_fail"],
            "fail_minus": minus["n_fail"],
            "n_total": base["n_total"],
        })

    return base, pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser(
        description="Sensitivity analysis (±delta per parameter). Uses RefProp + objective including T_dis (James et al. Eq. 40)."
    )

    ap.add_argument("--csv", required=True, help="Dataset CSV (with units row + header row)")
    ap.add_argument("--model", default="original", help="original | modified")
    ap.add_argument("--refrigerant", default="PROPANE")

    ap.add_argument("--oil", default="all", help="LPG100 | LPG68 | all")
    ap.add_argument("--params_csv", default=None, help="Optional fitted params CSV (one-row)")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0, help="Max speed [1/min] from datasheet")
    ap.add_argument("--V_h_cm3", type=float, default=30.7, help="Displacement [cm^3] from datasheet")

    ap.add_argument("--col_t_dis", default=T_DIS_MEAS_COL_DEFAULT, help="Measured discharge temperature column [°C]")

    ap.add_argument("--delta", type=float, default=0.05, help="Relative variation for sensitivity (default: 0.05 => ±5%)")
    ap.add_argument("--out", default="results/sensitivity", help="Output folder")
    ap.add_argument("--fail_penalty", type=float, default=10.0, help="Penalty (relative) used when a point fails to simulate")

    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y-%m-%d")

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

    # RefProp init (supports both calling conventions)
    try:
        med = RefProp(fluid_name=args.refrigerant)
    except TypeError:
        med = RefProp(args.refrigerant)

    # Decide which oil subsets to run
    oil_arg = args.oil.strip().lower()
    oil_values = sorted(df[OIL_COL].dropna().astype(str).unique().tolist())

    if oil_arg == "all":
        subsets = [("combined", df)]
        for ov in oil_values:
            subsets.append((ov, df[df[OIL_COL].astype(str) == ov]))
    else:
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
        rows = build_rows(df_sub, col_t_dis=args.col_t_dis)
        if len(rows) == 0:
            print(f"[WARN] no valid rows for subset: {subset_name}")
            continue

        base_metrics, sens_df = sensitivity_delta(
            rows=rows, med=med, model=args.model, params_base=params_base,
            N_max_hz=N_max_hz, V_h_m3=V_h_m3, delta=args.delta,
            fail_penalty=args.fail_penalty
        )

        out_csv = out_dir / f"sensitivity_{subset_name.lower()}_{args.model.lower()}_{run_id}.csv"
        sens_df.to_csv(out_csv, index=False)

        print(f"\n=== Sensitivity done: {subset_name} ===")
        print(f"base g           = {base_metrics['g']:.6e}")
        print(f"base em_rms      = {base_metrics['em_rms']:.6e}")
        print(f"base eW_rms      = {base_metrics['eW_rms']:.6e}")
        print(f"base eT_rms_norm = {base_metrics['eT_rms_norm']:.6e}  (=(Tdis_err/50K)_rms)")
        print(f"base Tdis_rmse_K = {base_metrics['Tdis_rmse_K']:.3f} K")
        print(f"fails            = {base_metrics['n_fail']}/{base_metrics['n_total']}")
        print(f"saved: {out_csv}")

if __name__ == "__main__":
    main()