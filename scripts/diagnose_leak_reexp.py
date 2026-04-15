# scripts/plotting_scripts/diagnose_leak_reexp.py
#
# Diagnose script for the I_leak_reexp irreversibility computation.
#
# Problem: For some models (especially "modified") the leakage+reexpansion
# irreversibility comes out as 0.0 for almost all operating points, while for
# the "original" model it is non-zero in many cases. Hypothesis: the raw s_gen
# value goes slightly negative due to numerical issues (small differences of
# similar-magnitude entropies), and the max(0.0, ...) clamping in the original
# script then sets it to zero.
#
# What this script does:
#   1. Runs a small sweep of operating points (default: T_evap = -5..25, fixed
#      T_cond, N, SH).
#   2. For each point, prints ALL intermediate quantities used in the
#      I_leak_reexp computation (states c1, c3, c4, mass flows, raw s_gen,
#      clamped value, etc.).
#   3. Saves all data to a CSV for further inspection.
#
# Goal: see whether s_gen is genuinely zero, slightly negative, or whether the
# computation has another issue (e.g. wrong f_hz, wrong state references).
#
# Activate REFPROP first:
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Modified model with LPG68 fit, default sweep:
#   python scripts/plotting_scripts/diagnose_leak_reexp.py \
#       --params_csv results/final_results/Modified_LPG68/Fitting/fitted_params_lpg68_modified_ga_2026-03-22_185546.csv \
#       --oil LPG68
#
#   # Original model for comparison:
#   python scripts/plotting_scripts/diagnose_leak_reexp.py \
#       --params_csv results/final_results/Molinaroli_LPG68/Fitting/fitted_params_lpg68_original_ga_2026-03-08_101308.csv \
#       --oil LPG68

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from vclibpy.media import RefProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import Molinaroli_2017_Compressor
from vclibpy.components.compressors.rolling_piston_Molinaroli_2017_modified import (
    Molinaroli_2017_Compressor_Modified,
)

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


F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0
T_AMB_REF_K = 273.15 + 25.0


# =========================================================
# Param defaults (must match GA fit)
# =========================================================
PARAM_NAMES = {
    "original": ["Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
                 "V_IC", "alpha_loss", "W_dot_loss_ref"],
    "modified": ["Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
                 "V_IC", "alpha_loss", "W_dot_loss_ref", "alpha_fric_tot"],
    "oil_path": ["Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
                 "V_IC", "alpha_loss", "W_dot_loss_ref", "alpha_fric_tot",
                 "m_dot_oil_ref", "Ua_suc_oil_ref"],
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

def c_to_k(t):
    return float(t) + 273.15

def rpm_to_hz(rpm):
    return float(rpm) / 60.0

def map_refrigerant(name):
    s = str(name).strip().upper()
    if s in {"PROPANE", "R290", "PROPAN"}:
        return "propane"
    return str(name).strip()

def map_oil(name):
    s = str(name).strip().lower().replace(" ", "")
    if s == "lpg68":
        return "LPG 68"
    if s == "lpg100":
        return "LPG 100"
    raise ValueError(f"Unsupported oil: {name}")


def make_compressor(model, N_max_hz, V_h_m3, params, refrigerant_name, oil_name):
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
    params = dict(DEFAULT_PARAMS[model])
    for k in PARAM_NAMES[model]:
        if k in row and pd.notna(row[k]):
            params[k] = float(row[k])
    if "f_ref" in row and pd.notna(row["f_ref"]):
        params["f_ref"] = float(row["f_ref"])
    meta = {k: row[k] for k in ("oil", "refrigerant", "model") if k in row}
    return params, meta


def compute_m_dot_ref(med, V_h_m3):
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * F_REF


def compute_suction_state(med, T_evap_C, SH_K):
    T_evap_K = c_to_k(T_evap_C)
    state_sat = med.calc_state("TQ", T_evap_K, Q_REF)
    return float(state_sat.p), T_evap_K + float(SH_K)


def compute_discharge_pressure(med, T_cond_C):
    T_cond_K = c_to_k(T_cond_C)
    state_sat = med.calc_state("TQ", T_cond_K, 0.0)
    return float(state_sat.p)


# =========================================================
# Diagnostic: full leak_reexp computation with intermediate values
# =========================================================
def diagnose_leak_reexp(comp, T_o_K=T_AMB_REF_K):
    """
    Replicate the I_leak_reexp computation step by step and return
    ALL intermediate quantities. Returns a dict.
    """
    out = {}

    m_dot_suc = _finite(getattr(comp, "m_flow", np.nan))
    out["m_dot_suc"] = m_dot_suc

    st_in = getattr(comp, "state_inlet", None)
    st_c1 = getattr(comp, "state_c_1", None)
    st_c3 = getattr(comp, "state_c_3", None)
    st_c4 = getattr(comp, "state_c_4", None)

    out["has_st_in"] = st_in is not None
    out["has_st_c1"] = st_c1 is not None
    out["has_st_c3"] = st_c3 is not None
    out["has_st_c4"] = st_c4 is not None

    if not all([st_in, st_c1, st_c3, st_c4]) or m_dot_suc <= 0:
        out["status"] = "missing_states_or_no_flow"
        return out

    # State properties
    out["T_in_K"]  = float(st_in.T)
    out["p_in_bar"] = float(st_in.p) / 1e5
    out["s_in"]    = float(st_in.s)
    out["h_in"]    = float(st_in.h)
    out["rho_in"]  = float(st_in.d)

    out["T_c1_K"]  = float(st_c1.T)
    out["p_c1_bar"] = float(st_c1.p) / 1e5
    out["s_c1"]    = float(st_c1.s)
    out["h_c1"]    = float(st_c1.h)
    out["rho_c1"]  = float(st_c1.d)

    out["T_c3_K"]  = float(st_c3.T)
    out["p_c3_bar"] = float(st_c3.p) / 1e5
    out["s_c3"]    = float(st_c3.s)
    out["h_c3"]    = float(st_c3.h)
    out["rho_c3"]  = float(st_c3.d)

    out["T_c4_K"]  = float(st_c4.T)
    out["p_c4_bar"] = float(st_c4.p) / 1e5
    out["s_c4"]    = float(st_c4.s)
    out["h_c4"]    = float(st_c4.h)
    out["rho_c4"]  = float(st_c4.d)

    # Compute frequency f_hz the same way as in the original script
    V_IC = comp.parameters["V_IC"]
    out["V_IC"] = V_IC

    # Method A: from internal state (mirrors the script logic)
    try:
        f_hz_method_a = comp.get_n_absolute(getattr(comp, "_n_rel_last", 1.0))
    except Exception:
        f_hz_method_a = None
    out["f_hz_method_a"] = f_hz_method_a

    # Method B: from continuity at suction (rho_suc * V_IC * f = m_dot_suc)
    rho_suc = out["rho_in"]
    f_hz_method_b = m_dot_suc / (rho_suc * V_IC) if rho_suc > 0 else None
    out["f_hz_method_b"] = f_hz_method_b

    # Method C: from c1 (rho_c1 * V_IC * f = ?)
    f_hz_method_c = m_dot_suc / (out["rho_c1"] * V_IC) if out["rho_c1"] > 0 else None
    out["f_hz_method_c"] = f_hz_method_c

    # We use method A if available (matches script), else B
    f_hz = f_hz_method_a if f_hz_method_a is not None else f_hz_method_b
    out["f_hz_used"] = f_hz

    # Compute m_dot_3 via continuity at c3 (rho_3 * V_IC * f)
    m_dot_3 = out["rho_c3"] * V_IC * f_hz
    out["m_dot_3"] = m_dot_3
    out["m_dot_3_minus_suc"] = m_dot_3 - m_dot_suc

    # Leakage mass flow (clamped at 0)
    m_dot_leak_clamped = max(0.0, m_dot_3 - m_dot_suc)
    m_dot_leak_raw = m_dot_3 - m_dot_suc
    out["m_dot_leak_raw"] = m_dot_leak_raw
    out["m_dot_leak_clamped"] = m_dot_leak_clamped

    # Entropy generation (mixing at suction)
    s_gen_raw = (m_dot_3 * out["s_c3"]
                 - m_dot_suc * out["s_c1"]
                 - m_dot_leak_clamped * out["s_c4"])
    out["s_gen_raw"] = s_gen_raw

    # Also compute alternate: use raw (unclamped) m_dot_leak
    s_gen_unclamped = (m_dot_3 * out["s_c3"]
                       - m_dot_suc * out["s_c1"]
                       - m_dot_leak_raw * out["s_c4"])
    out["s_gen_unclamped"] = s_gen_unclamped

    # Term-by-term breakdown
    out["term_3"]    = m_dot_3 * out["s_c3"]
    out["term_suc"]  = m_dot_suc * out["s_c1"]
    out["term_leak"] = m_dot_leak_clamped * out["s_c4"]

    # Final irreversibility
    irr_raw = T_o_K * s_gen_raw
    irr_clamped = max(0.0, irr_raw)
    out["I_leak_reexp_raw"] = irr_raw
    out["I_leak_reexp_clamped"] = irr_clamped

    # Diagnostic: mass continuity check
    # If we use c1 instead of in for upstream side
    s_gen_with_c1_density = (m_dot_3 * out["s_c3"]
                              - m_dot_suc * out["s_c1"]
                              - max(0.0, m_dot_3 - m_dot_suc) * out["s_c4"])
    out["s_gen_with_c1_density"] = s_gen_with_c1_density

    # Also compute reverse: Maybe the c3 state expects mixed mass flow through it
    # but the assumption m_dot_3 = rho_c3 * V_IC * f might be wrong. Try with c1:
    m_dot_3_via_c1 = out["rho_c1"] * V_IC * f_hz
    out["m_dot_3_via_c1"] = m_dot_3_via_c1

    out["status"] = "ok"
    return out


# =========================================================
# Run a sweep
# =========================================================
def run_sweep(args):
    # Resolve model
    params_peek = pd.read_csv(args.params_csv).iloc[0].to_dict()
    if args.model == "auto":
        args.model = str(params_peek.get("model", "modified")).lower().strip()
    if args.refrigerant == "auto":
        args.refrigerant = str(params_peek.get("refrigerant", "PROPANE"))

    model = args.model
    print(f"  Model:       {model}")
    print(f"  Refrigerant: {args.refrigerant}")
    print(f"  Oil:         {args.oil}")
    print(f"  Params CSV:  {args.params_csv}")

    # Load params
    params, meta = load_params_csv(args.params_csv, model)
    print(f"  Loaded params from oil={meta.get('oil', '?')}")

    # RefProp
    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6
    med = RefProp(fluid_name=args.refrigerant)
    params["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

    # Build compressor
    comp = make_compressor(model, N_max_hz, V_h_m3, params, args.refrigerant, args.oil)
    comp.med_prop = med
    if hasattr(comp, "debug_enabled"):
        comp.debug_enabled = False

    # Sweep T_evap
    T_evap_values = np.linspace(args.T_evap_min, args.T_evap_max, args.n_points)

    f_hz = rpm_to_hz(args.N_rpm)
    n_rel = f_hz / N_max_hz
    inputs = SimpleInputs(control=Control(n=n_rel), T_amb=c_to_k(args.T_amb_C))

    p_out = compute_discharge_pressure(med, args.T_cond_C)

    rows = []
    print(f"\n  Sweeping T_evap from {args.T_evap_min} to {args.T_evap_max} "
          f"({args.n_points} points) at T_cond={args.T_cond_C}, "
          f"N={args.N_rpm} rpm, SH={args.SH_K} K")
    print("  " + "-" * 95)
    print(f"  {'T_evap':>7} {'s_gen_raw':>14} {'I_raw':>14} {'I_clamped':>14} "
          f"{'m_dot_leak_raw':>16} {'m_dot_3-m_dot_suc':>20}")
    print("  " + "-" * 95)

    for T_evap_C in T_evap_values:
        try:
            p_suc, T_suc_K = compute_suction_state(med, T_evap_C, args.SH_K)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                comp.state_inlet = med.calc_state("PT", float(p_suc), float(T_suc_K))
                comp.calc_state_outlet(p_outlet=float(p_out), inputs=inputs,
                                       fs_state=FlowsheetState())

            diag = diagnose_leak_reexp(comp)
            diag["T_evap_C"] = T_evap_C
            diag["T_cond_C"] = args.T_cond_C
            diag["N_rpm"] = args.N_rpm
            diag["SH_K"] = args.SH_K
            diag["model"] = model
            diag["params_oil"] = meta.get("oil", "?")
            diag["data_oil"] = args.oil

            rows.append(diag)

            if diag.get("status") == "ok":
                print(f"  {T_evap_C:7.1f} {diag['s_gen_raw']:14.6e} "
                      f"{diag['I_leak_reexp_raw']:14.6e} "
                      f"{diag['I_leak_reexp_clamped']:14.6e} "
                      f"{diag['m_dot_leak_raw']:16.6e} "
                      f"{diag['m_dot_3_minus_suc']:20.6e}")
            else:
                print(f"  {T_evap_C:7.1f}  status={diag.get('status', '?')}")

        except Exception as e:
            print(f"  {T_evap_C:7.1f}  ERROR: {type(e).__name__}: {e}")
            rows.append({
                "T_evap_C": T_evap_C, "status": "exception",
                "error_msg": str(e), "model": model,
            })

    return pd.DataFrame(rows)


# =========================================================
# Summary statistics
# =========================================================
def print_summary(df):
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    ok = df[df["status"] == "ok"].copy()
    n_ok = len(ok)
    n_total = len(df)
    print(f"  Successful points: {n_ok}/{n_total}")

    if n_ok == 0:
        return

    # s_gen distribution
    s_gen = ok["s_gen_raw"].dropna()
    print(f"\n  s_gen_raw distribution (W/K):")
    print(f"    min:    {s_gen.min():.6e}")
    print(f"    max:    {s_gen.max():.6e}")
    print(f"    mean:   {s_gen.mean():.6e}")
    print(f"    median: {s_gen.median():.6e}")

    n_negative = (s_gen < 0).sum()
    n_positive = (s_gen > 0).sum()
    n_zero = (s_gen == 0).sum()
    print(f"\n  Sign breakdown of s_gen_raw:")
    print(f"    Negative: {n_negative}/{n_ok} ({100*n_negative/n_ok:.1f}%)")
    print(f"    Zero:     {n_zero}/{n_ok} ({100*n_zero/n_ok:.1f}%)")
    print(f"    Positive: {n_positive}/{n_ok} ({100*n_positive/n_ok:.1f}%)")

    # I after clamping
    i_clamp = ok["I_leak_reexp_clamped"].dropna()
    n_clamped_to_zero = (i_clamp == 0).sum()
    print(f"\n  After max(0, ...) clamping:")
    print(f"    Points clamped to 0: {n_clamped_to_zero}/{n_ok} "
          f"({100*n_clamped_to_zero/n_ok:.1f}%)")

    # m_dot_leak diagnostics
    m_leak_raw = ok["m_dot_leak_raw"].dropna()
    n_leak_negative = (m_leak_raw < 0).sum()
    print(f"\n  m_dot_leak_raw (before max(0,...) clamping):")
    print(f"    min:    {m_leak_raw.min():.6e} kg/s")
    print(f"    max:    {m_leak_raw.max():.6e} kg/s")
    print(f"    mean:   {m_leak_raw.mean():.6e} kg/s")
    print(f"    Negative: {n_leak_negative}/{n_ok} ({100*n_leak_negative/n_ok:.1f}%)")

    # f_hz comparison
    if "f_hz_method_a" in ok.columns and "f_hz_method_b" in ok.columns:
        a = ok["f_hz_method_a"].dropna()
        b = ok["f_hz_method_b"].dropna()
        if len(a) > 0 and len(b) > 0:
            print(f"\n  f_hz comparison:")
            print(f"    Method A (n_rel_last):       mean={a.mean():.4f} Hz")
            print(f"    Method B (continuity at in): mean={b.mean():.4f} Hz")
            if "f_hz_method_c" in ok.columns:
                c = ok["f_hz_method_c"].dropna()
                if len(c) > 0:
                    print(f"    Method C (continuity at c1): mean={c.mean():.4f} Hz")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Diagnose I_leak_reexp computation.")
    ap.add_argument("--params_csv", required=True, type=Path)
    ap.add_argument("--oil", required=True, help="LPG68 | LPG100")
    ap.add_argument("--model", default="auto",
                    help="original | modified | oil_path | auto")
    ap.add_argument("--refrigerant", default="auto")

    # Sweep parameters
    ap.add_argument("--T_evap_min", type=float, default=-5.0)
    ap.add_argument("--T_evap_max", type=float, default=25.0)
    ap.add_argument("--n_points", type=int, default=15)
    ap.add_argument("--T_cond_C", type=float, default=50.0)
    ap.add_argument("--N_rpm", type=float, default=3600.0)
    ap.add_argument("--SH_K", type=float, default=10.0)

    # Compressor geometry
    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)
    ap.add_argument("--T_amb_C", type=float, default=25.0)

    ap.add_argument("--out_dir", default="results/diagnose_leak_reexp", type=Path)

    args = ap.parse_args()

    if not args.params_csv.exists():
        raise FileNotFoundError(args.params_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Run sweep
    df = run_sweep(args)

    # Summary
    print_summary(df)

    # Save
    stamp = _ts()
    model_tag = args.model
    out_csv = args.out_dir / f"diagnose_leak_reexp_{model_tag}_{args.oil}_{stamp}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
