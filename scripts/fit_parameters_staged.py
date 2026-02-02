# Fits Molinaroli parameters using SciPy least_squares.
# Input:  CSV (with a units row -> header is 2nd line)
# Output: ONLY CSV files:
#   - results/fitted_params_<oil>_<model>_<tag>.csv   (one row with parameter values)
#   - results/fit_predictions_<oil>_<model>_<tag>.csv (measured vs predicted + residuals)
#
# Units (fixed to your dataset):
#   P1_mean, P2_mean: bar
#   T1_mean, Tamb_mean, T2_mean: °C
#   suction_mf_mean: g/s
#   Pel_mean: W
#   N: 1/min (rpm)
#
# Datasheet defaults:
#   N_max = 7200 rpm
#   V_h   = 30.7 cm^3

import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from vclibpy.media.cool_prop import CoolProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import (
    Molinaroli_2017_Compressor,
    Molinaroli_2017_Compressor_Modified,
)

# =========================
# CSV columns
# =========================
OIL_COL = "Ölbezeichnung"

P_SUC_COL = "P1_mean"            # bar
T_SUC_COL = "T1_mean"            # °C
P_OUT_COL = "P2_mean"            # bar
T_AMB_COL = "Tamb_mean"          # °C
T_DIS_COL = "T2_mean"            # °C measured discharge temperature

M_FLOW_MEAS_COL = "suction_mf_mean"   # g/s
P_EL_MEAS_COL = "Pel_mean"            # W

SPEED_COL = "N"  # 1/min (rpm)

# =========================
# Fixed reference values
# =========================
F_REF = 50.0      # Hz
T_REF = 273.15    # K
Q_REF = 1.0       # saturated vapor

# =========================
# Inputs wrapper for model
# =========================
@dataclass
class Control:
    n: float  # relative speed 0..1

@dataclass
class SimpleInputs:
    control: Control
    T_amb: float  # K

# =========================
# 8 fit parameters
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
    "A_dis": 86.1e-6,     # <- corrected
    "V_IC": 16.11e-6,
    "alpha_loss": 0.16,
    "W_dot_loss_ref": 83.0,
    "m_dot_ref": None,   # computed from paper definition
    "f_ref": F_REF,
}

# Bounds per parameter (same order as PARAM_NAMES)
LB_ALL = np.array([0.01, 0.01, 0.0,   1e-12, 1e-8,  1e-9,  0.0,  0.0], dtype=float)
UB_ALL = np.array([500.0, 500.0, 50.0, 1e-6,  1e-3,  1e-3,  1.0,  5000.0], dtype=float)

# Manual scaling per parameter (same order as PARAM_NAMES)
X_SCALE_MANUAL_ALL = np.array([
    16.0,   # Ua_suc_ref
    14.0,   # Ua_dis_ref
    0.36,   # Ua_amb
    1e-8,   # A_tot
    1e-4,   # A_dis (order near 1e-4..1e-5)
    1e-5,   # V_IC
    0.16,   # alpha_loss
    83.0    # W_dot_loss_ref
], dtype=float)

# Stages for stepwise fitting
STAGE_1 = ["A_tot", "A_dis", "V_IC", "alpha_loss", "W_dot_loss_ref"]
STAGE_2 = ["Ua_suc_ref", "Ua_dis_ref", "Ua_amb"]
STAGE_3 = PARAM_NAMES[:]  # all


def x_to_params(base_params: dict, x: np.ndarray, fit_names: list[str]) -> dict:
    """Return a full params dict: base + overwritten fit_names from x."""
    p = dict(base_params)
    for name, val in zip(fit_names, x):
        p[name] = float(val)
    return p


def pack_x(params: dict, fit_names: list[str]) -> np.ndarray:
    return np.array([float(params[n]) for n in fit_names], dtype=float)


def slice_by_names(arr_all: np.ndarray, fit_names: list[str]) -> np.ndarray:
    idx = [PARAM_NAMES.index(n) for n in fit_names]
    return arr_all[idx]


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
# Model selection
# =========================
def make_compressor(model: str, N_max_hz: float, V_h_m3: float, params: dict):
    m = model.lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    raise ValueError("Unknown --model. Use: original | modified")


# =========================
# calculation of m_dot_ref
# =========================
def compute_m_dot_ref(med: CoolProp, V_h_m3: float) -> float:
    st = med.calc_state("TQ", T_REF, Q_REF)
    rho_ref = st.d
    return rho_ref * V_h_m3 * F_REF


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


def load_x0_csv(path: Path) -> dict:
    """
    Load a one-row CSV that contains (some or all) PARAM_NAMES.
    Returns a dict with updates.
    """
    df0 = pd.read_csv(path)
    if len(df0) != 1:
        raise ValueError("x0 CSV must contain exactly one row.")
    row = df0.iloc[0].to_dict()

    out = {}
    for n in PARAM_NAMES:
        if n in row and pd.notna(row[n]):
            out[n] = float(row[n])
    return out


# =========================
# Residuals
# =========================
def residuals_for_dataset(
    params_full: dict, rows: list[dict], med: CoolProp,
    model: str, N_max_hz: float, V_h_m3: float,
    use_t_dis: bool, w_t_dis: float
) -> np.ndarray:
    res = []
    for row in rows:
        try:
            m_calc, P_calc, T_dis_calc = simulate_point(
                med=med, model=model, params_base=params_full,
                N_max_hz=N_max_hz, V_h_m3=V_h_m3,
                p_suc_pa=row["p_suc_pa"], T_suc_K=row["T_suc_K"], p_out_pa=row["p_out_pa"],
                f_oper_hz=row["f_oper_hz"], T_amb_K=row["T_amb_K"],
            )

            m_meas = row["m_meas"]
            P_meas = row["P_meas"]

            res.append((m_calc / m_meas) - 1.0)
            res.append((P_calc / P_meas) - 1.0)

            if use_t_dis:
                T_dis_meas = row["T_dis_meas_K"]
                eT = (T_dis_calc / T_dis_meas) - 1.0
                res.append(w_t_dis * eT)

        except Exception:
            # big penalty if point fails
            res.append(10.0)
            res.append(10.0)
            if use_t_dis:
                res.append(10.0)

    return np.asarray(res, dtype=float)


def run_least_squares_for_names(
    fit_names: list[str],
    params_start_full: dict,
    rows: list[dict],
    med: CoolProp,
    model: str,
    N_max_hz: float,
    V_h_m3: float,
    use_t_dis: bool,
    w_t_dis: float,
    x_scale_mode: str,
    ftol: float,
    xtol: float,
    gtol: float,
    max_nfev: int,
    debug: bool,
    stage_label: str = ""
):
    # initial x for subset
    x0 = pack_x(params_start_full, fit_names)

    lb = slice_by_names(LB_ALL, fit_names)
    ub = slice_by_names(UB_ALL, fit_names)

    # clip x0
    x0 = np.maximum(lb, np.minimum(ub, x0))

    if x_scale_mode == "jac":
        x_scale = "jac"
    else:
        x_scale = slice_by_names(X_SCALE_MANUAL_ALL, fit_names)

    last_x = {"x": None}
    call_counter = {"n": 0}

    def fun(x):
        params_full = x_to_params(params_start_full, x, fit_names)
        r = residuals_for_dataset(
            params_full=params_full, rows=rows, med=med,
            model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            use_t_dis=use_t_dis, w_t_dis=w_t_dis
        )

        call_counter["n"] += 1
        if debug:
            cost = 0.5 * float(np.dot(r, r))
            dx = np.zeros_like(x) if last_x["x"] is None else (x - last_x["x"])

            print("\n--- fun(x) call", call_counter["n"], "---")
            print(f"cost = {cost:.6e}")
            for name, xi, dxi in zip(fit_names, x, dx):
                print(f"  {name:>14s} = {xi: .6e}   Δ={dxi: .3e}")

            last_x["x"] = x.copy()

        return r

    if stage_label:
        print(f"\n===== {stage_label} =====")
    print(f"[Fit] params: {fit_names}")
    print(f"[Fit] x_scale: {x_scale_mode}")
    print(f"[Fit] use_t_dis: {use_t_dis} (w_T_dis={w_t_dis})")

    result = least_squares(
        fun,
        x0=x0,
        bounds=(lb, ub),
        method="trf",
        x_scale=x_scale,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        verbose=2,
    )

    # Print stage result (raw x) clearly
    if stage_label:
        print(f"\n=== {stage_label} RESULT (raw x) ===")
    else:
        print("\n=== RESULT (raw x) ===")
    for name, val in zip(fit_names, result.x):
        print(f"{name:>14s} = {val:.6e}")

    # return updated full params
    params_fitted_full = x_to_params(params_start_full, result.x, fit_names)
    return params_fitted_full, result


def main():
    ap = argparse.ArgumentParser(description="Fit Molinaroli parameters (SciPy least_squares), optional staged fit + x_scale=jac")

    ap.add_argument("--csv", required=True, help="Path to dataset CSV")
    ap.add_argument("--model", default="original", help="original | modified")
    ap.add_argument("--refrigerant", default="R290")

    ap.add_argument("--x0_csv", default=None, help="One-row CSV with start values (any subset of the 8 params)")
    ap.add_argument("--oil", default="all", help="LPG100 | LPG68 | all")

    ap.add_argument("--use_t_dis", action="store_true", help="Include T_dis as additional residual")
    ap.add_argument("--w_T_dis", type=float, default=1.0, help="Weight factor for T_dis residual (only if --use_t_dis)")

    ap.add_argument("--staged", action="store_true", help="Run staged fitting: Stage1->Stage2->Stage3")

    ap.add_argument("--x_scale", default="manual", choices=["manual", "jac"], help="Parameter scaling (manual array or jac)")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0, help="Max speed [1/min]")
    ap.add_argument("--V_h_cm3", type=float, default=30.7, help="Displacement volume [cm^3]")

    ap.add_argument("--max_rows", type=int, default=None, help="Optional limit for quick tests")
    ap.add_argument("--max_nfev", type=int, default=400, help="Max function evals")

    ap.add_argument("--out_params_csv", default=None, help="Output CSV for fitted params (one row)")
    ap.add_argument("--out_pred_csv", default=None, help="Output CSV with predictions/residuals")

    ap.add_argument("--ftol", type=float, default=1e-8)
    ap.add_argument("--xtol", type=float, default=1e-10)
    ap.add_argument("--gtol", type=float, default=1e-8)
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y-%m-%d")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = read_dataset_csv(csv_path)

    oil_sel = args.oil.strip().lower()
    if oil_sel != "all":
        df = df[df[OIL_COL].astype(str).str.lower() == oil_sel]

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    required = [OIL_COL, P_SUC_COL, T_SUC_COL, P_OUT_COL, T_AMB_COL,
                M_FLOW_MEAS_COL, P_EL_MEAS_COL, SPEED_COL]
    if args.use_t_dis:
        required.append(T_DIS_COL)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows left after filtering / NaN removal.")

    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6  # cm^3 -> m^3

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

        if args.use_t_dis:
            T_dis_meas_K = c_to_k(r[T_DIS_COL])
            if T_dis_meas_K <= 0:
                continue
            row["T_dis_meas_K"] = T_dis_meas_K

        rows.append(row)

    if len(rows) == 0:
        raise ValueError("No valid rows after unit conversion/filtering.")

    med = CoolProp(fluid_name=args.refrigerant)

    # base start params (full dict)
    params_start = dict(DEFAULT_PARAMS)

    # optional: overwrite starts from CSV
    if args.x0_csv is not None:
        x0_path = Path(args.x0_csv)
        if not x0_path.exists():
            raise FileNotFoundError(x0_path)
        upd = load_x0_csv(x0_path)
        params_start.update(upd)

    # run fit
    if args.staged:
        # Stage 1
        p1, r1 = run_least_squares_for_names(
            fit_names=STAGE_1,
            params_start_full=params_start,
            rows=rows, med=med, model=args.model,
            N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            use_t_dis=args.use_t_dis, w_t_dis=args.w_T_dis,
            x_scale_mode=args.x_scale,
            ftol=args.ftol, xtol=args.xtol, gtol=args.gtol,
            max_nfev=args.max_nfev,
            debug=args.debug,
            stage_label="STAGE 1"
        )
        # Stage 2
        p2, r2 = run_least_squares_for_names(
            fit_names=STAGE_2,
            params_start_full=p1,
            rows=rows, med=med, model=args.model,
            N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            use_t_dis=args.use_t_dis, w_t_dis=args.w_T_dis,
            x_scale_mode=args.x_scale,
            ftol=args.ftol, xtol=args.xtol, gtol=args.gtol,
            max_nfev=args.max_nfev,
            debug=args.debug,
            stage_label="STAGE 2"
        )
        # Stage 3 (polish)
        params_fitted, result = run_least_squares_for_names(
            fit_names=STAGE_3,
            params_start_full=p2,
            rows=rows, med=med, model=args.model,
            N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            use_t_dis=args.use_t_dis, w_t_dis=args.w_T_dis,
            x_scale_mode=args.x_scale,
            ftol=args.ftol, xtol=args.xtol, gtol=args.gtol,
            max_nfev=args.max_nfev,
            debug=args.debug,
            stage_label="STAGE 3"
        )
        tag = "staged"
    else:
        params_fitted, result = run_least_squares_for_names(
            fit_names=PARAM_NAMES,
            params_start_full=params_start,
            rows=rows, med=med, model=args.model,
            N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            use_t_dis=args.use_t_dis, w_t_dis=args.w_T_dis,
            x_scale_mode=args.x_scale,
            ftol=args.ftol, xtol=args.xtol, gtol=args.gtol,
            max_nfev=args.max_nfev,
            debug=args.debug,
            stage_label=""
        )
        tag = "single"

    # add metadata
    fitted_row = {k: float(params_fitted[k]) for k in PARAM_NAMES}
    fitted_row["f_ref"] = F_REF
    fitted_row["T_ref"] = T_REF
    fitted_row["m_dot_ref_definition"] = "rho_sat_vapor(T=273.15K,Q=1)*V_h*f_ref"
    fitted_row["oil_fit_mode"] = args.oil
    fitted_row["N_max_rpm"] = args.N_max_rpm
    fitted_row["N_max_hz"] = N_max_hz
    fitted_row["V_h_cm3"] = args.V_h_cm3
    fitted_row["V_h_m3"] = V_h_m3
    fitted_row["refrigerant"] = args.refrigerant
    fitted_row["model"] = args.model
    fitted_row["use_t_dis"] = bool(args.use_t_dis)
    fitted_row["w_T_dis"] = float(args.w_T_dis)
    fitted_row["fit_mode"] = tag
    fitted_row["x_scale"] = args.x_scale
    fitted_row["success"] = bool(result.success)
    fitted_row["message"] = str(result.message)
    fitted_row["cost"] = float(result.cost)
    fitted_row["n_points"] = int(len(rows))

    Path("results").mkdir(parents=True, exist_ok=True)
    suffix = f"{tag}" + ("_with_t_dis" if args.use_t_dis else "")

    if args.out_params_csv is None:
        args.out_params_csv = f"results/fitted_params_{args.oil.lower()}_{args.model.lower()}_{suffix}_{run_id}.csv"
    if args.out_pred_csv is None:
        args.out_pred_csv = f"results/fit_predictions_{args.oil.lower()}_{args.model.lower()}_{suffix}_{run_id}.csv"

    pd.DataFrame([fitted_row]).to_csv(args.out_params_csv, index=False)

    # predictions
    pred = []
    for row in rows:
        m_calc, P_calc, T_dis_calc = simulate_point(
            med=med, model=args.model, params_base=params_fitted,
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
        if args.use_t_dis:
            out["T_dis_meas_C"] = row["T_dis_meas_K"] - 273.15
            out["T_dis_calc_C"] = T_dis_calc - 273.15
            out["e_T_dis_rel"] = (T_dis_calc / row["T_dis_meas_K"]) - 1.0

        pred.append(out)

    pd.DataFrame(pred).to_csv(args.out_pred_csv, index=False)

    print("\n=== FIT DONE ===")
    print("oil:", args.oil)
    print("fit_mode:", tag)
    print("use_t_dis:", args.use_t_dis, "w_T_dis:", args.w_T_dis)
    print("x_scale:", args.x_scale)
    print("success:", result.success)
    print("message:", result.message)
    print("cost:", result.cost)
    print("saved params CSV:", args.out_params_csv)
    print("saved predictions CSV:", args.out_pred_csv)


if __name__ == "__main__":
    main()
