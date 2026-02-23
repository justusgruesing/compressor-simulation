# scripts/fit_molinaroli_ga.py
#
# Simple Genetic Algorithm fitter (James et al. 2016 style / Fig. A2),
# adapted to Molinaroli 8 parameters and your dataset layout.
#
# Objective includes: mass flow, electric power, and discharge temperature
# (temp error normalized by 50 K as in Eq. (40) in the paper).
#
# Outputs:
#   - fitted_params_<oil>_<model>_ga_<timestamp>.csv
#   - fit_predictions_<oil>_<model>_ga_<timestamp>.csv
#
# Uses REFPROP backend via vclibpy.media.RefProp
#
# Beispielufruf: python scripts/fit_parameters_ga.py --csv data/Datensatz_Fitting_1.csv --oil LPG68 --model original --refrigerant PROPANE --generations 50 --population 10
#
# Key performance decisions:
# - Build compressor ONCE per individual evaluation (objective_error).
# - Reuse SimpleInputs + FlowsheetState per individual evaluation (no per-point object creation).
# - m_dot_ref computed ONCE in main and passed into objective_error (constant here).
#
# Improvements included:
# - Log-uniform sampling for A_tot and A_dis (init + mutation).
# - Diversity check reuses already computed errors (no double objective eval).
# - Default training uses ALL points (n_train=0).
# - Elitism keeps elite_k individuals (paper-like 20% if elite_frac=0.2).

from __future__ import annotations

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

# -------------------------
# Defaults for YOUR dataset
# -------------------------
OIL_COL_DEFAULT = "Ölbezeichnung"
P_SUC_COL_DEFAULT = "P1_mean"        # bar
T_SUC_COL_DEFAULT = "T1_mean"        # °C
P_OUT_COL_DEFAULT = "P2_mean"        # bar
T_AMB_COL_DEFAULT = "Tamb_mean"      # °C
SPEED_COL_DEFAULT = "N"              # rpm

M_FLOW_MEAS_COL_DEFAULT = "suction_mf_mean"  # g/s
P_EL_MEAS_COL_DEFAULT = "Pel_mean"           # W

# Optional discharge temp column (if present, used in objective)
T_DIS_MEAS_COL_DEFAULT = "T2_mean"           # °C

# -------------------------
# Molinaroli references
# -------------------------
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0

# Large penalty for failed simulations (used inside squared error)
FAIL_E = 1e3  # relative-error magnitude -> squared becomes 1e6

# -------------------------
# 8 fitted parameters
# -------------------------
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
    "V_IC": 30.7e-6,
    "alpha_loss": 0.16,
    "W_dot_loss_ref": 83.0,
    "m_dot_ref": None,
    "f_ref": F_REF,
}

# -------------------------
# Log-uniform settings
# -------------------------
# A_tot index = 3, A_dis index = 4 (0-based in PARAM_NAMES)
LOG_UNIFORM_IDX = {3, 4}

# -------------------------
# Unit conversions
# -------------------------
def bar_to_pa(p_bar: float) -> float:
    return float(p_bar) * 100000.0

def c_to_k(t_c: float) -> float:
    return float(t_c) + 273.15

def rpm_to_hz(rpm: float) -> float:
    return float(rpm) / 60.0

def gs_to_kgps(g_s: float) -> float:
    return float(g_s) / 1000.0

# -------------------------
# Model plumbing
# -------------------------
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

def read_dataset_csv(path: Path, sep: str, header: int, decimal: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, header=header, decimal=decimal)

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

def simulate_point(
    comp,
    med,
    inputs: SimpleInputs,
    fs_state: FlowsheetState,
    p_suc_pa: float,
    T_suc_K: float,
    p_out_pa: float,
    n_rel: float,
    T_amb_K: float,
):
    # reuse objects: update inputs and re-use fs_state
    inputs.control.n = _clamp01(n_rel)
    inputs.T_amb = float(T_amb_K)

    comp.state_inlet = med.calc_state("PT", float(p_suc_pa), float(T_suc_K))
    comp.calc_state_outlet(p_outlet=float(p_out_pa), inputs=inputs, fs_state=fs_state)

    m_flow = float(comp.m_flow)
    P_el = float(comp.P_el)
    T_dis_K = float(comp.state_outlet.T)

    if (not np.isfinite(m_flow)) or (m_flow <= 0.0):
        raise ValueError("Invalid m_flow")
    if (not np.isfinite(P_el)) or (P_el <= 0.0):
        raise ValueError("Invalid P_el")
    if (not np.isfinite(T_dis_K)) or (T_dis_K <= 0.0):
        raise ValueError("Invalid T_dis")

    return m_flow, P_el, T_dis_K

# -------------------------
# Genetic Algorithm helpers
# -------------------------
def _sample_param(lo: float, hi: float, rng: np.random.Generator, log_uniform: bool) -> float:
    lo = float(lo)
    hi = float(hi)
    if not log_uniform:
        return float(rng.uniform(lo, hi))

    # Log-uniform only for strictly positive bounds
    if lo <= 0.0 or hi <= 0.0:
        return float(rng.uniform(lo, hi))

    return float(10.0 ** rng.uniform(np.log10(lo), np.log10(hi)))

def random_individual(bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.empty(bounds.shape[0], dtype=float)
    for j in range(bounds.shape[0]):
        x[j] = _sample_param(bounds[j, 0], bounds[j, 1], rng, log_uniform=(j in LOG_UNIFORM_IDX))
    return x

def uniform_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(p1.size) < 0.5
    return np.where(mask, p1, p2).astype(float)

def mutate(child: np.ndarray, bounds: np.ndarray, rng: np.random.Generator, p_mut_param: float) -> np.ndarray:
    for j in range(child.size):
        if rng.random() < p_mut_param:
            child[j] = _sample_param(bounds[j, 0], bounds[j, 1], rng, log_uniform=(j in LOG_UNIFORM_IDX))
    return child

def objective_error(
    x: np.ndarray,
    rows: list[dict],
    med,
    model: str,
    N_max_hz: float,
    V_h_m3: float,
    m_dot_ref: float,
    use_Tdis: bool,
    Tdis_norm_K: float,
) -> float:
    # Build compressor ONCE per individual evaluation
    params = _x_to_params(x)
    params["f_ref"] = F_REF
    params["m_dot_ref"] = float(m_dot_ref)

    comp = make_compressor(model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3, params=params)
    comp.med_prop = med

    # Reuse these objects across all data points for this individual
    inputs = SimpleInputs(control=Control(n=1e-6), T_amb=298.15)
    fs_state = FlowsheetState()

    err = 0.0
    for r in rows:
        try:
            m_calc, P_calc, T_dis_K = simulate_point(
                comp=comp,
                med=med,
                inputs=inputs,
                fs_state=fs_state,
                p_suc_pa=r["p_suc_pa"],
                T_suc_K=r["T_suc_K"],
                p_out_pa=r["p_out_pa"],
                n_rel=r["n_rel"],
                T_amb_K=r["T_amb_K"],
            )

            e_m = (m_calc - r["m_meas"]) / r["m_meas"]
            e_W = (P_calc - r["P_meas"]) / r["P_meas"]
            err += float(e_m * e_m) + float(e_W * e_W)

            if use_Tdis and (r.get("T_dis_meas_K") is not None):
                e_T = (T_dis_K - r["T_dis_meas_K"]) / float(Tdis_norm_K)
                err += float(e_T * e_T)

        except Exception:
            err += float(FAIL_E * FAIL_E) + float(FAIL_E * FAIL_E)
            if use_Tdis:
                err += float(FAIL_E * FAIL_E)

    return float(err)

def build_rows(df: pd.DataFrame, args, N_max_hz: float) -> tuple[list[dict], bool]:
    has_Tdis = args.col_T_dis in df.columns

    rows = []
    for _, r in df.iterrows():
        p_suc_pa = bar_to_pa(r[args.col_p_suc])
        p_out_pa = bar_to_pa(r[args.col_p_out])
        T_suc_K = c_to_k(r[args.col_T_suc])
        T_amb_K = c_to_k(r[args.col_T_amb])
        f_oper_hz = rpm_to_hz(r[args.col_speed])

        m_meas = gs_to_kgps(r[args.col_m_meas])
        P_meas = float(r[args.col_P_meas])

        if (m_meas <= 0) or (P_meas <= 0) or (not np.isfinite(m_meas)) or (not np.isfinite(P_meas)):
            continue

        n_rel = _clamp01(f_oper_hz / N_max_hz)

        T_dis_meas_K = None
        if has_Tdis and pd.notna(r[args.col_T_dis]):
            T_dis_meas_K = c_to_k(r[args.col_T_dis])

        rows.append({
            "p_suc_pa": float(p_suc_pa),
            "T_suc_K": float(T_suc_K),
            "p_out_pa": float(p_out_pa),
            "T_amb_K": float(T_amb_K),
            "f_oper_hz": float(f_oper_hz),
            "n_rel": float(n_rel),
            "m_meas": float(m_meas),
            "P_meas": float(P_meas),
            "T_dis_meas_K": float(T_dis_meas_K) if T_dis_meas_K is not None else None,
        })

    if len(rows) == 0:
        raise ValueError("No valid rows after filtering/NaN removal/unit conversion.")
    return rows, has_Tdis

def main():
    ap = argparse.ArgumentParser(description="GA fit (James et al. 2016 style) for Molinaroli 8-parameter model.")

    ap.add_argument("--csv", required=True, help="Input dataset CSV (units row + header row).")
    ap.add_argument("--oil", default="all", help="LPG68 | LPG100 | all")
    ap.add_argument("--oil_col", default=OIL_COL_DEFAULT)

    ap.add_argument("--model", default="original", help="original | modified")
    ap.add_argument("--refrigerant", default="PROPANE", help="REFPROP fluid name used by vclibpy RefProp")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)

    ap.add_argument("--sep", default=";")
    ap.add_argument("--decimal", default=",")
    ap.add_argument("--header", type=int, default=1)

    ap.add_argument("--col_p_suc", default=P_SUC_COL_DEFAULT)
    ap.add_argument("--col_T_suc", default=T_SUC_COL_DEFAULT)
    ap.add_argument("--col_p_out", default=P_OUT_COL_DEFAULT)
    ap.add_argument("--col_T_amb", default=T_AMB_COL_DEFAULT)
    ap.add_argument("--col_speed", default=SPEED_COL_DEFAULT)
    ap.add_argument("--col_m_meas", default=M_FLOW_MEAS_COL_DEFAULT)
    ap.add_argument("--col_P_meas", default=P_EL_MEAS_COL_DEFAULT)
    ap.add_argument("--col_T_dis", default=T_DIS_MEAS_COL_DEFAULT, help="Optional discharge temperature column (°C)")

    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--elite_frac", type=float, default=0.20)
    ap.add_argument("--random_keep_prob", type=float, default=0.10)
    ap.add_argument("--mutation_prob_param", type=float, default=0.20)
    ap.add_argument("--generations", type=int, default=1000)

    ap.add_argument("--n_train", type=int, default=0, help="Number of training points used in objective. Use 0 for all.")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed for determinism")

    ap.add_argument("--Tdis_norm_K", type=float, default=50.0)

    ap.add_argument("--vic_lo_scale", type=float, default=0.5)
    ap.add_argument("--vic_hi_scale", type=float, default=2.0)

    ap.add_argument("--out_dir", default="results/ga_fit")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset_csv(csv_path, sep=args.sep, header=args.header, decimal=args.decimal)

    oil_sel = str(args.oil).strip().lower()
    if oil_sel != "all":
        if args.oil_col not in df.columns:
            raise ValueError(f"Oil column '{args.oil_col}' not found, but --oil != all.")
        df = df[df[args.oil_col].astype(str).str.strip().str.lower() == oil_sel].copy()

    required = [
        args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_T_amb, args.col_speed,
        args.col_m_meas, args.col_P_meas
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows left after dropping NaNs (and oil filter).")

    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6

    rows_all, has_Tdis = build_rows(df, args, N_max_hz=N_max_hz)
    use_Tdis = bool(has_Tdis)
    if not use_Tdis:
        print(f"[INFO] Discharge temp column '{args.col_T_dis}' not found -> objective uses only m_dot + power.")

    rng = np.random.default_rng(args.seed)
    idx_all = np.arange(len(rows_all))
    if args.n_train and args.n_train > 0 and args.n_train < len(rows_all):
        train_idx = rng.choice(idx_all, size=int(args.n_train), replace=False)
        train_idx = np.sort(train_idx)
        rows_train = [rows_all[i] for i in train_idx]
        is_train_mask = np.zeros(len(rows_all), dtype=bool)
        is_train_mask[train_idx] = True
    else:
        rows_train = rows_all
        is_train_mask = np.ones(len(rows_all), dtype=bool)

    try:
        med = RefProp(fluid_name=args.refrigerant)
    except TypeError:
        med = RefProp(args.refrigerant)

    m_dot_ref = compute_m_dot_ref(med, V_h_m3)

    vic_lo = args.vic_lo_scale * V_h_m3
    vic_hi = args.vic_hi_scale * V_h_m3

    bounds = np.array([
        [2.0, 60.0],    # Ua_suc_ref
        [2.0, 60.0],    # Ua_dis_ref
        [0.05,  3.0],     # Ua_amb
        [5e-9, 1e-7],    # A_tot (LOG-UNIFORM)
        [2e-5,  2e-4],    # A_dis (LOG-UNIFORM)
        [vic_lo, vic_hi], # V_IC
        [0.05,  0.4],      # alpha_loss
        [0.0,  300.0],   # W_dot_loss_ref
    ], dtype=float)

    x0 = np.array([DEFAULT_PARAMS[n] for n in PARAM_NAMES], dtype=float)
    x0 = np.clip(x0, bounds[:, 0], bounds[:, 1])

    pop_size = int(args.population)
    elite_k = max(1, int(np.ceil(float(args.elite_frac) * pop_size)))

    population: list[np.ndarray] = [x0.copy()]
    while len(population) < pop_size:
        population.append(random_individual(bounds, rng))

    def eval_pop(pop: list[np.ndarray]) -> np.ndarray:
        errs = np.empty(len(pop), dtype=float)
        for i, ind in enumerate(pop):
            errs[i] = objective_error(
                x=ind,
                rows=rows_train,
                med=med,
                model=args.model,
                N_max_hz=N_max_hz,
                V_h_m3=V_h_m3,
                m_dot_ref=m_dot_ref,
                use_Tdis=use_Tdis,
                Tdis_norm_K=args.Tdis_norm_K,
            )
        return errs

    errors = eval_pop(population)
    best_idx = int(np.argmin(errors))
    best_x = population[best_idx].copy()
    best_err = float(errors[best_idx])
    print(f"[INIT] best_err={best_err:.6e}")

    for gen in range(1, int(args.generations) + 1):
        order = np.argsort(errors)
        population = [population[i] for i in order]
        errors = errors[order]

        if float(errors[0]) < best_err:
            best_err = float(errors[0])
            best_x = population[0].copy()

        selected: list[np.ndarray] = []
        selected_errs: list[float] = []

        for i in range(elite_k):
            selected.append(population[i])
            selected_errs.append(float(errors[i]))

        for i in range(elite_k, pop_size):
            if rng.random() < float(args.random_keep_prob):
                selected.append(population[i])
                selected_errs.append(float(errors[i]))

        if len(selected) < 2:
            selected = [population[0], population[1]]
            selected_errs = [float(errors[0]), float(errors[1])]

        sel_errs_arr = np.asarray(selected_errs, dtype=float)
        rounded = np.round(sel_errs_arr, decimals=12)

        seen = {}
        dup_idxs = []
        for i, val in enumerate(rounded):
            if val in seen:
                dup_idxs.append(i)
            else:
                seen[val] = i

        if len(dup_idxs) >= 3:
            for i in dup_idxs:
                if i >= 2:
                    selected[i] = random_individual(bounds, rng)

        n_children = pop_size - elite_k
        children: list[np.ndarray] = []
        while len(children) < n_children:
            p1 = selected[int(rng.integers(0, len(selected)))]
            p2 = selected[int(rng.integers(0, len(selected)))]
            child = uniform_crossover(p1, p2, rng)
            child = mutate(child, bounds, rng, p_mut_param=float(args.mutation_prob_param))
            child = np.clip(child, bounds[:, 0], bounds[:, 1])
            children.append(child)

        population = [population[i].copy() for i in range(elite_k)] + children
        errors = eval_pop(population)

        if gen % 25 == 0 or gen == 1:
            print(f"[GEN {gen:4d}] best_gen={float(np.min(errors)):.6e}  best_so_far={best_err:.6e}")

    tag = "ga"
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    out_params = out_dir / f"fitted_params_{str(args.oil).lower()}_{str(args.model).lower()}_{tag}_{run_id}.csv"
    out_pred   = out_dir / f"fit_predictions_{str(args.oil).lower()}_{str(args.model).lower()}_{tag}_{run_id}.csv"

    params = _x_to_params(best_x)
    params["f_ref"] = F_REF
    params["m_dot_ref"] = float(m_dot_ref)

    comp = make_compressor(model=args.model, N_max_hz=N_max_hz, V_h_m3=V_h_m3, params=params)
    comp.med_prop = med

    # Reuse objects for prediction export too
    inputs_pred = SimpleInputs(control=Control(n=1e-6), T_amb=298.15)
    fs_state_pred = FlowsheetState()

    pred_rows = []
    for i, r in enumerate(rows_all):
        ok = True
        try:
            m_calc, P_calc, T_dis_K = simulate_point(
                comp=comp, med=med,
                inputs=inputs_pred, fs_state=fs_state_pred,
                p_suc_pa=r["p_suc_pa"], T_suc_K=r["T_suc_K"],
                p_out_pa=r["p_out_pa"], n_rel=r["n_rel"], T_amb_K=r["T_amb_K"],
            )
        except Exception:
            ok = False
            m_calc = np.nan
            P_calc = np.nan
            T_dis_K = np.nan

        rec = {
            "idx": i,
            "is_train": bool(is_train_mask[i]),
            "f_oper_hz": r["f_oper_hz"],
            "p_suc_bar": r["p_suc_pa"] / 1e5,
            "T_suc_C": r["T_suc_K"] - 273.15,
            "p_out_bar": r["p_out_pa"] / 1e5,
            "T_amb_C": r["T_amb_K"] - 273.15,

            "m_meas_gps": r["m_meas"] * 1000.0,
            "m_calc_gps": (m_calc * 1000.0) if np.isfinite(m_calc) else np.nan,
            "e_m_rel": ((m_calc - r["m_meas"]) / r["m_meas"]) if (ok and r["m_meas"] > 0) else np.nan,

            "P_meas_W": r["P_meas"],
            "P_calc_W": P_calc,
            "e_P_rel": ((P_calc - r["P_meas"]) / r["P_meas"]) if (ok and r["P_meas"] > 0) else np.nan,

            "T_dis_meas_C": (r["T_dis_meas_K"] - 273.15) if (r.get("T_dis_meas_K") is not None) else np.nan,
            "T_dis_calc_C": (T_dis_K - 273.15) if np.isfinite(T_dis_K) else np.nan,
            "e_T_dis_norm": ((T_dis_K - r["T_dis_meas_K"]) / args.Tdis_norm_K)
                            if (ok and use_Tdis and r.get("T_dis_meas_K") is not None) else np.nan,
            "ok": ok,
        }
        pred_rows.append(rec)

    final_err = objective_error(
        x=best_x,
        rows=rows_train,
        med=med,
        model=args.model,
        N_max_hz=N_max_hz,
        V_h_m3=V_h_m3,
        m_dot_ref=m_dot_ref,
        use_Tdis=use_Tdis,
        Tdis_norm_K=args.Tdis_norm_K,
    )

    fitted_row = {k: float(v) for k, v in zip(PARAM_NAMES, best_x)}
    for name, val in zip(PARAM_NAMES, x0):
        fitted_row[f"x0_{name}"] = float(val)

    fitted_row.update({
        "f_ref": F_REF,
        "T_ref": T_REF,
        "m_dot_ref_definition": "rho_sat_vapor(T=273.15K,Q=1)*V_h*f_ref",
        "oil_fit_mode": str(args.oil),
        "refrigerant": str(args.refrigerant),
        "model": str(args.model),
        "fit_mode": f"{tag}_{run_id}",
        "error_sum_sq": float(final_err),
        "n_train": int(len(rows_train)),
        "n_points_total": int(len(rows_all)),
        "use_Tdis": bool(use_Tdis),
        "Tdis_norm_K": float(args.Tdis_norm_K),
        "seed": int(args.seed),
        "population": int(pop_size),
        "elite_frac": float(args.elite_frac),
        "random_keep_prob": float(args.random_keep_prob),
        "mutation_prob_param": float(args.mutation_prob_param),
        "generations": int(args.generations),
        "fail_penalty_e": float(FAIL_E),
        "log_uniform_params": "A_tot,A_dis",
    })

    pd.DataFrame([fitted_row]).to_csv(out_params, index=False)
    pd.DataFrame(pred_rows).to_csv(out_pred, index=False)

    print("\n=== GA FIT DONE ===")
    print(f"best error_sum_sq (train) = {float(final_err):.6e}")
    print(f"Saved params: {out_params}")
    print(f"Saved pred  : {out_pred}")

if __name__ == "__main__":
    main()