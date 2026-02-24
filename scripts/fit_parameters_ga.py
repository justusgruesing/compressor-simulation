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
# Beispielaufruf (seriell):
#   python scripts/fit_molinaroli_ga.py --csv data/Datensatz_Fitting_1.csv \
#       --oil LPG68 --model original --refrigerant PROPANE \
#       --generations 50 --population 10
#
# Beispielaufruf (parallel, alle Kerne):
#   python scripts/fit_molinaroli_ga.py --csv data/Datensatz_Fitting_1.csv \
#       --oil LPG68 --model original --refrigerant PROPANE \
#       --generations 1000 --population 20 --n_jobs 12
#
# Key design decisions:
# - Compressor built ONCE per individual (not per data point).
# - SimpleInputs + FlowsheetState reused across data points per individual.
# - m_dot_ref computed ONCE in main() from fixed V_h_geo (not from fit parameter V_IC).
# - Log-uniform sampling for A_tot and A_dis (init + mutation).
# - Diversity check reuses already-computed errors (no double objective eval).
# - Elitism keeps elite_k individuals (20% by default, paper-consistent).
# - Parallelization via ProcessPoolExecutor: one RefProp instance per worker process.
# - Bounds tightened to physically realistic ranges.

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tempfile
import shutil
import atexit
import uuid

import numpy as np
import pandas as pd

from vclibpy.components.compressors import (
    Molinaroli_2017_Compressor,
    Molinaroli_2017_Compressor_Modified,
)
from vclibpy.datamodels import FlowsheetState
from vclibpy.media import RefProp

# -------------------------
# CSV column defaults
# -------------------------
OIL_COL_DEFAULT       = "Ölbezeichnung"
P_SUC_COL_DEFAULT     = "P1_mean"           # bar
T_SUC_COL_DEFAULT     = "T1_mean"           # °C
P_OUT_COL_DEFAULT     = "P2_mean"           # bar
T_AMB_COL_DEFAULT     = "Tamb_mean"         # °C
SPEED_COL_DEFAULT     = "N"                 # rpm
M_FLOW_MEAS_COL_DEFAULT = "suction_mf_mean" # g/s
P_EL_MEAS_COL_DEFAULT = "Pel_mean"          # W
T_DIS_MEAS_COL_DEFAULT = "T2_mean"          # °C

# -------------------------
# Molinaroli references
# -------------------------
F_REF  = 50.0    # Hz
T_REF  = 273.15  # K
Q_REF  = 1.0     # saturated vapour

# Penalty for failed simulations
FAIL_E = 1e3     # squared → 1e6

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
    "Ua_suc_ref":     16.05,
    "Ua_dis_ref":     13.96,
    "Ua_amb":          0.36,
    "A_tot":           9.47e-9,
    "A_dis":           86.1e-6,
    "V_IC":            30.7e-6,
    "alpha_loss":      0.16,
    "W_dot_loss_ref":  83.0,
    "m_dot_ref":       None,
    "f_ref":           F_REF,
}

# Log-uniform sampling for A_tot (idx 3) and A_dis (idx 4)
LOG_UNIFORM_IDX = {3, 4}

# -------------------------
# Unit conversions
# -------------------------
def bar_to_pa(p):  return float(p) * 1e5
def c_to_k(t):     return float(t) + 273.15
def rpm_to_hz(n):  return float(n) / 60.0
def gs_to_kgps(m): return float(m) / 1000.0

# -------------------------
# Model helpers
# -------------------------
def make_compressor(model: str, N_max_hz: float, V_h_m3: float, params: dict):
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    raise ValueError("Unknown model. Use original | modified")

def compute_m_dot_ref(med, V_h_m3: float) -> float:
    """m_dot_ref = V_h_geo * f_ref * rho_sat(273.15 K)  [Paper Abschn. 3.1]"""
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * F_REF

def _x_to_params(x: np.ndarray) -> dict:
    p = dict(DEFAULT_PARAMS)
    for name, val in zip(PARAM_NAMES, x):
        p[name] = float(val)
    return p

def read_dataset_csv(path, sep, header, decimal):
    return pd.read_csv(path, sep=sep, header=header, decimal=decimal)

# -------------------------
# Inputs dataclasses
# -------------------------
@dataclass
class Control:
    n: float

@dataclass
class SimpleInputs:
    control: Control
    T_amb: float

def _clamp01(x): return max(1e-9, min(1.0, float(x)))

# -------------------------
# Single operating point
# -------------------------
def simulate_point(comp, med, inputs: SimpleInputs, fs_state: FlowsheetState,
                   p_suc_pa, T_suc_K, p_out_pa, n_rel, T_amb_K):
    """Reuses inputs and fs_state objects (no per-point allocation)."""
    inputs.control.n = _clamp01(n_rel)
    inputs.T_amb     = float(T_amb_K)

    comp.state_inlet = med.calc_state("PT", float(p_suc_pa), float(T_suc_K))
    comp.calc_state_outlet(p_outlet=float(p_out_pa), inputs=inputs, fs_state=fs_state)

    m_flow  = float(comp.m_flow)
    P_el    = float(comp.P_el)
    T_dis_K = float(comp.state_outlet.T)

    if not np.isfinite(m_flow)  or m_flow  <= 0: raise ValueError("Invalid m_flow")
    if not np.isfinite(P_el)    or P_el    <= 0: raise ValueError("Invalid P_el")
    if not np.isfinite(T_dis_K) or T_dis_K <= 0: raise ValueError("Invalid T_dis")

    return m_flow, P_el, T_dis_K

# -------------------------
# Objective function
# -------------------------
def objective_error(x, rows, med, model, N_max_hz, V_h_m3,
                    m_dot_ref, use_Tdis, Tdis_norm_K) -> float:
    """
    error = Σ_i [ (Δm/m_meas)² + (ΔW/W_meas)² + (ΔT/T_norm)² ]
    Compressor built ONCE per individual, inputs/fs_state reused per point.
    """
    params = _x_to_params(x)
    params["f_ref"]    = F_REF
    params["m_dot_ref"] = float(m_dot_ref)

    comp = make_compressor(model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3, params=params)
    comp.med_prop = med

    inputs   = SimpleInputs(control=Control(n=1e-6), T_amb=298.15)
    fs_state = FlowsheetState()

    err = 0.0
    for r in rows:
        try:
            m_c, P_c, T_c = simulate_point(
                comp, med, inputs, fs_state,
                r["p_suc_pa"], r["T_suc_K"], r["p_out_pa"], r["n_rel"], r["T_amb_K"],
            )
            err += ((m_c - r["m_meas"]) / r["m_meas"]) ** 2
            err += ((P_c - r["P_meas"]) / r["P_meas"]) ** 2
            if use_Tdis and r.get("T_dis_meas_K") is not None:
                err += ((T_c - r["T_dis_meas_K"]) / float(Tdis_norm_K)) ** 2
        except Exception:
            err += 2.0 * FAIL_E ** 2
            if use_Tdis:
                err += FAIL_E ** 2

    return float(err)

# -------------------------
# Parallelization: worker globals
# -------------------------
_WORK: dict = {}

def _init_worker(refrigerant, model, N_max_hz, V_h_m3,
                 m_dot_ref, rows_train, use_Tdis, Tdis_norm_K):
    """
    Runs ONCE per worker process.
    Creates a private RefProp instance AND a private working directory
    to avoid DLL copy collisions on Windows.
    """
    global _WORK

    # --- NEW: isolate each worker into its own temp directory ---
    # This prevents multiple processes from copying/loading the same
    # med_prop_<fluid>_REFPRP64.dll in the project folder.
    pid = os.getpid()
    unique = uuid.uuid4().hex[:8]
    worker_dir = Path(tempfile.gettempdir()) / f"refprop_worker_{pid}_{unique}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    # switch CWD so any "copy DLL to cwd" logic becomes per-worker
    os.chdir(worker_dir)

    # optional cleanup at process exit (safe even if folder stays)
    def _cleanup():
        try:
            shutil.rmtree(worker_dir, ignore_errors=True)
        except Exception:
            pass

    atexit.register(_cleanup)

    # --- create RefProp instance inside the worker (after chdir) ---
    try:
        med = RefProp(fluid_name=refrigerant)
    except TypeError:
        med = RefProp(refrigerant)

    _WORK = {
        "med":          med,
        "model":        str(model),
        "N_max_hz":     float(N_max_hz),
        "V_h_m3":       float(V_h_m3),
        "m_dot_ref":    float(m_dot_ref),
        "rows_train":   rows_train,
        "use_Tdis":     bool(use_Tdis),
        "Tdis_norm_K":  float(Tdis_norm_K),
        "worker_dir":   str(worker_dir),  # debug/trace
    }

def _objective_error_worker(x_in) -> float:
    """Worker-side objective using per-process globals (avoids pickling RefProp)."""
    x = np.asarray(x_in, dtype=float).reshape(-1)
    return objective_error(
        x, _WORK["rows_train"], _WORK["med"], _WORK["model"],
        _WORK["N_max_hz"], _WORK["V_h_m3"], _WORK["m_dot_ref"],
        _WORK["use_Tdis"], _WORK["Tdis_norm_K"],
    )

# -------------------------
# GA helpers
# -------------------------
def _sample_param(lo, hi, rng, log_uniform):
    lo, hi = float(lo), float(hi)
    if log_uniform and lo > 0 and hi > 0:
        return float(10.0 ** rng.uniform(np.log10(lo), np.log10(hi)))
    return float(rng.uniform(lo, hi))

def random_individual(bounds, rng):
    return np.array([
        _sample_param(bounds[j, 0], bounds[j, 1], rng, j in LOG_UNIFORM_IDX)
        for j in range(bounds.shape[0])
    ], dtype=float)

def uniform_crossover(p1, p2, rng):
    mask = rng.random(p1.size) < 0.5
    return np.where(mask, p1, p2).astype(float)

def mutate(child, bounds, rng, p_mut):
    for j in range(child.size):
        if rng.random() < p_mut:
            child[j] = _sample_param(bounds[j, 0], bounds[j, 1], rng, j in LOG_UNIFORM_IDX)
    return child

# -------------------------
# Data loading
# -------------------------
def build_rows(df, args, N_max_hz):
    has_Tdis = args.col_T_dis in df.columns
    rows = []
    for _, r in df.iterrows():
        m_meas = gs_to_kgps(r[args.col_m_meas])
        P_meas = float(r[args.col_P_meas])
        if m_meas <= 0 or P_meas <= 0 or not np.isfinite(m_meas) or not np.isfinite(P_meas):
            continue
        T_dis_meas_K = None
        if has_Tdis and pd.notna(r[args.col_T_dis]):
            T_dis_meas_K = c_to_k(r[args.col_T_dis])
        rows.append({
            "p_suc_pa":    bar_to_pa(r[args.col_p_suc]),
            "T_suc_K":     c_to_k(r[args.col_T_suc]),
            "p_out_pa":    bar_to_pa(r[args.col_p_out]),
            "T_amb_K":     c_to_k(r[args.col_T_amb]),
            "f_oper_hz":   rpm_to_hz(r[args.col_speed]),
            "n_rel":       _clamp01(rpm_to_hz(r[args.col_speed]) / N_max_hz),
            "m_meas":      m_meas,
            "P_meas":      P_meas,
            "T_dis_meas_K": float(T_dis_meas_K) if T_dis_meas_K is not None else None,
        })
    if not rows:
        raise ValueError("No valid rows after filtering.")
    return rows, has_Tdis

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="GA fit (James et al. 2016) for Molinaroli 8-parameter model.")

    ap.add_argument("--csv",       required=True, type=Path)
    ap.add_argument("--oil",       default="all")
    ap.add_argument("--oil_col",   default=OIL_COL_DEFAULT)
    ap.add_argument("--model",     default="original", choices=["original", "modified"])
    ap.add_argument("--refrigerant", default="PROPANE")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3",   type=float, default=30.7,
                    help="Geometrisches Hubvolumen [cm³] für m_dot_ref (kein Fit-Parameter)")

    ap.add_argument("--sep",     default=";")
    ap.add_argument("--decimal", default=",")
    ap.add_argument("--header",  type=int, default=1)

    ap.add_argument("--col_p_suc",   default=P_SUC_COL_DEFAULT)
    ap.add_argument("--col_T_suc",   default=T_SUC_COL_DEFAULT)
    ap.add_argument("--col_p_out",   default=P_OUT_COL_DEFAULT)
    ap.add_argument("--col_T_amb",   default=T_AMB_COL_DEFAULT)
    ap.add_argument("--col_speed",   default=SPEED_COL_DEFAULT)
    ap.add_argument("--col_m_meas",  default=M_FLOW_MEAS_COL_DEFAULT)
    ap.add_argument("--col_P_meas",  default=P_EL_MEAS_COL_DEFAULT)
    ap.add_argument("--col_T_dis",   default=T_DIS_MEAS_COL_DEFAULT)

    # GA settings (paper defaults)
    ap.add_argument("--population",          type=int,   default=20)
    ap.add_argument("--elite_frac",          type=float, default=0.20)
    ap.add_argument("--random_keep_prob",    type=float, default=0.10)
    ap.add_argument("--mutation_prob_param", type=float, default=0.20)
    ap.add_argument("--generations",         type=int,   default=1000)

    ap.add_argument("--n_train",    type=int,   default=0,
                    help="Anzahl Trainingspunkte (0 = alle)")
    ap.add_argument("--seed",       type=int,   default=1)
    ap.add_argument("--Tdis_norm_K", type=float, default=50.0)

    # Bounds scaling for V_IC
    ap.add_argument("--vic_lo_scale", type=float, default=0.5)
    ap.add_argument("--vic_hi_scale", type=float, default=2.0)

    ap.add_argument("--out_dir", default="results/ga_fit")

    # Parallelization
    ap.add_argument("--n_jobs", type=int, default=0,
                    help="Anzahl Worker-Prozesse. 0=auto (alle Kerne), 1=seriell")

    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # --- Daten laden ---
    df = read_dataset_csv(args.csv, args.sep, args.header, args.decimal)

    oil_sel = str(args.oil).strip().lower()
    if oil_sel != "all":
        if args.oil_col not in df.columns:
            raise ValueError(f"Öl-Spalte '{args.oil_col}' nicht gefunden.")
        df = df[df[args.oil_col].astype(str).str.strip().str.lower() == oil_sel].copy()

    required = [args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_T_amb,
                args.col_speed, args.col_m_meas, args.col_P_meas]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    if df.empty:
        raise ValueError("Keine Daten nach NaN-Filter.")

    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3   = args.V_h_cm3 * 1e-6

    rows_all, has_Tdis = build_rows(df, args, N_max_hz)
    use_Tdis = bool(has_Tdis)
    if not use_Tdis:
        print(f"[INFO] Spalte '{args.col_T_dis}' nicht gefunden — T_dis nicht in Zielfunktion.")

    rng = np.random.default_rng(args.seed)
    if args.n_train and 0 < args.n_train < len(rows_all):
        idx = np.sort(rng.choice(len(rows_all), size=args.n_train, replace=False))
        rows_train    = [rows_all[i] for i in idx]
        is_train_mask = np.zeros(len(rows_all), dtype=bool)
        is_train_mask[idx] = True
    else:
        rows_train    = rows_all
        is_train_mask = np.ones(len(rows_all), dtype=bool)

    # --- RefProp (Hauptprozess, für m_dot_ref und finale Vorhersage) ---
    try:
        med = RefProp(fluid_name=args.refrigerant)
    except TypeError:
        med = RefProp(args.refrigerant)

    # m_dot_ref einmalig aus festem V_h_geo berechnen
    m_dot_ref = compute_m_dot_ref(med, V_h_m3)
    print(f"  m_dot_ref = {m_dot_ref*1e3:.4f} g/s  (V_h={args.V_h_cm3} cm³, f_ref={F_REF} Hz)")

    # --- Bounds (physikalisch enge Grenzen) ---
    vic_lo = args.vic_lo_scale * V_h_m3
    vic_hi = args.vic_hi_scale * V_h_m3
    bounds = np.array([
        [2.0,   60.0  ],   # Ua_suc_ref
        [2.0,   60.0  ],   # Ua_dis_ref
        [0.05,   3.0  ],   # Ua_amb
        [5e-9,  1e-7  ],   # A_tot  (log-uniform)
        [2e-5,  1e-4  ],   # A_dis  (log-uniform)
        [vic_lo, vic_hi],  # V_IC
        [0.05,   0.4  ],   # alpha_loss
        [0.0,  300.0  ],   # W_dot_loss_ref
    ], dtype=float)

    # --- Initialpopulation ---
    x0 = np.clip(
        np.array([DEFAULT_PARAMS[n] for n in PARAM_NAMES], dtype=float),
        bounds[:, 0], bounds[:, 1]
    )
    pop_size = int(args.population)
    elite_k  = max(1, int(np.ceil(args.elite_frac * pop_size)))

    population: list[np.ndarray] = [x0.copy()]
    while len(population) < pop_size:
        population.append(random_individual(bounds, rng))

    # --- eval_pop: seriell oder parallel ---
    n_jobs = int(args.n_jobs)
    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 1

    if n_jobs == 1:
        # Serieller Pfad — kein Pool-Overhead
        def eval_pop(pop):
            errs = np.empty(len(pop), dtype=float)
            for i, ind in enumerate(pop):
                errs[i] = objective_error(
                    ind, rows_train, med, args.model,
                    N_max_hz, V_h_m3, m_dot_ref, use_Tdis, args.Tdis_norm_K,
                )
            return errs
        executor = None
    else:
        # Paralleler Pfad — ProcessPoolExecutor, einmal erstellt für alle Generationen
        executor = ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            initargs=(
                str(args.refrigerant), str(args.model),
                float(N_max_hz), float(V_h_m3), float(m_dot_ref),
                rows_train, bool(use_Tdis), float(args.Tdis_norm_K),
            ),
        )
        def eval_pop(pop):
            return np.asarray(
                list(executor.map(_objective_error_worker, pop, chunksize=1)),
                dtype=float,
            )

    print(f"  Parallelisierung: {'seriell' if n_jobs == 1 else f'{n_jobs} Worker-Prozesse'}")
    print(f"  Populationsgröße: {pop_size}  |  Eliten: {elite_k}  |  Generationen: {args.generations}")

    # --- GA-Loop ---
    try:
        errors  = eval_pop(population)
        best_x  = population[int(np.argmin(errors))].copy()
        best_err = float(np.min(errors))
        print(f"[INIT] best_err={best_err:.6e}")

        for gen in range(1, args.generations + 1):
            # Sortieren
            order      = np.argsort(errors)
            population = [population[i] for i in order]
            errors     = errors[order]

            if float(errors[0]) < best_err:
                best_err = float(errors[0])
                best_x   = population[0].copy()

            # Eltern-Pool aufbauen
            selected      = list(population[:elite_k])
            selected_errs = list(errors[:elite_k].astype(float))

            for i in range(elite_k, pop_size):
                if rng.random() < args.random_keep_prob:
                    selected.append(population[i])
                    selected_errs.append(float(errors[i]))

            if len(selected) < 2:
                selected      = population[:2]
                selected_errs = list(errors[:2].astype(float))

            # Diversity: Duplikate durch Zufalls-Individuen ersetzen
            rounded = np.round(np.asarray(selected_errs), decimals=12)
            seen, dup_idxs = {}, []
            for i, v in enumerate(rounded):
                if v in seen:
                    dup_idxs.append(i)
                else:
                    seen[v] = i
            if len(dup_idxs) >= 3:
                for i in dup_idxs:
                    if i >= 2:
                        selected[i] = random_individual(bounds, rng)

            # Kinder erzeugen
            children: list[np.ndarray] = []
            while len(children) < pop_size - elite_k:
                p1    = selected[int(rng.integers(0, len(selected)))]
                p2    = selected[int(rng.integers(0, len(selected)))]
                child = mutate(uniform_crossover(p1, p2, rng), bounds, rng,
                               args.mutation_prob_param)
                children.append(np.clip(child, bounds[:, 0], bounds[:, 1]))

            population = [population[i].copy() for i in range(elite_k)] + children
            errors     = eval_pop(population)

            if gen % 25 == 0 or gen == 1:
                print(f"[GEN {gen:4d}] "
                      f"best_gen={float(np.min(errors)):.6e}  "
                      f"best_so_far={best_err:.6e}")

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # --- Vorhersagen exportieren ---
    tag    = "ga"
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(args.out_dir)

    params_best = _x_to_params(best_x)
    params_best["f_ref"]    = F_REF
    params_best["m_dot_ref"] = float(m_dot_ref)

    comp_pred  = make_compressor(args.model, N_max_hz, V_h_m3, params_best)
    comp_pred.med_prop = med
    inputs_pred  = SimpleInputs(control=Control(n=1e-6), T_amb=298.15)
    fs_state_pred = FlowsheetState()

    pred_rows = []
    for i, r in enumerate(rows_all):
        ok = True
        try:
            m_c, P_c, T_c = simulate_point(
                comp_pred, med, inputs_pred, fs_state_pred,
                r["p_suc_pa"], r["T_suc_K"], r["p_out_pa"], r["n_rel"], r["T_amb_K"],
            )
        except Exception:
            ok, m_c, P_c, T_c = False, np.nan, np.nan, np.nan

        pred_rows.append({
            "idx":         i,
            "is_train":    bool(is_train_mask[i]),
            "f_oper_hz":   r["f_oper_hz"],
            "p_suc_bar":   r["p_suc_pa"] / 1e5,
            "T_suc_C":     r["T_suc_K"]  - 273.15,
            "p_out_bar":   r["p_out_pa"] / 1e5,
            "T_amb_C":     r["T_amb_K"]  - 273.15,

            "m_meas_gps":  r["m_meas"] * 1e3,
            "m_calc_gps":  m_c * 1e3 if ok else np.nan,
            "e_m_rel":     (m_c / r["m_meas"] - 1.0) if ok else np.nan,

            "P_meas_W":    r["P_meas"],
            "P_calc_W":    P_c if ok else np.nan,
            "e_P_rel":     (P_c / r["P_meas"] - 1.0) if ok else np.nan,

            "T_dis_meas_C": (r["T_dis_meas_K"] - 273.15)
                            if r.get("T_dis_meas_K") is not None else np.nan,
            "T_dis_calc_C": (T_c - 273.15) if ok else np.nan,
            "e_T_dis_K":    (T_c - r["T_dis_meas_K"])
                            if (ok and r.get("T_dis_meas_K") is not None) else np.nan,
            "ok":          ok,
        })

    final_err = objective_error(
        best_x, rows_train, med, args.model,
        N_max_hz, V_h_m3, m_dot_ref, use_Tdis, args.Tdis_norm_K,
    )

    # --- Statistik ---
    df_pred = pd.DataFrame(pred_rows).dropna(subset=["e_m_rel", "e_P_rel"])
    m5 = (df_pred["e_m_rel"].abs() <= 0.05).mean() * 100
    P5 = (df_pred["e_P_rel"].abs() <= 0.05).mean() * 100
    print(f"\n  Punkte innerhalb ±5 % (Massenstrom): {m5:.1f} %")
    print(f"  Punkte innerhalb ±5 % (Leistung):    {P5:.1f} %")
    if use_Tdis and "e_T_dis_K" in df_pred.columns:
        T3 = (df_pred["e_T_dis_K"].abs() <= 3.0).mean() * 100
        print(f"  Punkte innerhalb ±3 K (T_dis):       {T3:.1f} %")

    # --- Speichern ---
    suffix = f"{str(args.oil).lower()}_{args.model}_{tag}_{run_id}"
    fitted_row = {k: float(v) for k, v in zip(PARAM_NAMES, best_x)}
    fitted_row.update({
        "f_ref":               F_REF,
        "T_ref":               T_REF,
        "m_dot_ref":           float(m_dot_ref),
        "m_dot_ref_definition": "rho_sat(T=273.15K,Q=1)*V_h_geo*f_ref",
        "oil":                 str(args.oil),
        "refrigerant":         str(args.refrigerant),
        "model":               str(args.model),
        "error_sum_sq":        float(final_err),
        "n_train":             len(rows_train),
        "n_points_total":      len(rows_all),
        "use_Tdis":            bool(use_Tdis),
        "Tdis_norm_K":         float(args.Tdis_norm_K),
        "seed":                int(args.seed),
        "population":          pop_size,
        "elite_frac":          float(args.elite_frac),
        "random_keep_prob":    float(args.random_keep_prob),
        "mutation_prob_param": float(args.mutation_prob_param),
        "generations":         int(args.generations),
        "n_jobs":              n_jobs,
        "log_uniform_params":  "A_tot,A_dis",
    })

    out_params = out_dir / f"fitted_params_{suffix}.csv"
    out_pred   = out_dir / f"fit_predictions_{suffix}.csv"
    pd.DataFrame([fitted_row]).to_csv(out_params, index=False)
    pd.DataFrame(pred_rows).to_csv(out_pred,   index=False)

    print(f"\n=== GA FIT DONE ===")
    print(f"  Finaler Fehler (Trainingsdaten): {final_err:.6e}")
    print(f"  Parameter gespeichert: {out_params}")
    print(f"  Vorhersagen gespeichert: {out_pred}")


if __name__ == "__main__":
    mp.freeze_support()   # notwendig auf Windows
    main()