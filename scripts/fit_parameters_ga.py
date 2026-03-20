# scripts/fit_parameters_ga.py
#
# Genetic Algorithm fitter for Molinaroli compressor models.
#
# Data selection logic:
# - Training / validation points are not sampled randomly.
# - Instead, the script reads:
#     1) operating_points_rows.csv
#     2) operating_points_split_template.csv
# - Train / validation assignment is taken from split_role in the split template.
#
# Typical workflow:
# 1) Create operating_points_rows + operating_points_split_template
# 2) Fill split_role in the template, e.g.:
#       train
#       validation
# 3) Run this script with those two CSV files
#
# Activate REFPROP:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Limit BLAS threads:
#   $env:OPENBLAS_NUM_THREADS = "1"
#   $env:MKL_NUM_THREADS = "1"
#   $env:OMP_NUM_THREADS = "1"
#
# Example:
#   python scripts/fit_parameters_ga.py --op_rows_csv results/split_template/operating_points_rows_2026-03-12_112331.csv --split_csv results/split_template/operating_points_split_template_2026-03-12_112331.csv --oil LPG68 --model modified --refrigerant PROPANE --population 80 --generations 250 --n_jobs 20 --ind_timeout_s 40 --lsq_max_nfev 20000 --mutation_prob_param 0.30

from __future__ import annotations

import argparse
import atexit
import multiprocessing as mp
import os
import re
import shutil
import tempfile
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from vclibpy.components.compressors import Molinaroli_2017_Compressor
from vclibpy.components.compressors.rolling_piston_Molinaroli_2017_modified import (
    Molinaroli_2017_Compressor_Modified,
)
from vclibpy.datamodels import FlowsheetState
from vclibpy.media import RefProp


# =========================================================
# Warning shortening for RefProp messages
# =========================================================
def _short_refprop_warning(message: str) -> str:
    s = str(message)

    m = re.search(
        r"Temperature above upper limit: T\s*=\s*([0-9.]+)\s*K,\s*Tmax\s*=\s*([0-9.]+)\s*K",
        s,
    )
    if m:
        return f"RefProp: T>Tmax (T={m.group(1)}K, Tmax={m.group(2)}K)"

    first = s.splitlines()[0].strip()
    return (first[:140] + "…") if len(first) > 140 else first


def _warn_handler(message, category, filename, lineno, file=None, line=None):
    msg = str(message)

    if "ref_prop.py" in filename or "REFPROP" in msg or "PSFLSH" in msg or "PHFLSH" in msg:
        print(_short_refprop_warning(msg))
        return

    print(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = _warn_handler


# =========================================================
# Defaults for operating_points_rows.csv
# =========================================================
OP_ID_COL_DEFAULT = "op_id"
OIL_COL_DEFAULT = "Ölbezeichnung"

P_SUC_COL_DEFAULT = "P1_mean"                 # bar
T_SUC_COL_DEFAULT = "T1_mean"                 # °C
P_OUT_COL_DEFAULT = "P2_mean"                 # bar
T_AMB_COL_DEFAULT = "Tamb_mean"               # °C
SPEED_COL_DEFAULT = "N"                       # rpm
M_FLOW_MEAS_COL_DEFAULT = "suction_mf_mean"   # g/s
P_EL_MEAS_COL_DEFAULT = "Pel_mean"            # W
T_DIS_MEAS_COL_DEFAULT = "T2_mean"            # °C

SOURCE_ROW_COL_DEFAULT = "source_row_index"
FILTERED_ROW_COL_DEFAULT = "filtered_row_index"

# Split template columns
SPLIT_ROLE_COL_DEFAULT = "split_role"
SHARED_OK_COL_DEFAULT = "usable_for_shared_split"
SPLIT_NOTE_COL_DEFAULT = "split_note"


# =========================================================
# Molinaroli references
# =========================================================
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0

# Penalty for failed simulations
FAIL_E = 1e3


# =========================================================
# Parameter sets
# =========================================================
PARAM_NAMES_ORIGINAL = [
    "Ua_suc_ref",
    "Ua_dis_ref",
    "Ua_amb",
    "A_tot",
    "A_dis",
    "V_IC",
    "alpha_loss",
    "W_dot_loss_ref",
]

PARAM_NAMES_MODIFIED = [
    "Ua_suc_ref",
    "Ua_dis_ref",
    "Ua_amb",
    "A_tot",
    "A_dis",
    "V_IC",
    "alpha_loss",
    "W_dot_loss_ref",
    "alpha_fric_tot",
]

DEFAULT_PARAMS_ORIGINAL = {
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

DEFAULT_PARAMS_MODIFIED = {
    "Ua_suc_ref": 16.05,
    "Ua_dis_ref": 13.96,
    "Ua_amb": 0.36,
    "A_tot": 9.47e-9,
    "A_dis": 86.1e-6,
    "V_IC": 30.7e-6,
    "alpha_loss": 0.16,
    "W_dot_loss_ref": 10.0,
    "alpha_fric_tot": 120.0,
    "m_dot_ref": None,
    "f_ref": F_REF,
}

LOG_UNIFORM_PARAM_NAMES = {"A_tot", "A_dis"}


# =========================================================
# Unit conversions
# =========================================================
def bar_to_pa(p):
    return float(p) * 1e5


def c_to_k(t):
    return float(t) + 273.15


def rpm_to_hz(n):
    return float(n) / 60.0


def gs_to_kgps(m):
    return float(m) / 1000.0


# =========================================================
# Small helpers
# =========================================================
def norm_oil(s: str) -> str:
    return str(s).strip().lower().replace(" ", "")


def parse_bool_like(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x):
        return False

    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False

    raise ValueError(f"Could not parse boolean value from: {x}")


def parse_split_role(x: object) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip().lower()

    if s in {"", "unused"}:
        return ""
    if s in {"train", "training", "fit"}:
        return "train"
    if s in {"validation", "valid", "val", "test"}:
        return "validation"

    raise ValueError(
        f"Unsupported split_role value: {x!r}. "
        f"Allowed values are: train, validation, unused/blank."
    )


def _clamp01(x):
    return max(1e-9, min(1.0, float(x)))


# =========================================================
# Model helpers
# =========================================================
def map_refrigerant_for_modified_model(name: str) -> str:
    s = str(name).strip().upper()
    if s in {"PROPANE", "R290", "PROPAN"}:
        return "propane"
    return str(name).strip()


def map_oil_for_modified_model(name: str) -> str:
    s = norm_oil(name)
    if s == "lpg68":
        return "LPG 68"
    if s == "lpg100":
        return "LPG 100"
    raise ValueError(f"Unsupported oil for modified model: {name}")


def get_param_names(model: str) -> list[str]:
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return list(PARAM_NAMES_ORIGINAL)
    if m in ("mod", "modified"):
        return list(PARAM_NAMES_MODIFIED)
    raise ValueError("Unknown model. Use original | modified")


def get_default_params(model: str) -> dict:
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return dict(DEFAULT_PARAMS_ORIGINAL)
    if m in ("mod", "modified"):
        return dict(DEFAULT_PARAMS_MODIFIED)
    raise ValueError("Unknown model. Use original | modified")


def make_compressor(
    model: str,
    N_max_hz: float,
    V_h_m3: float,
    params: dict,
    refrigerant_name: str,
    oil_name: str | None = None,
):
    m = str(model).lower().strip()

    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(
            N_max=N_max_hz,
            V_h=V_h_m3,
            parameters=params,
        )

    if m in ("mod", "modified"):
        if oil_name is None:
            raise ValueError("Modified model requires an oil name.")
        return Molinaroli_2017_Compressor_Modified(
            N_max=N_max_hz,
            V_h=V_h_m3,
            fluid_name=map_refrigerant_for_modified_model(refrigerant_name),
            lub_name=map_oil_for_modified_model(oil_name),
            parameters=params,
        )

    raise ValueError("Unknown model. Use original | modified")


def compute_m_dot_ref(med, V_h_m3: float) -> float:
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * F_REF


def x_to_params(x: np.ndarray, param_names: list[str], default_params: dict) -> dict:
    p = dict(default_params)
    for name, val in zip(param_names, x):
        p[name] = float(val)
    return p


# =========================================================
# Inputs wrapper
# =========================================================
@dataclass
class Control:
    n: float


@dataclass
class SimpleInputs:
    control: Control
    T_amb: float
    lsq_max_nfev: int = 20000
    lsq_ftol: float = 1e-8
    lsq_xtol: float = 1e-8


# =========================================================
# Data loading
# =========================================================
def read_op_rows_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_split_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_merged_dataframe(op_rows_df: pd.DataFrame, split_df: pd.DataFrame, args) -> pd.DataFrame:
    required_op_cols = [
        args.op_id_col,
        args.oil_col,
        args.col_p_suc,
        args.col_T_suc,
        args.col_p_out,
        args.col_T_amb,
        args.col_speed,
        args.col_m_meas,
        args.col_P_meas,
    ]
    missing_op = [c for c in required_op_cols if c not in op_rows_df.columns]
    if missing_op:
        raise ValueError(f"operating_points_rows CSV missing required columns: {missing_op}")

    required_split_cols = [
        args.op_id_col,
        args.split_role_col,
        args.shared_ok_col,
    ]
    missing_split = [c for c in required_split_cols if c not in split_df.columns]
    if missing_split:
        raise ValueError(f"split CSV missing required columns: {missing_split}")

    op = op_rows_df.copy()
    sp = split_df.copy()

    sp[args.split_role_col] = sp[args.split_role_col].apply(parse_split_role)
    sp[args.shared_ok_col] = sp[args.shared_ok_col].apply(parse_bool_like)

    merge_cols = [args.op_id_col, args.split_role_col, args.shared_ok_col]
    if args.split_note_col in sp.columns:
        merge_cols.append(args.split_note_col)

    merged = op.merge(sp[merge_cols], on=args.op_id_col, how="left")

    merged["_oil_norm_fit"] = merged[args.oil_col].astype(str).map(norm_oil)

    oil_sel = norm_oil(args.oil)
    if oil_sel not in {"lpg68", "lpg100", "all"}:
        raise ValueError("--oil must be LPG68, LPG100 or all")

    if oil_sel != "all":
        merged = merged[merged["_oil_norm_fit"] == oil_sel].copy()
    else:
        merged = merged[merged[args.shared_ok_col] == True].copy()

        opid_oil_counts = merged.groupby(args.op_id_col)["_oil_norm_fit"].nunique()
        valid_shared_opids = opid_oil_counts[opid_oil_counts >= 2].index
        merged = merged[merged[args.op_id_col].isin(valid_shared_opids)].copy()

    numeric_cols = [
        args.col_p_suc,
        args.col_T_suc,
        args.col_p_out,
        args.col_T_amb,
        args.col_speed,
        args.col_m_meas,
        args.col_P_meas,
    ]
    if args.col_T_dis in merged.columns:
        numeric_cols.append(args.col_T_dis)

    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(
        subset=[
            args.col_p_suc,
            args.col_T_suc,
            args.col_p_out,
            args.col_T_amb,
            args.col_speed,
            args.col_m_meas,
            args.col_P_meas,
        ]
    ).reset_index(drop=True)

    if merged.empty:
        raise ValueError("No usable rows left after merging filters and numeric cleanup.")

    return merged


def build_row_records(df: pd.DataFrame, args, N_max_hz: float) -> tuple[list[dict], bool]:
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

        rows.append(
            {
                "source_row_index": int(r[args.source_row_col]) if args.source_row_col in df.columns else np.nan,
                "filtered_row_index": int(r[args.filtered_row_col]) if args.filtered_row_col in df.columns else np.nan,
                "op_id": str(r[args.op_id_col]),
                "oil_name": str(r[args.oil_col]),
                "oil_norm": norm_oil(r[args.oil_col]),
                "split_role": parse_split_role(r.get(args.split_role_col, "")),
                "split_note": r.get(args.split_note_col, ""),
                "p_suc_pa": bar_to_pa(r[args.col_p_suc]),
                "T_suc_K": c_to_k(r[args.col_T_suc]),
                "p_out_pa": bar_to_pa(r[args.col_p_out]),
                "T_amb_K": c_to_k(r[args.col_T_amb]),
                "f_oper_hz": rpm_to_hz(r[args.col_speed]),
                "n_rel": _clamp01(rpm_to_hz(r[args.col_speed]) / N_max_hz),
                "m_meas": m_meas,
                "P_meas": P_meas,
                "T_dis_meas_K": float(T_dis_meas_K) if T_dis_meas_K is not None else None,
            }
        )

    if not rows:
        raise ValueError("No valid simulation rows could be built from merged data.")

    return rows, has_Tdis


# =========================================================
# Single operating point
# =========================================================
def simulate_point(
    comp,
    med,
    inputs: SimpleInputs,
    fs_state: FlowsheetState,
    p_suc_pa,
    T_suc_K,
    p_out_pa,
    n_rel,
    T_amb_K,
):
    inputs.control.n = _clamp01(n_rel)
    inputs.T_amb = float(T_amb_K)

    comp.state_inlet = med.calc_state("PT", float(p_suc_pa), float(T_suc_K))
    comp.calc_state_outlet(p_outlet=float(p_out_pa), inputs=inputs, fs_state=fs_state)

    m_flow = float(comp.m_flow)
    P_el = float(comp.P_el)
    T_dis_K = float(comp.state_outlet.T)

    if not np.isfinite(m_flow) or m_flow <= 0:
        raise ValueError("Invalid m_flow")
    if not np.isfinite(P_el) or P_el <= 0:
        raise ValueError("Invalid P_el")
    if not np.isfinite(T_dis_K) or T_dis_K <= 0:
        raise ValueError("Invalid T_dis")

    return m_flow, P_el, T_dis_K


# =========================================================
# Runtime bundle per individual
# =========================================================
def build_runtime_bundle(
    model: str,
    rows: list[dict],
    med,
    refrigerant_name: str,
    N_max_hz: float,
    V_h_m3: float,
    params: dict,
    lsq_max_nfev: int,
):
    m = str(model).lower().strip()
    bundle = {}

    if m in ("orig", "original"):
        comp = make_compressor(
            model=model,
            N_max_hz=N_max_hz,
            V_h_m3=V_h_m3,
            params=params,
            refrigerant_name=refrigerant_name,
            oil_name=None,
        )
        comp.med_prop = med
        if hasattr(comp, "debug_enabled"):
            comp.debug_enabled = False

        bundle["single"] = {
            "comp": comp,
            "inputs": SimpleInputs(
                control=Control(n=1e-6),
                T_amb=298.15,
                lsq_max_nfev=int(lsq_max_nfev),
                lsq_ftol=1e-8,
                lsq_xtol=1e-8,
            ),
            "fs_state": FlowsheetState(),
        }
        return bundle

    unique_oils = sorted({r["oil_name"] for r in rows})
    for oil_name in unique_oils:
        comp = make_compressor(
            model=model,
            N_max_hz=N_max_hz,
            V_h_m3=V_h_m3,
            params=params,
            refrigerant_name=refrigerant_name,
            oil_name=oil_name,
        )
        comp.med_prop = med
        if hasattr(comp, "debug_enabled"):
            comp.debug_enabled = False

        bundle[norm_oil(oil_name)] = {
            "comp": comp,
            "inputs": SimpleInputs(
                control=Control(n=1e-6),
                T_amb=298.15,
                lsq_max_nfev=int(lsq_max_nfev),
                lsq_ftol=1e-8,
                lsq_xtol=1e-8,
            ),
            "fs_state": FlowsheetState(),
        }

    return bundle


def get_runtime_entry(bundle: dict, model: str, row: dict):
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return bundle["single"]
    return bundle[row["oil_norm"]]


# =========================================================
# Objective function
# =========================================================
def objective_error(
    x,
    rows,
    med,
    model,
    refrigerant_name,
    N_max_hz,
    V_h_m3,
    m_dot_ref,
    param_names,
    default_params,
    use_Tdis,
    Tdis_norm_K,
    lsq_max_nfev: int,
) -> float:
    params = x_to_params(np.asarray(x, dtype=float), param_names, default_params)
    params["f_ref"] = F_REF
    params["m_dot_ref"] = float(m_dot_ref)

    runtime = build_runtime_bundle(
        model=model,
        rows=rows,
        med=med,
        refrigerant_name=refrigerant_name,
        N_max_hz=N_max_hz,
        V_h_m3=V_h_m3,
        params=params,
        lsq_max_nfev=lsq_max_nfev,
    )

    err = 0.0
    for r in rows:
        try:
            entry = get_runtime_entry(runtime, model, r)
            m_c, P_c, T_c = simulate_point(
                entry["comp"],
                med,
                entry["inputs"],
                entry["fs_state"],
                r["p_suc_pa"],
                r["T_suc_K"],
                r["p_out_pa"],
                r["n_rel"],
                r["T_amb_K"],
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


# =========================================================
# Parallel worker globals
# =========================================================
_WORK: dict = {}


def _init_worker(
    refrigerant_name,
    model,
    N_max_hz,
    V_h_m3,
    m_dot_ref,
    rows_train,
    param_names,
    default_params,
    use_Tdis,
    Tdis_norm_K,
    lsq_max_nfev,
):
    global _WORK

    pid = os.getpid()
    unique = uuid.uuid4().hex[:8]
    worker_dir = Path(tempfile.gettempdir()) / f"refprop_worker_{pid}_{unique}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(worker_dir)

    def _cleanup():
        try:
            shutil.rmtree(worker_dir, ignore_errors=True)
        except Exception:
            pass

    atexit.register(_cleanup)

    med = RefProp(fluid_name=refrigerant_name)

    _WORK = {
        "med": med,
        "refrigerant_name": str(refrigerant_name),
        "model": str(model),
        "N_max_hz": float(N_max_hz),
        "V_h_m3": float(V_h_m3),
        "m_dot_ref": float(m_dot_ref),
        "rows_train": rows_train,
        "param_names": list(param_names),
        "default_params": dict(default_params),
        "use_Tdis": bool(use_Tdis),
        "Tdis_norm_K": float(Tdis_norm_K),
        "lsq_max_nfev": int(lsq_max_nfev),
        "worker_dir": str(worker_dir),
    }


def _objective_error_worker(x_in) -> float:
    x = np.asarray(x_in, dtype=float).reshape(-1)
    return objective_error(
        x=x,
        rows=_WORK["rows_train"],
        med=_WORK["med"],
        model=_WORK["model"],
        refrigerant_name=_WORK["refrigerant_name"],
        N_max_hz=_WORK["N_max_hz"],
        V_h_m3=_WORK["V_h_m3"],
        m_dot_ref=_WORK["m_dot_ref"],
        param_names=_WORK["param_names"],
        default_params=_WORK["default_params"],
        use_Tdis=_WORK["use_Tdis"],
        Tdis_norm_K=_WORK["Tdis_norm_K"],
        lsq_max_nfev=_WORK["lsq_max_nfev"],
    )


# =========================================================
# GA helpers
# =========================================================
def _sample_param(param_name: str, lo: float, hi: float, rng):
    lo = float(lo)
    hi = float(hi)
    if param_name in LOG_UNIFORM_PARAM_NAMES and lo > 0 and hi > 0:
        return float(10.0 ** rng.uniform(np.log10(lo), np.log10(hi)))
    return float(rng.uniform(lo, hi))


def random_individual(bounds: np.ndarray, param_names: list[str], rng) -> np.ndarray:
    vals = []
    for j, name in enumerate(param_names):
        vals.append(_sample_param(name, bounds[j, 0], bounds[j, 1], rng))
    return np.array(vals, dtype=float)


def uniform_crossover(p1, p2, rng):
    mask = rng.random(p1.size) < 0.5
    return np.where(mask, p1, p2).astype(float)


def mutate(child, bounds, param_names, rng, p_mut):
    for j, name in enumerate(param_names):
        if rng.random() < p_mut:
            child[j] = _sample_param(name, bounds[j, 0], bounds[j, 1], rng)
    return child


def build_bounds(model: str, V_h_m3: float, vic_lo_scale: float, vic_hi_scale: float, param_names: list[str]) -> np.ndarray:
    vic_lo = float(vic_lo_scale) * V_h_m3
    vic_hi = float(vic_hi_scale) * V_h_m3

    m = str(model).lower().strip()

    if m in ("orig", "original"):
        by_name = {
            "Ua_suc_ref": (8.0, 55.0),
            "Ua_dis_ref": (2.0, 20.0),
            "Ua_amb": (0.1, 3.0),
            "A_tot": (7e-9, 5e-7),
            "A_dis": (4e-6, 5e-4),
            "V_IC": (vic_lo, vic_hi),
            "alpha_loss": (0.10, 0.30),
            "W_dot_loss_ref": (40.0, 175.0),
        }
    elif m in ("mod", "modified"):
        by_name = {
            "Ua_suc_ref": (8.0, 55.0),
            "Ua_dis_ref": (2.0, 20.0),
            "Ua_amb": (0.1, 3.0),
            "A_tot": (2e-9, 5e-7),
            "A_dis": (4e-6, 5e-4),
            "V_IC": (vic_lo, vic_hi),
            "alpha_loss": (0.10, 0.30),
            "W_dot_loss_ref": (30.0, 80.0),
            "alpha_fric_tot": (500.0, 900.0),
        }
    else:
        raise ValueError("Unknown model. Use original | modified")

    missing = [name for name in param_names if name not in by_name]
    if missing:
        raise ValueError(f"No bounds defined for parameters: {missing}")

    return np.array([by_name[name] for name in param_names], dtype=float)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="GA fit for Molinaroli models using operating_points_rows + split_template."
    )

    ap.add_argument("--op_rows_csv", required=True, type=Path, help="Path to operating_points_rows.csv")
    ap.add_argument("--split_csv", required=True, type=Path, help="Path to operating_points_split_template.csv")

    ap.add_argument("--oil", default="all", help="LPG68 | LPG100 | all")
    ap.add_argument("--model", default="original", choices=["original", "modified"])
    ap.add_argument("--refrigerant", default="PROPANE")

    ap.add_argument("--op_id_col", default=OP_ID_COL_DEFAULT)
    ap.add_argument("--oil_col", default=OIL_COL_DEFAULT)

    ap.add_argument("--source_row_col", default=SOURCE_ROW_COL_DEFAULT)
    ap.add_argument("--filtered_row_col", default=FILTERED_ROW_COL_DEFAULT)

    ap.add_argument("--split_role_col", default=SPLIT_ROLE_COL_DEFAULT)
    ap.add_argument("--shared_ok_col", default=SHARED_OK_COL_DEFAULT)
    ap.add_argument("--split_note_col", default=SPLIT_NOTE_COL_DEFAULT)

    ap.add_argument("--col_p_suc", default=P_SUC_COL_DEFAULT)
    ap.add_argument("--col_T_suc", default=T_SUC_COL_DEFAULT)
    ap.add_argument("--col_p_out", default=P_OUT_COL_DEFAULT)
    ap.add_argument("--col_T_amb", default=T_AMB_COL_DEFAULT)
    ap.add_argument("--col_speed", default=SPEED_COL_DEFAULT)
    ap.add_argument("--col_m_meas", default=M_FLOW_MEAS_COL_DEFAULT)
    ap.add_argument("--col_P_meas", default=P_EL_MEAS_COL_DEFAULT)
    ap.add_argument("--col_T_dis", default=T_DIS_MEAS_COL_DEFAULT)

    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)

    # GA settings
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--elite_frac", type=float, default=0.20)
    ap.add_argument("--random_keep_prob", type=float, default=0.10)
    ap.add_argument("--mutation_prob_param", type=float, default=0.20)
    ap.add_argument("--generations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--Tdis_norm_K", type=float, default=50.0)

    # Bounds scaling for V_IC
    ap.add_argument("--vic_lo_scale", type=float, default=0.95)
    ap.add_argument("--vic_hi_scale", type=float, default=1.05)

    ap.add_argument("--out_dir", default="results/ga_fit")

    # Parallelization
    ap.add_argument("--n_jobs", type=int, default=0, help="0=auto, 1=serial")
    ap.add_argument("--lsq_max_nfev", type=int, default=1000)
    ap.add_argument("--ind_timeout_s", type=float, default=120.0)

    args = ap.parse_args()

    if not args.op_rows_csv.exists():
        raise FileNotFoundError(args.op_rows_csv)
    if not args.split_csv.exists():
        raise FileNotFoundError(args.split_csv)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load and merge selection data
    # -------------------------
    op_rows_df = read_op_rows_csv(args.op_rows_csv)
    split_df = read_split_csv(args.split_csv)

    merged_df = prepare_merged_dataframe(op_rows_df, split_df, args)

    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = args.V_h_cm3 * 1e-6

    rows_all, has_Tdis = build_row_records(merged_df, args, N_max_hz)
    use_Tdis = bool(has_Tdis)

    rows_train = [r for r in rows_all if r["split_role"] == "train"]
    rows_validation = [r for r in rows_all if r["split_role"] == "validation"]
    rows_unused = [r for r in rows_all if r["split_role"] == ""]

    if not rows_train:
        raise ValueError(
            "No training points found. Fill the split_role column in split CSV with 'train'."
        )

    print(f"  rows_all         = {len(rows_all)}")
    print(f"  rows_train       = {len(rows_train)}")
    print(f"  rows_validation  = {len(rows_validation)}")
    print(f"  rows_unused      = {len(rows_unused)}")

    if not use_Tdis:
        print(f"[INFO] Column '{args.col_T_dis}' not found — T_dis not used in objective.")

    train_op_ids = sorted({r["op_id"] for r in rows_train})
    val_op_ids = sorted({r["op_id"] for r in rows_validation})

    # -------------------------
    # RefProp in main process
    # -------------------------
    med = RefProp(fluid_name=args.refrigerant)

    m_dot_ref = compute_m_dot_ref(med, V_h_m3)
    print(f"  m_dot_ref = {m_dot_ref * 1e3:.4f} g/s  (V_h={args.V_h_cm3} cm³, f_ref={F_REF} Hz)")

    # -------------------------
    # Parameter setup
    # -------------------------
    param_names = get_param_names(args.model)
    default_params = get_default_params(args.model)

    bounds = build_bounds(
        model=args.model,
        V_h_m3=V_h_m3,
        vic_lo_scale=args.vic_lo_scale,
        vic_hi_scale=args.vic_hi_scale,
        param_names=param_names,
    )

    # -------------------------
    # Initial population
    # -------------------------
    x0 = np.clip(
        np.array([default_params[name] for name in param_names], dtype=float),
        bounds[:, 0],
        bounds[:, 1],
    )

    pop_size = int(args.population)
    elite_k = max(1, int(np.ceil(args.elite_frac * pop_size)))

    rng = np.random.default_rng(args.seed)

    population: list[np.ndarray] = [x0.copy()]
    while len(population) < pop_size:
        population.append(random_individual(bounds, param_names, rng))

    # -------------------------
    # Population evaluation
    # -------------------------
    n_jobs = int(args.n_jobs)
    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 1

    if n_jobs == 1:

        def eval_pop(pop):
            errs = np.empty(len(pop), dtype=float)
            for i, ind in enumerate(pop):
                errs[i] = objective_error(
                    x=ind,
                    rows=rows_train,
                    med=med,
                    model=args.model,
                    refrigerant_name=args.refrigerant,
                    N_max_hz=N_max_hz,
                    V_h_m3=V_h_m3,
                    m_dot_ref=m_dot_ref,
                    param_names=param_names,
                    default_params=default_params,
                    use_Tdis=use_Tdis,
                    Tdis_norm_K=args.Tdis_norm_K,
                    lsq_max_nfev=args.lsq_max_nfev,
                )
            return errs

        executor = None

    else:

        def make_executor():
            return ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_init_worker,
                initargs=(
                    str(args.refrigerant),
                    str(args.model),
                    float(N_max_hz),
                    float(V_h_m3),
                    float(m_dot_ref),
                    rows_train,
                    param_names,
                    default_params,
                    bool(use_Tdis),
                    float(args.Tdis_norm_K),
                    int(args.lsq_max_nfev),
                ),
            )

        executor = make_executor()

        def eval_pop(pop):
            nonlocal executor

            futs = {executor.submit(_objective_error_worker, ind): i for i, ind in enumerate(pop)}
            errs = np.full(len(pop), 3.0 * FAIL_E ** 2, dtype=float)

            timeout_s = float(args.ind_timeout_s)
            if timeout_s <= 0:
                for f, i in futs.items():
                    try:
                        errs[i] = float(f.result())
                    except Exception:
                        errs[i] = 3.0 * FAIL_E ** 2
                return errs

            total_budget = timeout_s * len(pop)
            done, not_done = wait(futs.keys(), timeout=total_budget)

            for f in done:
                i = futs[f]
                try:
                    errs[i] = float(f.result())
                except Exception:
                    errs[i] = 3.0 * FAIL_E ** 2

            if not_done:
                print(f"[WARN] Total timeout hit: {len(not_done)} individuals penalized -> restarting pool")
                executor.shutdown(wait=False, cancel_futures=True)
                executor = make_executor()

            return errs

    print(f"  Parallelization: {'serial' if n_jobs == 1 else f'{n_jobs} worker processes'}")
    print(f"  Population: {pop_size}  |  Elites: {elite_k}  |  Generations: {args.generations}")

    # -------------------------
    # GA loop
    # -------------------------
    try:
        errors = eval_pop(population)
        best_x = population[int(np.argmin(errors))].copy()
        best_err = float(np.min(errors))
        print(f"[INIT] best_err={best_err:.6e}")

        best_snapshots: list[dict] = []

        for gen in range(1, args.generations + 1):
            order = np.argsort(errors)
            population = [population[i] for i in order]
            errors = errors[order]

            if float(errors[0]) < best_err:
                best_err = float(errors[0])
                best_x = population[0].copy()

            selected = list(population[:elite_k])
            selected_errs = list(errors[:elite_k].astype(float))

            for i in range(elite_k, pop_size):
                if rng.random() < args.random_keep_prob:
                    selected.append(population[i])
                    selected_errs.append(float(errors[i]))

            if len(selected) < 2:
                selected = population[:2]
                selected_errs = list(errors[:2].astype(float))

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
                        selected[i] = random_individual(bounds, param_names, rng)

            children: list[np.ndarray] = []
            while len(children) < pop_size - elite_k:
                p1 = selected[int(rng.integers(0, len(selected)))]
                p2 = selected[int(rng.integers(0, len(selected)))]
                child = mutate(
                    uniform_crossover(p1, p2, rng),
                    bounds,
                    param_names,
                    rng,
                    args.mutation_prob_param,
                )
                children.append(np.clip(child, bounds[:, 0], bounds[:, 1]))

            population = [population[i].copy() for i in range(elite_k)] + children
            errors = eval_pop(population)

            print(
                f"[GEN {gen:4d}] "
                f"best_gen={float(np.min(errors)):.6e}  "
                f"best_so_far={best_err:.6e}"
            )

            if gen % 10 == 0:
                snap = {"gen": int(gen), "best_err_so_far": float(best_err)}
                snap.update({name: float(val) for name, val in zip(param_names, best_x)})
                best_snapshots.append(snap)

                pretty = ", ".join([f"{k}={snap[k]:.6g}" for k in param_names])
                print(f"[BEST @ GEN {gen:4d}] err={best_err:.6e} | {pretty}")

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # -------------------------
    # Final prediction on all available rows
    # -------------------------
    tag = "ga"
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(args.out_dir)

    params_best = x_to_params(best_x, param_names, default_params)
    params_best["f_ref"] = F_REF
    params_best["m_dot_ref"] = float(m_dot_ref)

    runtime_pred = build_runtime_bundle(
        model=args.model,
        rows=rows_all,
        med=med,
        refrigerant_name=args.refrigerant,
        N_max_hz=N_max_hz,
        V_h_m3=V_h_m3,
        params=params_best,
        lsq_max_nfev=20000,
    )

    pred_rows = []
    for r in rows_all:
        ok = True
        try:
            entry = get_runtime_entry(runtime_pred, args.model, r)
            m_c, P_c, T_c = simulate_point(
                entry["comp"],
                med,
                entry["inputs"],
                entry["fs_state"],
                r["p_suc_pa"],
                r["T_suc_K"],
                r["p_out_pa"],
                r["n_rel"],
                r["T_amb_K"],
            )
        except Exception:
            ok, m_c, P_c, T_c = False, np.nan, np.nan, np.nan

        e_m_rel = (m_c / r["m_meas"] - 1.0) if ok else np.nan
        e_P_rel = (P_c / r["P_meas"] - 1.0) if ok else np.nan
        e_T_dis_K = (
            (T_c - r["T_dis_meas_K"])
            if (ok and r.get("T_dis_meas_K") is not None)
            else np.nan
        )

        james_m_sq = (e_m_rel ** 2) if np.isfinite(e_m_rel) else np.nan
        james_P_sq = (e_P_rel ** 2) if np.isfinite(e_P_rel) else np.nan
        james_T_sq = (
            (e_T_dis_K / float(args.Tdis_norm_K)) ** 2
            if np.isfinite(e_T_dis_K)
            else np.nan
        )

        james_terms = [james_m_sq, james_P_sq, james_T_sq]
        james_error_point = (
            float(np.nansum(james_terms))
            if any(np.isfinite(x) for x in james_terms)
            else np.nan
        )

        pred_rows.append(
            {
                "source_row_index": r["source_row_index"],
                "filtered_row_index": r["filtered_row_index"],
                "op_id": r["op_id"],
                "oil": r["oil_name"],
                "split_role": r["split_role"],
                "split_note": r.get("split_note", ""),
                "is_train": bool(r["split_role"] == "train"),
                "is_validation": bool(r["split_role"] == "validation"),
                "f_oper_hz": r["f_oper_hz"],
                "p_suc_bar": r["p_suc_pa"] / 1e5,
                "T_suc_C": r["T_suc_K"] - 273.15,
                "p_out_bar": r["p_out_pa"] / 1e5,
                "T_amb_C": r["T_amb_K"] - 273.15,
                "m_meas_gps": r["m_meas"] * 1e3,
                "m_calc_gps": m_c * 1e3 if ok else np.nan,
                "e_m_rel": e_m_rel,
                "P_meas_W": r["P_meas"],
                "P_calc_W": P_c if ok else np.nan,
                "e_P_rel": e_P_rel,
                "T_dis_meas_C": (r["T_dis_meas_K"] - 273.15) if r.get("T_dis_meas_K") is not None else np.nan,
                "T_dis_calc_C": (T_c - 273.15) if ok else np.nan,
                "e_T_dis_K": e_T_dis_K,
                "james_m_sq": james_m_sq,
                "james_P_sq": james_P_sq,
                "james_T_sq": james_T_sq,
                "james_error_point": james_error_point,
                "ok": ok,
            }
        )

    final_err = objective_error(
        x=best_x,
        rows=rows_train,
        med=med,
        model=args.model,
        refrigerant_name=args.refrigerant,
        N_max_hz=N_max_hz,
        V_h_m3=V_h_m3,
        m_dot_ref=m_dot_ref,
        param_names=param_names,
        default_params=default_params,
        use_Tdis=use_Tdis,
        Tdis_norm_K=args.Tdis_norm_K,
        lsq_max_nfev=args.lsq_max_nfev,
    )

    # -------------------------
    # Statistics
    # -------------------------
    df_pred = pd.DataFrame(pred_rows)
    df_pred_ok = df_pred[df_pred["ok"] == True].copy()

    m5 = (df_pred_ok["e_m_rel"].abs() <= 0.05).mean() * 100 if not df_pred_ok.empty else np.nan
    P5 = (df_pred_ok["e_P_rel"].abs() <= 0.05).mean() * 100 if not df_pred_ok.empty else np.nan

    print(f"\n  Points within ±5 % (mass flow): {m5:.1f} %")
    print(f"  Points within ±5 % (power):     {P5:.1f} %")

    if use_Tdis and "e_T_dis_K" in df_pred_ok.columns:
        T3 = (df_pred_ok["e_T_dis_K"].abs() <= 3.0).mean() * 100
        print(f"  Points within ±3 K (T_dis):    {T3:.1f} %")

    # -------------------------
    # Save results
    # -------------------------
    suffix = f"{str(args.oil).lower()}_{args.model}_{tag}_{run_id}"

    fitted_row = {k: float(v) for k, v in zip(param_names, best_x)}
    fitted_row.update(
        {
            "f_ref": F_REF,
            "T_ref": T_REF,
            "m_dot_ref": float(m_dot_ref),
            "m_dot_ref_definition": "rho_sat(T=273.15K,Q=1)*V_h_geo*f_ref",
            "oil": str(args.oil),
            "refrigerant": str(args.refrigerant),
            "model": str(args.model),
            "error_sum_sq": float(final_err),
            "n_train_rows": len(rows_train),
            "n_validation_rows": len(rows_validation),
            "n_total_rows": len(rows_all),
            "n_train_operating_points": len(train_op_ids),
            "n_validation_operating_points": len(val_op_ids),
            "use_Tdis": bool(use_Tdis),
            "Tdis_norm_K": float(args.Tdis_norm_K),
            "seed": int(args.seed),
            "population": pop_size,
            "elite_frac": float(args.elite_frac),
            "random_keep_prob": float(args.random_keep_prob),
            "mutation_prob_param": float(args.mutation_prob_param),
            "generations": int(args.generations),
            "n_jobs": n_jobs,
            "log_uniform_params": ",".join(sorted(LOG_UNIFORM_PARAM_NAMES)),
            "op_rows_csv": str(args.op_rows_csv),
            "split_csv": str(args.split_csv),
        }
    )

    out_params = out_dir / f"fitted_params_{suffix}.csv"
    out_pred = out_dir / f"fit_predictions_{suffix}.csv"

    pd.DataFrame([fitted_row]).to_csv(out_params, index=False)
    df_pred.to_csv(out_pred, index=False)

    if best_snapshots:
        out_snap = out_dir / f"best_params_snapshots_{suffix}.csv"
        pd.DataFrame(best_snapshots).to_csv(out_snap, index=False)
        print(f"  Best snapshots saved: {out_snap}")

    print(f"\n=== GA FIT DONE ===")
    print(f"  Final training error: {final_err:.6e}")
    print(f"  Parameters saved:     {out_params}")
    print(f"  Predictions saved:    {out_pred}")


if __name__ == "__main__":
    mp.freeze_support()
    main()