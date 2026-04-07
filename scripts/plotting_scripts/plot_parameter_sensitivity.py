# scripts/plotting_scripts/plot_parameter_sensitivity.py
#
# Parameter sensitivity analysis for Molinaroli compressor models.
#
# Evaluates how the James-style objective function changes when each fitted
# parameter is varied around its identified value (±r_min..r_max).
#
# Input: operating_points_rows.csv + operating_points_split_template.csv
#        + fitted params CSV
#
# Activate REFPROP:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Sensitivity on training points only (recommended):
#   python scripts/plotting_scripts/plot_parameter_sensitivity.py \
#       --op_rows_csv results/split_template/operating_points_rows.csv \
#       --split_csv results/split_template/operating_points_split_template.csv \
#       --params_csv results/ga_fit/fitted_params_lpg68_original_ga_2026-03-08.csv \
#       --oil LPG68 --selection_mode train_only
#
#   # Sensitivity on all points:
#   python scripts/plotting_scripts/plot_parameter_sensitivity.py \
#       --op_rows_csv results/split_template/operating_points_rows.csv \
#       --split_csv results/split_template/operating_points_split_template.csv \
#       --params_csv results/ga_fit/fitted_params_lpg68_modified_ga_2026-03-19.csv \
#       --oil LPG68 --model modified --selection_mode all

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vclibpy.media import RefProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import Molinaroli_2017_Compressor
from vclibpy.components.compressors.rolling_piston_Molinaroli_2017_modified import (
    Molinaroli_2017_Compressor_Modified,
)

plt.style.use("ebc.paper.mplstyle")


# =========================================================
# Constants
# =========================================================
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0
T_DIS_NORM_K = 50.0


# =========================================================
# Parameter definitions (must match GA fitting script)
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


# =========================================================
# Column defaults
# =========================================================
OP_ID_COL_DEFAULT = "op_id"
OIL_COL_DEFAULT = "Ölbezeichnung"
SPLIT_ROLE_COL_DEFAULT = "split_role"
SHARED_OK_COL_DEFAULT = "usable_for_shared_split"
SPLIT_NOTE_COL_DEFAULT = "split_note"

P_SUC_COL_DEFAULT = "P1_mean"
T_SUC_COL_DEFAULT = "T1_mean"
P_OUT_COL_DEFAULT = "P2_mean"
T_AMB_COL_DEFAULT = "Tamb_mean"
SPEED_COL_DEFAULT = "N"
T_DIS_MEAS_COL_DEFAULT = "T2_mean"
M_FLOW_MEAS_COL_DEFAULT = "suction_mf_mean"
P_EL_MEAS_COL_DEFAULT = "Pel_mean"

SOURCE_ROW_COL_DEFAULT = "source_row_index"
FILTERED_ROW_COL_DEFAULT = "filtered_row_index"


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


def parse_split_role(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s in {"", "unused"}:
        return ""
    if s in {"train", "training", "fit"}:
        return "train"
    if s in {"validation", "valid", "val", "test"}:
        return "validation"
    raise ValueError(f"Unsupported split_role value: {x!r}")


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


def _clamp01(x):
    return max(1e-9, min(1.0, float(x)))


# =========================================================
# Model helpers (consistent with GA and validation scripts)
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
# Parameter loading
# =========================================================
def load_params_csv(path: Path, model: str) -> tuple[dict, dict]:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError("Params CSV must contain exactly one row.")
    row = df.iloc[0].to_dict()

    param_names = get_param_names(model)
    default_params = get_default_params(model)

    params = dict(default_params)
    for k in param_names:
        if k in row and pd.notna(row[k]):
            params[k] = float(row[k])
    if "f_ref" in row and pd.notna(row["f_ref"]):
        params["f_ref"] = float(row["f_ref"])

    meta = {}
    for key in ["oil", "refrigerant", "model"]:
        if key in row:
            meta[key] = row[key]

    return params, meta


# =========================================================
# Data loading (op_rows_csv + split_csv)
# =========================================================
def load_and_merge(args) -> pd.DataFrame:
    op_df = pd.read_csv(args.op_rows_csv)
    sp_df = pd.read_csv(args.split_csv)

    sp_df[args.split_role_col] = sp_df[args.split_role_col].apply(parse_split_role)
    if args.shared_ok_col in sp_df.columns:
        sp_df[args.shared_ok_col] = sp_df[args.shared_ok_col].apply(parse_bool_like)
    else:
        sp_df[args.shared_ok_col] = True

    merge_cols = [args.op_id_col, args.split_role_col, args.shared_ok_col]
    if args.split_note_col in sp_df.columns:
        merge_cols.append(args.split_note_col)

    merged = op_df.merge(sp_df[merge_cols], on=args.op_id_col, how="left")
    merged["_oil_norm"] = merged[args.oil_col].astype(str).map(norm_oil)

    # Filter by oil
    oil_sel = norm_oil(args.oil)
    if oil_sel not in {"lpg68", "lpg100", "all"}:
        raise ValueError("--oil must be LPG68, LPG100 or all")

    if oil_sel != "all":
        merged = merged[merged["_oil_norm"] == oil_sel].copy()
    else:
        merged = merged[merged[args.shared_ok_col] == True].copy()
        opid_oil_counts = merged.groupby(args.op_id_col)["_oil_norm"].nunique()
        valid_shared_opids = opid_oil_counts[opid_oil_counts >= 2].index
        merged = merged[merged[args.op_id_col].isin(valid_shared_opids)].copy()

    # Ensure numeric
    numeric_cols = [
        args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_T_amb,
        args.col_speed, args.col_m_meas, args.col_P_meas,
    ]
    if args.col_T_dis in merged.columns:
        numeric_cols.append(args.col_T_dis)

    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(
        subset=[args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_T_amb,
                args.col_speed, args.col_m_meas, args.col_P_meas]
    ).reset_index(drop=True)

    if merged.empty:
        raise ValueError("No usable rows left after merging and filtering.")

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

        rows.append({
            "op_id": str(r[args.op_id_col]),
            "oil_name": str(r[args.oil_col]),
            "oil_norm": norm_oil(r[args.oil_col]),
            "split_role": parse_split_role(r.get(args.split_role_col, "")),
            "p_suc_pa": bar_to_pa(r[args.col_p_suc]),
            "T_suc_K": c_to_k(r[args.col_T_suc]),
            "p_out_pa": bar_to_pa(r[args.col_p_out]),
            "T_amb_K": c_to_k(r[args.col_T_amb]),
            "f_oper_hz": rpm_to_hz(r[args.col_speed]),
            "n_rel": _clamp01(rpm_to_hz(r[args.col_speed]) / N_max_hz),
            "m_meas": m_meas,
            "P_meas": P_meas,
            "T_dis_meas_K": float(T_dis_meas_K) if T_dis_meas_K is not None else None,
        })

    if not rows:
        raise ValueError("No valid rows could be built from merged data.")

    return rows, has_Tdis


def select_rows(rows: list[dict], mode: str) -> list[dict]:
    m = str(mode).lower().strip()
    if m == "all":
        selected = rows
    elif m in {"train_only", "train"}:
        selected = [r for r in rows if r["split_role"] == "train"]
    elif m in {"validation_only", "validation"}:
        selected = [r for r in rows if r["split_role"] == "validation"]
    else:
        raise ValueError("Unknown --selection_mode. Use: train_only | validation_only | all")

    if not selected:
        raise ValueError(f"No rows selected for selection_mode='{mode}'.")

    return selected


# =========================================================
# Runtime bundle (same pattern as GA / validation scripts)
# =========================================================
def build_runtime_bundle(
    model: str,
    rows: list[dict],
    med,
    refrigerant_name: str,
    N_max_hz: float,
    V_h_m3: float,
    params: dict,
):
    m = str(model).lower().strip()
    bundle = {}

    if m in ("orig", "original"):
        comp = make_compressor(
            model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            params=params, refrigerant_name=refrigerant_name, oil_name=None,
        )
        comp.med_prop = med
        if hasattr(comp, "debug_enabled"):
            comp.debug_enabled = False

        bundle["single"] = {
            "comp": comp,
            "inputs": SimpleInputs(control=Control(n=1e-6), T_amb=298.15),
            "fs_state": FlowsheetState(),
        }
        return bundle

    unique_oils = sorted({r["oil_name"] for r in rows})
    for oil_name in unique_oils:
        comp = make_compressor(
            model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            params=params, refrigerant_name=refrigerant_name, oil_name=oil_name,
        )
        comp.med_prop = med
        if hasattr(comp, "debug_enabled"):
            comp.debug_enabled = False

        bundle[norm_oil(oil_name)] = {
            "comp": comp,
            "inputs": SimpleInputs(control=Control(n=1e-6), T_amb=298.15),
            "fs_state": FlowsheetState(),
        }

    return bundle


def get_bundle_entry(bundle: dict, model: str, row: dict):
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return bundle["single"]
    return bundle[row["oil_norm"]]


# =========================================================
# Objective function (James-style, using runtime bundle)
# =========================================================
def objective_g(
    rows: list[dict],
    med,
    model: str,
    refrigerant_name: str,
    params: dict,
    N_max_hz: float,
    V_h_m3: float,
    use_Tdis: bool,
    fail_penalty: float = 10.0,
) -> tuple[float, int, int, int]:
    """
    Returns: (g, n_fail, n_total, n_warn_total)
    """
    runtime = build_runtime_bundle(
        model=model, rows=rows, med=med,
        refrigerant_name=refrigerant_name,
        N_max_hz=N_max_hz, V_h_m3=V_h_m3, params=params,
    )

    em2 = 0.0
    eW2 = 0.0
    eT2 = 0.0
    n_fail = 0
    n_warn_total = 0
    n = 0

    for row in rows:
        try:
            entry = get_bundle_entry(runtime, model, row)
            comp = entry["comp"]
            inputs = entry["inputs"]
            fs_state = entry["fs_state"]

            inputs.control.n = _clamp01(row["n_rel"])
            inputs.T_amb = float(row["T_amb_K"])

            with warnings.catch_warnings(record=True) as wrec:
                warnings.simplefilter("always")
                comp.state_inlet = med.calc_state("PT", float(row["p_suc_pa"]), float(row["T_suc_K"]))
                comp.calc_state_outlet(p_outlet=float(row["p_out_pa"]), inputs=inputs, fs_state=fs_state)

            n_warn_total += len(wrec)

            m_c = float(comp.m_flow)
            P_c = float(comp.P_el)
            T_c = float(comp.state_outlet.T)

            if not np.isfinite(m_c) or m_c <= 0:
                raise ValueError("Invalid m_flow")
            if not np.isfinite(P_c) or P_c <= 0:
                raise ValueError("Invalid P_el")

            em = (m_c / row["m_meas"]) - 1.0
            eW = (P_c / row["P_meas"]) - 1.0

            em2 += em * em
            eW2 += eW * eW

            if use_Tdis and row.get("T_dis_meas_K") is not None:
                eT = (T_c - row["T_dis_meas_K"]) / T_DIS_NORM_K
                eT2 += eT * eT

        except Exception:
            n_fail += 1
            em2 += fail_penalty ** 2
            eW2 += fail_penalty ** 2
            if use_Tdis:
                eT2 += fail_penalty ** 2

        n += 1

    if n == 0:
        return float("inf"), 0, 0, 0

    g = float((em2 + eW2 + eT2) / n)
    return g, int(n_fail), int(n), int(n_warn_total)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Parameter sensitivity analysis (James objective) using op_rows + split_template."
    )

    # --- Input ---
    ap.add_argument("--op_rows_csv", required=True, type=Path, help="Path to operating_points_rows.csv")
    ap.add_argument("--split_csv", required=True, type=Path, help="Path to operating_points_split_template.csv")
    ap.add_argument("--params_csv", required=True, type=Path, help="One-row fitted parameter CSV")

    # --- Model / fluid / oil ---
    ap.add_argument("--model", default="auto", help="original | modified | auto (from params_csv)")
    ap.add_argument("--refrigerant", default="auto", help="RefProp fluid or auto (from params_csv)")
    ap.add_argument("--oil", default="auto", help="LPG68 | LPG100 | all | auto (from params_csv)")

    # --- Selection ---
    ap.add_argument(
        "--selection_mode",
        default="train_only",
        choices=["train_only", "validation_only", "all"],
        help="Which points to use for sensitivity (default: train_only)",
    )

    # --- Compressor geometry ---
    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)

    # --- Sweep settings ---
    ap.add_argument("--r_min", type=float, default=0.95, help="Min ratio (param/identified)")
    ap.add_argument("--r_max", type=float, default=1.05, help="Max ratio (param/identified)")
    ap.add_argument("--n_points", type=int, default=11, help="Number of ratio points")
    ap.add_argument("--fail_penalty", type=float, default=10.0)
    ap.add_argument(
        "--mask_if_fails", action="store_true",
        help="Set g_norm to NaN for ratio points where n_fail > 0",
    )

    # --- Column names ---
    ap.add_argument("--op_id_col", default=OP_ID_COL_DEFAULT)
    ap.add_argument("--oil_col", default=OIL_COL_DEFAULT)
    ap.add_argument("--split_role_col", default=SPLIT_ROLE_COL_DEFAULT)
    ap.add_argument("--shared_ok_col", default=SHARED_OK_COL_DEFAULT)
    ap.add_argument("--split_note_col", default=SPLIT_NOTE_COL_DEFAULT)
    ap.add_argument("--col_p_suc", default=P_SUC_COL_DEFAULT)
    ap.add_argument("--col_T_suc", default=T_SUC_COL_DEFAULT)
    ap.add_argument("--col_p_out", default=P_OUT_COL_DEFAULT)
    ap.add_argument("--col_T_amb", default=T_AMB_COL_DEFAULT)
    ap.add_argument("--col_speed", default=SPEED_COL_DEFAULT)
    ap.add_argument("--col_T_dis", default=T_DIS_MEAS_COL_DEFAULT)
    ap.add_argument("--col_m_meas", default=M_FLOW_MEAS_COL_DEFAULT)
    ap.add_argument("--col_P_meas", default=P_EL_MEAS_COL_DEFAULT)

    # --- Output ---
    ap.add_argument("--out_dir", default="results/sensitivity", help="Output folder")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    # -------------------------
    # Validate files
    # -------------------------
    if not args.op_rows_csv.exists():
        raise FileNotFoundError(args.op_rows_csv)
    if not args.split_csv.exists():
        raise FileNotFoundError(args.split_csv)
    if not args.params_csv.exists():
        raise FileNotFoundError(args.params_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Resolve auto values from params CSV
    # -------------------------
    params_peek = pd.read_csv(args.params_csv).iloc[0].to_dict()

    if args.model == "auto":
        args.model = str(params_peek.get("model", "original"))
    if args.refrigerant == "auto":
        args.refrigerant = str(params_peek.get("refrigerant", "PROPANE"))
    if args.oil == "auto":
        args.oil = str(params_peek.get("oil", "all"))

    params_base, params_meta = load_params_csv(args.params_csv, args.model)
    param_names = get_param_names(args.model)

    # -------------------------
    # Load and merge data
    # -------------------------
    merged_df = load_and_merge(args)

    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6

    rows_all, has_Tdis = build_row_records(merged_df, args, N_max_hz)
    use_Tdis = bool(has_Tdis)

    # Select rows
    rows_selected = select_rows(rows_all, args.selection_mode)

    n_train = len([r for r in rows_selected if r["split_role"] == "train"])
    n_val = len([r for r in rows_selected if r["split_role"] == "validation"])

    print(f"  Model:            {args.model}")
    print(f"  Params oil:       {params_meta.get('oil', 'unknown')}")
    print(f"  Data oil:         {args.oil}")
    print(f"  Selection mode:   {args.selection_mode}")
    print(f"  Selected rows:    {len(rows_selected)} (train={n_train}, validation={n_val})")
    print(f"  Use T_dis:        {use_Tdis}")

    # -------------------------
    # RefProp (single shared instance)
    # -------------------------
    med = RefProp(fluid_name=args.refrigerant)

    params_base["f_ref"] = F_REF
    params_base["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)
    print(f"  m_dot_ref:        {params_base['m_dot_ref'] * 1e3:.4f} g/s")

    # -------------------------
    # Baseline objective
    # -------------------------
    g_min, fail_min, n_tot, warn_min = objective_g(
        rows=rows_selected, med=med, model=args.model,
        refrigerant_name=args.refrigerant, params=params_base,
        N_max_hz=N_max_hz, V_h_m3=V_h_m3, use_Tdis=use_Tdis,
        fail_penalty=args.fail_penalty,
    )

    if not np.isfinite(g_min) or g_min <= 0:
        raise RuntimeError(f"Invalid g_min={g_min}. Check model stability / data / parameters.")

    print(f"  g_min (baseline): {g_min:.6e}")
    if fail_min > 0:
        print(f"  [WARN] Baseline has {fail_min}/{n_tot} failed points — normalization may be affected.")
    if warn_min > 0:
        print(f"  [INFO] Baseline RefProp warnings: {warn_min}")

    # Meta for output CSV
    csv_meta = {
        "params_csv": args.params_csv.name,
        "model": args.model,
        "oil": args.oil,
        "selection_mode": args.selection_mode,
        "n_selected": len(rows_selected),
        "g_min": float(g_min),
        "fail_min": int(fail_min),
    }
    csv_meta.update({f"base_{k}": float(params_base[k]) for k in param_names})

    # -------------------------
    # Sensitivity sweep
    # -------------------------
    ratios = np.linspace(args.r_min, args.r_max, args.n_points)

    records = []
    fig, ax = plt.subplots()

    for pname in param_names:
        p0 = float(params_base[pname])
        y = []

        for r in ratios:
            p = dict(params_base)
            p[pname] = p0 * float(r)

            g, n_fail, n_total, n_warn = objective_g(
                rows=rows_selected, med=med, model=args.model,
                refrigerant_name=args.refrigerant, params=p,
                N_max_hz=N_max_hz, V_h_m3=V_h_m3, use_Tdis=use_Tdis,
                fail_penalty=args.fail_penalty,
            )

            if args.mask_if_fails and n_fail > 0:
                g_norm = np.nan
            else:
                g_norm = (g / g_min) if np.isfinite(g) else np.nan

            y.append(g_norm)

            rec = dict(csv_meta)
            rec.update({
                "param": pname,
                "ratio": float(r),
                "param_value": p0 * float(r),
                "g": float(g) if np.isfinite(g) else np.nan,
                "g_norm": float(g_norm) if np.isfinite(g_norm) else np.nan,
                "n_fail": int(n_fail),
                "n_warn": int(n_warn),
            })
            records.append(rec)

        ax.plot(ratios, y, marker="o", label=pname)

    # -------------------------
    # Plot formatting
    # -------------------------
    ax.set_xlabel("Actual parameter / Identified parameter [-]")
    ax.set_ylabel("$g / g_{min}$ [-]")

    title = f"Sensitivity | {args.oil} | {args.refrigerant} | {args.model}"
    title += f"\n{args.selection_mode} ({len(rows_selected)} points)"
    ax.set_title(title)

    ax.legend(loc="best", frameon=True)

    # -------------------------
    # Save
    # -------------------------
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = f"{norm_oil(args.oil)}_{args.model}_{args.selection_mode}_{stamp}"

    out_plot = out_dir / f"sensitivity_{suffix}.{args.out_format}"
    out_csv = out_dir / f"sensitivity_{suffix}.csv"

    fig.savefig(out_plot, dpi=300)
    plt.close(fig)

    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)

    print(f"\n  Plot saved:  {out_plot}")
    print(f"  Data saved:  {out_csv}")

    if args.mask_if_fails:
        print("  [INFO] mask_if_fails active: points with n_fail > 0 shown as NaN.")


if __name__ == "__main__":
    main()
