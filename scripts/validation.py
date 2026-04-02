# scripts/validation.py
#
# Validation script for Molinaroli compressor models.
#
# Supports: original | modified | oil_path
#
# Supports two input modes:
#   A) New mode (like GA fitting script):
#      --op_rows_csv + --split_csv
#   B) Legacy mode:
#      --csv with --sep, --header, --decimal + optional --split_csv
#
# Cross-validation:
#   --oil controls which DATA rows are used for validation (LPG68 | LPG100 | all).
#   --params_csv provides the fitted parameters (can be from any oil).
#   This allows cross-validation, e.g. params fitted on LPG68, validated on LPG100.
#
# Example (new mode, oil_path model):
#   python scripts/validation.py --op_rows_csv results/split_template/operating_points_rows_2026-03-12_112331.csv --split_csv results/split_template/operating_points_split_template_2026-03-12_112331.csv --params_csv results/ga_fit/fitted_params_lpg68_oil_path_ga_2026-03-26.csv --model oil_path --oil LPG68 --selection_mode all
#
# Example (new mode, cross-validation: params from LPG68, validate on LPG100):
#   python scripts/validation.py --op_rows_csv results/split_template/operating_points_rows_2026-03-12_112331.csv --split_csv results/split_template/operating_points_split_template_2026-03-12_112331.csv --params_csv results/ga_fit/fitted_params_lpg68_modified_ga_2026-03-18_143802.csv --model modified --oil LPG68 --selection_mode all

from __future__ import annotations

import argparse
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
from vclibpy.components.compressors.rolling_piston_Molinaroli_oil_path import (
    Molinaroli_2017_Compressor_Oil_Path,
)


# =========================================================
# Constants
# =========================================================
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0

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

PARAM_NAMES_OIL_PATH = [
    "Ua_suc_ref",
    "Ua_dis_ref",
    "Ua_amb",
    "A_tot",
    "A_dis",
    "V_IC",
    "alpha_loss",
    "W_dot_loss_ref",
    "alpha_fric_tot",
    "m_dot_oil_ref",
    "Ua_suc_oil_ref",
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

DEFAULT_PARAMS_OIL_PATH = {
    "Ua_suc_ref": 16.05,
    "Ua_dis_ref": 13.96,
    "Ua_amb": 0.36,
    "A_tot": 9.47e-9,
    "A_dis": 86.1e-6,
    "V_IC": 30.7e-6,
    "alpha_loss": 0.16,
    "W_dot_loss_ref": 10.0,
    "alpha_fric_tot": 120.0,
    "m_dot_oil_ref": 0.001,
    "Ua_suc_oil_ref": 5.0,
    "mu_fallback": 5.0,
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
T_OIL_SUMP_COL_DEFAULT = "T7_mean"
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


def pa_to_bar(p):
    return float(p) / 1e5


def c_to_k(t):
    return float(t) + 273.15


def k_to_c(t):
    return float(t) - 273.15


def rpm_to_hz(n):
    return float(n) / 60.0


def gs_to_kgps(m):
    return float(m) / 1000.0


# =========================================================
# Small helpers
# =========================================================
def norm_oil(s: str) -> str:
    return str(s).strip().lower().replace(" ", "")


def _finite(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _model_key(model: str) -> str:
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return "original"
    if m in ("mod", "modified"):
        return "modified"
    if m in ("oil_path", "oilpath", "oil"):
        return "oil_path"
    raise ValueError("Unknown model. Use: original | modified | oil_path")


def _model_needs_oil(model: str) -> bool:
    return _model_key(model) in ("modified", "oil_path")


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
# Model helpers
# =========================================================
def map_refrigerant_for_oil_model(name: str) -> str:
    s = str(name).strip().upper()
    if s in {"PROPANE", "R290", "PROPAN"}:
        return "propane"
    return str(name).strip()


def map_oil_for_oil_model(name: str) -> str:
    s = norm_oil(name)
    if s == "lpg68":
        return "LPG 68"
    if s == "lpg100":
        return "LPG 100"
    raise ValueError(f"Unsupported oil: {name}")


def get_param_names(model: str) -> list[str]:
    k = _model_key(model)
    if k == "original":
        return list(PARAM_NAMES_ORIGINAL)
    if k == "modified":
        return list(PARAM_NAMES_MODIFIED)
    if k == "oil_path":
        return list(PARAM_NAMES_OIL_PATH)


def get_default_params(model: str) -> dict:
    k = _model_key(model)
    if k == "original":
        return dict(DEFAULT_PARAMS_ORIGINAL)
    if k == "modified":
        return dict(DEFAULT_PARAMS_MODIFIED)
    if k == "oil_path":
        return dict(DEFAULT_PARAMS_OIL_PATH)


def make_compressor(
    model: str,
    N_max_hz: float,
    V_h_m3: float,
    params: dict,
    refrigerant_name: str,
    oil_name: str | None = None,
):
    k = _model_key(model)

    if k == "original":
        return Molinaroli_2017_Compressor(
            N_max=N_max_hz,
            V_h=V_h_m3,
            parameters=params,
        )

    if k == "modified":
        if oil_name is None:
            raise ValueError("Modified model requires an oil name.")
        return Molinaroli_2017_Compressor_Modified(
            N_max=N_max_hz,
            V_h=V_h_m3,
            fluid_name=map_refrigerant_for_oil_model(refrigerant_name),
            lub_name=map_oil_for_oil_model(oil_name),
            parameters=params,
        )

    if k == "oil_path":
        if oil_name is None:
            raise ValueError("Oil path model requires an oil name.")
        return Molinaroli_2017_Compressor_Oil_Path(
            N_max=N_max_hz,
            V_h=V_h_m3,
            fluid_name=map_refrigerant_for_oil_model(refrigerant_name),
            lub_name=map_oil_for_oil_model(oil_name),
            parameters=params,
        )


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
    """Load fitted parameters and metadata from a one-row CSV."""
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
    for key in [
        "oil", "refrigerant", "model", "error_sum_sq",
        "n_train_rows", "n_validation_rows", "n_total_rows",
        "n_train_operating_points", "n_validation_operating_points",
        "use_Tdis", "Tdis_norm_K", "seed", "population",
        "elite_frac", "random_keep_prob", "mutation_prob_param",
        "generations", "n_jobs",
        "n_train", "n_points_total",
    ]:
        if key in row:
            meta[key] = row[key]

    return params, meta


# =========================================================
# Data loading: NEW mode (op_rows_csv + split_csv)
# =========================================================
def load_new_mode(args) -> pd.DataFrame:
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

    numeric_cols = [
        args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_T_amb,
        args.col_speed, args.col_m_meas, args.col_P_meas,
    ]
    if args.col_T_dis in merged.columns:
        numeric_cols.append(args.col_T_dis)
    if args.col_T_oil_sump in merged.columns:
        numeric_cols.append(args.col_T_oil_sump)

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


# =========================================================
# Data loading: LEGACY mode (raw CSV + optional split)
# =========================================================
def load_legacy_mode(args) -> pd.DataFrame:
    df = pd.read_csv(args.csv, sep=args.sep, header=args.header, decimal=args.decimal)

    if args.source_row_col not in df.columns:
        df.insert(0, args.source_row_col, np.arange(len(df), dtype=int))

    oil_sel = norm_oil(args.oil)
    if oil_sel != "all":
        if args.oil_col not in df.columns:
            raise ValueError(f"Oil column '{args.oil_col}' not found in CSV.")
        df = df[df[args.oil_col].astype(str).apply(norm_oil) == oil_sel].copy()

    required = [args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_T_amb,
                args.col_speed, args.col_m_meas, args.col_P_meas]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)

    if args.filtered_row_col not in df.columns:
        df.insert(1, args.filtered_row_col, np.arange(len(df), dtype=int))

    if args.split_csv is not None:
        split_df = pd.read_csv(args.split_csv)

        if "is_train" in split_df.columns:
            idx_col = args.legacy_split_idx_col
            if idx_col in split_df.columns:
                tmp = split_df[[idx_col, "is_train"]].copy()
                tmp["is_train"] = tmp["is_train"].apply(parse_bool_like)

                split_idx_numeric = pd.to_numeric(tmp[idx_col], errors="coerce")
                if split_idx_numeric.notna().all():
                    max_idx = int(split_idx_numeric.max())
                    merge_key = args.filtered_row_col if max_idx < len(df) else args.source_row_col
                else:
                    merge_key = args.filtered_row_col

                df = df.merge(
                    tmp.rename(columns={idx_col: merge_key}),
                    on=merge_key,
                    how="left",
                )

            if "is_train" in df.columns:
                df[args.split_role_col] = df["is_train"].apply(
                    lambda x: "train" if x else "validation"
                )
            else:
                df[args.split_role_col] = ""

        elif args.split_role_col in split_df.columns:
            if args.op_id_col in split_df.columns and args.op_id_col in df.columns:
                sp = split_df[[args.op_id_col, args.split_role_col]].copy()
                sp[args.split_role_col] = sp[args.split_role_col].apply(parse_split_role)
                df = df.merge(sp, on=args.op_id_col, how="left")
            else:
                df[args.split_role_col] = ""
        else:
            df[args.split_role_col] = ""
    else:
        df[args.split_role_col] = ""

    if args.split_role_col in df.columns:
        df[args.split_role_col] = df[args.split_role_col].fillna("")

    df["_oil_norm"] = df[args.oil_col].astype(str).map(norm_oil) if args.oil_col in df.columns else ""

    if df.empty:
        raise ValueError("No valid rows after filtering.")

    return df


# =========================================================
# Row selection by split_role
# =========================================================
def select_rows(df: pd.DataFrame, mode: str, split_role_col: str) -> pd.DataFrame:
    m = str(mode).lower().strip()

    if m == "all":
        out = df.copy()
    elif m in {"validation_only", "exclude_train"}:
        out = df[df[split_role_col] == "validation"].copy()
    elif m == "train_only":
        out = df[df[split_role_col] == "train"].copy()
    else:
        raise ValueError("Unknown --selection_mode. Use: validation_only | train_only | all")

    out = out.reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No rows selected for selection_mode='{mode}'.")
    return out


# =========================================================
# Runtime bundle
# =========================================================
def build_validation_bundle(
    model: str,
    oil_names: list[str],
    med,
    refrigerant_name: str,
    N_max_hz: float,
    V_h_m3: float,
    params: dict,
):
    k = _model_key(model)
    bundle = {}

    if k == "original":
        comp = make_compressor(
            model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            params=params, refrigerant_name=refrigerant_name, oil_name=None,
        )
        comp.med_prop = med
        if hasattr(comp, "debug_enabled"):
            comp.debug_enabled = True

        bundle["single"] = {
            "comp": comp,
            "inputs": SimpleInputs(control=Control(n=1e-6), T_amb=298.15),
            "fs_state": FlowsheetState(),
        }
        return bundle

    # modified and oil_path: one compressor per oil
    for oil_name in oil_names:
        comp = make_compressor(
            model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            params=params, refrigerant_name=refrigerant_name, oil_name=oil_name,
        )
        comp.med_prop = med
        if hasattr(comp, "debug_enabled"):
            comp.debug_enabled = True

        bundle[norm_oil(oil_name)] = {
            "comp": comp,
            "inputs": SimpleInputs(control=Control(n=1e-6), T_amb=298.15),
            "fs_state": FlowsheetState(),
        }

    return bundle


def get_bundle_entry(bundle: dict, model: str, oil_norm: str):
    k = _model_key(model)
    if k == "original":
        return bundle["single"]
    return bundle[oil_norm]


# =========================================================
# State / loss extraction helpers
# =========================================================
def _add_compact_state(rec: dict, prefix: str, st) -> None:
    if st is None:
        rec[f"{prefix}_p_bar"] = float("nan")
        rec[f"{prefix}_T_C"] = float("nan")
        rec[f"{prefix}_rho_kgpm3"] = float("nan")
        return

    p = _finite(getattr(st, "p", np.nan))
    T = _finite(getattr(st, "T", np.nan))
    rho = _finite(getattr(st, "d", np.nan))

    rec[f"{prefix}_p_bar"] = pa_to_bar(p) if np.isfinite(p) else float("nan")
    rec[f"{prefix}_T_C"] = k_to_c(T) if np.isfinite(T) else float("nan")
    rec[f"{prefix}_rho_kgpm3"] = rho


def _extract_internal_states(rec: dict, comp) -> None:
    for prefix, attr in [
        ("st_in", "state_inlet"),
        ("c1", "state_c_1"),
        ("c3", "state_c_3"),
        ("c4", "state_c_4"),
        ("c5", "state_c_5"),
        ("st_out", "state_outlet"),
    ]:
        _add_compact_state(rec, prefix, getattr(comp, attr, None))


def _extract_loss_terms(rec: dict, comp, model: str) -> None:
    k = _model_key(model)

    rec["W_dot_int_W"] = _finite(getattr(comp, "W_dot_int", np.nan))
    rec["W_dot_loss_W"] = _finite(getattr(comp, "W_dot_loss", np.nan))

    W_int = rec["W_dot_int_W"]
    W_loss = rec["W_dot_loss_W"]
    W_total = W_int + W_loss if (np.isfinite(W_int) and np.isfinite(W_loss)) else float("nan")
    rec["W_dot_int_plus_loss_W"] = W_total
    rec["W_dot_loss_share"] = (W_loss / W_total) if (np.isfinite(W_total) and W_total > 0) else float("nan")

    rec["W_dot_loss_load_W"] = _finite(getattr(comp, "W_dot_loss_load", np.nan))
    rec["W_dot_loss_ref_term_W"] = _finite(getattr(comp, "W_dot_loss_ref_term", np.nan))

    T_wall = _finite(getattr(comp, "T_w", np.nan))
    rec["T_wall_C"] = k_to_c(T_wall) if np.isfinite(T_wall) else float("nan")

    # Extended terms for modified and oil_path
    if k in ("modified", "oil_path"):
        rec["W_dot_loss_fric_W"] = _finite(getattr(comp, "W_dot_loss_fric", np.nan))
        rec["T_oil_sump_calc_C"] = k_to_c(_finite(getattr(comp, "T_oil_sump", np.nan)))
        rec["mu_oil_mPas"] = _finite(getattr(comp, "mu_oil", np.nan))
        rec["mu_mix_eff_Pas"] = _finite(getattr(comp, "mu_mix_eff", np.nan))

    # Oil path specific terms
    if k == "oil_path":
        rec["W_dot_oil_recirc_W"] = _finite(getattr(comp, "W_dot_oil_recirc", np.nan))
        rec["m_dot_oil_kg_s"] = _finite(getattr(comp, "m_dot_oil", np.nan))
        rec["m_dot_fl_kg_s"] = _finite(getattr(comp, "m_dot_fl", np.nan))
        rec["m_dot_gas_discharge_kg_s"] = _finite(getattr(comp, "m_dot_gas_discharge", np.nan))

        rec["Q_suc_oil_W"] = _finite(getattr(comp, "Q_dot_suc_oil", np.nan))
        rec["Q_dis_total_W"] = _finite(getattr(comp, "Q_dis_total", np.nan))

        rec["m_dot_KM_degas_thr_kg_s"] = _finite(getattr(comp, "m_dot_KM_degas_thr", np.nan))
        rec["m_dot_KM_degas_ht_kg_s"] = _finite(getattr(comp, "m_dot_KM_degas_ht", np.nan))
        rec["m_dot_KM_degas_total_kg_s"] = _finite(getattr(comp, "m_dot_KM_degas_total", np.nan))
        rec["m_dot_KM_degas_ht_raw_kg_s"] = _finite(getattr(comp, "m_dot_KM_degas_ht_raw", np.nan))

        rec["w_KM_sump"] = _finite(getattr(comp, "w_KM_sump", np.nan))
        rec["w_KM_after"] = _finite(getattr(comp, "w_KM_after", np.nan))
        rec["w_KM_after_raw"] = _finite(getattr(comp, "w_KM_after_raw", np.nan))

        T_oil_after = _finite(getattr(comp, "T_oil_after", np.nan))
        rec["T_oil_after_C"] = k_to_c(T_oil_after) if np.isfinite(T_oil_after) else float("nan")

        T_dis_est = _finite(getattr(comp, "T_dis_est", np.nan))
        T_dis_corr = _finite(getattr(comp, "T_dis_corr", np.nan))
        rec["T_dis_est_C"] = k_to_c(T_dis_est) if np.isfinite(T_dis_est) else float("nan")
        rec["T_dis_corr_C"] = k_to_c(T_dis_corr) if np.isfinite(T_dis_corr) else float("nan")

        m_KM_gas = _finite(getattr(comp, "m_dot_KM_gas", np.nan))
        T_KM_gas = _finite(getattr(comp, "T_KM_gas", np.nan))
        rec["m_dot_KM_gas_diag_kg_s"] = m_KM_gas
        rec["T_KM_gas_diag_C"] = k_to_c(T_KM_gas) if np.isfinite(T_KM_gas) else float("nan")


def _fill_nan_loss_terms(rec: dict, model: str) -> None:
    k = _model_key(model)

    for col in [
        "W_dot_int_W", "W_dot_loss_W", "W_dot_int_plus_loss_W", "W_dot_loss_share",
        "W_dot_loss_load_W", "W_dot_loss_ref_term_W", "T_wall_C",
    ]:
        rec[col] = float("nan")

    if k in ("modified", "oil_path"):
        for col in ["W_dot_loss_fric_W", "T_oil_sump_calc_C", "mu_oil_mPas", "mu_mix_eff_Pas"]:
            rec[col] = float("nan")

    if k == "oil_path":
        oil_nan_cols = [
            "W_dot_oil_recirc_W",
            "m_dot_oil_kg_s", "m_dot_fl_kg_s", "m_dot_gas_discharge_kg_s",
            "Q_suc_oil_W", "Q_dis_total_W",
            "m_dot_KM_degas_thr_kg_s", "m_dot_KM_degas_ht_kg_s",
            "m_dot_KM_degas_total_kg_s", "m_dot_KM_degas_ht_raw_kg_s",
            "w_KM_sump", "w_KM_after", "w_KM_after_raw",
            "T_oil_after_C", "T_dis_est_C", "T_dis_corr_C",
            "m_dot_KM_gas_diag_kg_s", "T_KM_gas_diag_C",
        ]
        for col in oil_nan_cols:
            rec[col] = float("nan")


# =========================================================
# Summary statistics
# =========================================================
def summarize_results(out_df: pd.DataFrame, args, params_meta: dict) -> pd.DataFrame:
    ok = out_df[out_df["success"]].copy()

    def _stat(col, func):
        if col in ok.columns and ok[col].notna().any():
            return float(func(ok[col].dropna()))
        return np.nan

    def _share(col, thr):
        if col in ok.columns and ok[col].notna().any():
            return float((ok[col].dropna().abs() <= thr).mean())
        return np.nan

    params_oil = params_meta.get("oil", "unknown")
    model_key = _model_key(args.model)

    summary = {
        "model": model_key,
        "params_oil": params_oil,
        "validation_oil": args.oil,
        "cross_validation": str(norm_oil(str(params_oil)) != norm_oil(args.oil)),
        "refrigerant": args.refrigerant,
        "selection_mode": args.selection_mode,
        "n_selected_points": int(len(out_df)),
        "n_successful_points": int(out_df["success"].sum()),
        "n_failed_points": int((~out_df["success"]).sum()),
        "Tdis_norm_K": float(params_meta.get("Tdis_norm_K", 50.0) or 50.0),

        "mean_e_m_rel": _stat("e_m_rel", np.mean),
        "mean_e_P_rel": _stat("e_P_rel", np.mean),
        "mean_e_T_dis_K": _stat("e_T_dis_K", np.mean),

        "mae_e_m_rel": _stat("e_m_rel", lambda s: s.abs().mean()),
        "mae_e_P_rel": _stat("e_P_rel", lambda s: s.abs().mean()),
        "mae_e_T_dis_K": _stat("e_T_dis_K", lambda s: s.abs().mean()),

        "rmse_e_m_rel": _stat("e_m_rel", lambda s: np.sqrt(np.mean(s**2))),
        "rmse_e_P_rel": _stat("e_P_rel", lambda s: np.sqrt(np.mean(s**2))),
        "rmse_e_T_dis_K": _stat("e_T_dis_K", lambda s: np.sqrt(np.mean(s**2))),

        "mean_james_m_sq": _stat("james_m_sq", np.mean),
        "mean_james_P_sq": _stat("james_P_sq", np.mean),
        "mean_james_T_sq": _stat("james_T_sq", np.mean),
        "james_error_mean": _stat("james_error_point", np.mean),
        "james_error_sum": _stat("james_error_point", np.sum),

        "share_m_within_3pct": _share("e_m_rel", 0.03),
        "share_m_within_4pct": _share("e_m_rel", 0.04),
        "share_m_within_5pct": _share("e_m_rel", 0.05),
        "share_P_within_3pct": _share("e_P_rel", 0.03),
        "share_P_within_4pct": _share("e_P_rel", 0.04),
        "share_P_within_5pct": _share("e_P_rel", 0.05),
        "share_Tdis_within_3K": _share("e_T_dis_K", 3.0),
        "share_Tdis_within_4K": _share("e_T_dis_K", 4.0),
        "share_Tdis_within_5K": _share("e_T_dis_K", 5.0),
    }
    return pd.DataFrame([summary])


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Validation script for Molinaroli compressor models (original / modified / oil_path)."
    )

    ap.add_argument("--op_rows_csv", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--sep", default=";")
    ap.add_argument("--decimal", default=",")
    ap.add_argument("--header", type=int, default=1)

    ap.add_argument("--split_csv", type=Path, default=None)
    ap.add_argument("--params_csv", required=True, type=Path)

    ap.add_argument("--model", default="auto",
                    help="original | modified | oil_path | auto (from params_csv)")
    ap.add_argument("--refrigerant", default="auto")
    ap.add_argument("--oil", default="auto",
                    help="LPG68 | LPG100 | all | auto. Controls which DATA is validated.")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)

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
    ap.add_argument("--col_T_oil_sump", default=T_OIL_SUMP_COL_DEFAULT)
    ap.add_argument("--col_T_dis", default=T_DIS_MEAS_COL_DEFAULT)
    ap.add_argument("--col_m_meas", default=M_FLOW_MEAS_COL_DEFAULT)
    ap.add_argument("--col_P_meas", default=P_EL_MEAS_COL_DEFAULT)

    ap.add_argument("--legacy_split_idx_col", default="idx")

    ap.add_argument("--selection_mode", default="all",
                    choices=["validation_only", "train_only", "all"])

    ap.add_argument("--out_dir", default="results/validation")

    args = ap.parse_args()

    if args.op_rows_csv is not None and args.csv is not None:
        raise ValueError("Specify either --op_rows_csv or --csv, not both.")
    if args.op_rows_csv is None and args.csv is None:
        raise ValueError("Specify either --op_rows_csv or --csv.")

    use_new_mode = args.op_rows_csv is not None

    if use_new_mode and not args.op_rows_csv.exists():
        raise FileNotFoundError(args.op_rows_csv)
    if not use_new_mode and not args.csv.exists():
        raise FileNotFoundError(args.csv)
    if not args.params_csv.exists():
        raise FileNotFoundError(args.params_csv)
    if args.split_csv is not None and not args.split_csv.exists():
        raise FileNotFoundError(args.split_csv)

    # Resolve "auto" values from params CSV
    params_peek = pd.read_csv(args.params_csv).iloc[0].to_dict()

    if args.model == "auto":
        args.model = str(params_peek.get("model", "original"))
    if args.refrigerant == "auto":
        args.refrigerant = str(params_peek.get("refrigerant", "PROPANE"))
    if args.oil == "auto":
        args.oil = str(params_peek.get("oil", "all"))

    model_key = _model_key(args.model)
    is_oil_path = model_key == "oil_path"

    params, params_meta = load_params_csv(args.params_csv, args.model)

    if use_new_mode:
        if args.split_csv is None:
            raise ValueError("New mode (--op_rows_csv) requires --split_csv.")
        df = load_new_mode(args)
    else:
        df = load_legacy_mode(args)

    if args.split_role_col not in df.columns:
        df[args.split_role_col] = ""

    if args.selection_mode != "all" and (df[args.split_role_col] == "").all():
        raise ValueError(
            f"--selection_mode='{args.selection_mode}' requires split info, "
            f"but no split_role data found."
        )

    selected_df = select_rows(df, args.selection_mode, args.split_role_col)

    print(f"  Input mode:       {'new (op_rows + split)' if use_new_mode else 'legacy (raw CSV)'}")
    print(f"  Model:            {model_key}")
    print(f"  Params oil:       {params_meta.get('oil', 'unknown')}")
    print(f"  Validation oil:   {args.oil}")
    print(f"  Refrigerant:      {args.refrigerant}")
    print(f"  Selection mode:   {args.selection_mode}")
    print(f"  Total rows:       {len(df)}")
    print(f"  Selected rows:    {len(selected_df)}")

    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6

    med = RefProp(fluid_name=args.refrigerant)

    params["f_ref"] = F_REF
    params["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)
    print(f"  m_dot_ref:        {params['m_dot_ref'] * 1e3:.4f} g/s")

    if "_oil_norm" in selected_df.columns:
        unique_oils = sorted(selected_df["_oil_norm"].unique())
    elif args.oil_col in selected_df.columns:
        unique_oils = sorted(selected_df[args.oil_col].astype(str).apply(norm_oil).unique())
    else:
        unique_oils = [norm_oil(args.oil)]

    oil_display_map = {"lpg68": "LPG68", "lpg100": "LPG100"}
    unique_oil_display = [oil_display_map.get(o, o) for o in unique_oils]

    bundle = build_validation_bundle(
        model=args.model,
        oil_names=unique_oil_display,
        med=med,
        refrigerant_name=args.refrigerant,
        N_max_hz=N_max_hz,
        V_h_m3=V_h_m3,
        params=params,
    )

    has_T_dis = args.col_T_dis in selected_df.columns
    has_T_oil = args.col_T_oil_sump in selected_df.columns
    has_m_meas = args.col_m_meas in selected_df.columns
    has_P_meas = args.col_P_meas in selected_df.columns
    tdis_norm_k = float(params_meta.get("Tdis_norm_K", 50.0) or 50.0)

    results = []

    for _, row in selected_df.iterrows():
        p_suc_pa = bar_to_pa(row[args.col_p_suc])
        p_out_pa = bar_to_pa(row[args.col_p_out])
        T_suc_K = c_to_k(row[args.col_T_suc])
        T_amb_K = c_to_k(row[args.col_T_amb])
        f_oper_hz = rpm_to_hz(row[args.col_speed])
        n_rel = _clamp01(f_oper_hz / N_max_hz)

        if "_oil_norm" in row.index:
            oil_norm = str(row["_oil_norm"])
        elif args.oil_col in row.index:
            oil_norm = norm_oil(str(row[args.oil_col]))
        else:
            oil_norm = norm_oil(args.oil)

        oil_display = str(row[args.oil_col]) if args.oil_col in row.index else args.oil

        try:
            st_sat = med.calc_state("PQ", float(p_suc_pa), 1.0)
            T_sat_suc_C = k_to_c(_finite(st_sat.T))
            superheat_C = float(row[args.col_T_suc]) - T_sat_suc_C if np.isfinite(T_sat_suc_C) else float("nan")
        except Exception:
            T_sat_suc_C = float("nan")
            superheat_C = float("nan")

        rec = {
            "source_row_index": int(row[args.source_row_col]) if args.source_row_col in row.index and pd.notna(row.get(args.source_row_col)) else np.nan,
            "filtered_row_index": int(row[args.filtered_row_col]) if args.filtered_row_col in row.index and pd.notna(row.get(args.filtered_row_col)) else np.nan,
            "op_id": str(row[args.op_id_col]) if args.op_id_col in row.index else "",
            "split_role": str(row[args.split_role_col]) if args.split_role_col in row.index else "",
            "split_note": str(row.get(args.split_note_col, "")) if args.split_note_col in row.index else "",
            "is_train": bool(row.get(args.split_role_col, "") == "train"),
            "is_validation": bool(row.get(args.split_role_col, "") == "validation"),
            "success": True,
            "error": "",
            "model": model_key,
            "params_oil": str(params_meta.get("oil", "unknown")),
            "oil": oil_display,
            "oil_norm": oil_norm,
            "refrigerant": args.refrigerant,
            "p_suc_bar": float(row[args.col_p_suc]),
            "T_suc_C": float(row[args.col_T_suc]),
            "p_out_bar": float(row[args.col_p_out]),
            "T_amb_C": float(row[args.col_T_amb]),
            "N_rpm": float(row[args.col_speed]),
            "f_oper_hz": f_oper_hz,
            "n_rel": n_rel,
            "T_oil_sump_C_meas": float(row[args.col_T_oil_sump]) if (has_T_oil and pd.notna(row.get(args.col_T_oil_sump))) else np.nan,
            "T_dis_meas_C": float(row[args.col_T_dis]) if (has_T_dis and pd.notna(row.get(args.col_T_dis))) else np.nan,
            "T_sat_suc_C": T_sat_suc_C,
            "superheat_C": superheat_C,
            "pressure_ratio": float(row[args.col_p_out]) / float(row[args.col_p_suc]),
        }

        try:
            entry = get_bundle_entry(bundle, args.model, oil_norm)
            comp = entry["comp"]
            inputs = entry["inputs"]
            fs_state = entry["fs_state"]

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

            rec["m_flow_kg_s"] = m_flow
            rec["m_flow_g_s"] = m_flow * 1e3
            rec["P_el_W"] = P_el
            rec["T_dis_calc_C"] = k_to_c(T_dis_K)

            _extract_internal_states(rec, comp)
            _extract_loss_terms(rec, comp, args.model)

            if has_m_meas and pd.notna(row.get(args.col_m_meas)):
                rec["m_meas_g_s"] = float(row[args.col_m_meas])
                m_meas = gs_to_kgps(row[args.col_m_meas])
                rec["e_m_rel"] = (m_flow / m_meas - 1.0) if m_meas > 0 else np.nan
            else:
                rec["m_meas_g_s"] = np.nan
                rec["e_m_rel"] = np.nan

            if has_P_meas and pd.notna(row.get(args.col_P_meas)):
                rec["P_meas_W"] = float(row[args.col_P_meas])
                P_meas = float(row[args.col_P_meas])
                rec["e_P_rel"] = (P_el / P_meas - 1.0) if P_meas > 0 else np.nan
            else:
                rec["P_meas_W"] = np.nan
                rec["e_P_rel"] = np.nan

            if np.isfinite(rec.get("T_dis_meas_C", np.nan)) and np.isfinite(rec.get("T_dis_calc_C", np.nan)):
                rec["e_T_dis_K"] = rec["T_dis_calc_C"] - rec["T_dis_meas_C"]
            else:
                rec["e_T_dis_K"] = np.nan

            rec["james_m_sq"] = rec["e_m_rel"] ** 2 if np.isfinite(rec.get("e_m_rel", np.nan)) else np.nan
            rec["james_P_sq"] = rec["e_P_rel"] ** 2 if np.isfinite(rec.get("e_P_rel", np.nan)) else np.nan
            rec["james_T_sq"] = (rec["e_T_dis_K"] / tdis_norm_k) ** 2 if np.isfinite(rec.get("e_T_dis_K", np.nan)) else np.nan

            james_terms = [rec["james_m_sq"], rec["james_P_sq"], rec["james_T_sq"]]
            rec["james_error_point"] = (
                float(np.nansum(james_terms))
                if any(np.isfinite(x) for x in james_terms)
                else np.nan
            )

        except Exception as e:
            rec["success"] = False
            rec["error"] = str(e)[:200]

            for col in ["m_flow_kg_s", "m_flow_g_s", "P_el_W", "T_dis_calc_C",
                        "m_meas_g_s", "e_m_rel", "P_meas_W", "e_P_rel", "e_T_dis_K",
                        "james_m_sq", "james_P_sq", "james_T_sq", "james_error_point"]:
                rec[col] = np.nan

            for prefix in ["st_in", "c1", "c3", "c4", "c5", "st_out"]:
                _add_compact_state(rec, prefix, None)

            _fill_nan_loss_terms(rec, args.model)

        results.append(rec)

    # Print debug report for oil_path
    if is_oil_path:
        for key, entry in bundle.items():
            comp = entry["comp"]
            if hasattr(comp, "get_debug_report"):
                print(f"\n--- Debug report for {key} ---")
                print(comp.get_debug_report())

    out_df = pd.DataFrame(results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    params_oil_tag = norm_oil(str(params_meta.get("oil", "unknown")))
    val_oil_tag = norm_oil(args.oil)
    suffix = f"params_{params_oil_tag}_val_{val_oil_tag}_{model_key}_{ts}"

    out_detail = out_dir / f"validation_detail_{suffix}.csv"
    out_summary = out_dir / f"validation_summary_{suffix}.csv"

    out_df.to_csv(out_detail, index=False)

    summary_df = summarize_results(out_df, args, params_meta)
    summary_df.to_csv(out_summary, index=False)

    n_ok = int(out_df["success"].sum())
    n_total = len(out_df)

    print(f"\n=== Validation done ===")
    print(f"  Points: {n_ok}/{n_total} successful")

    if n_ok > 0:
        ok = out_df[out_df["success"]]
        if ok["e_m_rel"].notna().any():
            m3 = (ok["e_m_rel"].abs() <= 0.03).mean() * 100
            m4 = (ok["e_m_rel"].abs() <= 0.04).mean() * 100
            m5 = (ok["e_m_rel"].abs() <= 0.05).mean() * 100
            print(f"  Mass flow  within ±3%: {m3:.1f}%  |  ±4%: {m4:.1f}%  |  ±5%: {m5:.1f}%")
        if ok["e_P_rel"].notna().any():
            P3 = (ok["e_P_rel"].abs() <= 0.03).mean() * 100
            P4 = (ok["e_P_rel"].abs() <= 0.04).mean() * 100
            P5 = (ok["e_P_rel"].abs() <= 0.05).mean() * 100
            print(f"  Power      within ±3%: {P3:.1f}%  |  ±4%: {P4:.1f}%  |  ±5%: {P5:.1f}%")
        if ok["e_T_dis_K"].notna().any():
            T3 = (ok["e_T_dis_K"].abs() <= 3.0).mean() * 100
            T4 = (ok["e_T_dis_K"].abs() <= 4.0).mean() * 100
            T5 = (ok["e_T_dis_K"].abs() <= 5.0).mean() * 100
            print(f"  T_dis      within ±3K: {T3:.1f}%  |  ±4K: {T4:.1f}%  |  ±5K: {T5:.1f}%")

    print(f"\n  Detail saved:  {out_detail}")
    print(f"  Summary saved: {out_summary}")


if __name__ == "__main__":
    main()
