# scripts/run_batch.py

import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import os
import tempfile
import uuid
import shutil

import numpy as np
import pandas as pd

from vclibpy.media import RefProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import (
    Molinaroli_2017_Compressor,
    Molinaroli_2017_Compressor_Modified,
)

#
# Beispielaufrufe:
# python scripts/run_batch.py --csv data/Datensatz_Fitting_1.csv --oil LPG68 --model original --refrigerant PROPANE
# python scripts/run_batch.py --csv data/Datensatz_Fitting_1.csv --oil LPG68 --model modified --refrigerant PROPANE --params_csv data/start_params_modified.csv
# python scripts/run_batch.py --csv results/split_template/operating_points_rows_2026-03-12_112331.csv --split_csv results/split_template/operating_points_split_template_2026-03-12_112331.csv --oil LPG68 --model modified --refrigerant PROPANE
#

# =========================
# Defaults for YOUR CSV
# =========================
OIL_COL_DEFAULT = "Ölbezeichnung"

P_SUC_COL_DEFAULT = "P1_mean"         # bar
T_SUC_COL_DEFAULT = "T1_mean"         # °C
P_OUT_COL_DEFAULT = "P2_mean"         # bar
T_AMB_COL_DEFAULT = "Tamb_mean"       # °C
SPEED_COL_DEFAULT = "N"               # rpm (1/min)

# Oil sump temperature
T_OIL_SUMP_COL_DEFAULT = "T7_mean"    # °C

# Measured discharge temperature from dataset
T_DIS_MEAS_COL_DEFAULT = "T2_mean"    # °C

# Optional measurement columns
M_FLOW_MEAS_COL_DEFAULT = "suction_mf_mean"  # g/s
P_EL_MEAS_COL_DEFAULT = "Pel_mean"           # W

# Optional operating-point / split metadata
OP_ID_COL_DEFAULT = "op_id"
SOURCE_ROW_COL_DEFAULT = "source_row_index"
FILTERED_ROW_COL_DEFAULT = "filtered_row_index"
SPLIT_ROLE_COL_DEFAULT = "split_role"
SPLIT_NOTE_COL_DEFAULT = "split_note"

# =========================
# Reference values (fixed)
# =========================
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0

# =========================
# Model parameter names
# =========================
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
    "mu_fallback",
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
    "mu_fallback": 5.0,
    "m_dot_ref": None,
    "f_ref": F_REF,
}


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
# Unit conversions
# =========================
def bar_to_pa(p_bar: float) -> float:
    return float(p_bar) * 100_000.0


def pa_to_bar(p_pa: float) -> float:
    return float(p_pa) / 100_000.0


def c_to_k(t_c: float) -> float:
    return float(t_c) + 273.15


def k_to_c(t_k: float) -> float:
    return float(t_k) - 273.15


def rpm_to_hz(rpm: float) -> float:
    return float(rpm) / 60.0


def gs_to_kgps(g_s: float) -> float:
    return float(g_s) / 1000.0


# =========================
# Small helpers
# =========================
def norm_oil(s: str) -> str:
    return str(s).strip().lower().replace(" ", "")


def _finite(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _clamp01(x: float) -> float:
    return max(1e-9, min(1.0, float(x)))


def parse_split_role(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s in {"train", "training", "fit"}:
        return "train"
    if s in {"validation", "valid", "val", "test"}:
        return "validation"
    if s in {"unused", "skip", "holdout", ""}:
        return ""
    raise ValueError(f"Unknown split_role value: {x}")


def _add_compact_state(rec: dict, prefix: str, st) -> None:
    """
    Output only:
      - pressure in bar
      - temperature in °C
      - density in kg/m³
    """
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


# =========================
# CSV I/O
# =========================
def read_dataset_csv(path: Path, sep: str, header: int, decimal: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, header=header, decimal=decimal)


def read_split_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_default_params(model: str) -> dict:
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return dict(DEFAULT_PARAMS_ORIGINAL)
    if m in ("mod", "modified"):
        return dict(DEFAULT_PARAMS_MODIFIED)
    raise ValueError("Unknown --model. Use: original | modified")


def get_param_names(model: str) -> list[str]:
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return list(PARAM_NAMES_ORIGINAL)
    if m in ("mod", "modified"):
        return list(PARAM_NAMES_MODIFIED)
    raise ValueError("Unknown --model. Use: original | modified")


def load_params_csv(path: Path, model: str) -> dict:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError("Params CSV must contain exactly one row.")

    row = df.iloc[0].to_dict()
    params = get_default_params(model=model)

    for k in get_param_names(model):
        if k in row and pd.notna(row[k]):
            params[k] = float(row[k])

    if "f_ref" in row and pd.notna(row["f_ref"]):
        params["f_ref"] = float(row["f_ref"])

    return params


def merge_split_information(df: pd.DataFrame, split_csv: Path | None, args) -> pd.DataFrame:
    out = df.copy()

    # If no split CSV is provided, still normalize an existing split_role column if present
    if split_csv is None:
        if args.split_role_col in out.columns:
            out[args.split_role_col] = out[args.split_role_col].apply(parse_split_role)
        else:
            out[args.split_role_col] = ""
        return out

    if args.op_id_col not in out.columns:
        raise ValueError(
            f"--split_csv was provided, but input CSV has no op_id column '{args.op_id_col}'."
        )

    df_split = read_split_csv(split_csv)
    required = [args.op_id_col, args.split_role_col]
    missing = [c for c in required if c not in df_split.columns]
    if missing:
        raise ValueError(f"Split CSV missing required columns: {missing}")

    df_split = df_split.copy()
    df_split[args.split_role_col] = df_split[args.split_role_col].apply(parse_split_role)

    keep_cols = [args.op_id_col, args.split_role_col]
    if args.split_note_col in df_split.columns:
        keep_cols.append(args.split_note_col)

    if df_split[args.op_id_col].duplicated().any():
        dup = df_split.loc[df_split[args.op_id_col].duplicated(), args.op_id_col].astype(str).tolist()
        raise ValueError(f"Split CSV contains duplicate op_id values, e.g. {dup[:10]}")

    out = out.drop(columns=[c for c in [args.split_role_col, args.split_note_col] if c in out.columns])
    out = out.merge(df_split[keep_cols], on=args.op_id_col, how="left")

    out[args.split_role_col] = out[args.split_role_col].fillna("").apply(parse_split_role)
    if args.split_note_col not in out.columns:
        out[args.split_note_col] = ""

    return out


# =========================
# Name mapping for modified model
# =========================
def map_refrigerant_for_modified_model(name: str) -> str:
    s = str(name).strip().upper()
    if s in ("PROPANE", "R290", "PROPAN"):
        return "propane"
    return str(name).strip().lower()


def map_oil_for_modified_model(name: str) -> str:
    s = norm_oil(name)
    if s == "lpg68":
        return "LPG 68"
    if s == "lpg100":
        return "LPG 100"
    raise ValueError(f"Unsupported oil for modified model: {name}")


# =========================
# Model helpers
# =========================
def pick_model(
    model_name: str,
    N_max_hz: float,
    V_h_m3: float,
    parameters: dict,
    fluid_name: str = None,
    lub_name: str = None,
):
    m = model_name.lower().strip()

    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(
            N_max=N_max_hz,
            V_h=V_h_m3,
            parameters=parameters,
        )

    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(
            N_max=N_max_hz,
            V_h=V_h_m3,
            fluid_name=fluid_name,
            lub_name=lub_name,
            parameters=parameters,
        )

    raise ValueError("Unknown --model. Use: original | modified")


def build_compressor(
    model_name: str,
    N_max_hz: float,
    V_h_m3: float,
    parameters: dict,
    med: RefProp,
    refrigerant_name: str,
    oil_name: str = None,
    debug_enabled: bool = False,
):
    m = model_name.lower().strip()

    if m in ("mod", "modified"):
        if oil_name is None:
            raise ValueError("Modified model requires an oil name.")
        fluid_name_mod = map_refrigerant_for_modified_model(refrigerant_name)
        lub_name_mod = map_oil_for_modified_model(oil_name)

        comp = pick_model(
            model_name=model_name,
            N_max_hz=N_max_hz,
            V_h_m3=V_h_m3,
            parameters=parameters,
            fluid_name=fluid_name_mod,
            lub_name=lub_name_mod,
        )
    else:
        comp = pick_model(
            model_name=model_name,
            N_max_hz=N_max_hz,
            V_h_m3=V_h_m3,
            parameters=parameters,
        )

    comp.med_prop = med
    if hasattr(comp, "debug_enabled"):
        comp.debug_enabled = bool(debug_enabled)
    return comp


def compute_m_dot_ref(med: RefProp, V_h_m3: float) -> float:
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * float(F_REF)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="Batch simulation for Molinaroli compressor models (original / modified, RefProp backend)."
    )

    ap.add_argument("--csv", required=True, help="Input CSV path.")
    ap.add_argument("--split_csv", default=None, help="Optional split CSV to merge via op_id.")
    ap.add_argument("--out", default=None, help="Output CSV path (default: results/batch_<timestamp>.csv)")

    ap.add_argument("--model", default="original", choices=["original", "modified"])
    ap.add_argument("--refrigerant", default="PROPANE")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0, help="Max speed [rpm] from datasheet")
    ap.add_argument("--V_h_cm3", type=float, default=30.7, help="Geometric displacement volume [cm³]")

    ap.add_argument("--sep", default=";", help="CSV separator for input CSV")
    ap.add_argument("--decimal", default=",", help="Decimal separator for input CSV")
    ap.add_argument("--header", type=int, default=1, help="Header row index")

    ap.add_argument("--oil_col", default=OIL_COL_DEFAULT)
    ap.add_argument("--oil", default="all", help="LPG68 | LPG100 | all")

    ap.add_argument("--op_id_col", default=OP_ID_COL_DEFAULT)
    ap.add_argument("--source_row_col", default=SOURCE_ROW_COL_DEFAULT)
    ap.add_argument("--filtered_row_col", default=FILTERED_ROW_COL_DEFAULT)
    ap.add_argument("--split_role_col", default=SPLIT_ROLE_COL_DEFAULT)
    ap.add_argument("--split_note_col", default=SPLIT_NOTE_COL_DEFAULT)

    ap.add_argument("--col_p_suc", default=P_SUC_COL_DEFAULT)
    ap.add_argument("--col_T_suc", default=T_SUC_COL_DEFAULT)
    ap.add_argument("--col_p_out", default=P_OUT_COL_DEFAULT)
    ap.add_argument("--col_T_amb", default=T_AMB_COL_DEFAULT)
    ap.add_argument("--col_speed", default=SPEED_COL_DEFAULT)

    ap.add_argument("--col_T_oil_sump", default=T_OIL_SUMP_COL_DEFAULT)
    ap.add_argument("--col_T_dis_meas", default=T_DIS_MEAS_COL_DEFAULT)
    ap.add_argument("--col_m_meas", default=M_FLOW_MEAS_COL_DEFAULT)
    ap.add_argument("--col_P_meas", default=P_EL_MEAS_COL_DEFAULT)

    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--params_csv", default=None, help="Optional one-row parameter CSV")
    ap.add_argument("--debug_model", action="store_true", help="Enable compressor debug output if supported")

    args = ap.parse_args()

    original_cwd = Path.cwd()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (original_cwd / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    split_csv_path = None
    if args.split_csv:
        split_csv_path = Path(args.split_csv)
        if not split_csv_path.is_absolute():
            split_csv_path = (original_cwd / split_csv_path).resolve()
        if not split_csv_path.exists():
            raise FileNotFoundError(split_csv_path)

    params_csv_path = None
    if args.params_csv:
        params_csv_path = Path(args.params_csv)
        if not params_csv_path.is_absolute():
            params_csv_path = (original_cwd / params_csv_path).resolve()
        if not params_csv_path.exists():
            raise FileNotFoundError(params_csv_path)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (original_cwd / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = (original_cwd / "results").resolve()
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_path = results_dir / f"batch_{args.oil.lower()}_{args.model.lower()}_{ts}.csv"

    # -------------------------
    # Read and prepare data
    # -------------------------
    df = read_dataset_csv(csv_path, sep=args.sep, header=args.header, decimal=args.decimal)
    df = merge_split_information(df, split_csv_path, args)

    oil_arg = norm_oil(args.oil)
    if oil_arg != "all":
        if args.oil_col not in df.columns:
            raise ValueError(f"--oil was set but oil column '{args.oil_col}' was not found in CSV.")
        df = df[df[args.oil_col].astype(str).apply(norm_oil) == oil_arg].copy()

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    required = [args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_T_amb, args.col_speed]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    optional_numeric = [args.col_T_oil_sump, args.col_T_dis_meas, args.col_m_meas, args.col_P_meas]
    for col in optional_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after dropping NaNs in required columns.")

    if args.split_role_col in df.columns:
        df[args.split_role_col] = df[args.split_role_col].fillna("").apply(parse_split_role)
    else:
        df[args.split_role_col] = ""

    if args.split_note_col not in df.columns:
        df[args.split_note_col] = ""

    # -------------------------
    # Basic conversions
    # -------------------------
    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6

    # -------------------------
    # Create isolated REFPROP workdir
    # -------------------------
    refprop_workdir = Path(tempfile.gettempdir()) / f"run_batch_refprop_{uuid.uuid4().hex[:8]}"
    refprop_workdir.mkdir(parents=True, exist_ok=True)

    try:
        os.chdir(refprop_workdir)

        med = RefProp(fluid_name=args.refrigerant)

        if params_csv_path is not None:
            params_base = load_params_csv(params_csv_path, model=args.model)
        else:
            params_base = get_default_params(model=args.model)

        params_base["f_ref"] = F_REF
        params_base["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

        has_m_meas = args.col_m_meas in df.columns
        has_P_meas = args.col_P_meas in df.columns
        has_T_oil = args.col_T_oil_sump in df.columns
        has_T_dis_meas = args.col_T_dis_meas in df.columns

        results = []
        comp_cache = {}

        for i, row in df.iterrows():
            p_suc_pa = bar_to_pa(row[args.col_p_suc])
            p_out_pa = bar_to_pa(row[args.col_p_out])
            T_suc_K = c_to_k(row[args.col_T_suc])
            T_amb_K = c_to_k(row[args.col_T_amb])
            f_oper_hz = rpm_to_hz(row[args.col_speed])
            n_rel = _clamp01(f_oper_hz / N_max_hz)

            oil_value = str(row[args.oil_col]) if args.oil_col in df.columns else ""

            if args.model.lower().strip() == "modified":
                if not oil_value:
                    raise ValueError("Modified model requires an oil column in the dataset.")
                comp_key = ("modified", norm_oil(oil_value))
            else:
                comp_key = ("original", "single")

            if comp_key not in comp_cache:
                comp_cache[comp_key] = build_compressor(
                    model_name=args.model,
                    N_max_hz=N_max_hz,
                    V_h_m3=V_h_m3,
                    parameters=params_base.copy(),
                    med=med,
                    refrigerant_name=args.refrigerant,
                    oil_name=oil_value if args.model == "modified" else None,
                    debug_enabled=args.debug_model,
                )
            comp = comp_cache[comp_key]

            # Suction superheat from measured suction pressure / temperature
            try:
                st_sat = med.calc_state("PQ", float(p_suc_pa), 1.0)
                T_sat_suc_K = _finite(getattr(st_sat, "T", np.nan))
                T_sat_suc_C = k_to_c(T_sat_suc_K) if np.isfinite(T_sat_suc_K) else float("nan")
                superheat_C = float(row[args.col_T_suc]) - T_sat_suc_C if np.isfinite(T_sat_suc_C) else float("nan")
            except Exception:
                superheat_C = float("nan")

            try:
                n_abs = float(comp.get_n_absolute(n_rel))
            except Exception:
                n_abs = float("nan")

            rec = {
                "row_index": int(i),
                "model": args.model,
                "backend": "RefProp",
                "refrigerant": args.refrigerant,
                "oil": oil_value,
                "success": True,
                "error": "",
                "p_suc_bar_in": float(row[args.col_p_suc]),
                "T_suc_C_in": float(row[args.col_T_suc]),
                "p_out_bar_in": float(row[args.col_p_out]),
                "T_amb_C_in": float(row[args.col_T_amb]),
                "superheat_C": float(superheat_C),
                "N_rpm_in": float(row[args.col_speed]),
                "f_oper_hz": float(f_oper_hz),
                "n_rel": float(n_rel),
                "n_abs_hz": float(n_abs),
                "split_role": str(row.get(args.split_role_col, "")),
                "is_train": bool(str(row.get(args.split_role_col, "")) == "train"),
                "is_validation": bool(str(row.get(args.split_role_col, "")) == "validation"),
                "split_note": row.get(args.split_note_col, ""),
            }

            if args.op_id_col in df.columns:
                rec["op_id"] = row.get(args.op_id_col, "")
            if args.source_row_col in df.columns:
                rec["source_row_index"] = row.get(args.source_row_col, np.nan)
            if args.filtered_row_col in df.columns:
                rec["filtered_row_index"] = row.get(args.filtered_row_col, np.nan)
            if "Bezeichnung" in df.columns:
                rec["Bezeichnung"] = row.get("Bezeichnung", "")

            rec["T_oil_sump_C_meas"] = (
                float(row[args.col_T_oil_sump])
                if (has_T_oil and pd.notna(row[args.col_T_oil_sump]))
                else np.nan
            )
            rec["T_dis_meas_C"] = (
                float(row[args.col_T_dis_meas])
                if (has_T_dis_meas and pd.notna(row[args.col_T_dis_meas]))
                else np.nan
            )

            fs_state = FlowsheetState()

            try:
                comp.state_inlet = med.calc_state("PT", float(p_suc_pa), float(T_suc_K))
                inputs = SimpleInputs(control=Control(n=n_rel), T_amb=float(T_amb_K))
                comp.calc_state_outlet(p_outlet=float(p_out_pa), inputs=inputs, fs_state=fs_state)

                _add_compact_state(rec, "st_in", getattr(comp, "state_inlet", None))
                _add_compact_state(rec, "c1", getattr(comp, "state_c_1", None))
                _add_compact_state(rec, "c3", getattr(comp, "state_c_3", None))
                _add_compact_state(rec, "c4", getattr(comp, "state_c_4", None))
                _add_compact_state(rec, "c5", getattr(comp, "state_c_5", None))
                _add_compact_state(rec, "st_out", getattr(comp, "state_outlet", None))

                rec["m_flow_kg_s"] = float(comp.m_flow)
                rec["m_flow_g_s"] = float(comp.m_flow) * 1000.0
                rec["P_el_W"] = float(comp.P_el)

                T_wall_K = _finite(getattr(comp, "T_w", np.nan))
                rec["T_wall_C"] = k_to_c(T_wall_K) if np.isfinite(T_wall_K) else float("nan")

                T_dis_K = _finite(getattr(comp.state_outlet, "T", np.nan))
                rec["T_dis_C"] = k_to_c(T_dis_K) if np.isfinite(T_dis_K) else float("nan")

                try:
                    rho3 = float(getattr(comp, "state_c_3").d)
                    V_IC = float(params_base["V_IC"])
                    m_dot_3 = rho3 * V_IC * float(n_abs)
                    rec["m_dot_3_kg_s"] = float(m_dot_3)
                except Exception:
                    rec["m_dot_3_kg_s"] = float("nan")

                W_dot_int = _finite(getattr(comp, "W_dot_int", np.nan))
                W_dot_loss = _finite(getattr(comp, "W_dot_loss", np.nan))
                W_dot_loss_load = _finite(getattr(comp, "W_dot_loss_load", np.nan))
                W_dot_loss_ref_term = _finite(getattr(comp, "W_dot_loss_ref_term", np.nan))
                W_dot_loss_fric = _finite(getattr(comp, "W_dot_loss_fric", np.nan))
                T_oil_sump_K = _finite(getattr(comp, "T_oil_sump", np.nan))
                mu_oil = _finite(getattr(comp, "mu_oil", np.nan))
                mu_mix_eff = _finite(getattr(comp, "mu_mix_eff", np.nan))

                # Fallback only for original model or if detailed terms are not exposed
                if not np.isfinite(W_dot_int) or not np.isfinite(W_dot_loss):
                    try:
                        rho3 = float(getattr(comp, "state_c_3").d)
                        h3 = float(getattr(comp, "state_c_3").h)
                        h4 = float(getattr(comp, "state_c_4").h)

                        V_IC = float(params_base["V_IC"])
                        m_dot_3 = rho3 * V_IC * float(n_abs)

                        W_dot_int = m_dot_3 * (h4 - h3)

                        alpha_loss = float(params_base["alpha_loss"])
                        W_dot_loss_ref = float(params_base["W_dot_loss_ref"])
                        W_dot_loss = (
                            W_dot_int * alpha_loss
                            + W_dot_loss_ref * (float(n_abs) / float(F_REF)) ** 2
                        )
                    except Exception:
                        W_dot_int = float("nan")
                        W_dot_loss = float("nan")

                rec["W_dot_int_W"] = float(W_dot_int) if np.isfinite(W_dot_int) else float("nan")
                rec["W_dot_loss_W"] = float(W_dot_loss) if np.isfinite(W_dot_loss) else float("nan")
                rec["W_dot_loss_load_W"] = float(W_dot_loss_load) if np.isfinite(W_dot_loss_load) else float("nan")
                rec["W_dot_loss_ref_term_W"] = float(W_dot_loss_ref_term) if np.isfinite(W_dot_loss_ref_term) else float("nan")
                rec["W_dot_loss_fric_W"] = float(W_dot_loss_fric) if np.isfinite(W_dot_loss_fric) else float("nan")

                if np.isfinite(W_dot_int) and np.isfinite(W_dot_loss):
                    rec["W_dot_int_plus_loss_W"] = float(W_dot_int + W_dot_loss)
                    rec["W_dot_loss_share"] = (
                        float(W_dot_loss / (W_dot_int + W_dot_loss))
                        if (W_dot_int + W_dot_loss) > 0
                        else float("nan")
                    )
                else:
                    rec["W_dot_int_plus_loss_W"] = float("nan")
                    rec["W_dot_loss_share"] = float("nan")

                rec["T_oil_sump_C"] = k_to_c(T_oil_sump_K) if np.isfinite(T_oil_sump_K) else float("nan")
                rec["mu_oil_mPas"] = float(mu_oil) if np.isfinite(mu_oil) else float("nan")
                rec["mu_mix_eff_Pa_s"] = float(mu_mix_eff) if np.isfinite(mu_mix_eff) else float("nan")

                if has_m_meas and pd.notna(row[args.col_m_meas]):
                    rec["m_meas_g_s"] = float(row[args.col_m_meas])
                    m_meas = gs_to_kgps(row[args.col_m_meas])
                    rec["e_m_rel"] = (rec["m_flow_kg_s"] / m_meas) - 1.0 if m_meas > 0 else np.nan

                if has_P_meas and pd.notna(row[args.col_P_meas]):
                    rec["P_meas_W"] = float(row[args.col_P_meas])
                    P_meas = float(row[args.col_P_meas])
                    rec["e_P_rel"] = (rec["P_el_W"] / P_meas) - 1.0 if P_meas > 0 else np.nan

                if np.isfinite(rec.get("T_dis_meas_C", np.nan)) and np.isfinite(rec.get("T_dis_C", np.nan)):
                    rec["e_T_dis_abs_C"] = float(rec["T_dis_C"] - rec["T_dis_meas_C"])
                    rec["e_T_dis_abs_abs_C"] = float(abs(rec["e_T_dis_abs_C"]))
                else:
                    rec["e_T_dis_abs_C"] = np.nan
                    rec["e_T_dis_abs_abs_C"] = np.nan

            except Exception as e:
                rec["success"] = False
                rec["error"] = str(e)

                for prefix in ["st_in", "c1", "c3", "c4", "c5", "st_out"]:
                    _add_compact_state(rec, prefix, None)

                rec["m_flow_kg_s"] = np.nan
                rec["m_flow_g_s"] = np.nan
                rec["P_el_W"] = np.nan
                rec["T_wall_C"] = np.nan
                rec["T_dis_C"] = np.nan
                rec["T_oil_sump_C"] = np.nan
                rec["mu_oil_mPas"] = np.nan
                rec["mu_mix_eff_Pa_s"] = np.nan

                rec["m_dot_3_kg_s"] = np.nan
                rec["W_dot_int_W"] = np.nan
                rec["W_dot_loss_W"] = np.nan
                rec["W_dot_loss_load_W"] = np.nan
                rec["W_dot_loss_ref_term_W"] = np.nan
                rec["W_dot_loss_fric_W"] = np.nan
                rec["W_dot_int_plus_loss_W"] = np.nan
                rec["W_dot_loss_share"] = np.nan

                rec["e_T_dis_abs_C"] = np.nan
                rec["e_T_dis_abs_abs_C"] = np.nan

                if has_m_meas:
                    rec["m_meas_g_s"] = float(row[args.col_m_meas]) if pd.notna(row[args.col_m_meas]) else np.nan
                    rec["e_m_rel"] = np.nan
                if has_P_meas:
                    rec["P_meas_W"] = float(row[args.col_P_meas]) if pd.notna(row[args.col_P_meas]) else np.nan
                    rec["e_P_rel"] = np.nan

            results.append(rec)

        out_df = pd.DataFrame(results)

        # -------------------------
        # Column order
        # -------------------------
        meta_cols = [
            "row_index",
            "source_row_index",
            "filtered_row_index",
            "op_id",
            "Bezeichnung",
            "split_role",
            "is_train",
            "is_validation",
            "split_note",
        ]

        input_cols = [
            "model",
            "backend",
            "refrigerant",
            "oil",
            "success",
            "error",
            "p_suc_bar_in",
            "T_suc_C_in",
            "p_out_bar_in",
            "T_amb_C_in",
            "T_oil_sump_C_meas",
            "T_dis_meas_C",
            "superheat_C",
            "N_rpm_in",
            "f_oper_hz",
            "n_rel",
            "n_abs_hz",
        ]

        state_prefixes = ["st_in", "c1", "c3", "c4", "c5", "st_out"]
        state_cols = []
        for p in state_prefixes:
            for suf in ["_p_bar", "_T_C", "_rho_kgpm3"]:
                col = f"{p}{suf}"
                if col in out_df.columns:
                    state_cols.append(col)

        output_cols = [
            "m_flow_kg_s",
            "m_flow_g_s",
            "P_el_W",
            "T_wall_C",
            "T_dis_C",
            "T_oil_sump_C",
            "mu_oil_mPas",
            "mu_mix_eff_Pa_s",
            "m_dot_3_kg_s",
            "W_dot_int_W",
            "W_dot_loss_W",
            "W_dot_loss_load_W",
            "W_dot_loss_ref_term_W",
            "W_dot_loss_fric_W",
            "W_dot_int_plus_loss_W",
            "W_dot_loss_share",
        ]

        error_cols = [
            "m_meas_g_s",
            "e_m_rel",
            "P_meas_W",
            "e_P_rel",
            "e_T_dis_abs_C",
            "e_T_dis_abs_abs_C",
        ]

        def _keep_existing(cols):
            return [c for c in cols if c in out_df.columns]

        ordered = (
            _keep_existing(meta_cols)
            + _keep_existing(input_cols)
            + _keep_existing(state_cols)
            + _keep_existing(output_cols)
            + _keep_existing(error_cols)
        )
        remaining = [c for c in out_df.columns if c not in ordered]
        out_df = out_df[ordered + remaining]

        out_df.to_csv(out_path, index=False)

        n_ok = int(out_df["success"].sum())
        n_total = len(out_df)

        print("\n=== Batch done ===")
        print(f"oil: {args.oil}, model: {args.model}, refrigerant: {args.refrigerant}, backend: RefProp")
        print(f"points: {n_ok}/{n_total} successful")
        print(f"params source: {params_csv_path if params_csv_path else 'model defaults'}")
        print(f"split source:  {split_csv_path if split_csv_path else 'none'}")
        print(f"saved: {out_path}")

    finally:
        try:
            os.chdir(original_cwd)
        except Exception:
            pass
        try:
            shutil.rmtree(refprop_workdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()