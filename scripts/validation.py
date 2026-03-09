# scripts/validation.py
# Beispielaufruf:
# python scripts/validation.py --csv data/Datensatz_Fitting_2.csv --params_csv results/final_results/Molinaroli_LPG68/fitted_params_lpg68_original_ga_2026-03-08_101308.csv --split_csv results/final_results/Molinaroli_LPG68/fit_predictions_lpg68_original_ga_2026-03-08_101308.csv
#
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

# =========================
# Defaults for CSV
# =========================
OIL_COL_DEFAULT = "Ölbezeichnung"
P_SUC_COL_DEFAULT = "P1_mean"         # bar
T_SUC_COL_DEFAULT = "T1_mean"         # °C
P_OUT_COL_DEFAULT = "P2_mean"         # bar
T_AMB_COL_DEFAULT = "Tamb_mean"       # °C
SPEED_COL_DEFAULT = "N"               # rpm
T_OIL_SUMP_COL_DEFAULT = "T7_mean"    # °C
T_DIS_MEAS_COL_DEFAULT = "T2_mean"    # °C
M_FLOW_MEAS_COL_DEFAULT = "suction_mf_mean"  # g/s
P_EL_MEAS_COL_DEFAULT = "Pel_mean"           # W

F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0

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
    "m_dot_ref": None,
    "f_ref": F_REF,
}


@dataclass
class Control:
    n: float


@dataclass
class SimpleInputs:
    control: Control
    T_amb: float


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


def norm_oil(s: str) -> str:
    return str(s).strip().lower().replace(" ", "")


def _finite(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _truthy(val) -> bool:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    s = str(val).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


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


def read_dataset_csv(path: Path, sep: str, header: int, decimal: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, header=header, decimal=decimal)
    df.insert(0, "source_row_index", np.arange(len(df), dtype=int))
    return df


def load_params_csv(path: Path):
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

    meta = {
        "oil": row.get("oil", None),
        "refrigerant": row.get("refrigerant", None),
        "model": row.get("model", None),
        "error_sum_sq": row.get("error_sum_sq", None),
        "n_train": row.get("n_train", None),
        "n_points_total": row.get("n_points_total", None),
        "use_Tdis": row.get("use_Tdis", None),
        "Tdis_norm_K": row.get("Tdis_norm_K", 50.0),
        "seed": row.get("seed", None),
        "population": row.get("population", None),
        "elite_frac": row.get("elite_frac", None),
        "random_keep_prob": row.get("random_keep_prob", None),
        "mutation_prob_param": row.get("mutation_prob_param", None),
        "generations": row.get("generations", None),
        "n_jobs": row.get("n_jobs", None),
    }

    return params, meta


def pick_model(model_name: str, N_max_hz: float, V_h_m3: float, parameters: dict):
    m = str(model_name).lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=parameters)
    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(N_max=N_max_hz, V_h=V_h_m3, parameters=parameters)
    raise ValueError("Unknown --model. Use: original | modified")


def compute_m_dot_ref(med: RefProp, V_h_m3: float) -> float:
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * float(F_REF)


def apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()

    oil_arg = args.oil.strip().lower()
    if oil_arg != "all":
        if args.oil_col not in out.columns:
            raise ValueError(f"--oil was set but oil column '{args.oil_col}' not found in CSV.")
        out = out[out[args.oil_col].astype(str).apply(norm_oil) == norm_oil(oil_arg)]

    required = [args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_T_amb, args.col_speed]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    out = out.dropna(subset=required).copy()
    out = out.reset_index(drop=True)
    out.insert(1, "filtered_row_index", np.arange(len(out), dtype=int))

    if args.max_rows is not None:
        out = out.head(args.max_rows).copy()
        out = out.reset_index(drop=True)
        out["filtered_row_index"] = np.arange(len(out), dtype=int)

    if len(out) == 0:
        raise ValueError("No valid rows after filtering and dropping NaNs.")

    return out


def read_split_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def attach_split_info(df: pd.DataFrame, split_df: pd.DataFrame, idx_col: str, is_train_col: str) -> pd.DataFrame:
    if idx_col not in split_df.columns:
        raise ValueError(f"Split CSV missing idx column '{idx_col}'.")
    if is_train_col not in split_df.columns:
        raise ValueError(f"Split CSV missing training flag column '{is_train_col}'.")

    tmp = split_df[[idx_col, is_train_col]].copy()
    tmp = tmp.rename(columns={idx_col: "split_idx", is_train_col: "is_train"})

    merged = df.copy()

    split_idx_numeric = pd.to_numeric(tmp["split_idx"], errors="coerce")
    if split_idx_numeric.notna().all() and len(df) > 0:
        max_idx = int(split_idx_numeric.max())
        if max_idx < len(df):
            merged = merged.merge(tmp, left_on="filtered_row_index", right_on="split_idx", how="left")
        else:
            merged = merged.merge(tmp, left_on="source_row_index", right_on="split_idx", how="left")
    else:
        merged = merged.merge(tmp, left_on="filtered_row_index", right_on="split_idx", how="left")

    merged["is_train"] = merged["is_train"].apply(_truthy) if "is_train" in merged.columns else False
    return merged


def select_validation_rows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    m = str(mode).lower().strip()
    if m == "all":
        out = df.copy()
    elif m in {"validation_only", "exclude_train"}:
        if "is_train" not in df.columns:
            raise ValueError("selection_mode requires split information, but no split file was provided.")
        out = df[~df["is_train"]].copy()
    elif m == "train_only":
        if "is_train" not in df.columns:
            raise ValueError("selection_mode requires split information, but no split file was provided.")
        out = df[df["is_train"]].copy()
    else:
        raise ValueError("Unknown --selection_mode. Use: validation_only | train_only | all")

    out = out.reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("No rows selected for validation.")
    return out


def summarize_results(out_df: pd.DataFrame, args, params_meta: dict) -> pd.DataFrame:
    ok = out_df[out_df["success"]].copy()

    def _mean(col):
        return float(ok[col].mean()) if col in ok.columns and ok[col].notna().any() else np.nan

    def _mae(col):
        if col in ok.columns and ok[col].notna().any():
            return float(ok[col].abs().mean())
        return np.nan

    def _rmse(col):
        if col in ok.columns and ok[col].notna().any():
            return float(np.sqrt(np.mean(np.square(ok[col].dropna()))))
        return np.nan

    def _share_rel(col, thr):
        if col in ok.columns and ok[col].notna().any():
            s = ok[col].dropna().abs()
            return float((s <= thr).mean())
        return np.nan

    def _share_abs(col, thr):
        if col in ok.columns and ok[col].notna().any():
            s = ok[col].dropna().abs()
            return float((s <= thr).mean())
        return np.nan

    summary = {
        "model": args.model,
        "oil": args.oil,
        "refrigerant": args.refrigerant,
        "selection_mode": args.selection_mode,
        "n_selected_points": int(len(out_df)),
        "n_successful_points": int(out_df["success"].sum()),
        "n_failed_points": int((~out_df["success"]).sum()),
        "n_train_from_params": params_meta.get("n_train", np.nan),
        "n_points_total_from_params": params_meta.get("n_points_total", np.nan),
        "ga_error_sum_sq_train": params_meta.get("error_sum_sq", np.nan),
        "james_Tdis_norm_K": params_meta.get("Tdis_norm_K", np.nan),
        # signed means
        "mean_e_m_rel": _mean("e_m_rel"),
        "mean_e_P_rel": _mean("e_P_rel"),
        "mean_e_T_dis_K": _mean("e_T_dis_K"),
        # MAE / RMSE
        "mae_e_m_rel": _mae("e_m_rel"),
        "mae_e_P_rel": _mae("e_P_rel"),
        "mae_e_T_dis_K": _mae("e_T_dis_K"),
        "rmse_e_m_rel": _rmse("e_m_rel"),
        "rmse_e_P_rel": _rmse("e_P_rel"),
        "rmse_e_T_dis_K": _rmse("e_T_dis_K"),
        # James-style averages
        "mean_james_m_sq": _mean("james_m_sq"),
        "mean_james_P_sq": _mean("james_P_sq"),
        "mean_james_T_sq": _mean("james_T_sq"),
        "james_error_mean": _mean("james_error_point"),
        "james_error_sum": float(ok["james_error_point"].sum()) if "james_error_point" in ok.columns else np.nan,
        # threshold shares
        "share_m_within_3pct": _share_rel("e_m_rel", 0.03),
        "share_P_within_2pct": _share_rel("e_P_rel", 0.02),
        "share_P_within_4pct": _share_rel("e_P_rel", 0.04),
        "share_Tdis_within_3K": _share_abs("e_T_dis_K", 3.0),
        "share_Tdis_within_4K": _share_abs("e_T_dis_K", 4.0),
        "share_Tdis_within_5K": _share_abs("e_T_dis_K", 5.0),
    }
    return pd.DataFrame([summary])


def main():
    ap = argparse.ArgumentParser(
        description="Validation script for Molinaroli compressor models using fitted parameters and split files."
    )
    ap.add_argument("--csv", required=True, help="Dataset CSV path (units row + header row).")
    ap.add_argument("--params_csv", required=True, help="ONE-ROW fitted parameter CSV from GA fitting.")
    ap.add_argument("--split_csv", default=None, help="Optional split CSV from fitting (e.g. idx,is_train,...).")

    ap.add_argument("--out_detail", default=None, help="Output CSV for validated points.")
    ap.add_argument("--out_summary", default=None, help="Output CSV for validation summary.")

    ap.add_argument("--model", default="auto", help="original | modified | auto (from params_csv)")
    ap.add_argument("--refrigerant", default="auto", help="RefProp fluid name or auto from params_csv")
    ap.add_argument("--oil", default="auto", help="LPG100 | LPG68 | all | auto (from params_csv)")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)

    ap.add_argument("--sep", default=";")
    ap.add_argument("--decimal", default=",")
    ap.add_argument("--header", type=int, default=1)

    ap.add_argument("--oil_col", default=OIL_COL_DEFAULT)
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

    ap.add_argument(
        "--selection_mode",
        default="validation_only",
        choices=["validation_only", "train_only", "all"],
        help="Which points to validate: validation_only (default), train_only, or all."
    )
    ap.add_argument("--split_idx_col", default="idx", help="Row index column in split CSV.")
    ap.add_argument("--split_train_col", default="is_train", help="Train flag column in split CSV.")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    params_path = Path(args.params_csv)
    split_path = Path(args.split_csv) if args.split_csv else None
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    if split_path is not None and not split_path.exists():
        raise FileNotFoundError(split_path)

    params_base, params_meta = load_params_csv(params_path)

    if args.model == "auto":
        args.model = str(params_meta.get("model") or "original")
    if args.refrigerant == "auto":
        args.refrigerant = str(params_meta.get("refrigerant") or "PROPANE")
    if args.oil == "auto":
        args.oil = str(params_meta.get("oil") or "all")

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    Path("results").mkdir(parents=True, exist_ok=True)
    if args.out_detail is None:
        args.out_detail = str(Path("results") / f"validation_points_{str(args.oil).lower()}_{str(args.model).lower()}_{ts}.csv")
    if args.out_summary is None:
        args.out_summary = str(Path("results") / f"validation_summary_{str(args.oil).lower()}_{str(args.model).lower()}_{ts}.csv")

    out_detail_path = Path(args.out_detail)
    out_summary_path = Path(args.out_summary)
    out_detail_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = read_dataset_csv(csv_path, sep=args.sep, header=args.header, decimal=args.decimal)
    df = apply_filters(raw_df, args)

    if split_path is not None:
        split_df = read_split_csv(split_path)
        df = attach_split_info(df, split_df, idx_col=args.split_idx_col, is_train_col=args.split_train_col)
        selected_df = select_validation_rows(df, args.selection_mode)
    else:
        if args.selection_mode != "all":
            raise ValueError("--selection_mode other than 'all' requires --split_csv.")
        selected_df = df.copy()

    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6
    med = RefProp(fluid_name=args.refrigerant)

    params_base["f_ref"] = F_REF
    params_base["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

    comp = pick_model(args.model, N_max_hz=N_max_hz, V_h_m3=V_h_m3, parameters=params_base)
    comp.med_prop = med
    comp.debug_enabled = True

    has_m_meas = args.col_m_meas in selected_df.columns
    has_P_meas = args.col_P_meas in selected_df.columns
    has_T_oil = args.col_T_oil_sump in selected_df.columns
    has_T_dis_meas = args.col_T_dis_meas in selected_df.columns
    tdis_norm_k = float(params_meta.get("Tdis_norm_K", 50.0) or 50.0)

    results = []

    for _, row in selected_df.iterrows():
        p_suc_pa = bar_to_pa(row[args.col_p_suc])
        p_out_pa = bar_to_pa(row[args.col_p_out])
        T_suc_K = c_to_k(row[args.col_T_suc])
        T_amb_K = c_to_k(row[args.col_T_amb])
        f_oper_hz = rpm_to_hz(row[args.col_speed])
        n_rel = float(max(1e-6, min(1.0, f_oper_hz / N_max_hz)))

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
            "source_row_index": int(row["source_row_index"]),
            "filtered_row_index": int(row["filtered_row_index"]),
            "idx_from_split": int(row["split_idx"]) if "split_idx" in row and pd.notna(row["split_idx"]) else np.nan,
            "is_train": bool(row["is_train"]) if "is_train" in row and pd.notna(row["is_train"]) else np.nan,
            "validated_in_this_run": True,
            "success": True,
            "error": "",
            "model": args.model,
            "backend": "RefProp",
            "refrigerant": args.refrigerant,
            "oil": str(row[args.oil_col]) if args.oil_col in row.index else "",
            "p_suc_bar_in": float(row[args.col_p_suc]),
            "T_suc_C_in": float(row[args.col_T_suc]),
            "p_out_bar_in": float(row[args.col_p_out]),
            "T_amb_C_in": float(row[args.col_T_amb]),
            "superheat_C": float(superheat_C),
            "N_rpm_in": float(row[args.col_speed]),
            "f_oper_hz": float(f_oper_hz),
            "n_rel": float(n_rel),
            "n_abs_hz": float(n_abs),
            "T_oil_sump_C_meas": float(row[args.col_T_oil_sump]) if (has_T_oil and pd.notna(row[args.col_T_oil_sump])) else np.nan,
            "T_dis_meas_C": float(row[args.col_T_dis_meas]) if (has_T_dis_meas and pd.notna(row[args.col_T_dis_meas])) else np.nan,
        }

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
                h3 = float(getattr(comp, "state_c_3").h)
                h4 = float(getattr(comp, "state_c_4").h)
                V_IC = float(params_base["V_IC"])
                m_dot_3 = rho3 * V_IC * float(n_abs)
                W_dot_int = m_dot_3 * (h4 - h3)
                alpha_loss = float(params_base["alpha_loss"])
                W_dot_loss_ref = float(params_base["W_dot_loss_ref"])
                W_dot_loss = (W_dot_int * alpha_loss + W_dot_loss_ref * (float(n_abs) / float(F_REF)) ** 2)
                rec["m_dot_3_kg_s"] = float(m_dot_3)
                rec["W_dot_int_W"] = float(W_dot_int)
                rec["W_dot_loss_W"] = float(W_dot_loss)
                rec["W_dot_int_plus_loss_W"] = float(W_dot_int + W_dot_loss)
                rec["W_dot_loss_share"] = float(W_dot_loss / (W_dot_int + W_dot_loss)) if (W_dot_int + W_dot_loss) > 0 else float("nan")
            except Exception:
                rec["m_dot_3_kg_s"] = np.nan
                rec["W_dot_int_W"] = np.nan
                rec["W_dot_loss_W"] = np.nan
                rec["W_dot_int_plus_loss_W"] = np.nan
                rec["W_dot_loss_share"] = np.nan

            if has_m_meas and pd.notna(row[args.col_m_meas]):
                rec["m_meas_g_s"] = float(row[args.col_m_meas])
                m_meas = gs_to_kgps(row[args.col_m_meas])
                rec["e_m_rel"] = (rec["m_flow_kg_s"] / m_meas) - 1.0 if m_meas > 0 else np.nan
            else:
                rec["m_meas_g_s"] = np.nan
                rec["e_m_rel"] = np.nan

            if has_P_meas and pd.notna(row[args.col_P_meas]):
                rec["P_meas_W"] = float(row[args.col_P_meas])
                P_meas = float(row[args.col_P_meas])
                rec["e_P_rel"] = (rec["P_el_W"] / P_meas) - 1.0 if P_meas > 0 else np.nan
            else:
                rec["P_meas_W"] = np.nan
                rec["e_P_rel"] = np.nan

            if np.isfinite(rec.get("T_dis_meas_C", np.nan)) and np.isfinite(rec.get("T_dis_C", np.nan)):
                rec["e_T_dis_K"] = float(rec["T_dis_C"] - rec["T_dis_meas_C"])
                rec["e_T_dis_abs_K"] = float(abs(rec["e_T_dis_K"]))
            else:
                rec["e_T_dis_K"] = np.nan
                rec["e_T_dis_abs_K"] = np.nan

            rec["james_m_sq"] = float(rec["e_m_rel"] ** 2) if np.isfinite(rec.get("e_m_rel", np.nan)) else np.nan
            rec["james_P_sq"] = float(rec["e_P_rel"] ** 2) if np.isfinite(rec.get("e_P_rel", np.nan)) else np.nan
            rec["james_T_sq"] = float((rec["e_T_dis_K"] / tdis_norm_k) ** 2) if np.isfinite(rec.get("e_T_dis_K", np.nan)) else np.nan

            james_terms = [rec["james_m_sq"], rec["james_P_sq"], rec["james_T_sq"]]
            rec["james_error_point"] = float(np.nansum(james_terms)) if any(np.isfinite(x) for x in james_terms) else np.nan

        except Exception as e:
            rec["success"] = False
            rec["error"] = str(e)
            for prefix in ["st_in", "c1", "c3", "c4", "c5", "st_out"]:
                _add_compact_state(rec, prefix, None)
            for col in [
                "m_flow_kg_s", "m_flow_g_s", "P_el_W", "T_wall_C", "T_dis_C",
                "m_dot_3_kg_s", "W_dot_int_W", "W_dot_loss_W", "W_dot_int_plus_loss_W", "W_dot_loss_share",
                "m_meas_g_s", "e_m_rel", "P_meas_W", "e_P_rel", "e_T_dis_K", "e_T_dis_abs_K",
                "james_m_sq", "james_P_sq", "james_T_sq", "james_error_point",
            ]:
                rec[col] = np.nan

        results.append(rec)

    out_df = pd.DataFrame(results)

    input_cols = [
        "source_row_index", "filtered_row_index", "idx_from_split", "is_train", "validated_in_this_run",
        "model", "backend", "refrigerant", "oil", "success", "error",
        "p_suc_bar_in", "T_suc_C_in", "p_out_bar_in", "T_amb_C_in",
        "T_oil_sump_C_meas", "T_dis_meas_C", "superheat_C", "N_rpm_in", "f_oper_hz", "n_rel", "n_abs_hz",
    ]
    state_prefixes = ["st_in", "c1", "c3", "c4", "c5", "st_out"]
    state_cols = []
    for p in state_prefixes:
        for suf in ["_p_bar", "_T_C", "_rho_kgpm3"]:
            col = f"{p}{suf}"
            if col in out_df.columns:
                state_cols.append(col)
    output_cols = [
        "m_flow_kg_s", "m_flow_g_s", "P_el_W", "T_wall_C", "T_dis_C",
        "m_dot_3_kg_s", "W_dot_int_W", "W_dot_loss_W", "W_dot_int_plus_loss_W", "W_dot_loss_share",
    ]
    error_cols = [
        "m_meas_g_s", "e_m_rel", "P_meas_W", "e_P_rel", "e_T_dis_K", "e_T_dis_abs_K",
        "james_m_sq", "james_P_sq", "james_T_sq", "james_error_point",
    ]

    def _keep_existing(cols):
        return [c for c in cols if c in out_df.columns]

    ordered = _keep_existing(input_cols) + _keep_existing(state_cols) + _keep_existing(output_cols) + _keep_existing(error_cols)
    remaining = [c for c in out_df.columns if c not in ordered]
    out_df = out_df[ordered + remaining]
    out_df.to_csv(out_detail_path, index=False)

    summary_df = summarize_results(out_df, args, params_meta)
    summary_df.to_csv(out_summary_path, index=False)

    n_ok = int(out_df["success"].sum())
    n_total = len(out_df)
    print("\n=== Validation done ===")
    print(f"oil: {args.oil}, model: {args.model}, refrigerant: {args.refrigerant}, backend: RefProp")
    print(f"selection_mode: {args.selection_mode}")
    print(f"points: {n_ok}/{n_total} successful")
    print(f"params source: {params_path}")
    print(f"split source: {split_path if split_path else 'None'}")
    print(f"detail saved: {out_detail_path}")
    print(f"summary saved: {out_summary_path}")


if __name__ == "__main__":
    main()
