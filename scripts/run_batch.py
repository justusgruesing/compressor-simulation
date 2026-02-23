# scripts/run_batch.py
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
#
# Beispielaufruf: python scripts/run_batch.py --csv data/Datensatz_Fitting_1.csv --oil LPG68 --model original --refrigerant PROPANE --params_csv results/matlab_fit/fitted_params_lpg68_original_....csv
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

# Optional measurement columns (if present -> compute relative errors)
M_FLOW_MEAS_COL_DEFAULT = "suction_mf_mean"  # g/s
P_EL_MEAS_COL_DEFAULT = "Pel_mean"           # W

# =========================
# Reference values (fixed)
# =========================
F_REF = 50.0      # Hz (fixed)
T_REF = 273.15    # K
Q_REF = 1.0       # saturated vapor

# =========================
# Model parameter names (8)
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
    "A_dis": 86.1e-6,   # important: e-6 (matches your fitting scripts)
    "V_IC": 16.11e-6,
    "alpha_loss": 0.16,
    "W_dot_loss_ref": 83.0,
    "m_dot_ref": None,   # computed
    "f_ref": F_REF,
}

# =========================
# Inputs wrapper (what model expects)
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
# CSV I/O
# =========================
def read_dataset_csv(path: Path, sep: str, header: int, decimal: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, header=header, decimal=decimal)

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

# =========================
# Model helpers
# =========================
def pick_model(model_name: str, N_max_hz: float, V_h_m3: float, parameters: dict):
    m = model_name.lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=parameters)
    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(N_max=N_max_hz, V_h=V_h_m3, parameters=parameters)
    raise ValueError("Unknown --model. Use: original | modified")

def compute_m_dot_ref(med: RefProp, V_h_m3: float) -> float:
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * float(F_REF)

def norm_oil(s: str) -> str:
    return str(s).strip().lower()

def _finite(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")

def _add_compact_state(rec: dict, prefix: str, st) -> None:
    """
    Only p and T (plus density for internal m_dot_3 calculation / debugging).
    """
    if st is None:
        rec[f"{prefix}_p_Pa"] = float("nan")
        rec[f"{prefix}_p_bar"] = float("nan")
        rec[f"{prefix}_T_K"] = float("nan")
        rec[f"{prefix}_T_C"] = float("nan")
        rec[f"{prefix}_rho_kgpm3"] = float("nan")
        return

    p = _finite(getattr(st, "p", np.nan))
    T = _finite(getattr(st, "T", np.nan))
    rho = _finite(getattr(st, "d", np.nan))

    rec[f"{prefix}_p_Pa"] = p
    rec[f"{prefix}_p_bar"] = pa_to_bar(p) if np.isfinite(p) else float("nan")
    rec[f"{prefix}_T_K"] = T
    rec[f"{prefix}_T_C"] = k_to_c(T) if np.isfinite(T) else float("nan")
    rec[f"{prefix}_rho_kgpm3"] = rho

def main():
    ap = argparse.ArgumentParser(
        description="Compact batch simulation for Molinaroli compressor model (RefProp backend)."
    )

    ap.add_argument("--csv", required=True, help="Input CSV path (units row + header row).")
    ap.add_argument("--out", default=None, help="Output CSV path (default: results/batch_<timestamp>.csv)")

    ap.add_argument("--model", default="original", help="original | modified")
    ap.add_argument("--refrigerant", default="PROPANE", help="RefProp fluid name (e.g. PROPANE)")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0, help="Max speed [rpm] from datasheet")
    ap.add_argument("--V_h_cm3", type=float, default=30.7, help="Displacement volume [cm^3] from datasheet")

    ap.add_argument("--sep", default=";", help="CSV separator (default: ';')")
    ap.add_argument("--decimal", default=",", help="Decimal separator (default: ',')")
    ap.add_argument("--header", type=int, default=1, help="Header row index (default: 1 because line 0 is units)")

    ap.add_argument("--oil_col", default=OIL_COL_DEFAULT, help="Oil column name (default: Ölbezeichnung)")
    ap.add_argument("--oil", default="all", help="LPG100 | LPG68 | all")

    ap.add_argument("--col_p_suc", default=P_SUC_COL_DEFAULT)
    ap.add_argument("--col_T_suc", default=T_SUC_COL_DEFAULT)
    ap.add_argument("--col_p_out", default=P_OUT_COL_DEFAULT)
    ap.add_argument("--col_T_amb", default=T_AMB_COL_DEFAULT)
    ap.add_argument("--col_speed", default=SPEED_COL_DEFAULT)

    ap.add_argument("--col_m_meas", default=M_FLOW_MEAS_COL_DEFAULT, help="Optional: measured m_dot column (g/s)")
    ap.add_argument("--col_P_meas", default=P_EL_MEAS_COL_DEFAULT, help="Optional: measured Pel column (W)")

    ap.add_argument("--max_rows", type=int, default=None, help="Optional: limit number of rows for testing")
    ap.add_argument("--params_csv", default=None, help="Optional: ONE-ROW params CSV (fitted/start params)")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Output path
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        Path("results").mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("results") / f"batch_{args.oil.lower()}_{args.model.lower()}_{ts}.csv"

    # Read dataset CSV
    df = read_dataset_csv(csv_path, sep=args.sep, header=args.header, decimal=args.decimal)

    # Filter by oil
    oil_arg = args.oil.strip().lower()
    if oil_arg != "all":
        if args.oil_col not in df.columns:
            raise ValueError(f"--oil was set but oil column '{args.oil_col}' not found in CSV.")
        df = df[df[args.oil_col].astype(str).apply(norm_oil) == oil_arg]

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    required = [args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_T_amb, args.col_speed]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid rows after dropping NaNs in required columns (and oil filter).")

    # Datasheet conversions
    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6

    # Media (RefProp)
    med = RefProp(fluid_name=args.refrigerant)

    # Parameters
    if args.params_csv:
        params_base = load_params_csv(Path(args.params_csv))
    else:
        params_base = DEFAULT_PARAMS.copy()

    params_base["f_ref"] = F_REF
    params_base["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

    # Build compressor ONCE
    comp = pick_model(args.model, N_max_hz=N_max_hz, V_h_m3=V_h_m3, parameters=params_base)
    comp.med_prop = med
    comp.debug_enabled = True

    has_m_meas = args.col_m_meas in df.columns
    has_P_meas = args.col_P_meas in df.columns

    results = []

    for i, row in df.iterrows():
        p_suc_pa = bar_to_pa(row[args.col_p_suc])
        p_out_pa = bar_to_pa(row[args.col_p_out])
        T_suc_K = c_to_k(row[args.col_T_suc])
        T_amb_K = c_to_k(row[args.col_T_amb])
        f_oper_hz = rpm_to_hz(row[args.col_speed])

        n_rel = float(max(1e-6, min(1.0, f_oper_hz / N_max_hz)))

        try:
            n_abs = float(comp.get_n_absolute(n_rel))
        except Exception:
            n_abs = float("nan")

        rec = {
            "row_index": int(i),
            "success": True,
            "error": "",
            "model": args.model,
            "backend": "RefProp",
            "refrigerant": args.refrigerant,
            "oil": str(row[args.oil_col]) if args.oil_col in df.columns else "",
            # Inputs
            "p_suc_bar_in": float(row[args.col_p_suc]),
            "T_suc_C_in": float(row[args.col_T_suc]),
            "p_out_bar_in": float(row[args.col_p_out]),
            "T_amb_C_in": float(row[args.col_T_amb]),
            "N_rpm_in": float(row[args.col_speed]),
            "f_oper_hz": float(f_oper_hz),
            "n_rel": float(n_rel),
            "n_abs_hz": float(n_abs),
        }

        fs_state = FlowsheetState()

        try:
            comp.state_inlet = med.calc_state("PT", float(p_suc_pa), float(T_suc_K))
            inputs = SimpleInputs(control=Control(n=n_rel), T_amb=float(T_amb_K))
            comp.calc_state_outlet(p_outlet=float(p_out_pa), inputs=inputs, fs_state=fs_state)

            # Compact thermodynamic states (p,T,rho)
            _add_compact_state(rec, "st_in", getattr(comp, "state_inlet", None))
            _add_compact_state(rec, "c1", getattr(comp, "state_c_1", None))
            _add_compact_state(rec, "c3", getattr(comp, "state_c_3", None))
            _add_compact_state(rec, "c4", getattr(comp, "state_c_4", None))
            _add_compact_state(rec, "c5", getattr(comp, "state_c_5", None))
            _add_compact_state(rec, "st_out", getattr(comp, "state_outlet", None))

            # Outputs
            rec["m_flow_kg_s"] = float(comp.m_flow)
            rec["m_flow_g_s"] = float(comp.m_flow) * 1000.0
            rec["P_el_W"] = float(comp.P_el)

            # Wall temperature
            rec["T_wall_K"] = _finite(getattr(comp, "T_w", np.nan))
            rec["T_wall_C"] = k_to_c(rec["T_wall_K"]) if np.isfinite(rec["T_wall_K"]) else float("nan")

            # Discharge temperature shortcut
            rec["T_dis_K"] = _finite(getattr(comp.state_outlet, "T", np.nan))
            rec["T_dis_C"] = k_to_c(rec["T_dis_K"]) if np.isfinite(rec["T_dis_K"]) else float("nan")

            # Internal mass flow & Wdot split (reconstructed from states, like in your model)
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
                rec["m_dot_3_kg_s"] = float("nan")
                rec["W_dot_int_W"] = float("nan")
                rec["W_dot_loss_W"] = float("nan")
                rec["W_dot_int_plus_loss_W"] = float("nan")
                rec["W_dot_loss_share"] = float("nan")

            # Measurements + residuals (optional)
            if has_m_meas and pd.notna(row[args.col_m_meas]):
                rec["m_meas_g_s"] = float(row[args.col_m_meas])
                m_meas = gs_to_kgps(row[args.col_m_meas])
                rec["e_m_rel"] = (rec["m_flow_kg_s"] / m_meas) - 1.0 if m_meas > 0 else np.nan

            if has_P_meas and pd.notna(row[args.col_P_meas]):
                rec["P_meas_W"] = float(row[args.col_P_meas])
                P_meas = float(row[args.col_P_meas])
                rec["e_P_rel"] = (rec["P_el_W"] / P_meas) - 1.0 if P_meas > 0 else np.nan

        except Exception as e:
            rec["success"] = False
            rec["error"] = str(e)

            # Fill compact outputs with NaNs
            for prefix in ["st_in", "c1", "c3", "c4", "c5", "st_out"]:
                _add_compact_state(rec, prefix, None)

            rec["m_flow_kg_s"] = np.nan
            rec["m_flow_g_s"] = np.nan
            rec["P_el_W"] = np.nan
            rec["T_wall_K"] = np.nan
            rec["T_wall_C"] = np.nan
            rec["T_dis_K"] = np.nan
            rec["T_dis_C"] = np.nan

            rec["m_dot_3_kg_s"] = np.nan
            rec["W_dot_int_W"] = np.nan
            rec["W_dot_loss_W"] = np.nan
            rec["W_dot_int_plus_loss_W"] = np.nan
            rec["W_dot_loss_share"] = np.nan

            if has_m_meas:
                rec["m_meas_g_s"] = float(row[args.col_m_meas]) if pd.notna(row[args.col_m_meas]) else np.nan
                rec["e_m_rel"] = np.nan
            if has_P_meas:
                rec["P_meas_W"] = float(row[args.col_P_meas]) if pd.notna(row[args.col_P_meas]) else np.nan
                rec["e_P_rel"] = np.nan

        results.append(rec)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)

    n_ok = int(out_df["success"].sum())
    n_total = len(out_df)

    print("\n=== Batch done ===")
    print(f"oil: {args.oil}, model: {args.model}, refrigerant: {args.refrigerant}, backend: RefProp")
    print(f"points: {n_ok}/{n_total} successful")
    print(f"params source: {args.params_csv if args.params_csv else 'DEFAULT_PARAMS'}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()