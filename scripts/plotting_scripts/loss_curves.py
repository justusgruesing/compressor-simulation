# scripts/plotting_scripts/loss_curves.py
#
# Simulates compressor operating points with fine resolution and plots
# the loss term breakdown as continuous curves.
#
# The script does NOT read measurement data. It takes fitted parameters,
# sweeps one operating parameter, and simulates each point itself.
#
# Activate REFPROP:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Sweep T_evap at multiple T_cond levels:
#   python scripts/plotting_scripts/loss_curves.py \
#       --params_csv results/ga_fit/fitted_params_lpg68_modified_ga_2026-03-19.csv \
#       --vary T_evap --T_evap_min -5 --T_evap_max 30 \
#       --T_cond 30 40 50 60 --N_rpm 3600 --SH_K 10
#
#   # Sweep T_cond at multiple T_evap levels:
#   python scripts/plotting_scripts/loss_curves.py --params_csv results/final_results/Modified_LPG100/Fitting/fitted_params_lpg100_modified_ga_2026-03-28_092941.csv --vary T_cond --T_cond_min 25 --T_cond_max 65 --T_evap 0 5 10 15 20 25 --N_rpm 3600 --SH_K 10
#
#   # Sweep speed at one condition, normalized:
#   python scripts/plotting_scripts/loss_curves.py \
#       --params_csv results/ga_fit/fitted_params_lpg68_modified_ga_2026-03-19.csv \
#       --vary speed --N_rpm_min 1800 --N_rpm_max 7200 \
#       --T_evap 10 --T_cond 50 --SH_K 10 --normalize
#
#   # Sweep superheat, lines mode:
#   python scripts/plotting_scripts/loss_curves.py \
#       --params_csv results/ga_fit/fitted_params_lpg68_modified_ga_2026-03-19.csv \
#       --vary superheat --SH_K_min 5 --SH_K_max 35 \
#       --T_evap 10 --T_cond 50 --N_rpm 3600 --plot_mode lines

from __future__ import annotations

import argparse
import itertools
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

# Optional: oil_path model
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

plt.style.use("ebc.paper.mplstyle")


# =========================================================
# Constants
# =========================================================
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0


# =========================================================
# Parameter definitions
# =========================================================
PARAM_NAMES_ORIGINAL = [
    "Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
    "V_IC", "alpha_loss", "W_dot_loss_ref",
]

PARAM_NAMES_MODIFIED = [
    "Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
    "V_IC", "alpha_loss", "W_dot_loss_ref", "alpha_fric_tot",
]

DEFAULT_PARAMS_ORIGINAL = {
    "Ua_suc_ref": 16.05, "Ua_dis_ref": 13.96, "Ua_amb": 0.36,
    "A_tot": 9.47e-9, "A_dis": 86.1e-6, "V_IC": 30.7e-6,
    "alpha_loss": 0.16, "W_dot_loss_ref": 83.0,
    "m_dot_ref": None, "f_ref": F_REF,
}

DEFAULT_PARAMS_MODIFIED = {
    "Ua_suc_ref": 16.05, "Ua_dis_ref": 13.96, "Ua_amb": 0.36,
    "A_tot": 9.47e-9, "A_dis": 86.1e-6, "V_IC": 30.7e-6,
    "alpha_loss": 0.16, "W_dot_loss_ref": 10.0, "alpha_fric_tot": 120.0,
    "m_dot_ref": None, "f_ref": F_REF,
}

PARAM_NAMES_OIL_PATH = [
    "Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
    "V_IC", "alpha_loss", "W_dot_loss_ref", "alpha_fric_tot",
    "m_dot_oil_ref", "Ua_suc_oil_ref",
]

DEFAULT_PARAMS_OIL_PATH = {
    "Ua_suc_ref": 16.05, "Ua_dis_ref": 13.96, "Ua_amb": 0.36,
    "A_tot": 9.47e-9, "A_dis": 86.1e-6, "V_IC": 30.7e-6,
    "alpha_loss": 0.16, "W_dot_loss_ref": 10.0, "alpha_fric_tot": 120.0,
    "m_dot_oil_ref": 0.005, "Ua_suc_oil_ref": 5.0,
    "m_dot_ref": None, "f_ref": F_REF,
}


# =========================================================
# Helpers
# =========================================================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _finite(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def c_to_k(t):
    return float(t) + 273.15


def k_to_c(t):
    return float(t) - 273.15


def rpm_to_hz(n):
    return float(n) / 60.0


# =========================================================
# Model helpers
# =========================================================
def map_refrigerant_for_modified_model(name: str) -> str:
    s = str(name).strip().upper()
    if s in {"PROPANE", "R290", "PROPAN"}:
        return "propane"
    return str(name).strip()


def map_oil_for_modified_model(name: str) -> str:
    s = str(name).strip().lower().replace(" ", "")
    if s == "lpg68":
        return "LPG 68"
    if s == "lpg100":
        return "LPG 100"
    raise ValueError(f"Unsupported oil: {name}")


def get_param_names(model: str) -> list[str]:
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return list(PARAM_NAMES_ORIGINAL)
    if m in ("mod", "modified"):
        return list(PARAM_NAMES_MODIFIED)
    if m in ("oil_path", "oilpath"):
        return list(PARAM_NAMES_OIL_PATH)
    raise ValueError("Unknown model. Use original | modified | oil_path")


def get_default_params(model: str) -> dict:
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return dict(DEFAULT_PARAMS_ORIGINAL)
    if m in ("mod", "modified"):
        return dict(DEFAULT_PARAMS_MODIFIED)
    if m in ("oil_path", "oilpath"):
        return dict(DEFAULT_PARAMS_OIL_PATH)
    raise ValueError("Unknown model. Use original | modified | oil_path")


def make_compressor(model, N_max_hz, V_h_m3, params, refrigerant_name, oil_name=None):
    m = str(model).lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max_hz, V_h=V_h_m3, parameters=params)
    if m in ("mod", "modified"):
        if oil_name is None:
            raise ValueError("Modified model requires oil_name.")
        return Molinaroli_2017_Compressor_Modified(
            N_max=N_max_hz, V_h=V_h_m3,
            fluid_name=map_refrigerant_for_modified_model(refrigerant_name),
            lub_name=map_oil_for_modified_model(oil_name),
            parameters=params,
        )
    if m in ("oil_path", "oilpath"):
        if not OIL_PATH_AVAILABLE:
            raise ImportError("oil_path model not available in this vclibpy installation.")
        if oil_name is None:
            raise ValueError("oil_path model requires oil_name.")
        return Molinaroli_2017_Compressor_OilPath(
            N_max=N_max_hz, V_h=V_h_m3,
            fluid_name=map_refrigerant_for_modified_model(refrigerant_name),
            lub_name=map_oil_for_modified_model(oil_name),
            parameters=params,
        )
    raise ValueError("Unknown model. Use original | modified | oil_path")


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
# Operating point computation
# =========================================================
def compute_suction_state(med, T_evap_C: float, SH_K: float):
    """Compute suction pressure and temperature from T_evap and superheat."""
    T_evap_K = c_to_k(T_evap_C)
    state_sat = med.calc_state("TQ", T_evap_K, Q_REF)
    p_suc = float(state_sat.p)
    T_suc_K = T_evap_K + float(SH_K)
    return p_suc, T_suc_K


def compute_discharge_pressure(med, T_cond_C: float):
    """Compute discharge pressure from T_cond (saturated liquid)."""
    T_cond_K = c_to_k(T_cond_C)
    state_sat = med.calc_state("TQ", T_cond_K, 0.0)
    return float(state_sat.p)


def simulate_single_point(comp, med, p_suc_pa, T_suc_K, p_out_pa, n_rel, T_amb_K):
    """
    Simulate one operating point. Returns dict with loss terms or None on failure.
    """
    inputs = SimpleInputs(
        control=Control(n=max(1e-9, min(1.0, n_rel))),
        T_amb=float(T_amb_K),
    )
    fs_state = FlowsheetState()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comp.state_inlet = med.calc_state("PT", float(p_suc_pa), float(T_suc_K))
            comp.calc_state_outlet(p_outlet=float(p_out_pa), inputs=inputs, fs_state=fs_state)

        result = {
            "m_flow": _finite(comp.m_flow),
            "P_el": _finite(comp.P_el),
            "T_dis_K": _finite(comp.state_outlet.T),
            "W_dot_int": _finite(getattr(comp, "W_dot_int", np.nan)),
            "W_dot_loss": _finite(getattr(comp, "W_dot_loss", np.nan)),
            "W_dot_loss_load": _finite(getattr(comp, "W_dot_loss_load", np.nan)),
            "W_dot_loss_ref_term": _finite(getattr(comp, "W_dot_loss_ref_term", np.nan)),
            "W_dot_loss_fric": _finite(getattr(comp, "W_dot_loss_fric", np.nan)),
            "W_dot_oil_recirc": _finite(getattr(comp, "W_dot_oil_recirc", np.nan)),
            "mu_mix_eff": _finite(getattr(comp, "mu_mix_eff", np.nan)),
            "T_oil_sump": _finite(getattr(comp, "T_oil_sump", np.nan)),
        }

        if not np.isfinite(result["P_el"]) or result["P_el"] <= 0:
            return None

        return result

    except Exception:
        return None


def run_sweep(
    med, model, refrigerant_name, oil_name,
    params, N_max_hz, V_h_m3,
    vary_name, vary_values,
    T_evap_C_fixed, T_cond_C_fixed, N_rpm_fixed, SH_K_fixed,
    T_amb_C=25.0,
):
    """
    Sweep one parameter over vary_values, simulate each point.
    Compressor is built once per sweep for efficiency.
    """
    comp = make_compressor(
        model=model, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
        params=params, refrigerant_name=refrigerant_name, oil_name=oil_name,
    )
    comp.med_prop = med
    if hasattr(comp, "debug_enabled"):
        comp.debug_enabled = False

    T_amb_K = c_to_k(T_amb_C)
    records = []

    for val in vary_values:
        if vary_name == "T_evap":
            T_evap_C, T_cond_C, N_rpm, SH_K = float(val), T_cond_C_fixed, N_rpm_fixed, SH_K_fixed
        elif vary_name == "T_cond":
            T_evap_C, T_cond_C, N_rpm, SH_K = T_evap_C_fixed, float(val), N_rpm_fixed, SH_K_fixed
        elif vary_name == "speed":
            T_evap_C, T_cond_C, N_rpm, SH_K = T_evap_C_fixed, T_cond_C_fixed, float(val), SH_K_fixed
        elif vary_name == "superheat":
            T_evap_C, T_cond_C, N_rpm, SH_K = T_evap_C_fixed, T_cond_C_fixed, N_rpm_fixed, float(val)
        else:
            raise ValueError(f"Unknown vary: {vary_name}")

        try:
            p_suc, T_suc_K = compute_suction_state(med, T_evap_C, SH_K)
            p_out = compute_discharge_pressure(med, T_cond_C)
        except Exception:
            continue

        f_hz = rpm_to_hz(N_rpm)
        n_rel = f_hz / N_max_hz

        if p_out <= p_suc or n_rel <= 0 or n_rel > 1.0:
            continue

        result = simulate_single_point(comp, med, p_suc, T_suc_K, p_out, n_rel, T_amb_K)
        if result is None:
            continue

        rec = {
            "vary_value": float(val),
            "T_evap_C": T_evap_C, "T_cond_C": T_cond_C,
            "N_rpm": N_rpm, "SH_K": SH_K,
            "p_suc_bar": p_suc / 1e5, "p_out_bar": p_out / 1e5,
            "pressure_ratio": p_out / p_suc,
            "T_suc_C": k_to_c(T_suc_K),
        }
        rec.update(result)
        records.append(rec)

    return pd.DataFrame(records)


# =========================================================
# Plot
# =========================================================
def plot_loss_curves(
    sweep_results: dict[str, pd.DataFrame],
    loss_cols: list[tuple[str, str]],
    vary_label: str,
    title: str,
    out_path: Path,
    normalize: bool = False,
    plot_mode: str = "stacked_area",
):
    series_list = list(sweep_results.items())
    n_series = len(series_list)

    if n_series == 0:
        print("  [SKIP] No series to plot.")
        return

    n_cols_layout = min(3, n_series)
    n_rows_layout = int(np.ceil(n_series / n_cols_layout))

    fig, axes = plt.subplots(
        n_rows_layout, n_cols_layout,
        figsize=(7 * n_cols_layout, 5.5 * n_rows_layout),
        squeeze=False,
    )

    colors = ["#EC635C", "#4B81C4", "#F49961", "#6EBB96"]

    for idx, (series_label, df) in enumerate(series_list):
        row_idx = idx // n_cols_layout
        col_idx = idx % n_cols_layout
        ax = axes[row_idx, col_idx]

        if df.empty:
            ax.set_title(f"{series_label}\n(keine gültigen Punkte)")
            continue

        x = df["vary_value"].to_numpy(dtype=float)

        loss_arrays = {}
        for col, label in loss_cols:
            if col in df.columns:
                vals = df[col].to_numpy(dtype=float)
                vals = np.where(np.isfinite(vals), vals, 0.0)
                loss_arrays[label] = vals

        if not loss_arrays:
            continue

        if normalize:
            total = np.zeros(len(x))
            for label in loss_arrays:
                total += np.abs(loss_arrays[label])
            total = np.where(total > 0, total, 1.0)
            for label in loss_arrays:
                loss_arrays[label] = np.abs(loss_arrays[label]) / total * 100.0

        if plot_mode == "stacked_area" and len(x) > 1:
            y_stack = [loss_arrays[label] for label in loss_arrays]
            labels = list(loss_arrays.keys())

            ax.stackplot(
                x, *y_stack,
                labels=labels,
                colors=colors[:len(labels)],
                alpha=0.85,
            )

            cumsum = np.zeros(len(x))
            for i, (label, vals) in enumerate(loss_arrays.items()):
                cumsum += vals
                ax.plot(x, cumsum, color=colors[i % len(colors)], linewidth=1.2, alpha=0.9)
        else:
            for i, (label, vals) in enumerate(loss_arrays.items()):
                ax.plot(
                    x, vals,
                    marker="o", markersize=3,
                    color=colors[i % len(colors)],
                    linewidth=1.8,
                    label=label,
                )

        ax.set_title(series_label, fontsize=12)
        ax.set_xlabel(vary_label)

        if normalize:
            ax.set_ylabel("Anteil [%]")
            ax.set_ylim(0, 105)
        else:
            ax.set_ylabel("Verlustleistung [W]")

        ax.grid(True, linewidth=0.6, alpha=0.35)

    # Hide empty subplots
    for idx in range(n_series, n_rows_layout * n_cols_layout):
        axes[idx // n_cols_layout, idx % n_cols_layout].set_visible(False)

    # Legend below figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(loss_cols),
            frameon=True,
            fontsize=11,
        )

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [OK] Saved: {out_path}  ({n_series} series)")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Simulate and plot continuous loss term curves by sweeping one operating parameter."
    )

    ap.add_argument("--params_csv", required=True, type=Path, help="Fitted parameter CSV")

    ap.add_argument("--model", default="auto", help="original | modified | oil_path | auto")
    ap.add_argument("--refrigerant", default="auto", help="RefProp fluid or auto")
    ap.add_argument("--oil", default="auto", help="LPG68 | LPG100 | auto")

    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)
    ap.add_argument("--T_amb_C", type=float, default=25.0, help="Ambient temperature [°C]")

    ap.add_argument(
        "--vary", required=True,
        choices=["T_evap", "T_cond", "speed", "superheat"],
        help="Which parameter to sweep",
    )

    # Sweep ranges
    ap.add_argument("--T_evap_min", type=float, default=-5.0)
    ap.add_argument("--T_evap_max", type=float, default=25.0)
    ap.add_argument("--T_cond_min", type=float, default=25.0)
    ap.add_argument("--T_cond_max", type=float, default=65.0)
    ap.add_argument("--N_rpm_min", type=float, default=1800.0)
    ap.add_argument("--N_rpm_max", type=float, default=7200.0)
    ap.add_argument("--SH_K_min", type=float, default=5.0)
    ap.add_argument("--SH_K_max", type=float, default=35.0)
    ap.add_argument("--n_points", type=int, default=50, help="Number of sweep points")

    # Fixed parameters (multiple values → multiple series)
    ap.add_argument("--T_evap", type=float, nargs="+", default=[10.0],
                    help="Fixed T_evap [°C]. Multiple values create multiple series.")
    ap.add_argument("--T_cond", type=float, nargs="+", default=[50.0],
                    help="Fixed T_cond [°C]. Multiple values create multiple series.")
    ap.add_argument("--N_rpm", type=float, nargs="+", default=[3600.0],
                    help="Fixed speed [rpm]. Multiple values create multiple series.")
    ap.add_argument("--SH_K", type=float, nargs="+", default=[10.0],
                    help="Fixed superheat [K]. Multiple values create multiple series.")

    # Plot options
    ap.add_argument("--normalize", action="store_true", help="Normalize to 100%%")
    ap.add_argument("--plot_mode", choices=["stacked_area", "lines"], default="stacked_area")
    ap.add_argument("--out_dir", default="results/loss_curves")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    if not args.params_csv.exists():
        raise FileNotFoundError(args.params_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Resolve auto values
    # -------------------------
    params_peek = pd.read_csv(args.params_csv).iloc[0].to_dict()

    if args.model == "auto":
        args.model = str(params_peek.get("model", "original"))
    if args.refrigerant == "auto":
        args.refrigerant = str(params_peek.get("refrigerant", "PROPANE"))
    if args.oil == "auto":
        args.oil = str(params_peek.get("oil", "LPG68"))

    params, params_meta = load_params_csv(args.params_csv, args.model)

    # -------------------------
    # RefProp (single shared instance)
    # -------------------------
    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6

    med = RefProp(fluid_name=args.refrigerant)
    params["f_ref"] = F_REF
    params["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)

    print(f"  Model:       {args.model}")
    print(f"  Oil:         {args.oil}")
    print(f"  Refrigerant: {args.refrigerant}")
    print(f"  m_dot_ref:   {params['m_dot_ref'] * 1e3:.4f} g/s")
    print(f"  Vary:        {args.vary}")

    # -------------------------
    # Build sweep values and series
    # -------------------------
    vary_config = {
        "T_evap": {
            "values": np.linspace(args.T_evap_min, args.T_evap_max, args.n_points),
            "label": "Verdampfungstemperatur [°C]",
            "series_params": ["T_cond", "N_rpm", "SH_K"],
        },
        "T_cond": {
            "values": np.linspace(args.T_cond_min, args.T_cond_max, args.n_points),
            "label": "Kondensationstemperatur [°C]",
            "series_params": ["T_evap", "N_rpm", "SH_K"],
        },
        "speed": {
            "values": np.linspace(args.N_rpm_min, args.N_rpm_max, args.n_points),
            "label": "Drehzahl [rpm]",
            "series_params": ["T_evap", "T_cond", "SH_K"],
        },
        "superheat": {
            "values": np.linspace(args.SH_K_min, args.SH_K_max, args.n_points),
            "label": "Überhitzung [K]",
            "series_params": ["T_evap", "T_cond", "N_rpm"],
        },
    }

    cfg = vary_config[args.vary]
    vary_values = cfg["values"]
    vary_label = cfg["label"]

    # Cartesian product of fixed parameter lists → series
    series_param_map = {
        "T_evap": args.T_evap,
        "T_cond": args.T_cond,
        "N_rpm": args.N_rpm,
        "SH_K": args.SH_K,
    }

    series_keys = cfg["series_params"]
    series_value_lists = [series_param_map[k] for k in series_keys]
    series_combinations = list(itertools.product(*series_value_lists))

    print(f"  Sweep: {len(vary_values)} points, {vary_values[0]:.1f} → {vary_values[-1]:.1f}")
    print(f"  Series: {len(series_combinations)} combinations of {series_keys}")

    # Oil name for modified model
    m = str(args.model).lower().strip()
    oil_name = args.oil if m in ("mod", "modified") else None

    # -------------------------
    # Run sweeps
    # -------------------------
    sweep_results = {}

    label_map = {
        "T_evap": ("$T_{evap}$", "°C"),
        "T_cond": ("$T_{cond}$", "°C"),
        "N_rpm": ("N", "rpm"),
        "SH_K": ("SH", "K"),
    }

    for combo in series_combinations:
        fixed = dict(zip(series_keys, combo))

        T_evap_fixed = fixed.get("T_evap", args.T_evap[0])
        T_cond_fixed = fixed.get("T_cond", args.T_cond[0])
        N_rpm_fixed = fixed.get("N_rpm", args.N_rpm[0])
        SH_K_fixed = fixed.get("SH_K", args.SH_K[0])

        # Series label
        label_parts = []
        for k, v in fixed.items():
            name, unit = label_map[k]
            label_parts.append(f"{name}={v:.0f} {unit}")
        series_label = ", ".join(label_parts)

        print(f"  Simulating: {series_label} ...")

        df_sweep = run_sweep(
            med=med, model=args.model,
            refrigerant_name=args.refrigerant, oil_name=oil_name,
            params=params, N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            vary_name=args.vary, vary_values=vary_values,
            T_evap_C_fixed=T_evap_fixed, T_cond_C_fixed=T_cond_fixed,
            N_rpm_fixed=N_rpm_fixed, SH_K_fixed=SH_K_fixed,
            T_amb_C=args.T_amb_C,
        )

        n_ok = len(df_sweep)
        n_fail = len(vary_values) - n_ok
        if n_fail > 0:
            print(f"    → {n_ok}/{len(vary_values)} successful ({n_fail} failed)")

        sweep_results[series_label] = df_sweep

    # -------------------------
    # Detect loss columns
    # -------------------------
    loss_col_candidates = [
        ("W_dot_loss_load", "Lastabhängig"),
        ("W_dot_loss_ref_term", "Drehzahlabhängig"),
        ("W_dot_loss_fric", "Viskositätsreibung"),
        ("W_dot_oil_recirc", "Öl-Hydraulik"),
    ]

    first_df = next((df for df in sweep_results.values() if not df.empty), None)
    if first_df is None:
        raise RuntimeError("All sweeps failed — no data to plot.")

    loss_cols = []
    for col, label in loss_col_candidates:
        if col in first_df.columns:
            vals = first_df[col].to_numpy(dtype=float)
            if np.any(np.isfinite(vals) & (vals != 0)):
                loss_cols.append((col, label))

    if not loss_cols:
        raise RuntimeError("No valid loss term data in simulation results.")

    print(f"  Loss terms: {[label for _, label in loss_cols]}")

    # -------------------------
    # Title
    # -------------------------
    norm_tag = " (normiert)" if args.normalize else ""
    title = f"Verlustanteile vs. {vary_label}{norm_tag}"
    title += f"\n{args.model.capitalize()} | Öl: {args.oil} | {args.refrigerant}"

    # -------------------------
    # Plot
    # -------------------------
    stamp = _ts()
    norm_suffix = "_norm" if args.normalize else ""
    out_path = out_dir / f"loss_curves_{args.vary}{norm_suffix}_{stamp}.{args.out_format}"

    plot_loss_curves(
        sweep_results=sweep_results,
        loss_cols=loss_cols,
        vary_label=vary_label,
        title=title,
        out_path=out_path,
        normalize=args.normalize,
        plot_mode=args.plot_mode,
    )

    # -------------------------
    # Save simulation data
    # -------------------------
    all_data = []
    for series_label, df in sweep_results.items():
        if not df.empty:
            df_out = df.copy()
            df_out["series"] = series_label
            all_data.append(df_out)

    if all_data:
        data_csv = out_path.with_suffix(".csv")
        pd.concat(all_data, ignore_index=True).to_csv(data_csv, index=False)
        print(f"  [OK] Data saved: {data_csv}")

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()