# scripts/plotting_scripts/oil_path_wKM.py
#
# Plots the dissolved refrigerant mass fraction w_KM at the various state
# points along the oil path through the compressor (oil_path model only).
#
# State points (x-axis, in order):
#   1. Ölsumpf          (w_KM_sump)   — p_dis, T_sump
#   2. Nach Drosselung   (w_KM_suc)    — p_suc, T_throttle
#   3. Nach Saug-WT      (w_KM_after)  — p_suc, T_oil_after
#   4. Nach Mischung     (w_KM_mix)    — p_dis, T_mix
#   5. Bei T_dis         (w_KM_dis)    — p_dis, T_dis
#   6. → Ölsumpf         (w_KM_sump)   — closes the loop
#
# Multiple operating points can be shown on the same plot, each as a
# separate colored line.
#
# Activate REFPROP first:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Single operating point:
#   python scripts/plotting_scripts/oil_path_wKM.py \
#       --params_csv results/.../fitted_params_lpg68_oil_path_ga_....csv \
#       --oil LPG68 --T_evap 0 --T_cond 50
#
#   # Multiple operating points (vary T_evap):
#   python scripts/plotting_scripts/oil_path_wKM.py --params_csv results/final_results/Oil_Path_LPG100/Fitting/fitted_params_lpg100_oil_path_ga_2026-04-18_041610.csv --oil PAG100 --T_evap -5 0 10 20 --T_cond 50 --out_format svg
#
#   # Vary T_cond instead:
#   python scripts/plotting_scripts/oil_path_wKM.py --params_csv results/final_results/Oil_Path_LPG68/Fitting/fitted_params_lpg68_oil_path_ga_2026-04-17_113953.csv --oil LPG68 --T_evap 10 --T_cond 30 40 50 60

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

# oil_path model import
try:
    from vclibpy.components.compressors.rolling_piston_Molinaroli_oil_path import (
        Molinaroli_2017_Compressor_Oil_Path as Molinaroli_2017_Compressor_OilPath,
    )
    OIL_PATH_AVAILABLE = True
except ImportError:
    try:
        from vclibpy.components.compressors.rolling_piston_Molinaroli_oil_path import (
            Molinaroli_2017_Compressor_OilPath,
        )
        OIL_PATH_AVAILABLE = True
    except ImportError:
        OIL_PATH_AVAILABLE = False
        Molinaroli_2017_Compressor_OilPath = None

plt.style.use("ebc.paper.mplstyle")
plt.rcParams["svg.fonttype"] = "none"

# =========================================================
# Constants
# =========================================================
F_REF = 50.0
T_REF = 273.15
Q_REF = 1.0

PARAM_NAMES = [
    "Ua_suc_ref", "Ua_dis_ref", "Ua_amb", "A_tot", "A_dis",
    "V_IC", "alpha_loss", "W_dot_loss_ref", "alpha_fric_tot",
    "m_dot_oil_ref", "Ua_suc_oil_ref",
]

DEFAULT_PARAMS = {
    "Ua_suc_ref": 16.05, "Ua_dis_ref": 13.96, "Ua_amb": 0.36,
    "A_tot": 9.47e-9, "A_dis": 86.1e-6, "V_IC": 30.7e-6,
    "alpha_loss": 0.16, "W_dot_loss_ref": 10.0, "alpha_fric_tot": 120.0,
    "m_dot_oil_ref": 0.005, "Ua_suc_oil_ref": 5.0,
    "m_dot_ref": None, "f_ref": F_REF,
}

# State point definitions along the oil path
STATE_POINTS = [
    {"key": "sump",     "label": "Ölsumpf",           "short": "Sumpf"},
    {"key": "throttle", "label": "Nach\nDrosselung",   "short": "Drossel"},
    {"key": "after_ht", "label": "Nach\nSaug-WÜ",      "short": "Saug-WÜ"},
    {"key": "mix",      "label": "Nach\nMischung",     "short": "Mischung"},
    {"key": "dis",      "label": "Nach\nDruck-WÜ",      "short": "T_dis"},
    {"key": "sump_ret", "label": "Ölsumpf\n(Rückkehr)","short": "Sumpf*"},
]

# EBC color palette
EBC_COLORS = [
    "#EC635C", "#4B81C4", "#F49961", "#6EBB96",
    "#8768B4", "#B45955", "#CB74F4",
]


# =========================================================
# Helpers
# =========================================================
def _ts():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def c_to_k(t):
    return float(t) + 273.15

def k_to_c(t):
    return float(t) - 273.15

def rpm_to_hz(n):
    return float(n) / 60.0

def map_refrigerant(name):
    s = str(name).strip().upper()
    return "propane" if s in {"PROPANE", "R290", "PROPAN"} else str(name).strip()

def map_oil(name):
    """Return the internal lubricant name expected by the simulation."""
    s = str(name).strip().lower().replace(" ", "")
    if s in ("lpg68", "pag68"): return "LPG 68"
    if s in ("lpg100", "pag100"): return "LPG 100"
    raise ValueError(f"Unsupported oil: {name}")

def display_oil(name):
    """Return the display name for plots and titles."""
    s = str(name).strip().lower().replace(" ", "")
    if s in ("lpg68", "pag68"): return "PAG 68"
    if s in ("lpg100", "pag100"): return "PAG 100"
    return str(name)


def load_params_csv(path):
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    params = dict(DEFAULT_PARAMS)
    for k in PARAM_NAMES:
        if k in row and pd.notna(row[k]):
            params[k] = float(row[k])
    if "f_ref" in row and pd.notna(row["f_ref"]):
        params["f_ref"] = float(row["f_ref"])
    meta = {k: row[k] for k in ("oil", "refrigerant", "model") if k in row}
    return params, meta


def compute_m_dot_ref(med, V_h_m3):
    st = med.calc_state("TQ", T_REF, Q_REF)
    return float(st.d) * float(V_h_m3) * F_REF


@dataclass
class Control:
    n: float

@dataclass
class SimpleInputs:
    control: Control
    T_amb: float
    lsq_max_nfev: int = 50000
    lsq_ftol: float = 1e-12
    lsq_xtol: float = 1e-12


# =========================================================
# Simulate one operating point and extract w_KM values
# =========================================================
def run_operating_point(
    med, params, refrigerant_name, oil_name,
    N_max_hz, V_h_m3,
    T_evap_C, T_cond_C, N_rpm, SH_K, T_amb_C=25.0,
):
    """
    Run the oil_path model for one operating point and extract all w_KM
    state points along the oil path.

    Returns dict with keys matching STATE_POINTS, plus metadata.
    Returns None if simulation fails.
    """
    if not OIL_PATH_AVAILABLE:
        raise ImportError("oil_path model not available.")

    f_hz = rpm_to_hz(N_rpm)
    n_rel = f_hz / N_max_hz

    # Suction state
    T_evap_K = c_to_k(T_evap_C)
    state_sat_suc = med.calc_state("TQ", T_evap_K, Q_REF)
    p_suc = float(state_sat_suc.p)
    T_suc_K = T_evap_K + SH_K

    # Discharge pressure
    T_cond_K = c_to_k(T_cond_C)
    state_sat_dis = med.calc_state("TQ", T_cond_K, 0.0)
    p_out = float(state_sat_dis.p)

    # Build compressor
    comp = Molinaroli_2017_Compressor_OilPath(
        N_max=N_max_hz, V_h=V_h_m3,
        fluid_name=map_refrigerant(refrigerant_name),
        lub_name=map_oil(oil_name),
        parameters=params,
    )
    comp.med_prop = med
    if hasattr(comp, "debug_enabled"):
        comp.debug_enabled = False

    inputs = SimpleInputs(
        control=Control(n=max(1e-9, min(1.0, n_rel))),
        T_amb=c_to_k(T_amb_C),
    )
    fs_state = FlowsheetState()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comp.state_inlet = med.calc_state("PT", p_suc, T_suc_K)
            comp.calc_state_outlet(p_outlet=p_out, inputs=inputs, fs_state=fs_state)
    except Exception as e:
        print(f"  [FAIL] T_evap={T_evap_C}, T_cond={T_cond_C}: {type(e).__name__}: {e}")
        return None

    # Extract w_KM values from internal dicts
    oil = getattr(comp, "_current_oil_path", None)
    dis = getattr(comp, "_dis_ht_result", None)

    if oil is None:
        print(f"  [FAIL] T_evap={T_evap_C}, T_cond={T_cond_C}: no oil path result")
        return None

    w_KM_sump = oil.get("w_KM_sump")
    w_KM_suc = oil.get("w_KM_suc")
    w_KM_after = oil.get("w_KM_after")
    w_KM_mix = dis.get("w_KM_mix") if dis else None
    w_KM_dis = dis.get("w_KM_dis") if dis else None

    # Temperatures at each state point
    T_sump = oil.get("T_oil_sump")
    T_throttle = oil.get("T_throttle")
    T_oil_after = oil.get("T_oil_after")

    return {
        # w_KM at each state point
        "sump": w_KM_sump,
        "throttle": w_KM_suc,
        "after_ht": w_KM_after,
        "mix": w_KM_mix,
        "dis": w_KM_dis,
        "sump_ret": w_KM_sump,  # closes the loop

        # Temperatures
        "T_sump_C": k_to_c(T_sump) if T_sump else None,
        "T_throttle_C": k_to_c(T_throttle) if T_throttle else None,
        "T_oil_after_C": k_to_c(T_oil_after) if T_oil_after else None,
        "T_dis_C": k_to_c(float(comp.state_outlet.T)) if comp.state_outlet else None,

        # Pressures
        "p_suc_bar": p_suc / 1e5,
        "p_dis_bar": p_out / 1e5,

        # Operating point
        "T_evap_C": T_evap_C,
        "T_cond_C": T_cond_C,
        "N_rpm": N_rpm,
        "SH_K": SH_K,

        # Mass flows
        "m_dot_oil_gps": oil.get("m_dot_oil", 0) * 1e3,
        "m_dot_KM_degas_thr_gps": oil.get("m_dot_KM_degas_thr", 0) * 1e3,
        "m_dot_KM_degas_ht_gps": oil.get("m_dot_KM_degas_ht", 0) * 1e3,
    }


# =========================================================
# Plot
# =========================================================
def plot_oil_path_wKM(
    results: list[dict],
    out_path: Path,
    oil_name: str,
    show_temperatures: bool = True,
):
    """
    Plot w_KM at each state point along the oil path.
    Each result dict represents one operating point (one line in the plot).
    """
    n_states = len(STATE_POINTS)
    x_pos = np.arange(n_states)

    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)

    for i, res in enumerate(results):
        color = EBC_COLORS[i % len(EBC_COLORS)]

        # Build label from operating conditions
        label = (f"$T_{{\\mathrm{{verd}}}}$={res['T_evap_C']:.0f} °C, "
                 f"$T_{{\\mathrm{{kond}}}}$={res['T_cond_C']:.0f} °C")
        if len(set(r["N_rpm"] for r in results)) > 1:
            label += f", N={res['N_rpm']:.0f}"
        if len(set(r["SH_K"] for r in results)) > 1:
            label += f", SH={res['SH_K']:.0f} K"

        # Collect w_KM values
        y_vals = []
        for sp in STATE_POINTS:
            val = res.get(sp["key"])
            y_vals.append(float(val) * 100 if val is not None else np.nan)
        y_vals = np.array(y_vals)

        # Plot line + markers
        ax.plot(x_pos, y_vals, color=color, linewidth=2.0, marker="o",
                markersize=9, markeredgecolor="white", markeredgewidth=1.2,
                label=label, zorder=3)

    # --- Annotate w_KM values with collision-aware placement ---
    # Collect all y-values per state point for smart offset calculation
    all_y_per_state = {}  # {state_idx: [(line_idx, y_val, color), ...]}
    all_y_global = []
    for i, res in enumerate(results):
        color = EBC_COLORS[i % len(EBC_COLORS)]
        for j, sp in enumerate(STATE_POINTS):
            val = res.get(sp["key"])
            yp = float(val) * 100 if val is not None else np.nan
            if np.isfinite(yp):
                all_y_per_state.setdefault(j, []).append((i, yp, color))
                all_y_global.append(yp)

    MIN_GAP_PT = 18  # minimum gap between labels in points

    for j, entries in all_y_per_state.items():
        # Sort by y-value (ascending)
        entries_sorted = sorted(entries, key=lambda e: e[1])

        # Assign offsets: bottom half gets labels below, top half above
        # Horizontal offset alternates within each side for spread
        n = len(entries_sorted)
        offsets = []
        below_count = 0
        above_count = 0
        for rank, (line_idx, yp, color) in enumerate(entries_sorted):
            if rank < n / 2:
                base_dy = -MIN_GAP_PT
                extra_dy = -MIN_GAP_PT * (n // 2 - 1 - rank) * 0.4
                dx = -8 if (below_count % 2 == 0) else 8
                below_count += 1
            else:
                base_dy = MIN_GAP_PT
                extra_dy = MIN_GAP_PT * (rank - n // 2) * 0.4
                dx = 8 if (above_count % 2 == 0) else -8
                above_count += 1

            dy = base_dy + extra_dy
            ha = "right" if dx < 0 else "left" if dx > 0 else "center"
            offsets.append((line_idx, yp, color, dx, dy, ha))

        for line_idx, yp, color, dx, dy, ha in offsets:
            ax.annotate(
                f"{yp:.1f}%",
                (j, yp),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8, color=color, ha=ha,
                fontweight="bold",
            )

    # Axis setup
    ax.set_xticks(x_pos)
    ax.set_xticklabels([sp["label"] for sp in STATE_POINTS],
                       fontsize=10, ha="center")

    ax.set_ylabel("Gelöster Kältemittelmassenanteil $w_{\\mathrm{KM}}$ in %")
    ax.set_xlabel("Zustandspunkt im Schmierstoffpfad")

    # Add pressure regions as background shading
    ax.axvspan(-0.5, 0.5, alpha=0.06, color="#4B81C4", zorder=0)
    ax.axvspan(0.5, 2.5, alpha=0.06, color="#EC635C", zorder=0)
    ax.axvspan(2.5, 5.5, alpha=0.06, color="#4B81C4", zorder=0)

    # Pressure labels at top of axes (use axes transform for y, data for x)
    ax.text(0.0, 0.97, "$p_{\\mathrm{aus}}$", ha="center", va="top",
            fontsize=9, color="#4B81C4", fontstyle="italic",
            transform=ax.get_xaxis_transform())
    ax.text(1.5, 0.97, "$p_{\\mathrm{ein}}$", ha="center", va="top",
            fontsize=9, color="#EC635C", fontstyle="italic",
            transform=ax.get_xaxis_transform())
    ax.text(4.0, 0.97, "$p_{\\mathrm{aus}}$", ha="center", va="top",
            fontsize=9, color="#4B81C4", fontstyle="italic",
            transform=ax.get_xaxis_transform())

    ax.set_xlim(-0.5, n_states - 0.5)
    if all_y_global:
        y_margin = 1.0  # 1 percentage point margin
        ax.set_ylim(min(all_y_global) - y_margin, max(all_y_global) + y_margin)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)

    # Legend
    ax.legend(loc="best", fontsize=9, frameon=True)

    # Title
    ax.set_title(
        f"Gelöster Kältemittelanteil entlang des Schmierstoffpfads — {display_oil(oil_name)}",
        fontsize=13, pad=15,
    )

    fig.savefig(out_path, format=out_path.suffix.lstrip("."), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    print(f"  [OK] Saved: {out_path}")


def print_table(results):
    """Print a summary table of w_KM values."""
    header = f"{'T_verd':>6} {'T_kond':>6}"
    for sp in STATE_POINTS:
        header += f" {sp['short']:>10}"
    print(f"\n  {header}")
    print("  " + "-" * len(header))

    for res in results:
        row = f"{res['T_evap_C']:>6.0f} {res['T_cond_C']:>6.0f}"
        for sp in STATE_POINTS:
            val = res.get(sp["key"])
            if val is not None:
                row += f" {val*100:>9.2f}%"
            else:
                row += f" {'—':>10}"
        print(f"  {row}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Plot dissolved refrigerant fraction along the oil path."
    )
    ap.add_argument("--params_csv", required=True, type=Path)
    ap.add_argument("--oil", required=True, help="PAG68 | PAG100 (or LPG68 | LPG100)")
    ap.add_argument("--refrigerant", default="auto")

    # Operating points — multiple values create multiple lines
    ap.add_argument("--T_evap", type=float, nargs="+", required=True,
                    help="Evaporation temperature(s) [°C]")
    ap.add_argument("--T_cond", type=float, nargs="+", required=True,
                    help="Condensation temperature(s) [°C]")
    ap.add_argument("--N_rpm", type=float, default=3600.0)
    ap.add_argument("--SH_K", type=float, default=10.0)
    ap.add_argument("--T_amb_C", type=float, default=25.0)

    # Geometry
    ap.add_argument("--N_max_rpm", type=float, default=7200.0)
    ap.add_argument("--V_h_cm3", type=float, default=30.7)

    # Output
    ap.add_argument("--out_dir", default="results/oil_path_wKM", type=Path)
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    if not args.params_csv.exists():
        raise FileNotFoundError(args.params_csv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    N_max_hz = rpm_to_hz(args.N_max_rpm)
    V_h_m3 = float(args.V_h_cm3) * 1e-6

    peek = pd.read_csv(args.params_csv).iloc[0].to_dict()
    if args.refrigerant == "auto":
        args.refrigerant = str(peek.get("refrigerant", "PROPANE"))

    med = RefProp(fluid_name=map_refrigerant(args.refrigerant))

    params, meta = load_params_csv(args.params_csv)
    params["m_dot_ref"] = compute_m_dot_ref(med, V_h_m3)
    params["f_ref"] = F_REF

    print(f"  Oil: {args.oil}")
    print(f"  Params from: {meta.get('oil', '?')}")

    # Build operating point combinations
    op_points = []
    for T_evap in args.T_evap:
        for T_cond in args.T_cond:
            if T_cond <= T_evap:
                print(f"  [SKIP] T_cond={T_cond} <= T_evap={T_evap}")
                continue
            op_points.append((T_evap, T_cond))

    if not op_points:
        raise ValueError("No valid operating points (T_cond must be > T_evap).")

    print(f"  Operating points: {len(op_points)}")

    # Run simulations
    results = []
    for T_evap, T_cond in op_points:
        print(f"  Simulating T_evap={T_evap:.0f}°C, T_cond={T_cond:.0f}°C ...")
        res = run_operating_point(
            med=med, params=params,
            refrigerant_name=args.refrigerant, oil_name=args.oil,
            N_max_hz=N_max_hz, V_h_m3=V_h_m3,
            T_evap_C=T_evap, T_cond_C=T_cond,
            N_rpm=args.N_rpm, SH_K=args.SH_K, T_amb_C=args.T_amb_C,
        )
        if res is not None:
            results.append(res)

    if not results:
        raise ValueError("All operating points failed. Check parameters.")

    # Print table
    print_table(results)

    # Plot
    stamp = _ts()
    oil_tag = args.oil.lower()
    out_name = f"oil_path_wKM_{oil_tag}_{stamp}.{args.out_format}"
    out_path = out_dir / out_name

    plot_oil_path_wKM(
        results=results,
        out_path=out_path,
        oil_name=args.oil,
    )

    # Save data CSV
    rows = []
    for res in results:
        row = {
            "T_evap_C": res["T_evap_C"],
            "T_cond_C": res["T_cond_C"],
            "N_rpm": res["N_rpm"],
            "SH_K": res["SH_K"],
            "p_suc_bar": res["p_suc_bar"],
            "p_dis_bar": res["p_dis_bar"],
        }
        for sp in STATE_POINTS:
            val = res.get(sp["key"])
            row[f"w_KM_{sp['key']}"] = val
        row.update({
            "T_sump_C": res.get("T_sump_C"),
            "T_throttle_C": res.get("T_throttle_C"),
            "T_oil_after_C": res.get("T_oil_after_C"),
            "T_dis_C": res.get("T_dis_C"),
            "m_dot_oil_gps": res.get("m_dot_oil_gps"),
            "m_dot_KM_degas_thr_gps": res.get("m_dot_KM_degas_thr_gps"),
            "m_dot_KM_degas_ht_gps": res.get("m_dot_KM_degas_ht_gps"),
        })
        rows.append(row)

    csv_path = out_dir / f"oil_path_wKM_{oil_tag}_{stamp}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  [OK] Data saved: {csv_path}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
