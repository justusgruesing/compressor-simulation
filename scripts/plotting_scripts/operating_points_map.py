# scripts/operating_points_map.py
#
# Plots operating points in a Tevap vs Tcond diagram (Molinaroli Fig. 3 style idea)
# - Tevap: saturation temperature at suction pressure p_suc (Q=1)
# - Tcond: saturation temperature at outlet pressure p_out (Q=0)
# - Superheat: T1_mean - Tevap
#
# Features:
# 1) Different marker shapes per speed group (e.g. N=3609 and N=4210 rpm)
# 3) If --oil is not set (None), ask interactively which oil to plot
# 4) No broad "except Exception" swallowing: only RefProp calc issues are handled
#    as warnings (short); other data/type errors raise.
# 6) Color scale = superheat at inlet; marker shape still indicates speed.
#
# Outputs (default):
#   results/operating_map/
#     - operating_map_<oil>_<refrigerant>_<timestamp>.png
#     - operating_map_<oil>_<refrigerant>_<timestamp>.csv
#
# Example:
#   python scripts/operating_points_map.py --csv data/Datensatz_Fitting_1.csv --refrigerant PROPANE --oil LPG68
#   python scripts/operating_points_map.py --oil LPG100 --csv data/Datensatz_Fitting_2.csv --xlim (-5, 30) --ylim (25, 65)
#
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from vclibpy.media import RefProp

# Use your style (adjust if your repo uses a different filename)
plt.style.use("ebc.paper.mplstyle")

# -------------------------
# Defaults for YOUR CSV
# -------------------------
OIL_COL_DEFAULT = "Ölbezeichnung"
P_SUC_COL_DEFAULT = "P1_mean"   # bar
T_SUC_COL_DEFAULT = "T1_mean"   # °C
P_OUT_COL_DEFAULT = "P2_mean"   # bar
SPEED_COL_DEFAULT = "N"         # rpm

# -------------------------
# Unit conversions
# -------------------------
def bar_to_pa(p_bar: float) -> float:
    return float(p_bar) * 100_000.0

def k_to_c(T_K: float) -> float:
    return float(T_K) - 273.15

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _prompt_oil_choice(oils: list[str]) -> str:
    oils_sorted = sorted({str(o).strip() for o in oils if str(o).strip()})
    if not oils_sorted:
        raise ValueError("Keine Ölwerte gefunden (leere Ölspalte?).")

    print("\nVerfügbare Öle im Datensatz:")
    for i, o in enumerate(oils_sorted, start=1):
        print(f"  [{i}] {o}")

    while True:
        s = input("Bitte Öl auswählen (Name oder Index): ").strip()
        if not s:
            continue

        # index?
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(oils_sorted):
                return oils_sorted[idx - 1]
            print("Ungültiger Index.")
            continue

        # name? (case-insensitive exact match)
        for o in oils_sorted:
            if o.lower() == s.lower():
                return o

        print("Ungültige Eingabe. Bitte exakt einen der Namen oder Index eingeben.")


def _short_msg(e: Exception, maxlen: int = 160) -> str:
    msg = str(e).replace("\n", " ").strip()
    return (msg[:maxlen] + "...") if len(msg) > maxlen else msg


def _calc_Tsat_C(med: RefProp, p_pa: float, Q: float, *, kind: str, row_idx: int) -> float:
    """
    Compute saturation temperature [°C] at pressure p_pa and quality Q.
    Handles only RefProp calc issues as warnings; returns NaN for those cases.
    """
    try:
        st = med.calc_state("PQ", float(p_pa), float(Q))
        T = float(getattr(st, "T", np.nan))
        if not np.isfinite(T):
            raise ValueError("RefProp returned non-finite T")
        return k_to_c(T)
    except Exception as e:
        logging.warning("RefProp failed (%s) row=%s: %s", kind, row_idx, _short_msg(e))
        return float("nan")


def main():
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(description="Plot operating points in Tevap vs Tcond diagram (colored by superheat).")
    ap.add_argument("--csv", required=True, help="Path to dataset CSV")
    ap.add_argument("--out_dir", default="results/operating_map", help="Output folder for PNG/CSV")

    ap.add_argument("--refrigerant", default="PROPANE", help="RefProp fluid name (e.g. PROPANE)")

    ap.add_argument("--sep", default=";", help="CSV separator (default ';')")
    ap.add_argument("--decimal", default=",", help="Decimal separator (default ',')")
    ap.add_argument("--header", type=int, default=1, help="Header row index (default 1 because row 0 is units)")

    ap.add_argument("--oil_col", default=OIL_COL_DEFAULT, help="Oil column name")
    ap.add_argument(
        "--oil",
        default=None,
        help="Oil to plot (e.g. LPG68/LPG100). If not set, you will be prompted.",
    )

    ap.add_argument("--col_p_suc", default=P_SUC_COL_DEFAULT, help="Suction pressure column [bar]")
    ap.add_argument("--col_T_suc", default=T_SUC_COL_DEFAULT, help="Suction temperature column [°C]")
    ap.add_argument("--col_p_out", default=P_OUT_COL_DEFAULT, help="Outlet pressure column [bar]")
    ap.add_argument("--col_speed", default=SPEED_COL_DEFAULT, help="Speed column [rpm]")

    ap.add_argument("--title", default=None, help="Optional plot title override")

    # axis limits (optional)
    ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"))
    ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"))

    # superheat color scale
    ap.add_argument("--cmap", default="viridis", help="Colormap for superheat")
    ap.add_argument("--cmin", type=float, default=None, help="Fixed min for color scale (superheat, °C)")
    ap.add_argument("--cmax", type=float, default=None, help="Fixed max for color scale (superheat, °C)")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=args.sep, header=args.header, decimal=args.decimal)

    # Validate required columns (hard errors, not swallowed)
    required = [args.oil_col, args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_speed]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Interactive oil selection if not provided
    if args.oil is None:
        oil_choice = _prompt_oil_choice(df[args.oil_col].dropna().astype(str).tolist())
    else:
        oil_choice = str(args.oil).strip()

    # Filter by oil
    df_oil = df[df[args.oil_col].astype(str).str.strip().str.lower() == oil_choice.lower()].copy()
    if df_oil.empty:
        raise ValueError(f"Keine Daten für Öl '{oil_choice}' gefunden.")

    # Coerce numeric columns (hard errors if impossible)
    df_oil[args.col_p_suc] = pd.to_numeric(df_oil[args.col_p_suc], errors="raise")
    df_oil[args.col_T_suc] = pd.to_numeric(df_oil[args.col_T_suc], errors="raise")
    df_oil[args.col_p_out] = pd.to_numeric(df_oil[args.col_p_out], errors="raise")
    df_oil[args.col_speed] = pd.to_numeric(df_oil[args.col_speed], errors="raise")

    # RefProp
    try:
        med = RefProp(fluid_name=args.refrigerant)
    except TypeError:
        med = RefProp(args.refrigerant)

    # Compute Tevap/Tcond/superheat
    Tevap = np.full(len(df_oil), np.nan, dtype=float)
    Tcond = np.full(len(df_oil), np.nan, dtype=float)
    superheat = np.full(len(df_oil), np.nan, dtype=float)

    # Keep original indices for clearer warnings
    for j, (idx, row) in enumerate(df_oil.iterrows()):
        p_suc_pa = bar_to_pa(row[args.col_p_suc])
        p_out_pa = bar_to_pa(row[args.col_p_out])

        # Tevap = Tsat at suction pressure, saturated vapour
        Tevap[j] = _calc_Tsat_C(med, p_suc_pa, Q=1.0, kind="Tevap(P,Q=1)", row_idx=int(idx))

        # Tcond = Tsat at outlet pressure, saturated liquid
        Tcond[j] = _calc_Tsat_C(med, p_out_pa, Q=0.0, kind="Tcond(P,Q=0)", row_idx=int(idx))

        # Superheat at inlet = T_suc - T_evap
        if np.isfinite(Tevap[j]):
            superheat[j] = float(row[args.col_T_suc]) - Tevap[j]

    df_oil = df_oil.reset_index(drop=False).rename(columns={"index": "row_index"})
    df_oil["T_evap_C"] = Tevap
    df_oil["T_cond_C"] = Tcond
    df_oil["superheat_C"] = superheat

    # Drop rows where RefProp failed
    df_plot = df_oil.dropna(subset=["T_evap_C", "T_cond_C", "superheat_C"]).copy()
    if df_plot.empty:
        raise ValueError("Alle Punkte sind NaN nach RefProp-Auswertung (keine plottbaren Betriebspunkte).")

    # ----- Plot with speed differentiation + color scale -----
    fig, ax = plt.subplots()

    # marker cycle for up to a few speed groups
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    # shared color mapping for superheat
    vmin = float(np.nanmin(df_plot["superheat_C"])) if args.cmin is None else float(args.cmin)
    vmax = float(np.nanmax(df_plot["superheat_C"])) if args.cmax is None else float(args.cmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps.get_cmap(args.cmap)

    # group by speed
    speeds = sorted(df_plot[args.col_speed].unique())
    for k, sp in enumerate(speeds):
        sub = df_plot[df_plot[args.col_speed] == sp]
        ax.scatter(
            sub["T_evap_C"].to_numpy(),
            sub["T_cond_C"].to_numpy(),
            c=sub["superheat_C"].to_numpy(),
            cmap=cmap,
            norm=norm,
            marker=markers[k % len(markers)],
            alpha=0.9,
            edgecolors="none",
            label=f"N = {sp:.0f} 1/min",
        )

    # Labels
    ax.set_xlabel("Evaporationstemperatur $T_{evap}$ [°C]")
    ax.set_ylabel("Kondensationstemperatur $T_{cond}$ [°C]")

    title = args.title
    if title is None:
        title = f"Betriebspunkte: {oil_choice} | {args.refrigerant}"
    ax.set_title(title)

    if args.xlim is not None:
        ax.set_xlim(float(args.xlim[0]), float(args.xlim[1]))
    if args.ylim is not None:
        ax.set_ylim(float(args.ylim[0]), float(args.ylim[1]))

    ax.legend(loc="best", frameon=True)

    # colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Überhitzung am Eintritt [°C]")

    # Save
    stamp = _ts()
    png_path = out_dir / f"operating_map_{oil_choice.lower()}_{args.refrigerant.lower()}_{stamp}.png"
    csv_path_out = out_dir / f"operating_map_{oil_choice.lower()}_{args.refrigerant.lower()}_{stamp}.csv"

    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    # Save CSV (like before): includes original columns + computed Tevap/Tcond/superheat,
    # including rows that failed (NaNs kept) for debugging/traceability.
    df_oil.to_csv(csv_path_out, index=False)

    print("Saved plot:", png_path)
    print("Saved data:", csv_path_out)


if __name__ == "__main__":
    main()