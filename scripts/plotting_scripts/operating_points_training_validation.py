# scripts/plotting_scripts/operating_points_map.py
#
# Plots operating points in a Tevap vs Tcond diagram (Molinaroli Fig. 3 style idea)
# - Tevap: saturation temperature at suction pressure p_suc (Q=1)
# - Tcond: saturation temperature at outlet pressure p_out (Q=0)
# - Superheat: T1_mean - Tevap
#
# Extended version:
# - Optional second CSV with train/validation assignment (e.g. fitting output)
# - Mapping is done via "idx" after:
#     1) filtering by oil
#     2) coercing numeric columns
#     3) dropping invalid rows
#     4) resetting index
#
# Visual encoding:
# - Color = superheat at inlet
# - Marker shape = speed group
# - Filled marker = training point
# - Open marker = validation point
#
# Outputs (default):
#   results/operating_map/
#     - operating_map_<oil>_<refrigerant>_<timestamp>.png
#     - operating_map_<oil>_<refrigerant>_<timestamp>.csv
#
# Example:
#   python scripts/plotting_scripts/operating_points_training_validation.py --csv data/Datensatz_Fitting_2.csv --refrigerant PROPANE --oil LPG68
#   python scripts/plotting_scripts/operating_points_training_validation.py --csv data/Datensatz_Fitting_1.csv --split_csv results/final_results/Molinaroli_LPG68/fit_predictions_lpg68_original_ga_2026-03-08_101308.csv --refrigerant PROPANE --oil LPG68 --xlim (-5, 30) --ylim (25, 70)
#
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

from vclibpy.media import RefProp

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

        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(oils_sorted):
                return oils_sorted[idx - 1]
            print("Ungültiger Index.")
            continue

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


def _parse_bool_like(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False

    raise ValueError(f"Could not parse boolean value from: {x}")


def _load_split_csv(path: Path) -> pd.DataFrame:
    """
    Load train/validation assignment CSV.

    Expected minimum columns:
      - idx
      - is_train

    Example:
      idx,is_train,...
      0,True,...
      1,False,...
    """
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = ["idx", "is_train"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Split CSV missing required columns: {missing}")

    df = df.copy()
    df["idx"] = pd.to_numeric(df["idx"], errors="raise").astype(int)
    df["is_train"] = df["is_train"].apply(_parse_bool_like)

    if df["idx"].duplicated().any():
        dup = df.loc[df["idx"].duplicated(), "idx"].tolist()
        raise ValueError(f"Split CSV contains duplicate idx values, e.g. {dup[:10]}")

    keep_cols = ["idx", "is_train"]
    if "ok" in df.columns:
        keep_cols.append("ok")

    return df[keep_cols]


def main():
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(
        description="Plot operating points in Tevap vs Tcond diagram (colored by superheat, optional train/validation split)."
    )
    ap.add_argument("--csv", required=True, help="Path to full dataset CSV")
    ap.add_argument(
        "--split_csv",
        default=None,
        help="Optional CSV with train/validation assignment (must contain columns: idx, is_train)",
    )
    ap.add_argument("--out_dir", default="results/operating_map", help="Output folder for PNG/CSV")

    ap.add_argument("--refrigerant", default="PROPANE", help="RefProp fluid name (e.g. PROPANE)")

    ap.add_argument("--sep", default=";", help="CSV separator for dataset CSV (default ';')")
    ap.add_argument("--decimal", default=",", help="Decimal separator for dataset CSV (default ',')")
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

    ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"))
    ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"))

    ap.add_argument("--cmap", default="viridis", help="Colormap for superheat")
    ap.add_argument("--cmin", type=float, default=None, help="Fixed min for color scale (superheat, °C)")
    ap.add_argument("--cmax", type=float, default=None, help="Fixed max for color scale (superheat, °C)")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    split_path = Path(args.split_csv) if args.split_csv else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=args.sep, header=args.header, decimal=args.decimal)

    required = [args.oil_col, args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_speed]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if args.oil is None:
        oil_choice = _prompt_oil_choice(df[args.oil_col].dropna().astype(str).tolist())
    else:
        oil_choice = str(args.oil).strip()

    # Filter by oil first
    df_oil = df[df[args.oil_col].astype(str).str.strip().str.lower() == oil_choice.lower()].copy()
    if df_oil.empty:
        raise ValueError(f"Keine Daten für Öl '{oil_choice}' gefunden.")

    # Keep original row index for traceability
    df_oil["source_row_index"] = df_oil.index

    # Coerce numeric columns
    df_oil[args.col_p_suc] = pd.to_numeric(df_oil[args.col_p_suc], errors="coerce")
    df_oil[args.col_T_suc] = pd.to_numeric(df_oil[args.col_T_suc], errors="coerce")
    df_oil[args.col_p_out] = pd.to_numeric(df_oil[args.col_p_out], errors="coerce")
    df_oil[args.col_speed] = pd.to_numeric(df_oil[args.col_speed], errors="coerce")

    # Drop invalid rows exactly before assigning idx
    df_oil = df_oil.dropna(subset=[args.col_p_suc, args.col_T_suc, args.col_p_out, args.col_speed]).reset_index(drop=True)

    # This idx is what the fitting output refers to
    df_oil["idx"] = df_oil.index

    # Optional merge with train/validation assignment
    if split_path is not None:
        df_split = _load_split_csv(split_path)

        df_oil = df_oil.merge(df_split, on="idx", how="left")

        df_oil["split"] = np.where(
            df_oil["is_train"] == True,
            "train",
            np.where(df_oil["is_train"] == False, "validation", "unassigned")
        )
    else:
        df_oil["is_train"] = np.nan
        df_oil["split"] = "all"

    # RefProp
    try:
        med = RefProp(fluid_name=args.refrigerant)
    except TypeError:
        med = RefProp(args.refrigerant)

    # Compute Tevap/Tcond/superheat
    Tevap = np.full(len(df_oil), np.nan, dtype=float)
    Tcond = np.full(len(df_oil), np.nan, dtype=float)
    superheat = np.full(len(df_oil), np.nan, dtype=float)

    for j, row in df_oil.iterrows():
        p_suc_pa = bar_to_pa(row[args.col_p_suc])
        p_out_pa = bar_to_pa(row[args.col_p_out])

        Tevap[j] = _calc_Tsat_C(med, p_suc_pa, Q=1.0, kind="Tevap(P,Q=1)", row_idx=int(row["source_row_index"]))
        Tcond[j] = _calc_Tsat_C(med, p_out_pa, Q=0.0, kind="Tcond(P,Q=0)", row_idx=int(row["source_row_index"]))

        if np.isfinite(Tevap[j]):
            superheat[j] = float(row[args.col_T_suc]) - Tevap[j]

    df_oil["T_evap_C"] = Tevap
    df_oil["T_cond_C"] = Tcond
    df_oil["superheat_C"] = superheat

    # Drop rows where RefProp failed
    df_plot = df_oil.dropna(subset=["T_evap_C", "T_cond_C", "superheat_C"]).copy()
    if df_plot.empty:
        raise ValueError("Alle Punkte sind NaN nach RefProp-Auswertung (keine plottbaren Betriebspunkte).")

    # ----- Plot -----
    fig, ax = plt.subplots()

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    vmin = float(np.nanmin(df_plot["superheat_C"])) if args.cmin is None else float(args.cmin)
    vmax = float(np.nanmax(df_plot["superheat_C"])) if args.cmax is None else float(args.cmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps.get_cmap(args.cmap)

    speeds = sorted(df_plot[args.col_speed].unique())

    # Plot points:
    # - train: filled marker
    # - validation: open marker with colored edge
    # - unassigned: open gray marker (only if split_csv provided and missing mapping)
    for k, sp in enumerate(speeds):
        marker = markers[k % len(markers)]
        sub_speed = df_plot[df_plot[args.col_speed] == sp]

        if split_path is None:
            ax.scatter(
                sub_speed["T_evap_C"].to_numpy(),
                sub_speed["T_cond_C"].to_numpy(),
                c=sub_speed["superheat_C"].to_numpy(),
                cmap=cmap,
                norm=norm,
                marker=marker,
                alpha=0.9,
                edgecolors="none",
            )
            continue

        sub_train = sub_speed[sub_speed["split"] == "train"]
        if not sub_train.empty:
            ax.scatter(
                sub_train["T_evap_C"].to_numpy(),
                sub_train["T_cond_C"].to_numpy(),
                c=sub_train["superheat_C"].to_numpy(),
                cmap=cmap,
                norm=norm,
                marker=marker,
                alpha=0.9,
                edgecolors="none",
            )

        sub_val = sub_speed[sub_speed["split"] == "validation"]
        if not sub_val.empty:
            edge_cols = cmap(norm(sub_val["superheat_C"].to_numpy()))
            ax.scatter(
                sub_val["T_evap_C"].to_numpy(),
                sub_val["T_cond_C"].to_numpy(),
                facecolors="none",
                edgecolors=edge_cols,
                marker=marker,
                alpha=0.95,
                linewidths=1.2,
            )

        sub_unassigned = sub_speed[sub_speed["split"] == "unassigned"]
        if not sub_unassigned.empty:
            ax.scatter(
                sub_unassigned["T_evap_C"].to_numpy(),
                sub_unassigned["T_cond_C"].to_numpy(),
                facecolors="none",
                edgecolors="0.5",
                marker=marker,
                alpha=0.95,
                linewidths=1.0,
            )

    # Labels
    ax.set_xlabel("Evaporationstemperatur $T_{evap}$ [°C]")
    ax.set_ylabel("Kondensationstemperatur $T_{cond}$ [°C]")

    if args.title is None:
        if split_path is None:
            title = f"Betriebspunkte: {oil_choice} | {args.refrigerant}"
        else:
            n_train = int((df_plot["split"] == "train").sum())
            n_val = int((df_plot["split"] == "validation").sum())
            n_unassigned = int((df_plot["split"] == "unassigned").sum())
            title = f"Betriebspunkte: {oil_choice} | {args.refrigerant} | Train={n_train}, Val={n_val}, Unassigned={n_unassigned}"
    else:
        title = args.title

    ax.set_title(title)

    if args.xlim is not None:
        ax.set_xlim(float(args.xlim[0]), float(args.xlim[1]))
    if args.ylim is not None:
        ax.set_ylim(float(args.ylim[0]), float(args.ylim[1]))

    # Two legends:
    # 1) Speed (marker shape)
    speed_handles = [
        Line2D(
            [0], [0],
            marker=markers[k % len(markers)],
            linestyle="None",
            color="black",
            markerfacecolor="0.5",
            markersize=7,
            label=f"N = {sp:.0f} 1/min",
        )
        for k, sp in enumerate(speeds)
    ]
    leg_speed = ax.legend(handles=speed_handles, title="Drehzahl", loc="upper left", frameon=True)
    ax.add_artist(leg_speed)

    # 2) Train/Validation style
    if split_path is not None:
        status_handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                color="black",
                markerfacecolor="0.5",
                markersize=7,
                label="Trainingspunkte",
            ),
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                color="black",
                markerfacecolor="none",
                markersize=7,
                label="Validierungspunkte",
            ),
        ]

        if (df_plot["split"] == "unassigned").any():
            status_handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    color="0.5",
                    markerfacecolor="none",
                    markersize=7,
                    label="Nicht zugeordnet",
                )
            )

        ax.legend(handles=status_handles, title="Datensatz", loc="lower right", frameon=True)

    # Colorbar
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

    # Save full oil-filtered, cleaned, merged dataset with computed values
    df_oil.to_csv(csv_path_out, index=False)

    print("Saved plot:", png_path)
    print("Saved data:", csv_path_out)


if __name__ == "__main__":
    main()