# scripts/plotting_scripts/operating_points_map.py
#
# Plot operating points in a Tevap vs Tcond diagram using the generated
# operating-point CSV files:
#   1) operating_points_rows.csv
#   2) operating_points_split_template.csv (filled with split_role)
#
# New logic:
# - Directly uses the rounded operating-point coordinates already stored in the CSV
# - Split mapping is done via "op_id"
#
# Visual encoding:
# - x-axis: T_evap
# - y-axis: T_cond
# - color: T1_SH
# - marker shape: speed group (Drehzahl)
# - filled marker: training point
# - open marker: validation point
# - gray open marker: unassigned point
#
# Typical workflow:
# 1) Generate operating_points_rows.csv and operating_points_split_template.csv
# 2) Fill column "split_role" in the split template with:
#       train
#       validation
#    (leave empty for unassigned)
# 3) Plot one oil or the shared unique operating points
#
# Example:
#   python scripts/plotting_scripts/operating_map_split.py --rows_csv results/split_template/operating_points_rows_2026-03-12_112331.csv --split_csv results/split_template/operating_points_split_template_2026-03-12_112331.csv --oil LPG68 --xlim (-5, 30) --ylim (25, 70)
#
#   python scripts/plotting_scripts/operating_map_split.py ^
#       --rows_csv results/operating_map/operating_points_rows.csv ^
#       --split_csv results/operating_map/operating_points_split_template_filled.csv ^
#       --oil LPG100
#
#   python scripts/plotting_scripts/operating_map_split.py ^
#       --rows_csv results/operating_map/operating_points_rows.csv ^
#       --split_csv results/operating_map/operating_points_split_template_filled.csv ^
#       --oil all
#
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

try:
    plt.style.use("ebc.paper.mplstyle")
except OSError:
    pass


# -------------------------
# Defaults for generated CSVs
# -------------------------
OP_ID_COL_DEFAULT = "op_id"
OIL_COL_DEFAULT = "_oil_norm"
OIL_FALLBACK_COL_DEFAULT = "Ölbezeichnung"

SPEED_COL_DEFAULT = "Drehzahl"   # rounded speed group, e.g. 60 / 70
TE_COL_DEFAULT = "T_evap"
TC_COL_DEFAULT = "T_cond"
SH_COL_DEFAULT = "T1_SH"

SPLIT_ROLE_COL_DEFAULT = "split_role"
SPLIT_NOTE_COL_DEFAULT = "split_note"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _norm_oil_name(x: str) -> str:
    s = str(x).strip().lower().replace(" ", "")
    if s == "lpg68":
        return "LPG68"
    if s == "lpg100":
        return "LPG100"
    if s == "all":
        return "all"
    return str(x).strip()


def _prompt_oil_choice(oils: list[str]) -> str:
    oils_sorted = sorted({_norm_oil_name(o) for o in oils if str(o).strip()})
    if not oils_sorted:
        raise ValueError("Keine Ölwerte gefunden.")

    print("\nVerfügbare Öle in der rows_csv:")
    for i, o in enumerate(oils_sorted, start=1):
        print(f"  [{i}] {o}")
    print("  [a] all")

    while True:
        s = input("Bitte Öl auswählen (Name, Index oder 'a'): ").strip()
        if not s:
            continue

        if s.lower() == "a":
            return "all"

        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(oils_sorted):
                return oils_sorted[idx - 1]
            print("Ungültiger Index.")
            continue

        s_norm = _norm_oil_name(s)
        for o in oils_sorted:
            if o.lower() == s_norm.lower():
                return o

        print("Ungültige Eingabe.")


def _parse_split_role(x) -> str:
    if pd.isna(x):
        return "unassigned"

    s = str(x).strip().lower()
    if s == "":
        return "unassigned"

    if s in {"train", "training"}:
        return "train"

    if s in {"validation", "valid", "val"}:
        return "validation"

    if s in {"test", "holdout"}:
        return "validation"

    return "unassigned"


def _load_rows_csv(path: Path, args) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [args.op_id_col, args.col_speed, args.col_T_evap, args.col_T_cond, args.col_T_sh]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Rows-CSV missing required columns: {missing}")

    df = df.copy()

    # Oil column handling
    oil_col = None
    if args.oil_col in df.columns:
        oil_col = args.oil_col
    elif args.oil_fallback_col in df.columns:
        oil_col = args.oil_fallback_col
    else:
        raise ValueError(
            f"Rows-CSV enthält weder '{args.oil_col}' noch '{args.oil_fallback_col}' als Ölspalte."
        )

    df["_plot_oil"] = df[oil_col].map(_norm_oil_name)

    # Numeric coercion
    for col in [args.col_speed, args.col_T_evap, args.col_T_cond, args.col_T_sh]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[args.op_id_col, args.col_speed, args.col_T_evap, args.col_T_cond, args.col_T_sh]).copy()
    df = df.reset_index(drop=True)

    return df


def _load_split_csv(path: Path, args) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [args.op_id_col, args.split_role_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Split-CSV missing required columns: {missing}")

    df = df.copy()
    df[args.op_id_col] = df[args.op_id_col].astype(str).str.strip()
    df["_split_role_norm"] = df[args.split_role_col].apply(_parse_split_role)

    if args.split_note_col in df.columns:
        keep_cols = [args.op_id_col, args.split_role_col, args.split_note_col, "_split_role_norm"]
    else:
        keep_cols = [args.op_id_col, args.split_role_col, "_split_role_norm"]

    if df[args.op_id_col].duplicated().any():
        dup = df.loc[df[args.op_id_col].duplicated(), args.op_id_col].tolist()
        raise ValueError(f"Split-CSV enthält doppelte op_id-Werte, z.B. {dup[:10]}")

    return df[keep_cols]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot operating points from operating_points_rows.csv and "
            "map train/validation assignment via op_id from split CSV."
        )
    )

    ap.add_argument(
        "--rows_csv", "--csv",
        dest="rows_csv",
        required=True,
        help="Path to operating_points_rows.csv"
    )
    ap.add_argument(
        "--split_csv",
        default=None,
        help="Optional path to filled operating_points_split_template.csv"
    )
    ap.add_argument(
        "--out_dir",
        default="results/operating_map",
        help="Output folder for PNG/CSV"
    )

    ap.add_argument("--op_id_col", default=OP_ID_COL_DEFAULT, help="op_id column name")
    ap.add_argument("--oil_col", default=OIL_COL_DEFAULT, help="Preferred oil column in rows CSV")
    ap.add_argument("--oil_fallback_col", default=OIL_FALLBACK_COL_DEFAULT, help="Fallback oil column in rows CSV")

    ap.add_argument("--oil", default=None, help="LPG68 | LPG100 | all. If not set, prompt interactively.")

    ap.add_argument("--col_speed", default=SPEED_COL_DEFAULT, help="Speed-group column")
    ap.add_argument("--col_T_evap", default=TE_COL_DEFAULT, help="Evaporation temperature column [°C]")
    ap.add_argument("--col_T_cond", default=TC_COL_DEFAULT, help="Condensation temperature column [°C]")
    ap.add_argument("--col_T_sh", default=SH_COL_DEFAULT, help="Superheat column [K or °C-difference]")

    ap.add_argument("--split_role_col", default=SPLIT_ROLE_COL_DEFAULT, help="split role column in split CSV")
    ap.add_argument("--split_note_col", default=SPLIT_NOTE_COL_DEFAULT, help="optional split note column in split CSV")

    ap.add_argument("--title", default=None, help="Optional plot title override")
    ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"))
    ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"))

    ap.add_argument("--cmap", default="viridis", help="Colormap for superheat")
    ap.add_argument("--cmin", type=float, default=None, help="Fixed min for color scale")
    ap.add_argument("--cmax", type=float, default=None, help="Fixed max for color scale")

    args = ap.parse_args()

    rows_path = Path(args.rows_csv)
    split_path = Path(args.split_csv) if args.split_csv else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_rows = _load_rows_csv(rows_path, args)

    if args.oil is None:
        oil_choice = _prompt_oil_choice(df_rows["_plot_oil"].dropna().tolist())
    else:
        oil_choice = _norm_oil_name(args.oil)

    if oil_choice != "all":
        df_plot_base = df_rows[df_rows["_plot_oil"] == oil_choice].copy()
        if df_plot_base.empty:
            raise ValueError(f"Keine Daten für Öl '{oil_choice}' in rows_csv gefunden.")
    else:
        # For a shared split plot, only unique operating points make sense.
        # LPG68 and LPG100 would otherwise lie exactly on top of each other.
        df_plot_base = (
            df_rows.sort_values([args.op_id_col, "_plot_oil"])
            .drop_duplicates(subset=[args.op_id_col])
            .copy()
        )

    if split_path is not None:
        df_split = _load_split_csv(split_path, args)
        df_plot = df_plot_base.merge(df_split, on=args.op_id_col, how="left")
        df_plot["_split_role_norm"] = df_plot["_split_role_norm"].fillna("unassigned")
    else:
        df_plot = df_plot_base.copy()
        df_plot[args.split_role_col] = np.nan
        df_plot["_split_role_norm"] = "all"

    if df_plot.empty:
        raise ValueError("Keine plottbaren Daten nach Filterung gefunden.")

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots()

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    speed_values = sorted(df_plot[args.col_speed].dropna().unique().tolist())

    vmin = float(np.nanmin(df_plot[args.col_T_sh])) if args.cmin is None else float(args.cmin)
    vmax = float(np.nanmax(df_plot[args.col_T_sh])) if args.cmax is None else float(args.cmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps.get_cmap(args.cmap)

    for k, speed in enumerate(speed_values):
        marker = markers[k % len(markers)]
        sub_speed = df_plot[df_plot[args.col_speed] == speed]

        if split_path is None:
            ax.scatter(
                sub_speed[args.col_T_evap].to_numpy(),
                sub_speed[args.col_T_cond].to_numpy(),
                c=sub_speed[args.col_T_sh].to_numpy(),
                cmap=cmap,
                norm=norm,
                marker=marker,
                alpha=0.9,
                edgecolors="none",
            )
            continue

        sub_train = sub_speed[sub_speed["_split_role_norm"] == "train"]
        if not sub_train.empty:
            ax.scatter(
                sub_train[args.col_T_evap].to_numpy(),
                sub_train[args.col_T_cond].to_numpy(),
                c=sub_train[args.col_T_sh].to_numpy(),
                cmap=cmap,
                norm=norm,
                marker=marker,
                alpha=0.9,
                edgecolors="none",
            )

        sub_val = sub_speed[sub_speed["_split_role_norm"] == "validation"]
        if not sub_val.empty:
            edge_cols = cmap(norm(sub_val[args.col_T_sh].to_numpy()))
            ax.scatter(
                sub_val[args.col_T_evap].to_numpy(),
                sub_val[args.col_T_cond].to_numpy(),
                facecolors="none",
                edgecolors=edge_cols,
                marker=marker,
                alpha=0.95,
                linewidths=1.2,
            )

        sub_unassigned = sub_speed[sub_speed["_split_role_norm"] == "unassigned"]
        if not sub_unassigned.empty:
            ax.scatter(
                sub_unassigned[args.col_T_evap].to_numpy(),
                sub_unassigned[args.col_T_cond].to_numpy(),
                facecolors="none",
                edgecolors="0.5",
                marker=marker,
                alpha=0.95,
                linewidths=1.0,
            )

    ax.set_xlabel("Verdampfungstemperatur $T_{evap}$ [°C]")
    ax.set_ylabel("Kondensationstemperatur $T_{cond}$ [°C]")

    if args.title is None:
        if split_path is None:
            title = f"Betriebspunkte: {oil_choice}"
        else:
            n_train = int((df_plot["_split_role_norm"] == "train").sum())
            n_val = int((df_plot["_split_role_norm"] == "validation").sum())
            n_unassigned = int((df_plot["_split_role_norm"] == "unassigned").sum())

            if oil_choice == "all":
                oil_txt = "shared operating points"
            else:
                oil_txt = oil_choice

            title = (
                f"Betriebspunkte: {oil_txt} | "
                f"Train={n_train}, Val={n_val}, Unassigned={n_unassigned}"
            )
    else:
        title = args.title

    ax.set_title(title)

    if args.xlim is not None:
        ax.set_xlim(float(args.xlim[0]), float(args.xlim[1]))
    if args.ylim is not None:
        ax.set_ylim(float(args.ylim[0]), float(args.ylim[1]))

    # Legend 1: speed / marker shape
    speed_handles = [
        Line2D(
            [0], [0],
            marker=markers[k % len(markers)],
            linestyle="None",
            color="black",
            markerfacecolor="0.5",
            markersize=7,
            label=f"Drehzahl = {speed:.0f} 1/s",
        )
        for k, speed in enumerate(speed_values)
    ]
    leg_speed = ax.legend(handles=speed_handles, title="Betriebspunktgruppe", loc="upper left", frameon=True)
    ax.add_artist(leg_speed)

    # Legend 2: split role
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

        if (df_plot["_split_role_norm"] == "unassigned").any():
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

        ax.legend(handles=status_handles, title="Split", loc="lower right", frameon=True)

    # Colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Überhitzung $T_{1,SH}$ [K]")

    # Save outputs
    stamp = _ts()
    oil_tag = str(oil_choice).lower()

    png_path = out_dir / f"operating_map_{oil_tag}_{stamp}.png"
    csv_path_out = out_dir / f"operating_map_{oil_tag}_{stamp}.csv"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    df_plot.to_csv(csv_path_out, index=False)

    print("Saved plot:", png_path)
    print("Saved merged plot data:", csv_path_out)


if __name__ == "__main__":
    main()