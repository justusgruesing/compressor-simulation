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
#   python scripts/plotting_scripts/operating_map_split.py --rows_csv results/split_template/operating_points_rows_2026-03-12_112331.csv --split_csv results/split_template/operating_points_split_template_2026-03-12_112331.csv --oil LPG68 --show_limits --xlim -5 30 --ylim 10 85
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


# =========================================================
# Operating limits (from Cui, Fig. 3.6 / Table 3.4)
# =========================================================
ENVELOPE_UPPER = np.array([
    [-22.0, 68.0],
    [ -5.0, 80.0],
    [ 10.0, 80.0],
    [ 25.0, 70.0],
])

P_SUC_MIN_BAR = 2.0
DELTA_P_MIN_BAR = 3.9
LIMIT_COLOR = "#D32F2F"


def _compute_min_pressure_curve(refrigerant="propane", delta_p_bar=3.9,
                                 T_evap_range=(-30, 27), n_points=100):
    from vclibpy.media import RefProp
    med = RefProp(fluid_name=refrigerant)
    T_evap_arr = np.linspace(T_evap_range[0], T_evap_range[1], n_points)
    T_evap_ok, T_cond_arr = [], []
    for T_evap_C in T_evap_arr:
        try:
            T_evap_K = T_evap_C + 273.15
            st_evap = med.calc_state("TQ", T_evap_K, 1.0)
            p_evap = float(st_evap.p) / 1e5
            p_cond_min = p_evap + delta_p_bar
            st_cond = med.calc_state("PQ", p_cond_min * 1e5, 0.0)
            T_cond_C = float(st_cond.T) - 273.15
            T_evap_ok.append(T_evap_C)
            T_cond_arr.append(T_cond_C)
        except Exception:
            pass
    return np.array(T_evap_ok), np.array(T_cond_arr)


def _compute_safety_T_evap(refrigerant="propane", p_min_bar=2.0):
    from vclibpy.media import RefProp
    med = RefProp(fluid_name=refrigerant)
    try:
        st = med.calc_state("PQ", p_min_bar * 1e5, 1.0)
        return float(st.T) - 273.15
    except Exception:
        return -25.3


def build_unified_boundary(refrigerant="propane"):
    T_evap_safety = _compute_safety_T_evap(refrigerant, P_SUC_MIN_BAR)
    T_evap_mp, T_cond_mp = _compute_min_pressure_curve(
        refrigerant=refrigerant, delta_p_bar=DELTA_P_MIN_BAR,
        T_evap_range=(T_evap_safety, ENVELOPE_UPPER[-1, 0]),
        n_points=80,
    )
    T_evap_right = float(ENVELOPE_UPPER[-1, 0])
    T_cond_right_mp = float(T_cond_mp[-1]) if len(T_cond_mp) > 0 else 25.0
    env_T_evap_left = ENVELOPE_UPPER[0, 0]
    env_T_cond_left = ENVELOPE_UPPER[0, 1]
    env_T_evap_next = ENVELOPE_UPPER[1, 0] if len(ENVELOPE_UPPER) > 1 else env_T_evap_left
    env_T_cond_next = ENVELOPE_UPPER[1, 1] if len(ENVELOPE_UPPER) > 1 else env_T_cond_left
    if T_evap_safety <= env_T_evap_left:
        T_cond_top_at_safety = env_T_cond_left
    else:
        frac = (T_evap_safety - env_T_evap_left) / max(1e-9, env_T_evap_next - env_T_evap_left)
        T_cond_top_at_safety = env_T_cond_left + frac * (env_T_cond_next - env_T_cond_left)
    T_cond_bot_at_safety = float(T_cond_mp[0]) if len(T_cond_mp) > 0 else 10.0
    vertices = []
    vertices.append([T_evap_safety, T_cond_bot_at_safety])
    vertices.append([T_evap_safety, T_cond_top_at_safety])
    for pt in ENVELOPE_UPPER:
        if pt[0] >= T_evap_safety:
            vertices.append([pt[0], pt[1]])
    vertices.append([T_evap_right, T_cond_right_mp])
    for te, tc in zip(T_evap_mp[::-1], T_cond_mp[::-1]):
        vertices.append([te, tc])
    vertices.append(vertices[0])
    return np.array(vertices), T_evap_safety


def draw_operating_limits(ax, refrigerant="propane"):
    legend_handles = []
    try:
        boundary, _ = build_unified_boundary(refrigerant)
        ax.fill(boundary[:, 0], boundary[:, 1],
                color=LIMIT_COLOR, alpha=0.06, zorder=0)
        ax.plot(boundary[:, 0], boundary[:, 1],
                color=LIMIT_COLOR, linewidth=1.8,
                linestyle="-", zorder=1, alpha=0.8)
        legend_handles.append(
            Line2D([0], [0], color=LIMIT_COLOR, linewidth=1.8,
                   label="Eingeschr\u00e4nkte Betriebsgrenzen"))
    except Exception as e:
        print(f"  [WARN] Could not draw operating limits: {e}")
    return legend_handles


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

P1_COL_DEFAULT = "P1_mean"  # suction pressure [bar]
P2_COL_DEFAULT = "P2_mean"  # discharge pressure [bar]


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
    for col in [args.col_speed, args.col_T_evap, args.col_T_cond, args.col_T_sh,
                args.col_P1, args.col_P2]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[args.op_id_col, args.col_speed, args.col_T_evap,
                           args.col_T_cond, args.col_T_sh,
                           args.col_P1, args.col_P2]).copy()
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
    ap.add_argument("--col_P1", default=P1_COL_DEFAULT, help="Suction pressure column [bar]")
    ap.add_argument("--col_P2", default=P2_COL_DEFAULT, help="Discharge pressure column [bar]")

    ap.add_argument("--split_role_col", default=SPLIT_ROLE_COL_DEFAULT, help="split role column in split CSV")
    ap.add_argument("--split_note_col", default=SPLIT_NOTE_COL_DEFAULT, help="optional split note column in split CSV")

    ap.add_argument("--title", default=None, help="Optional plot title override")
    ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"))
    ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"))
    ap.add_argument("--show_limits", action="store_true",
                    help="Draw operating limit boundaries (Cui Fig. 3.6)")
    ap.add_argument("--refrigerant", default="propane",
                    help="Refrigerant for operating limits computation (default: propane)")

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
    # Compute actual T_evap / T_cond from measured pressures
    # -------------------------
    print("  Computing T_sat from measured pressures via RefProp ...")
    from vclibpy.media import RefProp
    med = RefProp(fluid_name=args.refrigerant)

    T_evap_actual = []
    T_cond_actual = []
    for _, row in df_plot.iterrows():
        p1 = float(row[args.col_P1]) * 1e5  # bar → Pa
        p2 = float(row[args.col_P2]) * 1e5
        try:
            st_suc = med.calc_state("PQ", p1, 1.0)
            T_evap_actual.append(float(st_suc.T) - 273.15)
        except Exception:
            T_evap_actual.append(float(row[args.col_T_evap]))
        try:
            st_dis = med.calc_state("PQ", p2, 0.0)
            T_cond_actual.append(float(st_dis.T) - 273.15)
        except Exception:
            T_cond_actual.append(float(row[args.col_T_cond]))

    df_plot["T_evap_actual"] = T_evap_actual
    df_plot["T_cond_actual"] = T_cond_actual

    print(f"  T_evap actual: {df_plot['T_evap_actual'].min():.2f} to {df_plot['T_evap_actual'].max():.2f} °C")
    print(f"  T_cond actual: {df_plot['T_cond_actual'].min():.2f} to {df_plot['T_cond_actual'].max():.2f} °C")

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

    # All points filled, grouped by speed (marker shape)
    for k, speed in enumerate(speed_values):
        marker = markers[k % len(markers)]
        sub = df_plot[df_plot[args.col_speed] == speed]

        ax.scatter(
            sub["T_evap_actual"].to_numpy(),
            sub["T_cond_actual"].to_numpy(),
            c=sub[args.col_T_sh].to_numpy(),
            cmap=cmap,
            norm=norm,
            marker=marker,
            alpha=0.9,
            edgecolors="none",
        )

    ax.set_xlabel("Verdampfungstemperatur $T_{Verd}$ in °C")
    ax.set_ylabel("Kondensationstemperatur $T_{Kond}$ in °C")

    # Operating limits
    limit_handles = []
    if args.show_limits:
        limit_handles = draw_operating_limits(ax, refrigerant=args.refrigerant)

    if args.title is None:
        n_total = len(df_plot)
        if oil_choice == "all":
            oil_txt = "alle Öle"
        else:
            oil_txt = oil_choice
        title = f"Betriebspunkte: {oil_txt} (n={n_total})"
    else:
        title = args.title

    ax.set_title(title)

    if args.xlim is not None:
        ax.set_xlim(float(args.xlim[0]), float(args.xlim[1]))
    if args.ylim is not None:
        ax.set_ylim(float(args.ylim[0]), float(args.ylim[1]))

    # Legend: speed / marker shape
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
    ax.legend(handles=speed_handles, title="Betriebspunktgruppe", loc="upper left", frameon=True)

    # Colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Überhitzung $T_{1,SH}$ in K")

    # Operating limits legend at the bottom
    if limit_handles:
        fig.legend(handles=limit_handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.04), ncol=len(limit_handles),
                   fontsize=9, frameon=True)

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