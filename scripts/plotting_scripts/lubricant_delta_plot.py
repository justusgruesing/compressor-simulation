# scripts/plotting_scripts/lubricant_delta_plots.py
#
# Erstellt drei Diagramme auf Basis der operating_points_rows.csv:
#   1) ΔT2   über mittlerer Austrittstemperatur
#   2) Δṁ   über mittlerem Massenstrom
#   3) ΔPel über mittlerer Antriebsleistung
#
# Für jeden Betriebspunkt (op_id) werden die beiden Ölzeilen (LPG68, LPG100)
# zusammengeführt. Anschließend wird berechnet:
#
#   x = Mittelwert der Zielgröße aus LPG68 und LPG100
#   y = Zielgröße(LPG100) - Zielgröße(LPG68)
#
# Damit ist jeder Betriebspunkt genau einmal im jeweiligen Diagramm vertreten.
#
# Erwarteter Input:
#   operating_points_rows.csv
#
# Beispiel:
#   python scripts/plotting_scripts/lubricant_delta_plot.py --csv results/split_template/operating_points_rows_2026-03-12_112331.csv
#
# Outputs:
#   results/lubricant_delta_plots/
#       - oil_delta_pairs_<timestamp>.csv
#       - delta_T2_vs_mean_T2_<timestamp>.png
#       - delta_mdot_vs_mean_mdot_<timestamp>.png
#       - delta_Pel_vs_mean_Pel_<timestamp>.png
#

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ebc.paper.mplstyle")


# -------------------------
# Defaults
# -------------------------
OP_ID_COL_DEFAULT = "op_id"
OIL_COL_DEFAULT = "_oil_norm"

T2_COL_DEFAULT = "T2_mean"                 # °C
MASSFLOW_COL_DEFAULT = "suction_mf_mean"   # g/s
PEL_COL_DEFAULT = "Pel_mean"               # W
SPEED_COL_DEFAULT = "N"                    # 1/min

OIL_68_NAME = "LPG68"
OIL_100_NAME = "LPG100"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _norm_oil_name(x: str) -> str:
    s = str(x).strip().lower().replace(" ", "")
    if s == "lpg68":
        return OIL_68_NAME
    if s == "lpg100":
        return OIL_100_NAME
    return str(x).strip()


def _validate_required_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in CSV: {missing}")


def _pair_operating_points(
    df: pd.DataFrame,
    op_id_col: str,
    oil_col: str,
    t2_col: str,
    mdot_col: str,
    pel_col: str,
    speed_col: str,
) -> pd.DataFrame:
    """
    Baut aus zwei Zeilen pro Betriebspunkt (LPG68/LPG100) eine gepaarte Tabelle.

    Ergebnis je op_id:
      - jeweilige Rohwerte für LPG68 und LPG100
      - Mittelwerte
      - Deltas = LPG100 - LPG68
    """
    rows = []

    grouped = df.groupby(op_id_col, sort=True)
    for op_id, grp in grouped:
        grp = grp.copy()

        oils_present = set(grp[oil_col].astype(str))
        if OIL_68_NAME not in oils_present or OIL_100_NAME not in oils_present:
            logging.warning(
                "Betriebspunkt %s übersprungen: unvollständiges Öl-Paar vorhanden (%s)",
                op_id,
                sorted(oils_present),
            )
            continue

        grp_68 = grp[grp[oil_col] == OIL_68_NAME]
        grp_100 = grp[grp[oil_col] == OIL_100_NAME]

        if len(grp_68) != 1 or len(grp_100) != 1:
            logging.warning(
                "Betriebspunkt %s übersprungen: erwartet genau 1x LPG68 und 1x LPG100, "
                "gefunden %d / %d",
                op_id,
                len(grp_68),
                len(grp_100),
            )
            continue

        r68 = grp_68.iloc[0]
        r100 = grp_100.iloc[0]

        try:
            T2_68 = float(r68[t2_col])
            T2_100 = float(r100[t2_col])
            m_68 = float(r68[mdot_col])
            m_100 = float(r100[mdot_col])
            P_68 = float(r68[pel_col])
            P_100 = float(r100[pel_col])
        except Exception as e:
            logging.warning("Betriebspunkt %s übersprungen: numerischer Fehler: %s", op_id, e)
            continue

        if not all(np.isfinite(v) for v in [T2_68, T2_100, m_68, m_100, P_68, P_100]):
            logging.warning("Betriebspunkt %s übersprungen: nicht-finite Werte", op_id)
            continue

        speed_68 = float(r68[speed_col]) if speed_col in r68.index and pd.notna(r68[speed_col]) else np.nan
        speed_100 = float(r100[speed_col]) if speed_col in r100.index and pd.notna(r100[speed_col]) else np.nan
        speed_mean = np.nanmean([speed_68, speed_100])

        rows.append(
            {
                "op_id": op_id,
                "N_LPG68": speed_68,
                "N_LPG100": speed_100,
                "N_mean": speed_mean,

                "T2_LPG68_C": T2_68,
                "T2_LPG100_C": T2_100,
                "T2_mean_C": 0.5 * (T2_68 + T2_100),
                "delta_T2_C": T2_100 - T2_68,

                "m_dot_LPG68_gps": m_68,
                "m_dot_LPG100_gps": m_100,
                "m_dot_mean_gps": 0.5 * (m_68 + m_100),
                "delta_m_dot_gps": m_100 - m_68,

                "Pel_LPG68_W": P_68,
                "Pel_LPG100_W": P_100,
                "Pel_mean_W": 0.5 * (P_68 + P_100),
                "delta_Pel_W": P_100 - P_68,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Es konnten keine vollständigen LPG68/LPG100-Betriebspunktpaare gebildet werden.")

    return out.sort_values("op_id").reset_index(drop=True)


def _scatter_plot(
    df_pairs: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    fig, ax = plt.subplots()

    ax.scatter(
        df_pairs[x_col].to_numpy(),
        df_pairs[y_col].to_numpy(),
    )

    ax.axhline(0.0, linewidth=1.0)

    ax.margins(x=0.05, y=0.10)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(
        description=(
            "Erstellt Delta-Diagramme zwischen LPG100 und LPG68 je Betriebspunkt "
            "auf Basis der operating_points_rows.csv."
        )
    )
    ap.add_argument("--csv", required=True, help="Pfad zur operating_points_rows.csv")
    ap.add_argument("--out_dir", default="results/oil_delta_plots", help="Ausgabeordner")

    ap.add_argument("--sep", default=",", help="CSV-Trennzeichen der Inputdatei")
    ap.add_argument("--decimal", default=".", help="Dezimaltrennzeichen der Inputdatei")

    ap.add_argument("--col_op_id", default=OP_ID_COL_DEFAULT, help="Spaltenname für Betriebspunkt-ID")
    ap.add_argument("--col_oil", default=OIL_COL_DEFAULT, help="Spaltenname für Öl")
    ap.add_argument("--col_T2", default=T2_COL_DEFAULT, help="Spaltenname für Austrittstemperatur [°C]")
    ap.add_argument("--col_mdot", default=MASSFLOW_COL_DEFAULT, help="Spaltenname für Massenstrom [g/s]")
    ap.add_argument("--col_Pel", default=PEL_COL_DEFAULT, help="Spaltenname für Antriebsleistung [W]")
    ap.add_argument("--col_speed", default=SPEED_COL_DEFAULT, help="Spaltenname für Drehzahl [1/min]")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=args.sep, decimal=args.decimal)

    required = [
        args.col_op_id,
        args.col_oil,
        args.col_T2,
        args.col_mdot,
        args.col_Pel,
    ]
    _validate_required_columns(df, required)

    df = df.copy()
    df[args.col_oil] = df[args.col_oil].map(_norm_oil_name)

    for c in [args.col_T2, args.col_mdot, args.col_Pel]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if args.col_speed in df.columns:
        df[args.col_speed] = pd.to_numeric(df[args.col_speed], errors="coerce")

    df = df.dropna(subset=[args.col_op_id, args.col_oil, args.col_T2, args.col_mdot, args.col_Pel]).copy()
    if df.empty:
        raise ValueError("Keine gültigen Daten nach dem NaN-Filter vorhanden.")

    df_pairs = _pair_operating_points(
        df=df,
        op_id_col=args.col_op_id,
        oil_col=args.col_oil,
        t2_col=args.col_T2,
        mdot_col=args.col_mdot,
        pel_col=args.col_Pel,
        speed_col=args.col_speed,
    )

    stamp = _ts()

    # gepaarte CSV speichern
    pairs_csv_path = out_dir / f"oil_delta_pairs_{stamp}.csv"
    df_pairs.to_csv(pairs_csv_path, index=False)

    # 1) Austrittstemperatur
    plot_T2_path = out_dir / f"delta_T2_vs_mean_T2_{stamp}.png"
    _scatter_plot(
        df_pairs=df_pairs,
        x_col="T2_mean_C",
        y_col="delta_T2_C",
        xlabel="Mittlere Austrittstemperatur $\\overline{T_2}$ in °C",
        ylabel="$\\Delta T_2 = T_{2,\\mathrm{LPG100}} - T_{2,\\mathrm{LPG68}}$ in K",
        title="Delta der Austrittstemperatur zwischen LPG100 und LPG68",
        out_path=plot_T2_path,
    )

    # 2) Massenstrom
    plot_mdot_path = out_dir / f"delta_mdot_vs_mean_mdot_{stamp}.png"
    _scatter_plot(
        df_pairs=df_pairs,
        x_col="m_dot_mean_gps",
        y_col="delta_m_dot_gps",
        xlabel="Mittlerer Massenstrom $\\overline{\\dot{m}}$ in g/s",
        ylabel="$\\Delta \\dot{m} = \\dot{m}_{\\mathrm{LPG100}} - \\dot{m}_{\\mathrm{LPG68}}$ in g/s",
        title="Delta des Massenstroms zwischen LPG100 und LPG68",
        out_path=plot_mdot_path,
    )

    # 3) Antriebsleistung
    plot_Pel_path = out_dir / f"delta_Pel_vs_mean_Pel_{stamp}.png"
    _scatter_plot(
        df_pairs=df_pairs,
        x_col="Pel_mean_W",
        y_col="delta_Pel_W",
        xlabel="Mittlere Antriebsleistung $\\overline{P_{el}}$ in W",
        ylabel="$\\Delta P_{el} = P_{el,\\mathrm{LPG100}} - P_{el,\\mathrm{LPG68}}$ W",
        title="Delta der Antriebsleistung zwischen LPG100 und LPG68",
        out_path=plot_Pel_path,
    )

    print("Gespeichert:")
    print(f"  Gepaarte CSV: {pairs_csv_path}")
    print(f"  Plot T2:      {plot_T2_path}")
    print(f"  Plot m_dot:   {plot_mdot_path}")
    print(f"  Plot Pel:     {plot_Pel_path}")


if __name__ == "__main__":
    main()