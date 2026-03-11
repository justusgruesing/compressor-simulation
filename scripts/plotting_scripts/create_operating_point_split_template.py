# scripts/create_operating_point_split_template.py
#
# Erstellt aus dem Datensatz eine saubere Betriebspunkt-Tabelle und eine Split-Template-CSV.
#
# Logik:
# - Ein Betriebspunkt wird EXAKT definiert über:
#       Drehzahl, T1_SH, T_evap, T_cond
# - Daraus wird eine eindeutige op_id erzeugt.
# - Für jeden Betriebspunkt wird ausgewertet, ob LPG68 und/oder LPG100 vorhanden sind.
# - Es wird KEINE Toleranzlogik mehr verwendet.
# - Es wird KEIN RefProp und KEIN Plot mehr benötigt.
#
# Outputs:
#   1) operating_points_rows_<timestamp>.csv
#      -> zeilenweise Zuordnung jeder Messung zu einer op_id
#
#   2) operating_points_split_template_<timestamp>.csv
#      -> eine Zeile pro Betriebspunkt, zum manuellen Eintragen von split_role
#
# Empfohlener Workflow:
#   - split_template öffnen
#   - in "split_role" z. B. train / validation eintragen
#   - diese CSV später in den Fitting-Skripten als exakte Split-Definition verwenden
#
# Beispiel:
#   python scripts/plotting_scripts/create_operating_point_split_template.py --csv data/Datensatz_final.csv
#

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Defaults
# =========================
LABEL_COL_DEFAULT = "Bezeichnung"
SPEED_SET_COL_DEFAULT = "Drehzahl"       # 1/s bzw. Hz
OIL_COL_DEFAULT = "Ölbezeichnung"
REFRIGERANT_COL_DEFAULT = "Kältemittel"
SH_COL_DEFAULT = "T1_SH"                 # K
TEVAP_COL_DEFAULT = "T_evap"             # °C
TCOND_COL_DEFAULT = "T_cond"             # °C

# optionale Messgrößen, nur zur Ausgabe
P1_MEAN_COL_DEFAULT = "P1_mean"
P2_MEAN_COL_DEFAULT = "P2_mean"
T1_MEAN_COL_DEFAULT = "T1_mean"
T2_MEAN_COL_DEFAULT = "T2_mean"
T7_MEAN_COL_DEFAULT = "T7_mean"
TAMB_MEAN_COL_DEFAULT = "Tamb_mean"
PEL_MEAN_COL_DEFAULT = "Pel_mean"
MASSFLOW_MEAN_COL_DEFAULT = "suction_mf_mean"
N_RPM_COL_DEFAULT = "N"


# =========================
# Helper
# =========================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def norm_oil_name(s: str) -> str:
    s = str(s).strip().lower().replace(" ", "")
    if s == "lpg68":
        return "LPG68"
    if s == "lpg100":
        return "LPG100"
    return str(s).strip()


def _fmt_num(x: float) -> str:
    """
    Formatiert Zahlen kompakt für die op_id:
    10.0 -> 10
    10.5 -> 10p5
    -5.0 -> m5
    """
    x = float(x)
    if abs(x - round(x)) < 1e-12:
        s = str(int(round(x)))
    else:
        s = f"{x:.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def make_op_id(speed_set: float, superheat: float, T_evap: float, T_cond: float) -> str:
    return (
        f"N{_fmt_num(speed_set)}"
        f"_SH{_fmt_num(superheat)}"
        f"_Te{_fmt_num(T_evap)}"
        f"_Tc{_fmt_num(T_cond)}"
    )


def read_dataset_csv(path: Path, sep: str, decimal: str, header: int) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, decimal=decimal, header=header)


def clean_dataset(df: pd.DataFrame, args) -> pd.DataFrame:
    required = [
        args.col_label,
        args.col_speed_set,
        args.col_oil,
        args.col_refrigerant,
        args.col_superheat,
        args.col_T_evap,
        args.col_T_cond,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Pflichtspalten: {missing}")

    out = df.copy()
    out.insert(0, "source_row_index", np.arange(len(out), dtype=int))

    # Numerische Schlüsselspalten robust konvertieren
    num_cols = [
        args.col_speed_set,
        args.col_superheat,
        args.col_T_evap,
        args.col_T_cond,
    ]
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Leere Trennzeilen und unvollständige Schlüsselzeilen entfernen
    out[args.col_label] = out[args.col_label].astype(str).str.strip()
    out[args.col_oil] = out[args.col_oil].astype(str).str.strip()
    out[args.col_refrigerant] = out[args.col_refrigerant].astype(str).str.strip()

    out = out.dropna(subset=num_cols).copy()
    out = out[
        (out[args.col_label] != "")
        & (out[args.col_oil] != "")
        & (out[args.col_refrigerant] != "")
    ].copy()

    if out.empty:
        raise ValueError("Keine gültigen Datenzeilen nach Bereinigung übrig.")

    out = out.reset_index(drop=True)
    out.insert(1, "filtered_row_index", np.arange(len(out), dtype=int))
    return out


def attach_op_keys(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()

    out["_oil_norm"] = out[args.col_oil].map(norm_oil_name)

    out["op_id"] = out.apply(
        lambda r: make_op_id(
            speed_set=r[args.col_speed_set],
            superheat=r[args.col_superheat],
            T_evap=r[args.col_T_evap],
            T_cond=r[args.col_T_cond],
        ),
        axis=1,
    )

    return out


def build_split_template(df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []

    group_cols = [
        "op_id",
        args.col_speed_set,
        args.col_superheat,
        args.col_T_evap,
        args.col_T_cond,
    ]

    grouped = df.groupby(group_cols, dropna=False, sort=True)

    for key, sub in grouped:
        op_id, speed_set, superheat, T_evap, T_cond = key

        oils_present = sorted(sub["_oil_norm"].dropna().unique().tolist())
        labels_present = sorted(sub[args.col_label].astype(str).str.strip().unique().tolist())
        refs_present = sorted(sub[args.col_refrigerant].astype(str).str.strip().unique().tolist())

        n_lpg68 = int((sub["_oil_norm"] == "LPG68").sum())
        n_lpg100 = int((sub["_oil_norm"] == "LPG100").sum())

        has_lpg68 = n_lpg68 > 0
        has_lpg100 = n_lpg100 > 0

        if has_lpg68 and has_lpg100:
            pair_status = "both_oils_present"
        elif has_lpg68:
            pair_status = "only_LPG68"
        elif has_lpg100:
            pair_status = "only_LPG100"
        else:
            pair_status = "unknown"

        rows.append(
            {
                "op_id": op_id,
                "Bezeichnung_canonical": labels_present[0] if labels_present else "",
                "Bezeichnung_all": " | ".join(labels_present),
                "Kältemittel_all": " | ".join(refs_present),
                "Drehzahl_set": float(speed_set),
                "T1_SH_set": float(superheat),
                "T_evap_set_C": float(T_evap),
                "T_cond_set_C": float(T_cond),
                "n_rows_total": int(len(sub)),
                "n_rows_LPG68": n_lpg68,
                "n_rows_LPG100": n_lpg100,
                "has_LPG68": bool(has_lpg68),
                "has_LPG100": bool(has_lpg100),
                "pair_status": pair_status,
                "usable_for_shared_split": bool(has_lpg68 and has_lpg100),
                "split_role": "",          # hier manuell train / validation / ignore eintragen
                "split_note": "",          # optional freie Notiz
            }
        )

    out = pd.DataFrame(rows)

    sort_cols = ["Drehzahl_set", "T_cond_set_C", "T1_SH_set", "T_evap_set_C"]
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    out.insert(0, "split_row_index", np.arange(len(out), dtype=int))
    return out


def build_row_output(df: pd.DataFrame, args) -> pd.DataFrame:
    keep_cols = [
        "source_row_index",
        "filtered_row_index",
        "op_id",
        "_oil_norm",
        args.col_label,
        args.col_speed_set,
        args.col_oil,
        args.col_refrigerant,
        args.col_superheat,
        args.col_T_evap,
        args.col_T_cond,
        args.col_N_rpm,
        args.col_p1_mean,
        args.col_p2_mean,
        args.col_t1_mean,
        args.col_t2_mean,
        args.col_t7_mean,
        args.col_tamb_mean,
        args.col_pel_mean,
        args.col_mflow_mean,
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    out = df[keep_cols].copy()
    out = out.sort_values(
        by=[args.col_speed_set, args.col_T_cond, args.col_superheat, args.col_T_evap, "_oil_norm"],
        kind="stable",
    ).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Erzeugt aus einem Datensatz mit expliziten Betriebspunkt-Sollgrößen "
            "(Drehzahl, T1_SH, T_evap, T_cond) eine exakte Betriebspunkt-Tabelle "
            "und eine Split-Template-CSV."
        )
    )

    ap.add_argument("--csv", required=True, help="Pfad zur Datensatz-CSV")
    ap.add_argument("--out_dir", default="results/split_template", help="Ausgabeordner")

    ap.add_argument("--sep", default=";", help="CSV-Trennzeichen")
    ap.add_argument("--decimal", default=",", help="Dezimaltrennzeichen")
    ap.add_argument("--header", type=int, default=1, help="Header-Zeile (default: 1)")

    ap.add_argument("--col_label", default=LABEL_COL_DEFAULT)
    ap.add_argument("--col_speed_set", default=SPEED_SET_COL_DEFAULT)
    ap.add_argument("--col_oil", default=OIL_COL_DEFAULT)
    ap.add_argument("--col_refrigerant", default=REFRIGERANT_COL_DEFAULT)
    ap.add_argument("--col_superheat", default=SH_COL_DEFAULT)
    ap.add_argument("--col_T_evap", default=TEVAP_COL_DEFAULT)
    ap.add_argument("--col_T_cond", default=TCOND_COL_DEFAULT)

    ap.add_argument("--col_p1_mean", default=P1_MEAN_COL_DEFAULT)
    ap.add_argument("--col_p2_mean", default=P2_MEAN_COL_DEFAULT)
    ap.add_argument("--col_t1_mean", default=T1_MEAN_COL_DEFAULT)
    ap.add_argument("--col_t2_mean", default=T2_MEAN_COL_DEFAULT)
    ap.add_argument("--col_t7_mean", default=T7_MEAN_COL_DEFAULT)
    ap.add_argument("--col_tamb_mean", default=TAMB_MEAN_COL_DEFAULT)
    ap.add_argument("--col_pel_mean", default=PEL_MEAN_COL_DEFAULT)
    ap.add_argument("--col_mflow_mean", default=MASSFLOW_MEAN_COL_DEFAULT)
    ap.add_argument("--col_N_rpm", default=N_RPM_COL_DEFAULT)

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = read_dataset_csv(csv_path, sep=args.sep, decimal=args.decimal, header=args.header)
    df_clean = clean_dataset(df_raw, args)
    df_keyed = attach_op_keys(df_clean, args)

    rows_df = build_row_output(df_keyed, args)
    split_df = build_split_template(df_keyed, args)

    stamp = _ts()
    rows_out = out_dir / f"operating_points_rows_{stamp}.csv"
    split_out = out_dir / f"operating_points_split_template_{stamp}.csv"

    rows_df.to_csv(rows_out, index=False)
    split_df.to_csv(split_out, index=False)

    n_total_rows = len(df_keyed)
    n_ops_total = len(split_df)
    n_shared = int(split_df["usable_for_shared_split"].sum())
    n_only_lpg68 = int((split_df["pair_status"] == "only_LPG68").sum())
    n_only_lpg100 = int((split_df["pair_status"] == "only_LPG100").sum())

    print("\n=== Split-Template erstellt ===")
    print(f"Datensatz: {csv_path}")
    print(f"Gültige Messzeilen: {n_total_rows}")
    print(f"Eindeutige Betriebspunkte: {n_ops_total}")
    print(f"Betriebspunkte mit beiden Ölen: {n_shared}")
    print(f"Nur LPG68 vorhanden: {n_only_lpg68}")
    print(f"Nur LPG100 vorhanden: {n_only_lpg100}")
    print(f"Zeilen-Ausgabe: {rows_out}")
    print(f"Split-Template: {split_out}")
    print("\nNächster Schritt:")
    print("  In der Split-Template-CSV die Spalte 'split_role' manuell mit")
    print("  train / validation / ignore befüllen.")


if __name__ == "__main__":
    main()