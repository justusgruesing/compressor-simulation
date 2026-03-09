# scripts/compute_mu_ref.py
# Beispielaufruf:
# python scripts/compute_mu_ref.py --csv data/Datensatz_Fitting_2.csv --oil all --save_detail_csv data/mu_ref/mu_ref_all_not_balanced.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vclibpy.media import LubricantFitting, ThermodynamicState


class ConcreteLubricantFitting(LubricantFitting):
    def get_molar_mass(self):
        return None

    def get_critical_point(self):
        return None


# =========================
# Defaults for your CSV
# =========================
OIL_COL_DEFAULT = "Ölbezeichnung"
REFRIGERANT_COL_DEFAULT = "Kältemittel"
P_OUT_COL_DEFAULT = "P2_mean"      # bar
T_SUC_COL_DEFAULT = "T1_mean"      # °C
T_AMB_COL_DEFAULT = "Tamb_mean"    # °C
T_DIS_COL_DEFAULT = "T2_mean"      # °C

# Zhang unified correlation (in °C)
# T_oil = 0.914227*T_dis + 0.008136*T_in + 0.006144*T_amb
ZHANG_A = 0.914227
ZHANG_B = 0.008136
ZHANG_C = 0.006144

# Cache for lubricant models to avoid repeated REFPROP DLL copy attempts
_MODEL_CACHE = {}


def norm_oil_name(s: str) -> str:
    s = str(s).strip().lower().replace(" ", "")
    if s == "lpg68":
        return "LPG68"
    if s == "lpg100":
        return "LPG100"
    return str(s).strip()


def map_lubricant_name_for_vclibpy(s: str) -> str:
    s = norm_oil_name(s)
    if s == "LPG68":
        return "LPG 68"
    if s == "LPG100":
        return "LPG 100"
    raise ValueError(f"Unsupported oil for LubricantFitting: {s}")


def map_refrigerant_name_for_vclibpy(s: str) -> str:
    s = str(s).strip().upper()
    if s in ("R290", "PROPANE"):
        return "propane"
    raise ValueError(f"Unsupported refrigerant for LubricantFitting: {s}")


def get_lubricant_model(ref_name: str, oil_name: str) -> ConcreteLubricantFitting:
    key = (ref_name, oil_name)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = ConcreteLubricantFitting(
            fluid_name=ref_name,
            lub_name=oil_name,
        )
    return _MODEL_CACHE[key]


def c_to_k(t_c: float) -> float:
    return float(t_c) + 273.15


def bar_to_pa(p_bar: float) -> float:
    return float(p_bar) * 1e5


def zhang_oil_temp_C(T_dis_C: float, T_in_C: float, T_amb_C: float) -> float:
    return (
        ZHANG_A * float(T_dis_C)
        + ZHANG_B * float(T_in_C)
        + ZHANG_C * float(T_amb_C)
    )


def geometric_mean(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(vals))))


def read_dataset_csv(path: Path, sep: str, decimal: str, header: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, decimal=decimal, header=header)

    # Trennzeilen wie ;;;;;;;;;;; entfernen
    numeric_candidates = ["P1_mean", "T1_mean", "P2_mean", "Tamb_mean", "T2_mean"]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "P2_mean" in df.columns:
        df = df[df["P2_mean"].notna()].copy()

    return df.reset_index(drop=True)


def compute_mu_for_row(
    row: pd.Series,
    oil_col: str,
    ref_col: str,
    p_out_col: str,
    T_suc_col: str,
    T_amb_col: str,
    T_dis_col: str,
) -> float:
    oil_name = map_lubricant_name_for_vclibpy(row[oil_col])
    ref_name = map_refrigerant_name_for_vclibpy(row[ref_col])

    # Reuse cached model instead of instantiating one per row
    model = get_lubricant_model(ref_name=ref_name, oil_name=oil_name)

    T_oil_C = zhang_oil_temp_C(
        T_dis_C=row[T_dis_col],
        T_in_C=row[T_suc_col],
        T_amb_C=row[T_amb_col],
    )

    state = ThermodynamicState(
        p=bar_to_pa(row[p_out_col]),
        T=c_to_k(T_oil_C),
    )

    props = model.calc_transport_properties(state=state, phase="liquid")

    if props is None or getattr(props, "dyn_vis", None) is None:
        return float("nan")

    return float(props.dyn_vis)


def main():
    ap = argparse.ArgumentParser(
        description="Compute mu_ref from dataset using Zhang oil temperature correlation and geometric mean."
    )
    ap.add_argument("--csv", required=True, help="Path to dataset CSV")
    ap.add_argument("--sep", default=";", help="CSV separator")
    ap.add_argument("--decimal", default=",", help="Decimal separator")
    ap.add_argument("--header", type=int, default=1, help="Header row index (default: 1)")

    ap.add_argument("--oil", default="all", help="LPG68 | LPG100 | all")
    ap.add_argument("--oil_col", default=OIL_COL_DEFAULT)
    ap.add_argument("--ref_col", default=REFRIGERANT_COL_DEFAULT)
    ap.add_argument("--col_p_out", default=P_OUT_COL_DEFAULT)
    ap.add_argument("--col_T_suc", default=T_SUC_COL_DEFAULT)
    ap.add_argument("--col_T_amb", default=T_AMB_COL_DEFAULT)
    ap.add_argument("--col_T_dis", default=T_DIS_COL_DEFAULT)

    ap.add_argument(
        "--balance_by_oil",
        action="store_true",
        help="If --oil all: LPG68 and LPG100 contribute equally, regardless of point count."
    )
    ap.add_argument(
        "--save_detail_csv",
        default=None,
        help="Optional path to save row-wise T_oil and mu values."
    )

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = read_dataset_csv(csv_path, sep=args.sep, decimal=args.decimal, header=args.header)

    required = [
        args.oil_col,
        args.ref_col,
        args.col_p_out,
        args.col_T_suc,
        args.col_T_amb,
        args.col_T_dis,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["_oil_norm"] = df[args.oil_col].map(norm_oil_name)

    oil_arg = norm_oil_name(args.oil)
    if oil_arg != "all":
        df = df[df["_oil_norm"] == oil_arg].copy()

    df = df.dropna(
        subset=[args.col_p_out, args.col_T_suc, args.col_T_amb, args.col_T_dis]
    ).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No usable rows after filtering.")

    mus = []
    T_oils_C = []

    for _, row in df.iterrows():
        T_oil_C = zhang_oil_temp_C(
            T_dis_C=row[args.col_T_dis],
            T_in_C=row[args.col_T_suc],
            T_amb_C=row[args.col_T_amb],
        )
        T_oils_C.append(T_oil_C)

        try:
            mu = compute_mu_for_row(
                row=row,
                oil_col=args.oil_col,
                ref_col=args.ref_col,
                p_out_col=args.col_p_out,
                T_suc_col=args.col_T_suc,
                T_amb_col=args.col_T_amb,
                T_dis_col=args.col_T_dis,
            )
        except Exception as e:
            print("Fehler in Zeile:")
            print(row)
            print("Exception:", repr(e))
            raise
        mus.append(mu)

    df["T_oil_zhang_C"] = T_oils_C
    df["mu_from_zhang"] = mus

    valid = df["mu_from_zhang"].notna() & np.isfinite(df["mu_from_zhang"]) & (df["mu_from_zhang"] > 0)
    df_valid = df.loc[valid].copy()

    if len(df_valid) == 0:
        raise ValueError("No valid viscosity values could be computed.")

    oils_present = sorted(df_valid["_oil_norm"].dropna().unique().tolist())

    # geometrische Mittel je Öl
    oil_geom_means = {}
    for oil in oils_present:
        vals = df_valid.loc[df_valid["_oil_norm"] == oil, "mu_from_zhang"].to_numpy()
        oil_geom_means[oil] = geometric_mean(vals)

    # globales mu_ref
    if args.balance_by_oil and oil_arg == "all":
        if len(oils_present) < 2:
            print("Warning: --balance_by_oil set, but only one oil present after filtering.")
            mu_ref = geometric_mean(df_valid["mu_from_zhang"].to_numpy())
        else:
            oil_log_means = []
            for oil in oils_present:
                vals = df_valid.loc[df_valid["_oil_norm"] == oil, "mu_from_zhang"].to_numpy()
                oil_log_means.append(np.mean(np.log(vals)))
            mu_ref = float(np.exp(np.mean(oil_log_means)))
    else:
        mu_ref = geometric_mean(df_valid["mu_from_zhang"].to_numpy())

    # Ergebnisse auch in df_valid schreiben
    df_valid["mu_ref_overall"] = mu_ref
    df_valid["mu_ref_geom_current_oil"] = df_valid["_oil_norm"].map(oil_geom_means)
    df_valid["mu_ref_geom_LPG68"] = oil_geom_means.get("LPG68", np.nan)
    df_valid["mu_ref_geom_LPG100"] = oil_geom_means.get("LPG100", np.nan)
    df_valid["balance_by_oil"] = bool(args.balance_by_oil)

    print("\n=== mu_ref from dataset ===")
    print(f"Rows used: {len(df_valid)} / {len(df)}")
    print(f"Oil filter: {args.oil}")
    print(f"Balanced by oil: {args.balance_by_oil}")
    print(f"Geometric mean mu_ref: {mu_ref}")

    for oil in oils_present:
        vals = df_valid.loc[df_valid["_oil_norm"] == oil, "mu_from_zhang"].to_numpy()
        print(f"{oil}: n={len(vals)}, geometric mean={oil_geom_means[oil]}")

    print(f"T_oil,zhang mean [°C]: {df_valid['T_oil_zhang_C'].mean():.4f}")
    print(f"T_oil,zhang median [°C]: {df_valid['T_oil_zhang_C'].median():.4f}")

    if args.save_detail_csv:
        out_path = Path(args.save_detail_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            args.oil_col,
            args.ref_col,
            args.col_p_out,
            args.col_T_suc,
            args.col_T_amb,
            args.col_T_dis,
            "T_oil_zhang_C",
            "mu_from_zhang",
            "_oil_norm",
            "mu_ref_overall",
            "mu_ref_geom_current_oil",
            "mu_ref_geom_LPG68",
            "mu_ref_geom_LPG100",
            "balance_by_oil",
        ]
        keep = [c for c in cols if c in df_valid.columns]
        df_valid[keep].to_csv(out_path, index=False, sep=";", decimal=",")
        print(f"Detail CSV saved to: {out_path}")


if __name__ == "__main__":
    main()