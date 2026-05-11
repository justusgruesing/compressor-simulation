"""
scripts/analyze_oil_path_validation.py
Analyse-Skript für Stufe-II-Validierungsergebnisse.

Liest die validation_detail_*.csv-Dateien des Oil-Path-Modells ein und
erzeugt aggregierte Tabellen und Plots für die Diskussion in Kapitel 4.4.2.

Aufruf:
    python scripts/analyze_oil_path_validation.py --files results/validation/oil_path/detail/validation_detail_params_lpg100_val_all_oil_path_validation_only_2026-05-09_191905.csv results/validation/oil_path/detail/validation_detail_params_lpg68_val_all_oil_path_validation_only_2026-05-09_193015.csv results/validation/oil_path/detail/validation_detail_params_all_val_all_oil_path_validation_only_2026-05-09_193622.csv --out_dir results/analysis_4.4.2 --plots
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Plotting ist optional. Wenn matplotlib nicht installiert ist,
# laufen die Tabellen trotzdem.
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =========================================================
# Cluster-Definitionen
# =========================================================
SH_BINS = [(7.5, 12.5, "SH10"),
           (17.5, 22.5, "SH20"),
           (27.5, 32.5, "SH30")]

TC_BINS = [(25.0, 35.0, "Tc30"),
           (35.0, 45.0, "Tc40"),
           (45.0, 55.0, "Tc50"),
           (55.0, 65.0, "Tc60")]


def assign_sh_cluster(sh: float) -> str:
    for lo, hi, label in SH_BINS:
        if lo <= sh < hi:
            return label
    return "SH_other"


def assign_tc_cluster(p_out_bar: float) -> str:
    """
    Cluster nach Kondensationstemperatur, abgeleitet aus p_out_bar.
    Faustregel für Propan:
        Tc=30 °C → ~10.7 bar
        Tc=40 °C → ~13.7 bar
        Tc=50 °C → ~17.2 bar
        Tc=60 °C → ~21.2 bar
    """
    if p_out_bar < 12.0:
        return "Tc30"
    if p_out_bar < 15.5:
        return "Tc40"
    if p_out_bar < 19.0:
        return "Tc50"
    return "Tc60"


# =========================================================
# Globale Fehlermetriken
# =========================================================
def compute_global_metrics(df: pd.DataFrame) -> dict:
    ok = df[df["success"]].copy()

    def stats(col, abs_tol=None):
        s = ok[col].dropna()
        if s.empty:
            return {"n": 0, "mean": np.nan, "mae": np.nan, "rmse": np.nan,
                    "min": np.nan, "max": np.nan, "share_within": np.nan}
        return {
            "n": int(len(s)),
            "mean": float(s.mean()),
            "mae": float(s.abs().mean()),
            "rmse": float(np.sqrt((s ** 2).mean())),
            "min": float(s.min()),
            "max": float(s.max()),
            "share_within": float((s.abs() <= abs_tol).mean()) if abs_tol is not None else np.nan,
        }

    return {
        "n_total": int(len(df)),
        "n_success": int(ok.shape[0]),
        "n_failed": int((~df["success"]).sum()),
        "m_flow":  stats("e_m_rel",  abs_tol=0.05),
        "P_el":    stats("e_P_rel",  abs_tol=0.05),
        "T_dis":   stats("e_T_dis_K", abs_tol=3.0),
    }


def metrics_to_row(name: str, metrics: dict) -> dict:
    row = {"file": name,
           "n_total": metrics["n_total"],
           "n_success": metrics["n_success"]}
    for key, label in [("m_flow", "m"), ("P_el", "P"), ("T_dis", "T_dis")]:
        m = metrics[key]
        row[f"mae_{label}"] = m["mae"]
        row[f"rmse_{label}"] = m["rmse"]
        row[f"min_{label}"] = m["min"]
        row[f"max_{label}"] = m["max"]
        row[f"share_within_{label}"] = m["share_within"]
    return row


# =========================================================
# Cluster-Statistik
# =========================================================
def cluster_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["success"]].copy()
    if ok.empty:
        return pd.DataFrame()

    ok["sh_cluster"] = ok["superheat_C"].apply(assign_sh_cluster)
    ok["tc_cluster"] = ok["p_out_bar"].apply(assign_tc_cluster)

    grouped = ok.groupby(["oil_norm", "sh_cluster", "tc_cluster"], dropna=False)

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(g),
            "mae_T_dis_K":  g["e_T_dis_K"].abs().mean(),
            "max_T_dis_K":  g["e_T_dis_K"].abs().max(),
            "mean_e_T_dis_K_signed": g["e_T_dis_K"].mean(),
            "mae_m_pct":   g["e_m_rel"].abs().mean()  * 100,
            "mae_P_pct":   g["e_P_rel"].abs().mean()  * 100,
            "mean_pc_gap_K":  g["pc_convergence_gap_K"].mean(),
            "max_pc_gap_K":   g["pc_convergence_gap_K"].max(),
            "mean_dw_KM_stage1": (g["w_KM_mix"] - g["w_KM_after"]).mean(),
            "max_dw_KM_stage1":  (g["w_KM_mix"] - g["w_KM_after"]).max(),
            "n_stage1_fb":   int((g["stage1_fallback_count"]   > 0).sum()),
            "n_throttle_fb": int((g["throttle_fallback_count"] > 0).sum()),
            "n_w_KM_fb":     int((g["w_KM_after_fallback_count"] > 0).sum()),
            "n_corrector_fb": int((g["corrector_fallback_count"] > 0).sum()),
        })

    return grouped.apply(agg).reset_index()


# =========================================================
# Plausibilitäts-/Energiebilanz-Kennzahlen
# =========================================================
def physical_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["success"]].copy()
    if ok.empty:
        return pd.DataFrame()

    # Verhältnisse zu W_dot_int
    ok["Q_dissolve3_over_Wint"]  = ok["Q_dissolve_3_W"]    / ok["W_dot_int_W"]
    ok["Q_oil_sump_over_Wint"]   = ok["Q_oil_sump_W"]      / ok["W_dot_int_W"]
    ok["Q_dis_total_over_Wint"]  = ok["Q_dis_total_W"]     / ok["W_dot_int_W"]
    ok["W_recirc_over_Pel"]      = ok["W_dot_oil_recirc_W"] / ok["P_el_W"]

    # Schmierstoff-/Kältemittelverhältnisse
    ok["mdot_oil_over_msuc"]     = ok["m_dot_oil_kg_s"]    / ok["m_flow_kg_s"]
    ok["mdot_degas_over_msuc"]   = ok["m_dot_KM_degas_total_kg_s"] / ok["m_flow_kg_s"]

    # Rücklösungssprung in Stage 1
    ok["dw_KM_stage1"] = ok["w_KM_mix"] - ok["w_KM_after"]

    cols = ["oil_norm", "Q_dissolve3_over_Wint", "Q_oil_sump_over_Wint",
            "Q_dis_total_over_Wint", "W_recirc_over_Pel",
            "mdot_oil_over_msuc", "mdot_degas_over_msuc", "dw_KM_stage1"]

    return ok.groupby("oil_norm")[cols[1:]].agg(["mean", "median", "min", "max"]).reset_index()


# =========================================================
# Plot-Funktionen (optional)
# =========================================================
def plot_e_T_dis_vs_pressure_ratio(df: pd.DataFrame, out_path: Path, title: str):
    if not HAS_MPL: return
    ok = df[df["success"]].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for oil, marker in [("lpg68", "o"), ("lpg100", "s")]:
        sub = ok[ok["oil_norm"] == oil]
        if sub.empty: continue
        sc = ax.scatter(sub["pressure_ratio"], sub["e_T_dis_K"],
                        c=sub["superheat_C"], cmap="viridis",
                        marker=marker, label=f"PAG {oil[3:]}",
                        edgecolors="black", linewidth=0.4, s=45)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Überhitzung [K]")
    ax.axhspan(-3, 3, color="lightgray", alpha=0.4)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Druckverhältnis p_dis / p_suc")
    ax.set_ylabel("Fehler T_dis [K]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pc_gap_vs_e_T_dis(df: pd.DataFrame, out_path: Path, title: str):
    if not HAS_MPL: return
    ok = df[df["success"]].copy()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(ok["pc_convergence_gap_K"].abs(),
                    ok["e_T_dis_K"].abs(),
                    c=ok["pressure_ratio"], cmap="plasma",
                    edgecolors="black", linewidth=0.4, s=45)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Druckverhältnis [-]")
    ax.set_xlabel("|Predictor-Corrector-Gap| [K]")
    ax.set_ylabel("|Fehler T_dis| [K]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dw_KM_vs_pressure_ratio(df: pd.DataFrame, out_path: Path, title: str):
    if not HAS_MPL: return
    ok = df[df["success"]].copy()
    ok["dw"] = ok["w_KM_mix"] - ok["w_KM_after"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for oil, marker in [("lpg68", "o"), ("lpg100", "s")]:
        sub = ok[ok["oil_norm"] == oil]
        if sub.empty: continue
        ax.scatter(sub["pressure_ratio"], sub["dw"],
                   c=sub["superheat_C"], cmap="viridis", marker=marker,
                   label=f"PAG {oil[3:]}", edgecolors="black",
                   linewidth=0.4, s=45)
    ax.set_xlabel("Druckverhältnis p_dis / p_suc")
    ax.set_ylabel("Rücklösungssprung w_KM(Stage 1) - w_KM(after suc)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eps_dis_proxy(df: pd.DataFrame, out_path: Path, title: str):
    """
    Rekonstruiert eine Proxy-Effektivität aus den vorhandenen Größen:
        eps_proxy = Q_dis_total / (m_dot_total * cp_comb * (T_mix - T_w))
    Da T_mix nicht direkt ausgegeben wird, nehmen wir T_dis_corr_C als Proxy
    für T_mix (vor dem druckseitigen WT) und vergleichen ihn mit T_wall_C.
    """
    if not HAS_MPL: return
    ok = df[df["success"]].copy()
    ok["dT_drive"] = ok["T_dis_corr_C"] - ok["T_wall_C"]
    ok["dT_observed"] = ok["T_dis_corr_C"] - ok["T_dis_calc_C"]
    ok["eps_proxy"] = ok["dT_observed"] / ok["dT_drive"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(ok["pressure_ratio"], ok["eps_proxy"],
                    c=ok["superheat_C"], cmap="viridis",
                    edgecolors="black", linewidth=0.4, s=45)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Überhitzung [K]")
    ax.set_xlabel("Druckverhältnis p_dis / p_suc")
    ax.set_ylabel("Effektivität ε_dis (Proxy)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Analyseskript für Stufe-II-Validierungsergebnisse.")
    ap.add_argument("--files", nargs="+", required=True, type=Path,
                    help="Eine oder mehrere validation_detail_*.csv-Dateien.")
    ap.add_argument("--out_dir", type=Path, default=Path("analysis_out"),
                    help="Verzeichnis für die Ausgabedateien.")
    ap.add_argument("--plots", action="store_true",
                    help="Erzeugt zusätzlich Diagnose-Plots als PNG.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    global_rows = []
    all_cluster = []
    all_physical = []

    for path in args.files:
        if not path.exists():
            print(f"  Datei nicht gefunden: {path}")
            continue

        df = pd.read_csv(path)
        tag = path.stem  # zur Identifikation
        print(f"\n=== {tag} ===")
        print(f"  Zeilen gesamt: {len(df)},  davon erfolgreich: {int(df['success'].sum())}")

        # --- Globale Metriken ---
        m = compute_global_metrics(df)
        global_rows.append(metrics_to_row(tag, m))

        print(f"  Massenstrom:  MAE = {m['m_flow']['mae']*100:.2f} %  | RMSE = {m['m_flow']['rmse']*100:.2f} %"
              f"  | Anteil ±5% = {m['m_flow']['share_within']*100:.1f} %")
        print(f"  El. Leistung: MAE = {m['P_el']['mae']*100:.2f} %  | RMSE = {m['P_el']['rmse']*100:.2f} %"
              f"  | Anteil ±5% = {m['P_el']['share_within']*100:.1f} %")
        print(f"  T_dis:        MAE = {m['T_dis']['mae']:.2f} K   | RMSE = {m['T_dis']['rmse']:.2f} K"
              f"   | Anteil ±3K = {m['T_dis']['share_within']*100:.1f} %")

        # --- Cluster-Tabelle ---
        ct = cluster_table(df)
        ct.insert(0, "file", tag)
        all_cluster.append(ct)

        # --- Plausibilitätskennzahlen ---
        pt = physical_table(df)
        pt.insert(0, "file", tag)
        all_physical.append(pt)

        # --- Plots ---
        if args.plots and HAS_MPL:
            plot_e_T_dis_vs_pressure_ratio(
                df, args.out_dir / f"{tag}__eTdis_vs_PR.png",
                title=f"T_dis-Fehler über Druckverhältnis ({tag})")
            plot_pc_gap_vs_e_T_dis(
                df, args.out_dir / f"{tag}__pcgap_vs_eTdis.png",
                title=f"Predictor-Corrector-Gap vs. T_dis-Fehler ({tag})")
            plot_dw_KM_vs_pressure_ratio(
                df, args.out_dir / f"{tag}__dwKM_vs_PR.png",
                title=f"Rücklösungssprung in Stage 1 ({tag})")
            plot_eps_dis_proxy(
                df, args.out_dir / f"{tag}__eps_proxy_vs_PR.png",
                title=f"Effektivitäts-Proxy ({tag})")

    # --- Zusammenfassende CSVs ---
    pd.DataFrame(global_rows).to_csv(args.out_dir / "global_metrics.csv", index=False)
    if all_cluster:
        pd.concat(all_cluster, ignore_index=True).to_csv(
            args.out_dir / "cluster_metrics.csv", index=False)
    if all_physical:
        pd.concat(all_physical, ignore_index=True).to_csv(
            args.out_dir / "physical_metrics.csv", index=False)

    print(f"\nFertig. Ergebnisse in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()