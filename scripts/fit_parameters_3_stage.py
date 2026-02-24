"""
fit_parameters.py
=================
Parameter-Fitting für das semi-empirische Verdichtermodell nach Molinaroli et al. (2017)
implementiert in VCLibPy (Branch: 40-add-the-lubricant-influence).

Methodik:
  - Zielfunktion: Gl. (31)-(33) aus Molinaroli et al. (2017)
      g = sqrt( 0.5 * [1/n * sum(e_m_i^2)] + 0.5 * [1/n * sum(e_W_i^2)] )
      e_m = 1 - m_suc_calc / m_suc_meas
      e_W = 1 - W_comp_calc / W_comp_meas
  - Optimierer: scipy.optimize.minimize mit Bounds (äquivalent zur MATLAB Optimization Toolbox)
  - Multi-Start: n_starts Zufallsstarts zur Absicherung des globalen Minimums (wie Sec. 3.4 im Paper)
  - Referenzwerte: m_REF = V_IC * f_REF * rho_sat(T=273.15 K), f_REF = 50 Hz (Netzfrequenz)

8 Fit-Parameter (Benennung exakt wie im VCLibPy-Parameterdikt):
  Ua_suc_ref     [W/K]      – Referenz-Wärmeübergangskoeffizient Saugseite
  Ua_dis_ref     [W/K]      – Referenz-Wärmeübergangskoeffizient Druckseite
  Ua_amb         [W/K^1.25] – Wärmeübergangskoeffizient Verdichter-Umgebung
  A_tot          [m²]       – Äquivalente Gesamtfläche (Leckage + Re-Expansion)
  A_dis          [m²]       – Fläche Druckventil
  V_IC           [m³]       – Hubvolumen des isentropen Verdichters
  alpha_loss     [-]        – Proportionalitätsfaktor elektromech. Verluste
  W_dot_loss_ref [W]        – Referenz elektromech. Verluste

Referenzwerte (werden aus den Daten/Parametern berechnet, nicht gefittet):
  m_dot_ref  [kg/s] – Referenz-Massenstrom = V_IC * f_ref * rho_sat(273.15 K)
  f_ref      [Hz]   – Referenz-Drehfrequenz (Standard: 50 Hz)

Aufruf:
  python scripts/fit_parameters.py --csv data/Datensatz_Fitting_1.csv --oil LPG100
  python scripts/fit_parameters.py --csv data/Datensatz_Fitting_1.csv --oil all --model original
  python scripts/fit_parameters_3_stage.py --csv data/Datensatz_Fitting_1.csv --oil LPG68 --x0_csv data/start_params.csv

CSV-Format (Zeile 1: Einheiten, Zeile 2: Spaltennamen, ab Zeile 3: Daten):
  Spalten: oil, P1_mean [bar], T1_mean [°C], P2_mean [bar],
           Tamb_mean [°C], T2_mean [°C], suction_mf_mean [g/s],
           Pel_mean [W], N [1/min]
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Pfad-Setup: VCLibPy und compressor_models müssen im Python-Pfad sein.
# Passe ggf. den relativen Pfad zu VCLibPy an.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# Falls VCLibPy als Paket installiert ist (pip install -e vclibpy), ist diese Zeile nicht nötig.
# Andernfalls: sys.path.insert(0, str(REPO_ROOT.parent / "vclibpy"))

from dataclasses import dataclass

@dataclass
class _Control:
    n: float   # relatives Drehzahlsignal 0..1  -> get_n_absolute(n) = n * N_max

@dataclass
class _SimpleInputs:
    control: _Control
    T_amb: float   # Umgebungstemperatur [K]

try:
    # Verdichterklassen direkt aus dem compressors-Package (wie im funktionierenden Skript)
    from vclibpy.components.compressors import (
        Molinaroli_2017_Compressor,
        Molinaroli_2017_Compressor_Modified,
    )
    # CoolProp-Wrapper aus VCLibPy
    from vclibpy.media.cool_prop import CoolProp as VCLibCoolProp
    # FlowsheetState als Ergebnisspeicher
    from vclibpy.datamodels import FlowsheetState
    VCLIBPY_AVAILABLE = True
except ImportError as e:
    warnings.warn(
        f"VCLibPy konnte nicht importiert werden: {e}\n"
        "Stelle sicher, dass VCLibPy installiert ist (pip install -e vclibpy) "
        "oder der Pfad korrekt gesetzt ist."
    )
    VCLIBPY_AVAILABLE = False


# ===========================================================================
# Hilfsfunktion: Modellaufruf
# ===========================================================================

def run_compressor_model(params: np.ndarray, p_suc: float, T_suc: float,
                         p_dis: float, f_Hz: float, T_amb: float,
                         medium: str, model_type: str = "original",
                         f_ref: float = 50.0,
                         m_dot_ref: float = None) -> tuple[float, float]:
    """
    Ruft Molinaroli_2017_Compressor aus VCLibPy für einen Betriebspunkt auf.

    Parameters
    ----------
    params     : Array der 8 Fit-Parameter in der Reihenfolge von PARAM_NAMES:
                 [Ua_suc_ref, Ua_dis_ref, Ua_amb, A_tot, A_dis, V_IC, alpha_loss, W_dot_loss_ref]
    p_suc      : Saugdruck [Pa]
    T_suc      : Saugtemperatur [K]
    p_dis      : Austrittsdruck [Pa]
    f_Hz       : Drehfrequenz [Hz]
    T_amb      : Umgebungstemperatur [K]
    medium     : Kältemittelbezeichnung (CoolProp-String, z.B. 'R290')
    model_type : 'original' oder 'modified' (mit Öleinfluss)
    f_ref      : Referenz-Drehfrequenz [Hz] (Standard: 50 Hz)
    m_dot_ref  : Referenz-Massenstrom [kg/s] = V_h_geo * f_ref * rho_sat(273.15 K)

    Returns
    -------
    (m_suc_calc [kg/s], W_comp_calc [W])
    """
    Ua_suc_ref, Ua_dis_ref, Ua_amb, A_tot, A_dis, V_IC, alpha_loss, W_dot_loss_ref = params

    med_prop = VCLibCoolProp(fluid_name=medium)
    state_inlet = med_prop.calc_state("PT", p_suc, T_suc)

    # Parameter-Dict (Schlüsselnamen exakt wie im VCLibPy-Konstruktor):
    parameters = {
        "Ua_suc_ref":     Ua_suc_ref,
        "Ua_dis_ref":     Ua_dis_ref,
        "Ua_amb":         Ua_amb,
        "A_tot":          A_tot,
        "A_dis":          A_dis,
        "V_IC":           V_IC,
        "alpha_loss":     alpha_loss,
        "W_dot_loss_ref": W_dot_loss_ref,
        "m_dot_ref":      m_dot_ref,
        "f_ref":          f_ref,
    }

    # Verdichter instanziieren (N_max=f_Hz → get_n_absolute(1.0)=f_Hz):
    compressor = Molinaroli_2017_Compressor(
        N_max=f_Hz,
        V_h=V_IC,
        parameters=parameters,
    )

    # Med-Prop und Eintrittszustand setzen (normalerweise vom Flowsheet gesetzt):
    compressor.med_prop = med_prop
    compressor.state_inlet = state_inlet

    # Inputs: n=1.0 → get_n_absolute(1.0) = N_max = f_Hz
    inputs = _SimpleInputs(
        control=_Control(n=1.0),
        T_amb=T_amb,
    )

    # FlowsheetState als Ergebnisspeicher:
    fs_state = FlowsheetState()


    compressor.calc_state_outlet(
        p_outlet=p_dis,
        inputs=inputs,
        fs_state=fs_state,
    )

    # Ergebnisse auslesen:
    m_suc_calc  = compressor.m_flow
    W_comp_calc = compressor.P_el

    return m_suc_calc, W_comp_calc


# ===========================================================================
# Zielfunktion  (Gleichungen 31–33 aus Molinaroli et al. 2017)
# ===========================================================================

def objective_function(params: np.ndarray, data: pd.DataFrame,
                       medium: str, model_type: str,
                       f_ref: float = 50.0,
                       m_dot_ref: float = None,
                       return_details: bool = False):
    """
    Zielfunktion g nach Gl. (31):
        g = sqrt( 0.5 * mean(e_m^2) + 0.5 * mean(e_W^2) )

    mit:
        e_m = 1 - m_calc / m_meas    (Gl. 32)
        e_W = 1 - W_calc / W_meas    (Gl. 33)

    Parameters
    ----------
    params       : 1-D Array mit den 8 Fit-Parametern
    data         : DataFrame mit den Messpunkten
    medium       : Kältemittelbezeichnung
    model_type   : 'original' oder 'modified'
    return_details: Falls True, werden Einzelfehler zurückgegeben

    Returns
    -------
    g (float) oder (g, details_dict) falls return_details=True
    """
    # Parameterwerte < 0 direkt bestrafen (physikalisch sinnlos)
    if np.any(params <= 0):
        return 1e6

    errors_m = []
    errors_W = []

    for _, row in data.iterrows():
        p_suc = row["P1_mean"] * 1e5        # bar -> Pa
        T_suc = row["T1_mean"] + 273.15     # °C -> K
        p_dis = row["P2_mean"] * 1e5        # bar -> Pa
        T_amb = row["Tamb_mean"] + 273.15   # °C -> K
        f_Hz  = row["N"] / 60.0             # 1/min -> Hz
        m_meas = row["suction_mf_mean"] * 1e-3  # g/s -> kg/s
        W_meas = row["Pel_mean"]            # W

        try:
            m_calc, W_calc = run_compressor_model(
                params, p_suc, T_suc, p_dis, f_Hz, T_amb, medium, model_type,
                f_ref=f_ref, m_dot_ref=m_dot_ref,
            )
        except Exception:
            # Falls Modell für diesen Punkt nicht konvergiert: Fehler aufschlagen
            errors_m.append(1.0)
            errors_W.append(1.0)
            continue

        if m_calc <= 0 or W_calc <= 0:
            errors_m.append(1.0)
            errors_W.append(1.0)
            continue

        e_m = 1.0 - m_calc / m_meas
        e_W = 1.0 - W_calc / W_meas
        errors_m.append(e_m)
        errors_W.append(e_W)

    if len(errors_m) == 0:
        return 1e6

    errors_m = np.array(errors_m)
    errors_W = np.array(errors_W)

    g = np.sqrt(
        0.5 * np.mean(errors_m**2) +
        0.5 * np.mean(errors_W**2)
    )

    if return_details:
        return g, {"errors_m": errors_m, "errors_W": errors_W}
    return g


# ===========================================================================
# Standard-Startparameter (können via --x0_csv überschrieben werden)
# Benennung EXAKT wie in VCLibPy parameters-Dict  (s. Molinaroli_2017_Compressor.__init__)
# Startwerte orientiert an Kompressor B aus Paper Tab. 2 (R290, 16.1 cm³)
# ===========================================================================

DEFAULT_PARAMS = {
    # Name              Startwert    Untergrenze   Obergrenze
    "Ua_suc_ref":     (16.05,        1e-3,          1e3),
    "Ua_dis_ref":     (13.96,        1e-3,          1e3),
    "Ua_amb":         (0.36,         1e-4,          100.0),
    "A_tot":          (9.47e-9,      1e-12,         1e-4),
    "A_dis":          (86.1e-9,      1e-12,         1e-4),
    "V_IC":           (16.11e-6,     1e-6,          1e-3),
    "alpha_loss":     (0.16,         0.0,           1.0),
    "W_dot_loss_ref": (83.0,         1.0,           5000.0),
}
PARAM_NAMES = list(DEFAULT_PARAMS.keys())


# ===========================================================================
# Optimierungs-Routine mit Multi-Start
# ===========================================================================

def fit_parameters(data: pd.DataFrame, medium: str, model_type: str,
                   x0: np.ndarray = None, bounds=None,
                   n_starts: int = 10, use_global: bool = True,
                   f_ref: float = 50.0,
                   m_dot_ref: float = None,
                   verbose: bool = True) -> dict:
    """
    Führt das Parameter-Fitting durch.

    Strategie (analog MATLAB Optimization Toolbox im Paper):
      1. Optional: Globale Vorsuche mit differential_evolution (empfohlen)
      2. Lokale Verfeinerung mit L-BFGS-B (gradient-basiert, mit Bounds)
      3. n_starts zufällige Neustarts zur Verifikation des globalen Minimums

    Parameters
    ----------
    data        : Fitting-Datenpunkte
    medium      : Kältemittelbezeichnung
    model_type  : 'original' oder 'modified'
    x0          : Startparameter (falls None: DEFAULT_PARAMS)
    bounds      : Liste von (lb, ub) Tupeln (falls None: DEFAULT_PARAMS Grenzen)
    n_starts    : Anzahl Zufallsstarts
    use_global  : Falls True, differential_evolution als globale Vorsuche
    verbose     : Ausgabe des Fortschritts

    Returns
    -------
    dict mit Feldern: params, param_names, g_opt, result_scipy
    """
    if x0 is None:
        x0 = np.array([v[0] for v in DEFAULT_PARAMS.values()])
    if bounds is None:
        bounds = [(v[1], v[2]) for v in DEFAULT_PARAMS.values()]

    obj = lambda p: objective_function(p, data, medium, model_type, f_ref=f_ref, m_dot_ref=m_dot_ref)

    best_result = None
    best_g = np.inf

    # --- Schritt 1: Globale Suche via Differential Evolution ---
    if use_global:
        if verbose:
            print("  [1/3] Globale Suche mit Differential Evolution ...")
        try:
            de_result = differential_evolution(
                obj, bounds,
                seed=42,
                maxiter=300,
                tol=1e-6,
                popsize=12,
                mutation=(0.5, 1.0),
                recombination=0.9,
                workers=1,
            )
            if de_result.fun < best_g:
                best_g = de_result.fun
                best_result = de_result
            if verbose:
                print(f"      g = {de_result.fun:.6f}")
        except Exception as e:
            warnings.warn(f"Differential Evolution fehlgeschlagen: {e}")

    # --- Schritt 2: Lokale Verfeinerung vom besten bisherigen Punkt ---
    x_init = best_result.x if best_result is not None else x0
    if verbose:
        print("  [2/3] Lokale Verfeinerung mit L-BFGS-B ...")
    local_result = minimize(
        obj, x_init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if local_result.fun < best_g:
        best_g = local_result.fun
        best_result = local_result
    if verbose:
        print(f"      g = {local_result.fun:.6f}")

    # --- Schritt 3: Multi-Start Zufallsstarts (Verifikation globales Minimum) ---
    if verbose:
        print(f"  [3/3] Multi-Start Verifikation ({n_starts} Zufallsstarts) ...")
    rng = np.random.default_rng(seed=0)

    for i in range(n_starts):
        # Zufälliger Startpunkt im Bereich [lb, ub] (log-uniform für Größenordnungen)
        x_rand = np.array([
            np.exp(rng.uniform(np.log(max(lb, 1e-12)), np.log(ub)))
            for lb, ub in bounds
        ])
        res = minimize(
            obj, x_rand,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if res.fun < best_g:
            best_g = res.fun
            best_result = res
            if verbose:
                print(f"      Neues Minimum bei Start {i+1}: g = {res.fun:.6f}")

    if verbose:
        print(f"\n  Bestes Ergebnis: g = {best_g:.6f}")

    return {
        "params": best_result.x,
        "param_names": PARAM_NAMES,
        "g_opt": best_g,
        "result_scipy": best_result,
    }


# ===========================================================================
# Evaluation: Berechne alle Punkte mit gefundenen Parametern
# ===========================================================================

def evaluate_model(params: np.ndarray, data: pd.DataFrame,
                   medium: str, model_type: str,
                   f_ref: float = 50.0,
                   m_dot_ref: float = None) -> pd.DataFrame:
    """
    Berechnet m_suc und W_comp für alle Datenpunkte mit den gefundenen Parametern.
    Gibt DataFrame mit Messwerten, Vorhersagen und Residuen zurück.
    """
    rows = []
    for idx, row in data.iterrows():
        p_suc = row["P1_mean"] * 1e5
        T_suc = row["T1_mean"] + 273.15
        p_dis = row["P2_mean"] * 1e5
        T_amb = row["Tamb_mean"] + 273.15
        f_Hz  = row["N"] / 60.0
        m_meas = row["suction_mf_mean"] * 1e-3
        W_meas = row["Pel_mean"]

        try:
            m_calc, W_calc = run_compressor_model(
                params, p_suc, T_suc, p_dis, f_Hz, T_amb, medium, model_type,
                f_ref=f_ref, m_dot_ref=m_dot_ref,
            )
            e_m = (1.0 - m_calc / m_meas) * 100.0   # % Fehler
            e_W = (1.0 - W_calc / W_meas) * 100.0   # % Fehler
        except Exception as ex:
            m_calc, W_calc, e_m, e_W = np.nan, np.nan, np.nan, np.nan

        # Öl-Spalte: "Ölbezeichnung" (wie im Datensatz), Fallback auf "oil"
        oil_val = row.get("Ölbezeichnung", row.get("oil", ""))
        rows.append({
            "index":            idx,
            "oil":              oil_val,
            "T_suc_C":          row["T1_mean"],
            "T_dis_meas_C":     row.get("T2_mean", np.nan),
            "N_rpm":            row["N"],
            "m_meas_gs":        m_meas * 1e3,
            "m_calc_gs":        m_calc * 1e3 if not np.isnan(m_calc) else np.nan,
            "e_m_pct":          e_m,
            "W_meas_W":         W_meas,
            "W_calc_W":         W_calc,
            "e_W_pct":          e_W,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Plots: Parity-Plots (wie Fig. 4 und 5 im Paper)
# ===========================================================================

def plot_parity(df_eval: pd.DataFrame, output_dir: Path, tag: str):
    """Erstellt Parity-Plots für Massenstrom und elektrische Leistung."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (col_meas, col_calc, label, unit) in zip(
        axes,
        [
            ("m_meas_gs", "m_calc_gs", "Massenstrom", "g/s"),
            ("W_meas_W",  "W_calc_W",  "Elektrische Leistung", "W"),
        ]
    ):
        df_clean = df_eval.dropna(subset=[col_meas, col_calc])
        x = df_clean[col_meas].values
        y = df_clean[col_calc].values
        vmin, vmax = min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.05

        ax.scatter(x, y, s=20, alpha=0.7, label="Datenpunkte")
        ax.plot([vmin, vmax], [vmin, vmax], "k-", lw=1, label="Ideale Linie")
        ax.plot([vmin, vmax], [vmin * 1.05, vmax * 1.05], "r--", lw=0.8, label="+5%")
        ax.plot([vmin, vmax], [vmin * 0.95, vmax * 0.95], "r--", lw=0.8, label="−5%")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel(f"{label} gemessen [{unit}]")
        ax.set_ylabel(f"{label} berechnet [{unit}]")
        ax.set_title(f"Parity-Plot: {label}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Fitting-Ergebnis – Modell: {tag}", fontsize=12)
    plt.tight_layout()
    fname = output_dir / f"parity_plot_{tag}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Parity-Plot gespeichert: {fname}")


def plot_error_distribution(df_eval: pd.DataFrame, output_dir: Path, tag: str):
    """Fehlerverteilung als Histogramm."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (col, label) in zip(axes, [("e_m_pct", "Massenstrom"), ("e_W_pct", "Leistung")]):
        vals = df_eval[col].dropna().values
        ax.hist(vals, bins=20, edgecolor="black", alpha=0.8)
        ax.axvline(-5, color="r", linestyle="--", linewidth=1)
        ax.axvline(+5, color="r", linestyle="--", linewidth=1, label="±5%-Grenze")
        ax.set_xlabel(f"Relativer Fehler {label} [%]")
        ax.set_ylabel("Häufigkeit [-]")
        ax.set_title(f"Fehlerverteilung: {label}")
        within = np.sum(np.abs(vals) <= 5)
        ax.legend(title=f"{within}/{len(vals)} innerhalb ±5%", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Fehlerverteilung – Modell: {tag}", fontsize=12)
    plt.tight_layout()
    fname = output_dir / f"error_distribution_{tag}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fehlerverteilung gespeichert: {fname}")


# ===========================================================================
# CSV laden
# ===========================================================================

def load_data(csv_path: Path, oil_filter: str = "all") -> pd.DataFrame:
    """
    Lädt Messdaten aus CSV (deutsches Format: Semikolon-Trenner, Komma als Dezimalzeichen).
    Zeile 1: Einheiten (übersprungen via header=1)
    Zeile 2: Spaltennamen
    ab Zeile 3: Daten

    Spaltenname für Öl: "Ölbezeichnung" (wie im Datensatz)
    """
    # sep=";" und decimal="," entsprechen dem deutschen CSV-Format des Datensatzes
    # header=1 überspringt die Einheitenzeile (Zeile 1) und nutzt Zeile 2 als Header
    df = pd.read_csv(csv_path, sep=";", header=1, decimal=",")
    df.columns = df.columns.str.strip()

    print(f"  Gelesene Spalten: {list(df.columns)}")

    # Numerische Spalten sicherstellen
    numeric_cols = ["P1_mean", "T1_mean", "P2_mean", "Tamb_mean",
                    "T2_mean", "suction_mf_mean", "Pel_mean", "N"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Zeilen mit fehlenden Pflichtdaten entfernen
    required = ["P1_mean", "T1_mean", "P2_mean", "suction_mf_mean", "Pel_mean", "N"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Folgende Pflicht-Spalten fehlen in der CSV: {missing_cols}\n"
            f"Vorhandene Spalten: {list(df.columns)}\n"
            f"Hinweis: Prüfe ob sep=';' und decimal=',' korrekt sind."
        )
    df = df.dropna(subset=required)

    # Öl-Filter: Spalte heißt "Ölbezeichnung" (wie im Datensatz)
    oil_col = "Ölbezeichnung"
    if oil_filter != "all":
        if oil_col not in df.columns:
            # Fallback: nach einer Spalte suchen die "l" enthält (Öl)
            candidates = [c for c in df.columns if "l" in c.lower() and "bezeichnung" in c.lower()]
            if candidates:
                oil_col = candidates[0]
                print(f"  Öl-Spalte gefunden: '{oil_col}'")
            else:
                raise KeyError(
                    f"Öl-Spalte '{oil_col}' nicht gefunden. "
                    f"Verfügbare Spalten: {list(df.columns)}"
                )
        df = df[df[oil_col].astype(str).str.strip() == oil_filter]
        if df.empty:
            all_oils = df[oil_col].unique() if oil_col in df.columns else "?"
            raise ValueError(
                f"Keine Daten für Öl '{oil_filter}' gefunden. "
                f"Verfügbare Öle: {all_oils}"
            )

    print(f"  {len(df)} Datenpunkte geladen (Ölfilter: '{oil_filter}')")
    return df.reset_index(drop=True)


def load_start_params(x0_csv: Path) -> np.ndarray:
    """
    Lädt Startparameter aus CSV. Unterstützt zwei Formate:

    Format A – eine Zeile, Parameternamen als Spaltenköpfe (wie im funktionierenden Skript):
        Ua_suc_ref,Ua_dis_ref,...
        16.05,13.96,...

    Format B – zwei Spalten param_name/value (Zeilenformat):
        param_name,value
        Ua_suc_ref,16.05
        ...
    """
    df = pd.read_csv(x0_csv)
    df.columns = df.columns.str.strip()

    # Format A erkennen: Parameternamen direkt als Spaltenköpfe
    if any(name in df.columns for name in PARAM_NAMES):
        row = df.iloc[0]
        x0 = []
        missing = []
        for name in PARAM_NAMES:
            if name in df.columns and pd.notna(row[name]):
                x0.append(float(row[name]))
            else:
                missing.append(name)
        if missing:
            raise ValueError(
                f"Startparameter-CSV (Format A): fehlende Spalten: {missing}\n"
                f"Vorhandene Spalten: {list(df.columns)}"
            )
        return np.array(x0)

    # Format B: Spalten param_name + value
    if "param_name" in df.columns and "value" in df.columns:
        x0 = []
        for name in PARAM_NAMES:
            match = df[df["param_name"].str.strip() == name]
            if match.empty:
                raise ValueError(f"Startparameter '{name}' nicht in {x0_csv} gefunden.")
            x0.append(float(match["value"].iloc[0]))
        return np.array(x0)

    raise ValueError(
        f"Unbekanntes Format in {x0_csv}.\n"
        f"Erwartet entweder Parameternamen als Spaltenköpfe (Format A) "
        f"oder Spalten 'param_name'+'value' (Format B).\n"
        f"Gefundene Spalten: {list(df.columns)}"
    )


# ===========================================================================
# Zusammenfassungs-Statistik
# ===========================================================================

def print_summary(df_eval: pd.DataFrame, fit_result: dict):
    """Gibt Fitting-Statistik auf der Konsole aus."""
    print("\n" + "=" * 60)
    print("FITTING-ERGEBNIS")
    print("=" * 60)
    print(f"  Zielfunktion g = {fit_result['g_opt']:.6f}")
    print("\n  Identifizierte Parameter:")
    for name, val in zip(fit_result["param_names"], fit_result["params"]):
        print(f"    {name:<16} = {val:.6e}")

    print("\n  Gütekriterien (alle Punkte):")
    for col, label in [("e_m_pct", "Massenstrom"), ("e_W_pct", "Leistung")]:
        vals = df_eval[col].dropna().values
        if len(vals) == 0:
            continue
        within_5 = np.sum(np.abs(vals) <= 5)
        print(f"    {label}:")
        print(f"      Punkte innerhalb ±5%: {within_5}/{len(vals)} "
              f"({within_5/len(vals)*100:.1f}%)")
        print(f"      Fehlerbereich: {vals.min():.2f}% bis {vals.max():.2f}%")
        print(f"      RMSE: {np.sqrt(np.mean(vals**2)):.2f}%")


# ===========================================================================
# CLI-Argumente
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameter-Fitting des Molinaroli-Verdichtermodells nach "
                    "Molinaroli et al. (2017), Gl. 31-33.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",       required=True, type=Path,
                        help="Pfad zur Mess-CSV-Datei")
    parser.add_argument("--oil",       default="all",
                        help="Ölbezeichnung zum Filtern (z.B. LPG100) oder 'all'")
    parser.add_argument("--medium",    default="R290",
                        help="Kältemittel (CoolProp-String, z.B. R290, R134a, R404A)")
    parser.add_argument("--model",     default="original",
                        choices=["original", "modified"],
                        help="Modellvariante: 'original' (Molinaroli 2017) oder "
                             "'modified' (mit Öleinfluss)")
    parser.add_argument("--x0_csv",    default=Path("data/start_params.csv"), type=Path,
                        help="CSV mit Startparametern (Spalten: param_name, value). "
                             "Standard: data/start_params.csv")
    parser.add_argument("--output_dir", default="results", type=Path,
                        help="Ausgabeordner für CSV und Plots")
    parser.add_argument("--n_starts",  default=10, type=int,
                        help="Anzahl Zufallsstarts für Multi-Start-Optimierung")
    parser.add_argument("--no_global", action="store_true",
                        help="Globale Vorsuche (differential_evolution) überspringen")
    parser.add_argument("--f_ref",     default=50.0, type=float,
                        help="Referenz-Drehfrequenz [Hz] (Netzfrequenz)")
    parser.add_argument("--V_h_cm3",   default=30.7, type=float,
                        help="Geometrisches Hubvolumen aus Herstellerdaten [cm³] "
                             "fuer m_dot_ref = V_h * f_ref * rho_sat(273.15 K) "
                             "(Paper Abschn. 3.1). Kein Fit-Parameter. Standard: 30.7 cm³")
    parser.add_argument("--quiet",     action="store_true",
                        help="Minimale Konsolenausgabe")
    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    verbose = not args.quiet

    if not VCLIBPY_AVAILABLE:
        print("FEHLER: VCLibPy ist nicht verfügbar. Bitte installieren.")
        sys.exit(1)

    # Ausgabeordner anlegen
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tag für Dateinamen
    tag = f"{args.model}_{args.oil}_{args.medium}"

    print(f"\nFitting gestartet:")
    print(f"  Modell:    {args.model}")
    print(f"  Kältemittel: {args.medium}")
    print(f"  Öl:        {args.oil}")
    print(f"  CSV:       {args.csv}")

    # --- Daten laden ---
    data = load_data(args.csv, oil_filter=args.oil)

    # --- Startparameter laden (immer aus start_params.csv, außer explizit anders) ---
    x0 = None
    if args.x0_csv.exists():
        x0 = load_start_params(args.x0_csv)
        print(f"  Startparameter aus: {args.x0_csv}")
    else:
        print(f"  Hinweis: {args.x0_csv} nicht gefunden – verwende Default-Startwerte.")
        print(f"  (DEFAULT_PARAMS aus Paper Tab. 2, Kompressor B, R290)")

    # Referenz-Massenstrom: V_h_geo * f_ref * rho_sat(273.15 K)  [Paper Abschn. 3.1]
    V_h_m3 = args.V_h_cm3 * 1e-6
    _med_ref = VCLibCoolProp(fluid_name=args.medium)
    _state_ref = _med_ref.calc_state("TQ", 273.15, 1)
    m_dot_ref = V_h_m3 * args.f_ref * float(_state_ref.d)
    print(f"  m_dot_ref = {m_dot_ref*1000:.4f} g/s  (V_h={args.V_h_cm3:.1f} cm³, f_ref={args.f_ref} Hz)")

    # --- Fitting ---
    print("\nOptimierung läuft ...")
    fit_result = fit_parameters(
        data=data,
        medium=args.medium,
        model_type=args.model,
        x0=x0,
        n_starts=args.n_starts,
        use_global=not args.no_global,
        f_ref=args.f_ref,
        m_dot_ref=m_dot_ref,
        verbose=verbose,
    )

    # --- Evaluation aller Punkte ---
    df_eval = evaluate_model(
        fit_result["params"], data, args.medium, args.model,
        f_ref=args.f_ref, m_dot_ref=m_dot_ref,
    )

    # --- Zusammenfassung ---
    if verbose:
        print_summary(df_eval, fit_result)

    # --- Ergebnisse speichern ---
    # 1) Parameter-CSV
    params_csv = output_dir / f"fitted_params_{tag}.csv"
    param_df = pd.DataFrame({
        "param_name": fit_result["param_names"],
        "value":      fit_result["params"],
    })
    param_df.to_csv(params_csv, index=False)
    print(f"\n  Parameter gespeichert: {params_csv}")

    # 2) Vorhersage-CSV
    pred_csv = output_dir / f"predictions_{tag}.csv"
    df_eval.to_csv(pred_csv, index=False)
    print(f"  Vorhersagen gespeichert: {pred_csv}")

    # 3) Plots
    plot_parity(df_eval, output_dir, tag)
    plot_error_distribution(df_eval, output_dir, tag)

    print("\nFertig.")
    return fit_result


if __name__ == "__main__":
    main()