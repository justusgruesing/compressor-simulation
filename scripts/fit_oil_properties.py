"""
python scripts/fit_oil_properties.py
python scripts/fit_oil_properties.py --plots cp
Fitting der linearen Korrelationen für die spezifische Wärmekapazität cp
und die Wärmeleitfähigkeit lambda von Reniso LPG 68 und LPG 100.

Datenquelle:
    - Reniso LPG 100: Fuchs Datenblatt VP.FLG.4.2022/250
      (2023_09_19_Reniso_LPG_100_Cp_Lambda_Visko_Dichte_Oberflächenspannung_Molekulargewicht.xlsx)
    - Reniso LPG 68:  Fuchs Datenblatt VP.FLG.4.2022/276
      (2025_12_02_Reniso_LPG_68_100_150_220_Kältemaschinenöle_Kenndaten.xlsx)

Methode:
    Lineare Regression (Methode der kleinsten Fehlerquadrate) mit numpy.polyfit.
    Korrelationsform:  y(T) = a + b * T,  T in Kelvin

Ergebnis:
    Die Koeffizienten werden in das LubricantFitting-Modul eingetragen als
    self._cp_a, self._cp_b, self._lam_a, self._lam_b.

Ausgabe:
    results/oil_properties/oil_property_fits.pdf              — Fit-Plots für cp und lambda
    results/oil_properties/oil_property_fits_residuals.pdf    — Residuenplots für cp und lambda
    results/oil_properties/oil_property_fits_cp.pdf           — Fit-Plots nur für cp bei --plots cp
    results/oil_properties/oil_property_fits_cp_residuals.pdf — Residuenplots nur für cp bei --plots cp
    results/oil_properties/fit_summary.txt                    — Koeffizienten und Fehlermaße

Autor: [Name eintragen]
Datum: [Datum eintragen]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# =====================================================================
#  Matplotlib-Style laden (EBC Paper-Style)
# =====================================================================
# Pfad ggf. anpassen — Standard: Style-Datei liegt im selben Ordner wie das Skript
_STYLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ebc_paper.mplstyle")
if os.path.exists(_STYLE_FILE):
    plt.style.use(_STYLE_FILE)
else:
    # Fallback: nach Style-Datei im aktuellen Arbeitsverzeichnis suchen
    if os.path.exists("ebc_paper.mplstyle"):
        plt.style.use("ebc_paper.mplstyle")


# =====================================================================
#  Deutsche Zahlenformatierung (Dezimalkomma statt -punkt)
# =====================================================================

def _de_tick(x, pos=None):
    """
    Tick-Formatter: ersetzt den Dezimalpunkt durch ein Komma und
    verwendet ein typographisches Minuszeichen (U+2212).
    """
    return f"{x:g}".replace("-", "\u2212").replace(".", ",")


_DE_FORMATTER = mticker.FuncFormatter(_de_tick)


def _apply_de_formatter(ax):
    """Setzt deutschen Dezimaltrenner auf x- und y-Achsen-Ticks."""
    ax.xaxis.set_major_formatter(_DE_FORMATTER)
    ax.yaxis.set_major_formatter(_DE_FORMATTER)


def _de(value, fmt):
    """Formatiert einen Float mit dt. Dezimaltrenner gemäß format spec."""
    return format(value, fmt).replace(".", ",")


def _fit_label(a, b, b_fmt=".6f"):
    """
    Erzeugt einen Legenden-Eintrag der Form  'Fit: a +/- |b|·T'
    mit deutschem Dezimalkomma und typographischem Minus.
    """
    sign = "+" if b >= 0 else "\u2212"
    return f"Fit: {_de(a, '.4f')} {sign} {_de(abs(b), b_fmt)}·T"


def _savefig_multi(fig, basename):
    """Speichert eine Figure in allen Formaten aus OUTPUT_FORMATS."""
    for ext in OUTPUT_FORMATS:
        path = os.path.join(OUTPUT_DIR, f"{basename}.{ext}")
        # dpi nur für Raster-Formate relevant; bei Vektor-Ausgaben ignoriert
        fig.savefig(path, dpi=150, bbox_inches="tight")


# =====================================================================
#  Ausgabeverzeichnis und -formate (hier anpassen)
# =====================================================================
OUTPUT_DIR = os.path.join("results", "oil_properties")

# Gewünschte Dateiformate für die Plots — beliebig erweiterbar.
# Unterstützt: 'pdf', 'png', 'svg' (und alle weiteren Matplotlib-Backends).
OUTPUT_FORMATS = ("pdf", "png", "svg")


def parse_args():
    """Liest optionale Kommandozeilenargumente ein."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit der Öl-Stoffwerte und Ausgabe der zugehörigen Plotdateien. "
            "Standardmäßig werden cp und lambda geplottet."
        )
    )
    parser.add_argument(
        "--plots",
        choices=("all", "cp", "lambda"),
        default="all",
        help=(
            "Auswahl der auszugebenden Plots: "
            "'all' für cp und lambda, 'cp' nur für spezifische Wärmekapazität, "
            "'lambda' nur für Wärmeleitfähigkeit."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Ausgabeverzeichnis für Summary und Plotdateien.",
    )
    return parser.parse_args()


ARGS = parse_args()
OUTPUT_DIR = ARGS.output_dir
PLOT_SELECTION = ARGS.plots

# =====================================================================
#  Messdaten aus Herstellerdatenblättern
# =====================================================================

# --- LPG 68 (Fuchs VP.FLG.4.2022/276) ---
T_celsius_68 = np.array([-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

cp_68 = np.array([1.63, 1.66, 1.68, 1.71, 1.73, 1.77, 1.80,
                  1.84, 1.88, 1.91, 1.95, 1.99, 2.03])  # kJ/(kg*K)

lam_68 = np.array([0.15695, 0.15541, 0.15327, 0.15220, 0.15032,
                   0.14916, 0.14815, 0.14691, 0.14609, 0.14500,
                   0.14365, 0.14270, 0.14124])  # W/(m*K)

# --- LPG 100 (Fuchs VP.FLG.4.2022/250) ---
T_celsius_100 = np.array([-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

cp_100 = np.array([1.64, 1.66, 1.68, 1.71, 1.74, 1.77, 1.81,
                   1.84, 1.88, 1.92, 1.96, 2.00, 2.04])  # kJ/(kg*K)

lam_100 = np.array([0.15797, 0.15587, 0.15434, 0.15304, 0.15106,
                    0.15003, 0.14930, 0.14787, 0.14701, 0.14597,
                    0.14454, 0.14353, 0.14235])  # W/(m*K)

# Umrechnung in Kelvin
T_K_68 = T_celsius_68 + 273.15
T_K_100 = T_celsius_100 + 273.15


# =====================================================================
#  Lineare Regression: y = a + b * T
# =====================================================================

def fit_linear(T, y, name):
    """
    Führt eine lineare Regression durch und gibt Koeffizienten
    sowie Fehlermaße zurück.

    Args:
        T: Temperatur-Array [K]
        y: Messwert-Array
        name: Bezeichnung für die Ausgabe

    Returns:
        dict mit Koeffizienten und Fehlermaßen
    """
    b, a = np.polyfit(T, y, 1)

    y_pred = a + b * T
    residuals = y - y_pred
    max_abs_err = np.max(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    max_rel_err = np.max(np.abs(residuals / y)) * 100
    mean_abs_err = np.mean(np.abs(residuals))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    n = len(y)
    r_squared_adj = 1.0 - (1.0 - r_squared) * (n - 1) / (n - 2)

    result = {
        "name": name,
        "a": a,
        "b": b,
        "r_squared": r_squared,
        "r_squared_adj": r_squared_adj,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "mean_abs_err": mean_abs_err,
        "rmse": rmse,
        "n_points": n,
        "T_min_C": T[0] - 273.15,
        "T_max_C": T[-1] - 273.15,
    }

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Korrelation:  y(T) = a + b * T")
    print(f"  a = {a:.6f}")
    print(f"  b = {b:.8f}")
    print(f"  R²       = {r_squared:.6f}")
    print(f"  R²_adj   = {r_squared_adj:.6f}")
    print(f"  Max. absoluter Fehler:      {max_abs_err:.6f}")
    print(f"  Max. relativer Fehler:      {max_rel_err:.2f} %")
    print(f"  Mittlerer absoluter Fehler: {mean_abs_err:.6f}")
    print(f"  RMSE:                       {rmse:.6f}")
    print(f"  Anzahl Messpunkte:          {n}")
    print(f"  Gültigkeitsbereich:         {T[0] - 273.15:.0f} °C bis {T[-1] - 273.15:.0f} °C")

    return result


# =====================================================================
#  Fits durchführen
# =====================================================================

print("=" * 60)
print("  LINEARER FIT DER ÖL-STOFFWERTE")
print("  Reniso LPG 68 und LPG 100")
print("=" * 60)

fit_cp_68 = fit_linear(T_K_68, cp_68, "LPG 68 — cp [kJ/(kg*K)]")
fit_cp_100 = fit_linear(T_K_100, cp_100, "LPG 100 — cp [kJ/(kg*K)]")
fit_lam_68 = fit_linear(T_K_68, lam_68, "LPG 68 — λ [W/(m*K)]")
fit_lam_100 = fit_linear(T_K_100, lam_100, "LPG 100 — λ [W/(m*K)]")

all_fits = [fit_cp_68, fit_cp_100, fit_lam_68, fit_lam_100]

a_cp_68, b_cp_68 = fit_cp_68["a"], fit_cp_68["b"]
a_cp_100, b_cp_100 = fit_cp_100["a"], fit_cp_100["b"]
a_lam_68, b_lam_68 = fit_lam_68["a"], fit_lam_68["b"]
a_lam_100, b_lam_100 = fit_lam_100["a"], fit_lam_100["b"]


# =====================================================================
#  Summary-Datei schreiben
# =====================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_path = os.path.join(OUTPUT_DIR, "fit_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:

    f.write("=" * 72 + "\n")
    f.write("  Linearer Fit der Öl-Stoffwerte — Zusammenfassung\n")
    f.write("  Reniso LPG 68 und LPG 100\n")
    f.write("=" * 72 + "\n\n")

    f.write("  Korrelationsform:  y(T) = a + b * T,   T in Kelvin\n")
    f.write("  Methode:           Lineare Regression (numpy.polyfit, Grad 1)\n")
    f.write("  Referenzzustand:   T_ref = 273.15 K,  h_oil(T_ref) = 0\n\n")

    # --- Einzelergebnisse ---
    for fit in all_fits:
        f.write("-" * 72 + "\n")
        f.write(f"  {fit['name']}\n")
        f.write("-" * 72 + "\n")
        f.write(f"  a (Achsenabschnitt)         = {fit['a']:.6f}\n")
        f.write(f"  b (Steigung)                = {fit['b']:.8f}\n")
        f.write(f"  R²                          = {fit['r_squared']:.6f}\n")
        f.write(f"  R²_adj                      = {fit['r_squared_adj']:.6f}\n")
        f.write(f"  Max. absoluter Fehler       = {fit['max_abs_err']:.6f}\n")
        f.write(f"  Max. relativer Fehler       = {fit['max_rel_err']:.2f} %\n")
        f.write(f"  Mittlerer absoluter Fehler  = {fit['mean_abs_err']:.6f}\n")
        f.write(f"  RMSE                        = {fit['rmse']:.6f}\n")
        f.write(f"  Anzahl Messpunkte           = {fit['n_points']}\n")
        f.write(f"  Gültigkeitsbereich          = {fit['T_min_C']:.0f} °C bis {fit['T_max_C']:.0f} °C\n\n")

    # --- Übersichtstabelle ---
    f.write("=" * 72 + "\n")
    f.write("  Übersichtstabelle\n")
    f.write("=" * 72 + "\n\n")

    header = (f"  {'Größe':<20s} {'Öl':<10s} {'a':>12s} {'b':>14s} "
              f"{'R²':>8s} {'R²_adj':>8s} {'RMSE':>10s} {'Max.rel.':>10s}\n")
    f.write(header)
    f.write("  " + "-" * 70 + "\n")

    for fit in all_fits:
        groesse = fit["name"].split("—")[1].strip()
        oel = fit["name"].split("—")[0].strip()
        row = (f"  {groesse:<20s} {oel:<10s} "
               f"{fit['a']:>12.6f} {fit['b']:>14.8f} "
               f"{fit['r_squared']:>8.4f} {fit['r_squared_adj']:>8.4f} "
               f"{fit['rmse']:>10.6f} {fit['max_rel_err']:>9.2f} %\n")
        f.write(row)

    # --- Koeffizienten für LubricantFitting ---
    f.write("\n\n")
    f.write("=" * 72 + "\n")
    f.write("  Koeffizienten für LubricantFitting-Modul\n")
    f.write("=" * 72 + "\n\n")

    f.write("  # LPG 68\n")
    f.write(f"  self._cp_a  = {a_cp_68:.6f}    # kJ/(kg*K)\n")
    f.write(f"  self._cp_b  = {b_cp_68:.8f}  # kJ/(kg*K²)\n")
    f.write(f"  self._lam_a = {a_lam_68:.6f}   # W/(m*K)\n")
    f.write(f"  self._lam_b = {b_lam_68:.8f} # W/(m*K²)\n\n")

    f.write("  # LPG 100\n")
    f.write(f"  self._cp_a  = {a_cp_100:.6f}    # kJ/(kg*K)\n")
    f.write(f"  self._cp_b  = {b_cp_100:.8f}  # kJ/(kg*K²)\n")
    f.write(f"  self._lam_a = {a_lam_100:.6f}   # W/(m*K)\n")
    f.write(f"  self._lam_b = {b_lam_100:.8f} # W/(m*K²)\n\n")

    # --- Enthalpie-Verifikation ---
    f.write("=" * 72 + "\n")
    f.write("  Verifikation der Enthalpie-Integration\n")
    f.write("=" * 72 + "\n\n")
    f.write("  h_oil(T) = a*(T - T_ref) + b/2*(T² - T_ref²)\n")
    f.write("  T_ref = 273.15 K,  h_oil(T_ref) = 0\n\n")

    T_ref = 273.15
    for oil_name, a, b in [("LPG 68", a_cp_68, b_cp_68),
                            ("LPG 100", a_cp_100, b_cp_100)]:
        f.write(f"  {oil_name}:\n")
        for T_C in [0, 20, 50, 80, 100]:
            T_K = T_C + 273.15
            h = a * (T_K - T_ref) + b / 2.0 * (T_K ** 2 - T_ref ** 2)
            f.write(f"    T = {T_C:4d} °C  →  h_oil = {h:8.2f} kJ/kg = {h * 1e3:10.1f} J/kg\n")
        f.write("\n")

print(f"\nSummary gespeichert: {summary_path}")


# =====================================================================
#  Plots
# =====================================================================

T_plot = np.linspace(T_K_68[0], T_K_68[-1], 200)
T_plot_C = T_plot - 273.15


def _plot_fit_axis(ax, T_C, y_data, marker, color, a, b, b_fmt,
                   ylabel, title):
    """Zeichnet Herstellerdaten und linearen Fit in eine Achse."""
    ax.plot(T_C, y_data, marker, color=color, markersize=6,
            label='Herstellerdaten')
    ax.plot(T_plot_C, a + b * T_plot, '-', color=color, linewidth=1.5,
            label=_fit_label(a, b, b_fmt))
    ax.set_xlabel('Temperatur in °C')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _apply_de_formatter(ax)


def _plot_residual_axis(ax, T_C, y_data, fit, T_K, title, color):
    """Zeichnet relative Residuen eines linearen Fits in eine Achse."""
    y_fit = fit["a"] + fit["b"] * T_K
    residuals = y_data - y_fit
    rel_residuals = residuals / y_data * 100

    ax.bar(T_C, rel_residuals, width=8, color=color, alpha=0.7,
           edgecolor=color)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Temperatur in °C')
    ax.set_ylabel('Relativer Fehler in %')
    ax.set_title(title)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True, alpha=0.3)
    _apply_de_formatter(ax)


def _plot_fit_selection(selection):
    """Erzeugt die Fit-Plots abhängig von der gewählten Stoffgröße."""
    if selection == "all":
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
        ]
        datasets = [
            (axes[0], T_celsius_68, cp_68, 'o', '#534AB7',
             a_cp_68, b_cp_68, ".6f", '$c_p$ in kJ/(kg·K)',
             'PAG 68 — Spezifische Wärmekapazität'),
            (axes[1], T_celsius_100, cp_100, 's', '#0F6E56',
             a_cp_100, b_cp_100, ".6f", '$c_p$ in kJ/(kg·K)',
             'PAG 100 — Spezifische Wärmekapazität'),
            (axes[2], T_celsius_68, lam_68, 'o', '#D85A30',
             a_lam_68, b_lam_68, ".8f", '$\\lambda$ in W/(m·K)',
             'PAG 68 — Wärmeleitfähigkeit'),
            (axes[3], T_celsius_100, lam_100, 's', '#993556',
             a_lam_100, b_lam_100, ".8f", '$\\lambda$ in W/(m·K)',
             'PAG 100 — Wärmeleitfähigkeit'),
        ]
        basename = "oil_property_fits"

    elif selection == "cp":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        datasets = [
            (axes[0], T_celsius_68, cp_68, 'o', '#534AB7',
             a_cp_68, b_cp_68, ".6f", '$c_p$ in kJ/(kg·K)',
             'PAG 68 — Spezifische Wärmekapazität'),
            (axes[1], T_celsius_100, cp_100, 's', '#0F6E56',
             a_cp_100, b_cp_100, ".6f", '$c_p$ in kJ/(kg·K)',
             'PAG 100 — Spezifische Wärmekapazität'),
        ]
        basename = "oil_property_fits_cp"

    elif selection == "lambda":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        datasets = [
            (axes[0], T_celsius_68, lam_68, 'o', '#D85A30',
             a_lam_68, b_lam_68, ".8f", '$\\lambda$ in W/(m·K)',
             'PAG 68 — Wärmeleitfähigkeit'),
            (axes[1], T_celsius_100, lam_100, 's', '#993556',
             a_lam_100, b_lam_100, ".8f", '$\\lambda$ in W/(m·K)',
             'PAG 100 — Wärmeleitfähigkeit'),
        ]
        basename = "oil_property_fits_lambda"

    else:
        raise ValueError(f"Unbekannte Plot-Auswahl: {selection}")

    for data in datasets:
        _plot_fit_axis(*data)

    fig.tight_layout()
    _savefig_multi(fig, basename)
    plt.close(fig)
    print(f"Fit-Plots gespeichert als '{basename}.*' in: "
          f"{OUTPUT_DIR} ({', '.join(OUTPUT_FORMATS)})")


def _plot_residual_selection(selection):
    """Erzeugt die Residuenplots abhängig von der gewählten Stoffgröße."""
    if selection == "all":
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes_flat = axes.ravel()
        datasets = [
            (axes_flat[0], T_celsius_68, cp_68, fit_cp_68, T_K_68,
             'LPG 68 — $c_p$', '#534AB7'),
            (axes_flat[1], T_celsius_100, cp_100, fit_cp_100, T_K_100,
             'LPG 100 — $c_p$', '#0F6E56'),
            (axes_flat[2], T_celsius_68, lam_68, fit_lam_68, T_K_68,
             'LPG 68 — $\\lambda$', '#D85A30'),
            (axes_flat[3], T_celsius_100, lam_100, fit_lam_100, T_K_100,
             'LPG 100 — $\\lambda$', '#993556'),
        ]
        basename = "oil_property_fits_residuals"

    elif selection == "cp":
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        datasets = [
            (axes[0], T_celsius_68, cp_68, fit_cp_68, T_K_68,
             'LPG 68 — $c_p$', '#534AB7'),
            (axes[1], T_celsius_100, cp_100, fit_cp_100, T_K_100,
             'LPG 100 — $c_p$', '#0F6E56'),
        ]
        basename = "oil_property_fits_cp_residuals"

    elif selection == "lambda":
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        datasets = [
            (axes[0], T_celsius_68, lam_68, fit_lam_68, T_K_68,
             'LPG 68 — $\\lambda$', '#D85A30'),
            (axes[1], T_celsius_100, lam_100, fit_lam_100, T_K_100,
             'LPG 100 — $\\lambda$', '#993556'),
        ]
        basename = "oil_property_fits_lambda_residuals"

    else:
        raise ValueError(f"Unbekannte Plot-Auswahl: {selection}")

    for data in datasets:
        _plot_residual_axis(*data)

    fig.tight_layout()
    _savefig_multi(fig, basename)
    plt.close(fig)
    print(f"Residuenplots gespeichert als '{basename}.*' in: "
          f"{OUTPUT_DIR} ({', '.join(OUTPUT_FORMATS)})")


_plot_fit_selection(PLOT_SELECTION)
_plot_residual_selection(PLOT_SELECTION)

print(f"\nAlle Ausgaben in: {os.path.abspath(OUTPUT_DIR)}")
