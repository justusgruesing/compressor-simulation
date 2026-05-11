"""
Plotting-Skript für die zentrale Diagnose-Abbildung von Kapitel 4.3:
Mischtemperatur T^(6) nach Schritt 6 als Funktion der gemessenen
Austrittstemperatur. Dokumentiert die Plateaubildung von T^(6) an
der kritischen Temperatur des Propans.

Erzeugte Datei
--------------
    results/plot_tmis_vs_tdis/oil_path_tmix_vs_tdis.pdf

Voraussetzungen
---------------
    - pandas, numpy, matplotlib
    - validation_detail-CSV mit den vier Diagnose-Spalten:
        T_mix_C, cp_comb_J_kgK, eps_dis, m_dot_total_kg_s

Konfiguration
-------------
Pfade werden NICHT über die Kommandozeile gesteuert, sondern direkt
in den drei Konstanten unten gesetzt. Anpassen, falls die CSV oder
der Style an einem anderen Ort liegen.

Aufruf
------
    python scripts/plotting_scripts/plot_tmix_vs_tdis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# KONFIGURATION  --  hier den Pfad zur eigenen Datei eintragen
# =====================================================================

# 1) Pfad zur Validierungs-CSV mit den vier neuen Diagnose-Spalten
CSV_PATH = Path("results/validation/oil_path/detail/validation_detail_params_all_val_all_oil_path_validation_only_2026-05-10_220136.csv")

# 2) Pfad zur ebc_paper.mplstyle-Datei
STYLE_PATH = Path("ebc.paper.mplstyle")

# 3) Ausgabeordner (wird angelegt, falls noch nicht vorhanden)
RESULTS_DIR = Path("results/plot_tmis_vs_tdis")

# 4) Name der erzeugten Plot-Datei
PLOT_FILENAME = "oil_path_tmix_vs_tdis.pdf"


# =====================================================================
# Konstanten
# =====================================================================
T_CRIT_PROPANE_C = 96.74          # kritische Temperatur Propan
HOTSPOT_THRESH_K = 5.0            # |e_T_dis| > 5 K = Hotspot
PLOT_RANGE_C = (40.0, 110.0)      # Achsenbereich


# =====================================================================
# Style laden
# =====================================================================
if STYLE_PATH.is_file():
    plt.style.use(str(STYLE_PATH))
    print(f"Style geladen: {STYLE_PATH}")
else:
    print(f"WARNUNG: Style-Datei nicht gefunden unter {STYLE_PATH} -- nutze matplotlib-Default")


# =====================================================================
# Daten einlesen
# =====================================================================
if not CSV_PATH.is_file():
    raise SystemExit(f"CSV nicht gefunden: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
required = ["T_mix_C", "T_dis_meas_C", "pressure_ratio", "e_T_dis_K"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Fehlende Spalten in CSV: {missing}")

df["abs_e"] = df["e_T_dis_K"].abs()
df["is_hotspot"] = df["abs_e"] > HOTSPOT_THRESH_K


# =====================================================================
# Plot
# =====================================================================
fig, ax = plt.subplots(figsize=(9.0, 7.0))

# Winkelhalbierende
diag = np.linspace(*PLOT_RANGE_C, 100)
ax.plot(diag, diag, color="0.4", lw=1.0, linestyle="-",
        label=r"Winkelhalbierende $T^{(6)} = T_{\mathrm{dis,meas}}$",
        zorder=1)

# Kritische Temperatur als horizontale Linie
ax.axhline(T_CRIT_PROPANE_C, color="C3", lw=1.2, linestyle="--",
           label=fr"$T_{{\mathrm{{krit}}}} = {T_CRIT_PROPANE_C:.2f}\,^\circ$C",
           zorder=2)

# Datenpunkte, farbcodiert nach Druckverhältnis
sc = ax.scatter(df["T_dis_meas_C"], df["T_mix_C"],
                c=df["pressure_ratio"],
                cmap="viridis",
                s=55, edgecolors="black", linewidths=0.6,
                zorder=3)

# Hotspot-Punkte zusätzlich markieren
hot = df[df["is_hotspot"]]
ax.scatter(hot["T_dis_meas_C"], hot["T_mix_C"],
           facecolors="none", edgecolors="C3", s=140, lw=1.4,
           label=fr"Hotspot ($|e_{{T_{{\mathrm{{dis}}}}}}| > {HOTSPOT_THRESH_K:.0f}\,$K)",
           zorder=4)

# Colorbar
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label(r"Druckverhältnis $\Pi_{\mathrm{c}} = p_{\mathrm{aus}}/p_{\mathrm{ein}}$")

# Achsen
ax.set_xlabel(r"$T_{\mathrm{dis,meas}}$ in $^\circ$C")
ax.set_ylabel(r"$T^{(6)}$ in $^\circ$C")
ax.set_xlim(*PLOT_RANGE_C)
ax.set_ylim(*PLOT_RANGE_C)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)

# Legende in der leeren unteren rechten Ecke (Daten liegen oben links)
ax.legend(loc="lower right", framealpha=0.95)

# Layout
fig.tight_layout()


# =====================================================================
# Ausgabeordner anlegen und speichern
# =====================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out_path = RESULTS_DIR / PLOT_FILENAME
fig.savefig(out_path, bbox_inches="tight")
print(f"Plot gespeichert: {out_path}")


# =====================================================================
# Kurze Statistik zur Bestätigung
# =====================================================================
n_plateau = (df["T_mix_C"] > 96.5).sum()
n_total = len(df)
plateau = df[df["T_mix_C"] > 96.5]["T_mix_C"]
print(f"\nPunkte am Plateau (T_mix > 96.5 °C): {n_plateau}/{n_total}")
if n_plateau > 0:
    print(f"  Mittelwert: {plateau.mean():.4f} °C")
    print(f"  Std:        {plateau.std():.4f} K")
    print(f"  Min/Max:    {plateau.min():.4f} / {plateau.max():.4f} °C")

n_hot = df["is_hotspot"].sum()
n_hot_plateau = ((df["T_mix_C"] > 96.5) & df["is_hotspot"]).sum()
print(f"\nHotspot-Punkte: {n_hot}")
print(f"  davon am Plateau: {n_hot_plateau}")
