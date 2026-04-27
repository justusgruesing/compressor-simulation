"""
Plot-Skript für Abschnitt 3.4: Gemischeigenschaften
====================================================
Erzeugt zwei Abbildungen für die Bachelorarbeit:
  1. Gleichgewichtsmassenanteil w_KM(T, p)  — Löslichkeitsplot
  2. Kinematische Viskosität nu(T, p)       — Viskosität

Stoffpaarung: Propan (R-290) / Fuchs Reniso LPG 68
Kältemittelstoffdaten: CoolProp (ersetzt REFPROP für die Ploterzeugung)
Stil: ebc_paper.mplstyle
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import CoolProp.CoolProp as CP
import math
import os

# =================================================================
#  STYLE
# =================================================================
STYLE_PATH = os.path.join(os.path.dirname(__file__), "ebc_paper.mplstyle")
if os.path.exists(STYLE_PATH):
    plt.style.use(STYLE_PATH)

# EBC-Farbpalette (aus der .mplstyle, kommentiert)
EBC_RED    = "#E53027"
EBC_BLUE   = "#1058B0"
EBC_ORANGE = "#F47328"
EBC_PURPLE = "#5F379B"
EBC_DKRED  = "#9B231E"
EBC_PINK   = "#BE4198"
EBC_GREEN  = "#008746"

# Sequentielle Farbpalette für Isobaren (dunkelblau → hellblau)
_N_CMAP = 256
cmap_blue = LinearSegmentedColormap.from_list(
    "ebc_blue_seq",
    ["#0A2A66", "#1058B0", "#3B8DD4", "#7BBCE8", "#B0D8F0"],
    N=_N_CMAP,
)

# =================================================================
#  STOFFDATEN: Propan / LPG 68
# =================================================================
FLUID = "Propane"
T_CRIT = CP.PropsSI("Tcrit", FLUID)   # ≈ 369.89 K
M_KM   = 44.096e-3   # kg/mol  (Propan)
M_OIL  = 0.400       # kg/mol  (LPG 68, effektive Molmasse)

# --- Löslichkeitskoeffizienten (modifizierter Raoult, LPG 68) ---
SAT_A, SAT_B, SAT_C =  7.11, -2.34, -4.96
SAT_D, SAT_E, SAT_F = -4.02, -0.436, 5.05

# --- Kinematische Viskosität (doppelt-log Polynom, LPG 68) ---
KV_A, KV_B, KV_C =  1.3893e1, -7.8106e0,  9.4155e-1
KV_D, KV_E, KV_F = -5.2681e-2, -1.6857e0,  3.5412e-1
KV_G, KV_H, KV_I = -4.0291e-2, -1.2629e-1,  2.0426e-1


# =================================================================
#  HILFSFUNKTIONEN
# =================================================================
def p_sat_propane(T_K: float) -> float:
    """Sättigungsdampfdruck Propan [Pa] bei T [K] über CoolProp."""
    if T_K >= T_CRIT:
        return float("nan")
    return CP.PropsSI("P", "T", T_K, "Q", 1, FLUID)


def p_mix(w_KM: float, T_K: float) -> float:
    """Gemischdruck [Pa] nach modifiziertem Raoult-Ansatz."""
    if T_K >= T_CRIT or w_KM <= 0 or w_KM >= 1:
        return float("nan")
    Tr = T_K / T_CRIT
    x = w_KM / (w_KM + (1.0 - w_KM) * (M_KM / M_OIL))
    ps = p_sat_propane(T_K)
    if not np.isfinite(ps):
        return float("nan")
    f_corr = (SAT_A + SAT_B * Tr + SAT_C * Tr**2
              + SAT_D * x + SAT_E * x * Tr + SAT_F * x * Tr**2)
    return x * ps + x * (1.0 - x) * f_corr * ps


def solve_w_KM(T_K: float, p_Pa: float,
               w_lo: float = 0.001, w_hi: float = 0.999) -> float:
    """Gleichgewichtsmassenanteil w_KM bei (T, p) durch Brentq."""
    from scipy.optimize import brentq
    def obj(w):
        return p_mix(w, T_K) - p_Pa
    try:
        fa, fb = obj(w_lo), obj(w_hi)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return float("nan")
        if fa * fb > 0:
            return float("nan")
        return brentq(obj, w_lo, w_hi, maxiter=400)
    except Exception:
        return float("nan")


def kin_visc_mix(T_K: float, w_KM: float) -> float:
    """Kinematische Viskosität [mm²/s] des Gemischs."""
    logT = math.log10(T_K)
    y = ((KV_A + KV_B * logT + KV_C * logT**2)
         + w_KM * (KV_D + KV_E * logT + KV_F * logT**2)
         + w_KM**2 * (KV_G + KV_H * logT + KV_I * logT**2))
    return 10.0**(10.0**y) - 0.7


# =================================================================
#  PLOT 1: Löslichkeit  w_KM(T) bei verschiedenen Drücken
# =================================================================
def plot_solubility(save_path: str = "plot_loeslichkeit.pdf"):
    pressures_bar = [1, 1.5, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25]
    T_C = np.linspace(-40, 140, 400)
    T_K = T_C + 273.15

    fig, ax = plt.subplots(figsize=(10, 6.5))

    colors = cmap_blue(np.linspace(0.05, 0.95, len(pressures_bar)))

    for i, p_bar in enumerate(pressures_bar):
        p_Pa = p_bar * 1e5
        w_arr = np.array([solve_w_KM(T, p_Pa) for T in T_K])
        mask = np.isfinite(w_arr) & (w_arr > 0) & (w_arr < 1)
        if mask.sum() < 2:
            continue
        ax.plot(T_C[mask], w_arr[mask] * 100, color=colors[i],
                linewidth=1.8)

        # Label: bei hohen Drücken am rechten Rand, bei niedrigen am oberen Rand
        idx_valid = np.where(mask)[0]
        if p_bar >= 10:
            # Label am rechten Ende der Kurve
            idx_label = idx_valid[-1]
            ax.annotate(
                f"{p_bar} bar",
                xy=(T_C[idx_label], w_arr[idx_label] * 100),
                fontsize=10, color=colors[i],
                ha="left", va="bottom",
                xytext=(4, 2), textcoords="offset points",
            )
        else:
            # Label oben auf der Kurve bei fester y-Position
            target_w = min(0.48, w_arr[mask].max() * 0.85)
            diffs = np.abs(w_arr[mask] - target_w)
            idx_in_masked = np.argmin(diffs)
            idx_label = idx_valid[idx_in_masked]
            if w_arr[idx_label] > 0.05:
                ax.annotate(
                    f"{p_bar} bar",
                    xy=(T_C[idx_label], w_arr[idx_label] * 100),
                    fontsize=9, color=colors[i],
                    ha="center", va="bottom",
                    rotation=0,
                    xytext=(0, 4), textcoords="offset points",
                )

    ax.set_xlabel(r"Temperatur $T$ [°C]")
    ax.set_ylabel(r"Kältemittelmassenanteil $w_{\mathrm{KM}}$ [%]")
    ax.set_xlim(-40, 145)
    ax.set_ylim(0, 55)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  → {save_path}")
    plt.close(fig)


# =================================================================
#  PLOT 2: Kinematische Viskosität ν(T) bei verschiedenen Drücken
# =================================================================
def plot_viscosity(save_path: str = "plot_viskositaet.pdf"):
    pressures_bar = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
    w_KM_lines = [0, 1, 2.5, 5, 10, 20, 30]  # konstante w_KM [%]
    T_C = np.linspace(-40, 145, 500)
    T_K = T_C + 273.15

    fig, ax = plt.subplots(figsize=(10, 7.5))

    # --- Isobaren (hellblau, mit Druck-Labels) ---
    colors_p = cmap_blue(np.linspace(0.35, 0.85, len(pressures_bar)))

    for i, p_bar in enumerate(pressures_bar):
        p_Pa = p_bar * 1e5
        nu_arr = []
        for T in T_K:
            w = solve_w_KM(T, p_Pa)
            if np.isfinite(w) and 0 < w < 1:
                nu_arr.append(kin_visc_mix(T, w))
            else:
                nu_arr.append(float("nan"))
        nu_arr = np.array(nu_arr)
        mask = np.isfinite(nu_arr) & (nu_arr > 0.3)
        if mask.sum() < 2:
            continue
        ax.plot(T_C[mask], nu_arr[mask], color=colors_p[i],
                linewidth=1.2, alpha=0.85)

        # Label
        idx_valid = np.where(mask)[0]
        idx_label = idx_valid[len(idx_valid) // 4]
        ax.annotate(
            f"{p_bar} bar",
            xy=(T_C[idx_label], nu_arr[idx_label]),
            fontsize=8,
            color=colors_p[i],
            ha="center",
            va="bottom",
            rotation=0,
            xytext=(0, 3),
            textcoords="offset points",
        )

    # --- Iso-w_KM Linien (dunkelblau, fett) ---
    for w_pct in w_KM_lines:
        w = w_pct / 100.0
        nu_arr = np.array([kin_visc_mix(T, w) for T in T_K])
        mask = np.isfinite(nu_arr) & (nu_arr > 0.3)
        if mask.sum() < 2:
            continue
        ax.plot(T_C[mask], nu_arr[mask], color="#0A2A66",
                linewidth=1.8)

        # Label rechts
        idx_valid = np.where(mask)[0]
        idx_label = idx_valid[-1]
        label_text = (f"{w_pct:g} %" if w_pct > 0 else "0 % Propan")
        ax.annotate(
            label_text,
            xy=(T_C[idx_label], nu_arr[idx_label]),
            fontsize=10,
            fontweight="bold" if w_pct == 0 else "normal",
            color="#0A2A66",
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"Temperatur $T$ [°C]")
    ax.set_ylabel(r"Kinematische Viskosität $\nu$ [mm²/s]")
    ax.set_xlim(-40, 145)
    ax.set_ylim(0.5, 1000)

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(
        base=10, subs=np.arange(2, 10) * 0.1, numticks=50))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    # Manuelle Major-Tick-Labels für bessere Lesbarkeit
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:g}" if y in [0.5, 0.75, 1, 2, 3, 5, 7, 10,
                                        20, 30, 50, 100, 200, 300,
                                        500, 1000] else ""))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  → {save_path}")
    plt.close(fig)


# =================================================================
#  MAIN
# =================================================================
if __name__ == "__main__":
    print("Erzeuge Plots für Abschnitt 3.4 ...")
    plot_solubility("plot_loeslichkeit.pdf")
    plot_viscosity("plot_viskositaet.pdf")
    print("Fertig.")
