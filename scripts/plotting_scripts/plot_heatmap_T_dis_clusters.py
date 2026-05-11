"""
Heatmap-Plot der T_dis-Fehler über Überhitzung und Kondensationstemperatur
für die Modellausbaustufe II.

Liest cluster_metrics.csv, filtert auf den kombinierten Fit (params=all),
und erzeugt zwei Heatmaps (PAG 68 / PAG 100) nebeneinander.

Aufruf:
    python scripts/plotting_scripts/plot_heatmap_T_dis_clusters.py --cluster_csv results/analysis_4.4.2/cluster_metrics.csv --out_path results/analysis_4.4.2/heatmap_T_dis_SH_Tc.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Reihenfolge der Cluster-Achsen
SH_ORDER = ["SH10", "SH20", "SH30"]
TC_ORDER = ["Tc30", "Tc40", "Tc50", "Tc60"]
OIL_LABELS = {"lpg68": "PAG 68", "lpg100": "PAG 100"}


def make_pivot(df_oil: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Erzeugt eine Pivot-Tabelle SH × Tc."""
    p = df_oil.pivot_table(
        index="sh_cluster", columns="tc_cluster",
        values=value_col, aggfunc="mean")
    return p.reindex(index=SH_ORDER, columns=TC_ORDER)


def make_count_pivot(df_oil: pd.DataFrame) -> pd.DataFrame:
    p = df_oil.pivot_table(
        index="sh_cluster", columns="tc_cluster",
        values="n", aggfunc="sum")
    return p.reindex(index=SH_ORDER, columns=TC_ORDER)


def annotate_heatmap(ax, mae_pivot, count_pivot, vmax):
    """Beschriftet jede Zelle mit MAE und Punktzahl."""
    nrows, ncols = mae_pivot.shape
    for i in range(nrows):
        for j in range(ncols):
            mae = mae_pivot.values[i, j]
            n = count_pivot.values[i, j]
            if pd.isna(mae) or pd.isna(n):
                continue
            # Textfarbe weiß bei dunklem Hintergrund
            color = "white" if mae > 0.6 * vmax else "black"
            ax.text(j, i - 0.12, f"{mae:.1f} K",
                    ha="center", va="center", color=color,
                    fontsize=10, fontweight="bold")
            ax.text(j, i + 0.20, f"n={int(n)}",
                    ha="center", va="center", color=color,
                    fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_csv", type=Path, required=True)
    ap.add_argument("--out_path", type=Path,
                    default=Path("heatmap_T_dis_SH_Tc.png"))
    ap.add_argument("--params_filter", default="all",
                    help="Filter auf params_oil (Standard: 'all' = kombinierter Fit)")
    args = ap.parse_args()

    df = pd.read_csv(args.cluster_csv)

    # Auf den gewünschten Fit filtern (Default: kombinierter Fit)
    if args.params_filter:
        mask = df["file"].str.contains(f"params_{args.params_filter}", case=False)
        df = df[mask].copy()
        if df.empty:
            raise ValueError(
                f"Keine Zeilen für params_filter='{args.params_filter}' gefunden.")

    # Globalen Farbskalen-Bereich bestimmen, damit beide Plots vergleichbar sind
    vmax = float(np.nanmax(df["mae_T_dis_K"]))
    vmin = 0.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1, 1]})

    for ax, oil_norm in zip(axes, ["lpg68", "lpg100"]):
        sub = df[df["oil_norm"] == oil_norm].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"keine Daten für {OIL_LABELS[oil_norm]}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        mae_p = make_pivot(sub, "mae_T_dis_K")
        cnt_p = make_count_pivot(sub)

        im = ax.imshow(mae_p.values, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(len(TC_ORDER)))
        ax.set_xticklabels(TC_ORDER)
        ax.set_yticks(range(len(SH_ORDER)))
        ax.set_yticklabels(SH_ORDER)
        ax.set_title(OIL_LABELS[oil_norm])
        ax.set_xlabel("Kondensationstemperaturcluster")
        ax.set_ylabel("Überhitzungscluster")

        # Gridlinien zwischen Zellen
        ax.set_xticks(np.arange(-0.5, len(TC_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SH_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        annotate_heatmap(ax, mae_p, cnt_p, vmax)

    # Gemeinsame Farbskala rechts
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"MAE($T_\mathrm{dis}$) in K")

    fig.suptitle(
        "Mittlerer absoluter Fehler der Austrittstemperatur über "
        "Überhitzungs- und Kondensationscluster\n"
        f"(Modellausbaustufe II, kombinierter Fit, params_oil='{args.params_filter}')",
        fontsize=11, y=1.02)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap gespeichert: {args.out_path.resolve()}")


if __name__ == "__main__":
    main()