# scripts/plotting_scripts/operating_point_diagram.py
#
# Plots operating points in a T_evap × T_cond diagram, showing which points
# were used for training and which for validation.
#
# Two layout modes:
#   combined: single plot, train and validation points with different markers
#   split:    two plots side by side (Training | Validation)
#
# Points can optionally be colored by superheat or speed to show the
# multi-dimensional structure of the test matrix.
#
# Input: operating_points_split_template_*.csv
#
# Examples:
#   # Combined plot:
#   python scripts/plotting_scripts/operating_point_diagram.py \
#       --split_csv results/split_template/operating_points_split_template_2026-03-12_112331.csv
#
#   # Side-by-side:
#   python scripts/plotting_scripts/operating_point_diagram.py --split_csv results/split_template/operating_points_split_template_2026-03-12_112331.csv --op_rows_csv results/split_template/operating_points_rows_2026-03-12_112331.csv --mode split --use_measured --filter_oil LPG68 --color_by superheat_cbar --show_limits --xlim -5 30 --ylim 10 85
#
#   # Color by superheat:
#   python scripts/plotting_scripts/operating_point_diagram.py \
#       --split_csv results/split_template/operating_points_split_template_2026-03-12_112331.csv \
#       --color_by superheat

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use("ebc.paper.mplstyle")


# =========================================================
# Constants
# =========================================================
TRAIN_COLOR = "#EC635C"      # EBC red
VAL_COLOR = "#4B81C4"        # EBC blue

TRAIN_MARKER = "o"
VAL_MARKER = "s"

# Colors for superheat / speed / oil categories
CATEGORY_COLORS = [
    "#EC635C", "#4B81C4", "#6EBB96", "#F49961",
    "#8768B4", "#B45955", "#CB74F4",
]

OIL_COLORS = {
    "LPG68": "#EC635C",
    "LPG100": "#4B81C4",
}
OIL_DISPLAY = {"LPG68": "LPG 68", "LPG100": "LPG 100"}

# =========================================================
# Operating limits (from Cui, Fig. 3.6 / Table 3.4)
# =========================================================
# Upper/right envelope vertices from manufacturer (T_evap, T_cond)
# These define the outer boundary where it is NOT determined by
# min-pressure or safety limits. Order: top-left → top-right → bottom-right.
ENVELOPE_UPPER = np.array([
    [-22.0, 68.0],
    [ -5.0, 80.0],
    [ 10.0, 80.0],
    [ 25.0, 70.0],
])

# Safety system limits (Table 3.4)
P_SUC_MIN_BAR = 2.0     # p1 untere Grenze → left vertical boundary
DELTA_P_MIN_BAR = 3.9    # Mindestdruckunterschied → bottom curve

LIMIT_COLOR = "#D32F2F"  # red


def _compute_min_pressure_curve(refrigerant="propane", delta_p_bar=3.9,
                                 T_evap_range=(-30, 27), n_points=100):
    """
    Compute T_cond = f(T_evap) where p_cond - p_evap = delta_p_bar.
    Returns arrays (T_evap_C, T_cond_C).
    """
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
    """Compute T_evap at which p_sat = p_min."""
    from vclibpy.media import RefProp
    med = RefProp(fluid_name=refrigerant)
    try:
        st = med.calc_state("PQ", p_min_bar * 1e5, 1.0)
        return float(st.T) - 273.15
    except Exception:
        return -25.3  # fallback for propane


def build_unified_boundary(refrigerant="propane"):
    """
    Build a single closed polygon representing the combined operating boundary.

    The boundary is assembled from three constraints:
      - Left edge: T_evap >= T_sat(p_suc_min)  (safety system)
      - Bottom edge: T_cond >= T_cond_min(T_evap) from Δp >= 3.9 bar  (min pressure)
      - Upper/right edges: manufacturer envelope

    Returns polygon as (N, 2) array of (T_evap, T_cond) vertices.
    """
    T_evap_safety = _compute_safety_T_evap(refrigerant, P_SUC_MIN_BAR)

    # Min-pressure curve from safety limit to right envelope edge
    T_evap_mp, T_cond_mp = _compute_min_pressure_curve(
        refrigerant=refrigerant, delta_p_bar=DELTA_P_MIN_BAR,
        T_evap_range=(T_evap_safety, ENVELOPE_UPPER[-1, 0]),
        n_points=80,
    )

    # The upper envelope has a top-right corner; the min-pressure curve
    # ends at the right side. We need to find the T_cond of the envelope's
    # bottom-right corner to connect properly.
    T_evap_right = float(ENVELOPE_UPPER[-1, 0])  # 25°C
    T_cond_right_mp = float(T_cond_mp[-1]) if len(T_cond_mp) > 0 else 25.0

    # Build the envelope top portion:
    # The upper envelope's left edge starts at T_evap_safety.
    # Interpolate T_cond at T_evap_safety from the leftmost upper envelope segment.
    env_T_evap_left = ENVELOPE_UPPER[0, 0]
    env_T_cond_left = ENVELOPE_UPPER[0, 1]
    env_T_evap_next = ENVELOPE_UPPER[1, 0] if len(ENVELOPE_UPPER) > 1 else env_T_evap_left
    env_T_cond_next = ENVELOPE_UPPER[1, 1] if len(ENVELOPE_UPPER) > 1 else env_T_cond_left

    # If safety limit is to the left of the first envelope point, extend vertically
    if T_evap_safety <= env_T_evap_left:
        T_cond_top_at_safety = env_T_cond_left
    else:
        # Interpolate linearly
        frac = (T_evap_safety - env_T_evap_left) / max(1e-9, env_T_evap_next - env_T_evap_left)
        T_cond_top_at_safety = env_T_cond_left + frac * (env_T_cond_next - env_T_cond_left)

    # T_cond at safety T_evap on the min-pressure curve
    T_cond_bot_at_safety = float(T_cond_mp[0]) if len(T_cond_mp) > 0 else 10.0

    # Assemble polygon clockwise:
    # 1. Left edge: vertical from bottom (min-pressure) to top (envelope) at T_evap_safety
    # 2. Upper envelope: from safety to right
    # 3. Right edge: vertical from envelope down to min-pressure curve
    # 4. Bottom: min-pressure curve from right to left (reversed)

    vertices = []

    # Bottom-left corner (safety × min-pressure intersection)
    vertices.append([T_evap_safety, T_cond_bot_at_safety])

    # Left edge up to envelope top
    vertices.append([T_evap_safety, T_cond_top_at_safety])

    # Upper envelope points (only those to the right of safety limit)
    for pt in ENVELOPE_UPPER:
        if pt[0] >= T_evap_safety:
            vertices.append([pt[0], pt[1]])

    # Right edge down to min-pressure curve
    vertices.append([T_evap_right, T_cond_right_mp])

    # Bottom edge: min-pressure curve reversed (right to left)
    for te, tc in zip(T_evap_mp[::-1], T_cond_mp[::-1]):
        vertices.append([te, tc])

    # Close polygon
    vertices.append(vertices[0])

    return np.array(vertices), T_evap_safety


def draw_operating_limits(ax, refrigerant="propane"):
    """
    Draw the unified operating boundary as a single red solid polygon.
    Returns legend handles.
    """
    legend_handles = []

    try:
        boundary, T_evap_safety = build_unified_boundary(refrigerant)

        # Fill
        ax.fill(boundary[:, 0], boundary[:, 1],
                color=LIMIT_COLOR, alpha=0.06, zorder=0)

        # Solid red outline
        ax.plot(boundary[:, 0], boundary[:, 1],
                color=LIMIT_COLOR, linewidth=1.8,
                linestyle="-", zorder=1, alpha=0.8)

        legend_handles.append(
            Line2D([0], [0], color=LIMIT_COLOR, linewidth=1.8,
                   label="Betriebsgrenzen"))

    except Exception as e:
        print(f"  [WARN] Could not draw operating limits: {e}")

    return legend_handles


# =========================================================
# Helpers
# =========================================================
def _ts():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def load_split_template(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("t_evap_set_c", "t_evap_c", "t_evap"):
            col_map[c] = "T_evap"
        elif cl in ("t_cond_set_c", "t_cond_c", "t_cond"):
            col_map[c] = "T_cond"
        elif cl in ("split_role",):
            col_map[c] = "split_role"
        elif cl in ("t1_sh_set", "sh_k", "superheat"):
            col_map[c] = "SH"
        elif cl in ("drehzahl_set", "n_rpm", "speed"):
            col_map[c] = "speed"

    df = df.rename(columns=col_map)

    if "T_evap" not in df.columns or "T_cond" not in df.columns:
        raise ValueError("Could not find T_evap and T_cond columns.")
    if "split_role" not in df.columns:
        raise ValueError("No split_role column found.")

    df["is_train"] = df["split_role"].astype(str).str.lower().str.strip() == "train"
    df["is_val"] = ~df["is_train"]

    return df


def load_measured_points(
    op_rows_path: Path,
    split_path: Path,
    refrigerant: str = "propane",
) -> pd.DataFrame:
    """
    Load actual measured operating points from op_rows CSV, join with
    split_template for train/validation assignment, and compute actual
    T_evap and T_cond from measured pressures via RefProp.
    """
    from vclibpy.media import RefProp

    rows = pd.read_csv(op_rows_path)
    split = pd.read_csv(split_path)

    # Join on op_id
    split_slim = split[["op_id", "split_role"]].copy()
    df = rows.merge(split_slim, on="op_id", how="left")

    df["is_train"] = df["split_role"].astype(str).str.lower().str.strip() == "train"
    df["is_val"] = ~df["is_train"]

    # Normalize oil column
    oil_col = "_oil_norm" if "_oil_norm" in df.columns else "Ölbezeichnung"
    df["oil"] = df[oil_col].astype(str).str.strip().str.upper().str.replace(" ", "")

    # Normalize SH and speed
    if "T1_SH" in df.columns:
        df["SH"] = df["T1_SH"]
    if "N" in df.columns:
        df["speed"] = df["N"]
    elif "Drehzahl" in df.columns:
        df["speed"] = df["Drehzahl"]

    # Compute actual T_sat from measured pressures
    print("  Computing T_sat from measured pressures via RefProp ...")
    med = RefProp(fluid_name=refrigerant)

    T_evap_actual = []
    T_cond_actual = []
    for _, row in df.iterrows():
        p_suc = float(row["P1_mean"]) * 1e5   # bar → Pa
        p_dis = float(row["P2_mean"]) * 1e5

        try:
            st_suc = med.calc_state("PQ", p_suc, 1.0)
            T_evap_actual.append(float(st_suc.T) - 273.15)
        except Exception:
            T_evap_actual.append(float(row.get("T_evap", np.nan)))

        try:
            st_dis = med.calc_state("PQ", p_dis, 0.0)
            T_cond_actual.append(float(st_dis.T) - 273.15)
        except Exception:
            T_cond_actual.append(float(row.get("T_cond", np.nan)))

    df["T_evap"] = T_evap_actual
    df["T_cond"] = T_cond_actual

    # Keep set values for reference
    if "T_evap" in rows.columns:
        df["T_evap_set"] = rows["T_evap"]
    if "T_cond" in rows.columns:
        df["T_cond_set"] = rows["T_cond"]

    print(f"  T_evap actual range: {df['T_evap'].min():.2f} to {df['T_evap'].max():.2f} °C")
    print(f"  T_cond actual range: {df['T_cond'].min():.2f} to {df['T_cond'].max():.2f} °C")
    print(f"  Points: {len(df)} ({df['is_train'].sum()} train, {df['is_val'].sum()} val)")

    return df


# =========================================================
# Plot: combined (one plot, different markers)
# =========================================================
def plot_combined(df, out_path, color_by=None, point_size=120,
                  continuous_cbar=False, cmap="viridis", cbar_label="",
                  show_limits=False, refrigerant="propane",
                  xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 8))

    train = df[df["is_train"]].copy()
    val = df[df["is_val"]].copy()

    if continuous_cbar and color_by is not None and color_by in df.columns:
        _plot_continuous_cbar(ax, train, val, color_by, point_size, fig,
                              cmap=cmap, cbar_label=cbar_label)
    elif color_by is not None and color_by in df.columns:
        _plot_colored(ax, train, val, color_by, point_size, fig)
    else:
        ax.scatter(
            train["T_evap"], train["T_cond"],
            s=point_size, marker=TRAIN_MARKER, color=TRAIN_COLOR,
            edgecolors="white", linewidths=0.8, zorder=3,
            label=f"Training (n={len(train)})",
        )
        ax.scatter(
            val["T_evap"], val["T_cond"],
            s=point_size, marker=VAL_MARKER, color=VAL_COLOR,
            edgecolors="white", linewidths=0.8, zorder=3,
            label=f"Validierung (n={len(val)})",
        )
        ax.legend(loc="upper left", fontsize=10, frameon=True)

    # Operating limits
    if show_limits:
        limit_handles = draw_operating_limits(ax, refrigerant=refrigerant)

    _setup_axes(ax, df, xlim=xlim, ylim=ylim)
    ax.set_title("Betriebspunkte — Training & Validierung", fontsize=13)

    # Merge limit legend handles with existing legend
    if show_limits and limit_handles:
        existing_handles, existing_labels = ax.get_legend_handles_labels()
        ax.legend(handles=existing_handles + limit_handles,
                  loc="upper left", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                format=out_path.suffix.lstrip("."))
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# =========================================================
# Plot: split (two plots side by side)
# =========================================================
def plot_split(df, out_path, color_by=None, point_size=120,
               continuous_cbar=False, cmap="viridis", cbar_label="",
               show_limits=False, refrigerant="propane",
               xlim=None, ylim=None):
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(16, 7),
                                            sharey=True, sharex=True)

    train = df[df["is_train"]].copy()
    val = df[df["is_val"]].copy()

    if continuous_cbar and color_by is not None and color_by in df.columns:
        # Shared vmin/vmax across both panels
        all_vals = pd.concat([train[color_by], val[color_by]]).dropna()
        vmin = float(all_vals.min())
        vmax = float(all_vals.max())
        _plot_continuous_cbar_single(ax_train, train, color_by, point_size,
                                     TRAIN_MARKER, fig, cmap=cmap,
                                     cbar_label=cbar_label,
                                     vmin=vmin, vmax=vmax, show_cbar=False)
        _plot_continuous_cbar_single(ax_val, val, color_by, point_size,
                                     VAL_MARKER, fig, cmap=cmap,
                                     cbar_label=cbar_label,
                                     vmin=vmin, vmax=vmax, show_cbar=True)
    elif color_by is not None and color_by in df.columns:
        _plot_colored_single(ax_train, train, color_by, point_size,
                             TRAIN_MARKER, fig, show_legend=True)
        _plot_colored_single(ax_val, val, color_by, point_size,
                             VAL_MARKER, fig, show_legend=False)
    else:
        ax_train.scatter(
            train["T_evap"], train["T_cond"],
            s=point_size, marker=TRAIN_MARKER, color=TRAIN_COLOR,
            edgecolors="white", linewidths=0.8, zorder=3,
        )
        ax_val.scatter(
            val["T_evap"], val["T_cond"],
            s=point_size, marker=VAL_MARKER, color=VAL_COLOR,
            edgecolors="white", linewidths=0.8, zorder=3,
        )

    # Operating limits on both panels
    if show_limits:
        draw_operating_limits(ax_train, refrigerant=refrigerant)
        limit_handles = draw_operating_limits(ax_val, refrigerant=refrigerant)

    _setup_axes(ax_train, df, xlim=xlim, ylim=ylim)
    _setup_axes(ax_val, df, xlim=xlim, ylim=ylim)

    ax_train.set_title(f"Training (n={len(train)})", fontsize=13)
    ax_val.set_title(f"Validierung (n={len(val)})", fontsize=13)
    ax_val.set_ylabel("")  # shared y axis

    # Add limits legend below both panels
    if show_limits and limit_handles:
        fig.legend(handles=limit_handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.04), ncol=len(limit_handles),
                   fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                format=out_path.suffix.lstrip("."))
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# =========================================================
# Color by category (superheat or speed)
# =========================================================
def _plot_colored(ax, train, val, color_by, point_size, fig):
    """Combined plot with color by category, markers distinguish train/val."""
    all_data = pd.concat([train, val], ignore_index=True)
    categories = sorted(all_data[color_by].dropna().unique())

    # Use oil-specific colors if color_by is oil, otherwise generic
    if color_by == "oil":
        color_map = {cat: OIL_COLORS.get(cat, CATEGORY_COLORS[i % len(CATEGORY_COLORS)])
                     for i, cat in enumerate(categories)}
        display_map = OIL_DISPLAY
        label_prefix = ""
        label_unit = ""
    else:
        color_map = {cat: CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
                     for i, cat in enumerate(categories)}
        display_map = {}
        label_prefix = {
            "SH": "$\\Delta T_{SH}$",
            "speed": "$N$",
        }.get(color_by, color_by)
        label_unit = {
            "SH": "K",
            "speed": "Hz",
        }.get(color_by, "")

    for cat in categories:
        color = color_map[cat]
        if color_by == "oil":
            cat_label = display_map.get(cat, cat)
        else:
            cat_label = f"{label_prefix}={cat:.0f} {label_unit}".strip()

        mask_t = train[color_by] == cat
        mask_v = val[color_by] == cat

        if mask_t.any():
            ax.scatter(
                train.loc[mask_t, "T_evap"], train.loc[mask_t, "T_cond"],
                s=point_size, marker=TRAIN_MARKER, color=color,
                edgecolors="white", linewidths=0.8, zorder=3,
            )
        if mask_v.any():
            ax.scatter(
                val.loc[mask_v, "T_evap"], val.loc[mask_v, "T_cond"],
                s=point_size, marker=VAL_MARKER, color=color,
                edgecolors="white", linewidths=0.8, zorder=3,
            )

    # Build clean legend
    handles = []
    for cat in categories:
        color = color_map[cat]
        if color_by == "oil":
            cat_label = display_map.get(cat, cat)
        else:
            cat_label = f"{label_prefix}={cat:.0f} {label_unit}".strip()
        handles.append(Line2D([0], [0], linestyle="None",
                              marker="o", markersize=9, color=color,
                              markeredgecolor="white", markeredgewidth=0.8,
                              label=cat_label))
    handles.append(Line2D([0], [0], linestyle="None",
                          marker=TRAIN_MARKER, markersize=9, color="0.4",
                          markeredgecolor="white", label="Training"))
    handles.append(Line2D([0], [0], linestyle="None",
                          marker=VAL_MARKER, markersize=9, color="0.4",
                          markeredgecolor="white", label="Validierung"))
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)


def _plot_colored_single(ax, data, color_by, point_size, marker, fig,
                         show_legend=True):
    """Single panel with color by category."""
    categories = sorted(data[color_by].dropna().unique())

    if color_by == "oil":
        color_map = {cat: OIL_COLORS.get(cat, CATEGORY_COLORS[i % len(CATEGORY_COLORS)])
                     for i, cat in enumerate(categories)}
        display_map = OIL_DISPLAY
    else:
        all_cats = sorted(data[color_by].dropna().unique())
        color_map = {cat: CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
                     for i, cat in enumerate(all_cats)}
        display_map = {}

    label_prefix = {
        "SH": "$\\Delta T_{SH}$",
        "speed": "$N$",
    }.get(color_by, color_by)
    label_unit = {"SH": "K", "speed": "Hz"}.get(color_by, "")

    for cat in categories:
        color = color_map[cat]
        mask = data[color_by] == cat
        if color_by == "oil":
            cat_label = display_map.get(cat, cat)
        else:
            cat_label = f"{label_prefix}={cat:.0f} {label_unit}".strip()
        if mask.any():
            ax.scatter(
                data.loc[mask, "T_evap"], data.loc[mask, "T_cond"],
                s=point_size, marker=marker, color=color,
                edgecolors="white", linewidths=0.8, zorder=3,
                label=cat_label if show_legend else "_nolegend_",
            )

    if show_legend:
        ax.legend(loc="upper left", fontsize=9, frameon=True)


# =========================================================
# Continuous colorbar mode (like operating_points_map.py)
# =========================================================
def _plot_continuous_cbar(ax, train, val, color_col, point_size, fig,
                          cmap="viridis", cbar_label=""):
    """Combined plot with continuous colorbar, markers distinguish train/val."""
    import matplotlib.colors as mcolors

    all_vals = pd.concat([train[color_col], val[color_col]]).dropna()
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if len(train) > 0:
        ax.scatter(
            train["T_evap"], train["T_cond"],
            c=train[color_col], cmap=cmap, norm=norm,
            s=point_size, marker=TRAIN_MARKER,
            edgecolors="none", alpha=0.9, zorder=3,
        )
    if len(val) > 0:
        ax.scatter(
            val["T_evap"], val["T_cond"],
            c=val[color_col], cmap=cmap, norm=norm,
            s=point_size, marker=VAL_MARKER,
            edgecolors="none", alpha=0.9, zorder=3,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    # Marker legend
    handles = [
        Line2D([0], [0], linestyle="None", marker=TRAIN_MARKER,
               markersize=9, color="0.4", markeredgecolor="white",
               label=f"Training (n={len(train)})"),
        Line2D([0], [0], linestyle="None", marker=VAL_MARKER,
               markersize=9, color="0.4", markeredgecolor="white",
               label=f"Validierung (n={len(val)})"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)


def _plot_continuous_cbar_single(ax, data, color_col, point_size, marker,
                                  fig, cmap="viridis", cbar_label="",
                                  vmin=None, vmax=None, show_cbar=True):
    """Single panel with continuous colorbar."""
    import matplotlib.colors as mcolors

    vals = data[color_col].dropna()
    if vmin is None:
        vmin = float(vals.min())
    if vmax is None:
        vmax = float(vals.max())
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if len(data) > 0:
        sc = ax.scatter(
            data["T_evap"], data["T_cond"],
            c=data[color_col], cmap=cmap, norm=norm,
            s=point_size, marker=marker,
            edgecolors="none", alpha=0.9, zorder=3,
        )
        if show_cbar:
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(cbar_label)


# =========================================================
# Axes setup
# =========================================================
def _setup_axes(ax, df, xlim=None, ylim=None):
    ax.set_xlabel("Verdampfungstemperatur $T_{Verd}$ in °C")
    ax.set_ylabel("Kondensationstemperatur $T_{Kond}$ in °C")
    ax.grid(True, linewidth=0.5, alpha=0.3)

    t_evap_vals = sorted(df["T_evap"].dropna().unique())
    t_cond_vals = sorted(df["T_cond"].dropna().unique())

    # Use set values for ticks if available (round numbers), else auto
    if "T_evap_set" in df.columns:
        tick_evap = sorted(df["T_evap_set"].dropna().unique())
        tick_cond = sorted(df["T_cond_set"].dropna().unique())
    else:
        tick_evap = t_evap_vals
        tick_cond = t_cond_vals

    # Only set manual ticks if there aren't too many
    if len(tick_evap) <= 15:
        ax.set_xticks(tick_evap)
    if len(tick_cond) <= 15:
        ax.set_yticks(tick_cond)

    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    else:
        pad_x = max(2, (max(t_evap_vals) - min(t_evap_vals)) * 0.08)
        ax.set_xlim(min(t_evap_vals) - pad_x, max(t_evap_vals) + pad_x)

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        pad_y = max(2, (max(t_cond_vals) - min(t_cond_vals)) * 0.08)
        ax.set_ylim(min(t_cond_vals) - pad_y, max(t_cond_vals) + pad_y)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Plot operating points in a T_evap × T_cond diagram "
                    "with train/validation split."
    )
    ap.add_argument("--split_csv", required=True, type=Path,
                    help="Path to operating_points_split_template_*.csv")
    ap.add_argument("--mode", choices=["combined", "split"], default="combined",
                    help="combined: one plot with markers. "
                         "split: two plots side by side. (default: combined)")
    ap.add_argument("--color_by",
                    choices=["superheat", "speed", "oil", "superheat_cbar", "none"],
                    default="none",
                    help="Color points by category (discrete) or with colorbar (continuous). "
                         "superheat/speed/oil = discrete colors. "
                         "superheat_cbar = continuous colorbar like operating_points_map.py.")
    ap.add_argument("--cmap", default="viridis",
                    help="Colormap for superheat_cbar mode (default: viridis)")

    # Filter
    ap.add_argument("--filter_oil", default=None,
                    help="Show only points for this oil (LPG68 | LPG100). "
                         "Only works with --use_measured.")
    ap.add_argument("--show_limits", action="store_true",
                    help="Draw operating limit boundaries (Cui Fig. 3.6): "
                         "manufacturer envelope, min pressure difference, "
                         "safety system limit. Requires REFPROP for pressure curves.")

    # Axis limits
    ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"),
                    help="Override x-axis limits (T_evap) [°C], e.g. --xlim -30 30")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                    help="Override y-axis limits (T_cond) [°C], e.g. --ylim 0 85")

    # Measured mode
    ap.add_argument("--use_measured", action="store_true",
                    help="Use actual measured pressures to compute T_evap/T_cond "
                         "(requires --op_rows_csv and REFPROP)")
    ap.add_argument("--op_rows_csv", type=Path, default=None,
                    help="Path to operating_points_rows_*.csv (for --use_measured)")
    ap.add_argument("--refrigerant", default="propane",
                    help="Refrigerant name for RefProp (default: propane)")

    ap.add_argument("--point_size", type=float, default=120)
    ap.add_argument("--out_dir", default="results/operating_point_diagram", type=Path)
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    args = ap.parse_args()

    if not args.split_csv.exists():
        raise FileNotFoundError(args.split_csv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.use_measured:
        if args.op_rows_csv is None:
            raise ValueError("--use_measured requires --op_rows_csv")
        if not args.op_rows_csv.exists():
            raise FileNotFoundError(args.op_rows_csv)

        df = load_measured_points(
            op_rows_path=args.op_rows_csv,
            split_path=args.split_csv,
            refrigerant=args.refrigerant,
        )
        data_tag = "measured"
    else:
        df = load_split_template(args.split_csv)
        data_tag = "set"

    n_train = df["is_train"].sum()
    n_val = df["is_val"].sum()
    print(f"  Loaded {len(df)} operating points: {n_train} train, {n_val} validation")

    # Filter by oil if requested
    if args.filter_oil:
        oil_norm = args.filter_oil.strip().upper().replace(" ", "")
        if "oil" not in df.columns:
            print("  [WARN] No oil column — --filter_oil requires --use_measured")
        else:
            before = len(df)
            df = df[df["oil"] == oil_norm].copy()
            print(f"  Filtered to oil={oil_norm}: {len(df)}/{before} points")

    # Determine color mode
    continuous_cbar = False
    color_col = None
    cbar_label = ""

    if args.color_by == "superheat_cbar":
        color_col = "SH" if "SH" in df.columns else None
        if color_col is None:
            print("  [WARN] No superheat column found, ignoring --color_by")
        else:
            continuous_cbar = True
            cbar_label = "Überhitzung $\\Delta T_{ÜH}$ in K"
            print(f"  Color by: superheat (continuous colorbar, cmap={args.cmap})")
    elif args.color_by == "superheat":
        color_col = "SH" if "SH" in df.columns else None
        if color_col is None:
            print("  [WARN] No superheat column found, ignoring --color_by")
    elif args.color_by == "speed":
        color_col = "speed" if "speed" in df.columns else None
        if color_col is None:
            print("  [WARN] No speed column found, ignoring --color_by")
    elif args.color_by == "oil":
        color_col = "oil" if "oil" in df.columns else None
        if color_col is None:
            print("  [WARN] No oil column found, ignoring --color_by. "
                  "Use --use_measured to include oil information.")

    if color_col and not continuous_cbar:
        cats = sorted(df[color_col].dropna().unique())
        print(f"  Color by: {args.color_by} → {cats}")

    stamp = _ts()
    color_tag = f"_by_{args.color_by}" if args.color_by != "none" else ""
    oil_tag = f"_{args.filter_oil.lower()}" if args.filter_oil else ""
    limits_tag = "_limits" if args.show_limits else ""

    if args.mode == "combined":
        out_path = out_dir / f"op_diagram_combined_{data_tag}{oil_tag}{color_tag}{limits_tag}_{stamp}.{args.out_format}"
        plot_combined(df, out_path, color_by=color_col, point_size=args.point_size,
                      continuous_cbar=continuous_cbar, cmap=args.cmap,
                      cbar_label=cbar_label,
                      show_limits=args.show_limits, refrigerant=args.refrigerant,
                      xlim=args.xlim, ylim=args.ylim)

    elif args.mode == "split":
        out_path = out_dir / f"op_diagram_split_{data_tag}{oil_tag}{color_tag}{limits_tag}_{stamp}.{args.out_format}"
        plot_split(df, out_path, color_by=color_col, point_size=args.point_size,
                   continuous_cbar=continuous_cbar, cmap=args.cmap,
                   cbar_label=cbar_label,
                   show_limits=args.show_limits, refrigerant=args.refrigerant,
                   xlim=args.xlim, ylim=args.ylim)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
