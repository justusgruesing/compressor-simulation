# scripts/batch_run_thesis_plots.py
#
# Generates all plots needed for the thesis:
#   Phase A: Parity plots (train_vs_validation + oil coloring)
#   Phase B: Error analysis curves (oil, superheat, T_evap, T_dis_vs_P_el)
#   Phase C: Cross-validation heatmaps (per-model + model comparison)
#
# Prerequisites:
#   - All 54 standard validations from batch_run_validations.py
#   - Additionally: 3 validations with selection_mode=all
#     (params=all, val=all, for each model)
#     Run: python scripts/batch_run_validations.py --include_all_mode
#          --filter_params_oil all --filter_val_oil all
#
# Activate REFPROP first:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Full run:
#   python scripts/batch_run_thesis_plots.py --filter_model original
#
#   # Only parity plots:
#   python scripts/batch_run_thesis_plots.py --phase A
#
#   # Only for one model:
#   python scripts/batch_run_thesis_plots.py --filter_model modified
#
#   # Dry run:
#   python scripts/batch_run_thesis_plots.py --dry_run

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================
VALIDATION_BASE = Path("results/validation")
OUTPUT_BASE = Path("results/thesis_plots")

PARITY_SCRIPT = Path("scripts/plotting_scripts/parity_plot.py")
ERROR_SCRIPT = Path("scripts/plotting_scripts/error_analysis_curves.py")
HEATMAP_SCRIPT = Path("scripts/plotting_scripts/cross_validation_heatmap.py")

MODELS = ["original", "modified", "oil_path"]
METRICS = ["mae", "rmse", "mae_combined", "rmse_combined"]
SELECTION_MODES = ["train_only", "validation_only"]

# Filename pattern for validation_detail CSVs
DETAIL_PATTERN = re.compile(
    r"^validation_detail_"
    r"params_(?P<params_oil>[^_]+)_"
    r"val_(?P<val_oil>[^_]+)_"
    r"(?P<model>original|modified|oil_path)_"
    r"(?P<mode>train_only|validation_only|all)_"
    r"(?P<stamp>\d{4}-\d{2}-\d{2}_\d{6})\.csv$"
)


# =========================================================
# Helpers
# =========================================================
def _ts():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def find_detail_csv(model, params_oil, val_oil, mode):
    """
    Find the validation_detail CSV matching the given criteria.
    Returns Path or None.
    """
    detail_dir = VALIDATION_BASE / model / "detail"
    if not detail_dir.exists():
        return None

    pattern = f"validation_detail_params_{params_oil}_val_{val_oil}_{model}_{mode}_*.csv"
    matches = sorted(detail_dir.glob(pattern))
    if matches:
        return matches[-1]  # latest
    return None


def run_job(cmd, tag, args, log_lines):
    """Run a single subprocess job. Returns result dict."""
    if args.dry_run:
        msg = f"  [DRY-RUN] {tag}"
        print(msg)
        log_lines.append(msg)
        log_lines.append(f"          {' '.join(str(c) for c in cmd)}")
        return {"status": "dry_run", "duration_s": 0.0}

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    start = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=os.getcwd(),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=args.timeout_s, env=env,
        )
        duration = time.time() - start

        if result.returncode != 0:
            msg = f"  [FAIL] {tag}  (returncode={result.returncode}, {duration:.1f}s)"
            print(msg)
            log_lines.append(msg)
            log_lines.append(f"    STDERR: {result.stderr.strip()[:500]}")
            return {"status": "failed", "duration_s": duration}

        msg = f"  [OK]   {tag}  ({duration:.1f}s)"
        print(msg)
        log_lines.append(msg)
        return {"status": "ok", "duration_s": duration}

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        msg = f"  [TIMEOUT] {tag}  (>{args.timeout_s}s)"
        print(msg)
        log_lines.append(msg)
        return {"status": "timeout", "duration_s": duration}

    except Exception as e:
        duration = time.time() - start
        msg = f"  [ERROR] {tag}  ({type(e).__name__}: {e})"
        print(msg)
        log_lines.append(msg)
        return {"status": "error", "duration_s": duration}


def has_output(out_dir, prefix=""):
    """Check if directory has any PNG/SVG output."""
    if not out_dir.exists():
        return False
    return (any(out_dir.glob(f"{prefix}*.png")) or
            any(out_dir.glob(f"{prefix}*.svg")))


# =========================================================
# Phase A: Parity Plots
# =========================================================
def build_parity_jobs(models, args):
    """
    A1: train_vs_validation coloring (params=all, val=all, mode=all)
    A2: oil coloring (params=lpg68/lpg100, val=all, mode=validation_only)
    """
    jobs = []

    for model in models:
        # --- A1: Train vs Validation ---
        csv_path = find_detail_csv(model, "all", "all", "all")
        if csv_path is None:
            print(f"  [WARN] Missing CSV for parity A1: {model} params=all val=all mode=all")
            print(f"         Run: python scripts/batch_run_validations.py "
                  f"--include_all_mode --filter_model {model} --filter_params_oil all --filter_val_oil all")
        else:
            out_dir = OUTPUT_BASE / "parity" / "train_vs_validation" / model
            tag = f"parity A1 | {model} | train_vs_validation"
            cmd = [
                sys.executable, str(PARITY_SCRIPT),
                "--pred_csv", str(csv_path),
                "--out_dir", str(out_dir),
                "--color_by", "train_validation",
                "--out_format", args.out_format,
            ]
            jobs.append({"cmd": cmd, "tag": tag, "out_dir": out_dir, "prefix": "parity"})

        # --- A2: Oil coloring ---
        for params_oil in ["lpg68", "lpg100"]:
            csv_path = find_detail_csv(model, params_oil, "all", "validation_only")
            if csv_path is None:
                print(f"  [WARN] Missing CSV for parity A2: {model} params={params_oil} val=all mode=validation_only")
                continue

            out_dir = OUTPUT_BASE / "parity" / "by_oil" / model / f"params_{params_oil}"
            tag = f"parity A2 | {model} | params={params_oil} | by_oil"
            cmd = [
                sys.executable, str(PARITY_SCRIPT),
                "--pred_csv", str(csv_path),
                "--out_dir", str(out_dir),
                "--color_by", "oil",
                "--out_format", args.out_format,
            ]
            jobs.append({"cmd": cmd, "tag": tag, "out_dir": out_dir, "prefix": "parity"})

    return jobs


# =========================================================
# Phase B: Error Analysis Curves
# =========================================================
def build_error_analysis_jobs(models, args):
    """
    All use params=all, val=all, mode=validation_only.
    B1: color_by oil
    B2: color_by superheat, default x-axes
    B3: color_by T_evap, all x-axes = T_cond
    B4: color_by superheat, x_T_dis = P_el
    """
    jobs = []

    for model in models:
        csv_path = find_detail_csv(model, "all", "all", "validation_only")
        if csv_path is None:
            print(f"  [WARN] Missing CSV for error analysis: {model} params=all val=all mode=validation_only")
            continue

        base_args = [
            sys.executable, str(ERROR_SCRIPT),
            "--pred_csv", str(csv_path),
            "--out_format", args.out_format,
        ]

        # B1: Oil coloring
        out_dir = OUTPUT_BASE / "error_analysis" / "by_oil" / model
        tag = f"error B1 | {model} | by_oil"
        cmd = base_args + [
            "--out_dir", str(out_dir),
            "--color_by", "oil",
            "--x_m_dot", "T_evap",
            "--x_P_el", "pressure_ratio",
            "--x_T_dis", "pressure_ratio",
        ]
        jobs.append({"cmd": cmd, "tag": tag, "out_dir": out_dir, "prefix": "err"})

        # B2: Superheat coloring, default axes
        out_dir = OUTPUT_BASE / "error_analysis" / "by_superheat" / model
        tag = f"error B2 | {model} | by_superheat"
        cmd = base_args + [
            "--out_dir", str(out_dir),
            "--color_by", "superheat",
            "--x_m_dot", "T_evap",
            "--x_P_el", "pressure_ratio",
            "--x_T_dis", "pressure_ratio",
        ]
        jobs.append({"cmd": cmd, "tag": tag, "out_dir": out_dir, "prefix": "err"})

        # B3: T_evap on colorbar, T_cond on x-axis
        out_dir = OUTPUT_BASE / "error_analysis" / "by_T_evap_vs_T_cond" / model
        tag = f"error B3 | {model} | T_evap vs T_cond"
        cmd = base_args + [
            "--out_dir", str(out_dir),
            "--color_by", "T_evap",
            "--x_m_dot", "T_cond",
            "--x_P_el", "T_cond",
            "--x_T_dis", "T_cond",
        ]
        jobs.append({"cmd": cmd, "tag": tag, "out_dir": out_dir, "prefix": "err"})

        # B4: T_dis vs P_el
        out_dir = OUTPUT_BASE / "error_analysis" / "T_dis_vs_P_el" / model
        tag = f"error B4 | {model} | T_dis vs P_el"
        cmd = base_args + [
            "--out_dir", str(out_dir),
            "--color_by", "superheat",
            "--x_m_dot", "T_evap",
            "--x_P_el", "pressure_ratio",
            "--x_T_dis", "P_el",
        ]
        jobs.append({"cmd": cmd, "tag": tag, "out_dir": out_dir, "prefix": "err"})

    return jobs


# =========================================================
# Phase C: Cross-Validation Heatmaps
# =========================================================
def build_heatmap_jobs(models, args):
    """
    C1: Per-model heatmaps (3 models × 4 metrics × 2 modes = 24)
    C2: Model comparison (1 × 4 metrics × 2 modes = 8)
    """
    jobs = []

    # C1: Per-model cross-validation
    for model in models:
        summary_dir = VALIDATION_BASE / model / "summary"
        if not summary_dir.exists():
            print(f"  [WARN] Summary dir not found: {summary_dir}")
            continue

        for sel_mode in SELECTION_MODES:
            for metric in METRICS:
                out_dir = OUTPUT_BASE / "cross_validation_heatmap" / "per_model" / model / sel_mode
                tag = f"heatmap C1 | {model} | {metric} | {sel_mode}"
                cmd = [
                    sys.executable, str(HEATMAP_SCRIPT),
                    "--mode", "cross_validation",
                    "--summary_dir", str(summary_dir),
                    "--metric", metric,
                    "--selection_mode", sel_mode,
                    "--out_dir", str(out_dir),
                    "--out_format", args.out_format,
                ]
                jobs.append({"cmd": cmd, "tag": tag, "out_dir": out_dir,
                             "prefix": f"cross_validation_{metric}"})

    # C2: Model comparison
    for sel_mode in SELECTION_MODES:
        for metric in METRICS:
            out_dir = OUTPUT_BASE / "cross_validation_heatmap" / "model_comparison" / sel_mode
            tag = f"heatmap C2 | model_comparison | {metric} | {sel_mode}"
            cmd = [
                sys.executable, str(HEATMAP_SCRIPT),
                "--mode", "model_comparison",
                "--base_dir", str(VALIDATION_BASE),
                "--params_oil", "all",
                "--metric", metric,
                "--selection_mode", sel_mode,
                "--out_dir", str(out_dir),
                "--out_format", args.out_format,
            ]
            jobs.append({"cmd": cmd, "tag": tag, "out_dir": out_dir,
                         "prefix": f"model_comparison_{metric}"})

    return jobs


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Generate all thesis plots: parity, error analysis, cross-validation heatmaps."
    )
    ap.add_argument("--phase", choices=["A", "B", "C", "all"], default="all",
                    help="Which phase: A=parity, B=error_analysis, C=heatmap (default: all)")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")
    ap.add_argument("--force", action="store_true",
                    help="Re-plot even if outputs already exist")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--timeout_s", type=int, default=600)
    ap.add_argument("--filter_model", default=None,
                    choices=["original", "modified", "oil_path"])

    args = ap.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Determine models to process
    models = [args.filter_model] if args.filter_model else MODELS

    # Sanity checks
    if not VALIDATION_BASE.exists():
        raise FileNotFoundError(f"Validation base not found: {VALIDATION_BASE}")
    for script in [PARITY_SCRIPT, ERROR_SCRIPT, HEATMAP_SCRIPT]:
        if not script.exists():
            print(f"  [WARN] Script not found: {script}")

    # Build all jobs
    all_jobs = []

    if args.phase in ("A", "all"):
        print("\n  Building Phase A jobs (parity plots) ...")
        all_jobs.extend(build_parity_jobs(models, args))

    if args.phase in ("B", "all"):
        print("\n  Building Phase B jobs (error analysis) ...")
        all_jobs.extend(build_error_analysis_jobs(models, args))

    if args.phase in ("C", "all"):
        print("\n  Building Phase C jobs (cross-validation heatmaps) ...")
        all_jobs.extend(build_heatmap_jobs(models, args))

    n_jobs = len(all_jobs)
    if n_jobs == 0:
        print("\nNo jobs to run.")
        return

    # Count by phase
    n_a = sum(1 for j in all_jobs if "parity" in j["tag"])
    n_b = sum(1 for j in all_jobs if "error" in j["tag"])
    n_c = sum(1 for j in all_jobs if "heatmap" in j["tag"])

    print("\n" + "=" * 70)
    print(f"Thesis plot batch: {n_jobs} jobs")
    print("=" * 70)
    print(f"  Phase A (parity):          {n_a}")
    print(f"  Phase B (error analysis):  {n_b}")
    print(f"  Phase C (heatmaps):        {n_c}")
    print(f"  Models:                    {models}")
    print(f"  Skip-if-exists:            {not args.force}")
    print(f"  Dry run:                   {args.dry_run}")
    print("=" * 70)

    # Run
    log_lines = [f"Thesis plot run started: {_ts()}", f"Total jobs: {n_jobs}", "=" * 70]
    t_start = time.time()
    results = []

    for i, job in enumerate(all_jobs, 1):
        print(f"\n[{i}/{n_jobs}]")
        log_lines.append(f"\n[{i}/{n_jobs}]")

        # Skip-if-exists
        if not args.force and has_output(job["out_dir"], job.get("prefix", "")):
            msg = f"  [SKIP] {job['tag']}  (already exists)"
            print(msg)
            log_lines.append(msg)
            results.append({"status": "skipped", "duration_s": 0.0})
            continue

        # Ensure output dir
        job["out_dir"].mkdir(parents=True, exist_ok=True)

        res = run_job(job["cmd"], job["tag"], args, log_lines)
        results.append(res)

    duration_total = time.time() - t_start

    # Summary
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_fail = sum(1 for r in results if r["status"] in ("failed", "error", "timeout"))
    n_dry = sum(1 for r in results if r["status"] == "dry_run")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total:    {n_jobs}")
    print(f"  OK:       {n_ok}")
    print(f"  Skipped:  {n_skip}")
    print(f"  Failed:   {n_fail}")
    if args.dry_run:
        print(f"  Dry-run:  {n_dry}")
    print(f"  Duration: {duration_total:.1f}s ({duration_total/60:.1f} min)")
    print("=" * 70)

    log_lines.append("")
    log_lines.append("=" * 70)
    log_lines.append(f"Total: {n_jobs} | OK: {n_ok} | Skipped: {n_skip} | Failed: {n_fail}")
    log_lines.append(f"Duration: {duration_total:.1f}s ({duration_total/60:.1f} min)")
    log_lines.append(f"Finished: {_ts()}")

    # Write log
    log_path = OUTPUT_BASE / f"thesis_plot_log_{_ts()}.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nLog saved: {log_path}")

    if n_fail > 0:
        print(f"\n[WARN] {n_fail} job(s) failed. See log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
