# scripts/batch_run_plots.py
#
# Batch-runs parity_plot.py and error_analysis_curves.py for all
# validation_detail_*.csv files produced by batch_run_validations.py.
#
# Input structure (produced by batch_run_validations.py):
#   results/validation/<model>/detail/validation_detail_params_<p>_val_<v>_<model>_<mode>_<stamp>.csv
#
# Output structure:
#   results/plots/
#   ├── parity/
#   │   └── <model>/
#   │       └── params_<p>_val_<v>_<mode>/
#   │           ├── by_superheat/
#   │           │   ├── parity_m_dot_<stamp>.png
#   │           │   ├── parity_P_el_<stamp>.png
#   │           │   └── parity_T_dis_<stamp>.png
#   │           └── by_pressure_ratio/
#   │               └── ...
#   └── error_analysis/
#       └── <model>/
#           └── params_<p>_val_<v>_<mode>/
#               ├── vs_T_evap/
#               │   ├── err_m_dot_vs_T_evap_<stamp>.png
#               │   ├── err_P_el_vs_T_evap_<stamp>.png
#               │   └── err_T_dis_vs_T_evap_<stamp>.png
#               ├── vs_T_cond/
#               └── vs_pressure_ratio/
#
# Activate REFPROP first:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Full batch run (default: 2 parity colors + 3 error-analysis x-axes):
#   python scripts/batch_run_plots.py --force
#
#   # Only one model, 4 parallel workers:
#   python scripts/batch_run_plots.py --filter_model modified --n_workers 4
#
#   # Skip error analysis, only parity plots:
#   python scripts/batch_run_plots.py --skip_error_analysis --filter_model original
#
#   # Dry-run:
#   python scripts/batch_run_plots.py --dry_run

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================
VALIDATION_BASE = Path("results/validation")
OUTPUT_BASE = Path("results/plots")

PARITY_SCRIPT = Path("scripts/plotting_scripts/parity_plot.py")
ERROR_ANALYSIS_SCRIPT = Path("scripts/plotting_scripts/error_analysis_curves.py")

# Default plot options
PARITY_COLOR_BY = ["superheat", "pressure_ratio"]
ERROR_X_AXES = ["T_evap", "T_cond", "pressure_ratio"]
ERROR_COLOR_BY = "superheat"

MODELS = ["original", "modified", "oil_path"]

# Filename pattern to parse: validation_detail_params_<p>_val_<v>_<model>_<mode>_<stamp>.csv
DETAIL_PATTERN = re.compile(
    r"^validation_detail_"
    r"params_(?P<params_oil>[^_]+)_"
    r"val_(?P<val_oil>[^_]+)_"
    r"(?P<model>original|modified|oil_path|oilpath)_"
    r"(?P<mode>train_only|validation_only|all)_"
    r"(?P<stamp>\d{4}-\d{2}-\d{2}_\d{6})\.csv$"
)


# =========================================================
# Helpers
# =========================================================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _norm_model(s: str) -> str:
    s = s.strip().lower()
    return "oil_path" if s == "oilpath" else s


def parse_detail_filename(path: Path):
    """Extract metadata from validation_detail_*.csv filename."""
    m = DETAIL_PATTERN.match(path.name)
    if not m:
        return None
    return {
        "params_oil": m.group("params_oil"),
        "val_oil": m.group("val_oil"),
        "model": _norm_model(m.group("model")),
        "mode": m.group("mode"),
        "stamp": m.group("stamp"),
        "detail_csv": path,
    }


def discover_detail_csvs(base: Path) -> list[dict]:
    """Find all validation_detail_*.csv files under results/validation/<model>/detail/."""
    found = []
    if not base.exists():
        return found

    for model in MODELS:
        detail_dir = base / model / "detail"
        if not detail_dir.exists():
            continue
        for csv_path in sorted(detail_dir.glob("validation_detail_*.csv")):
            meta = parse_detail_filename(csv_path)
            if meta is None:
                print(f"  [WARN] Could not parse filename, skipping: {csv_path.name}")
                continue
            # Cross-check model in filename vs. folder
            if meta["model"] != model:
                print(f"  [WARN] Model mismatch (folder={model}, filename={meta['model']}): "
                      f"{csv_path.name}")
            found.append(meta)
    return found


def config_folder(meta: dict) -> str:
    """Return folder name for one validation config."""
    return f"params_{meta['params_oil']}_val_{meta['val_oil']}_{meta['mode']}"


def parity_output_exists(out_dir: Path) -> bool:
    """Check if parity plots already exist in out_dir."""
    if not out_dir.exists():
        return False
    # Expect at least one parity_*.png file
    return any(out_dir.glob("parity_*.png")) or any(out_dir.glob("parity_*.svg"))


def error_output_exists(out_dir: Path) -> bool:
    """Check if error analysis plots already exist in out_dir."""
    if not out_dir.exists():
        return False
    return any(out_dir.glob("err_*.png")) or any(out_dir.glob("err_*.svg"))


# =========================================================
# Build jobs
# =========================================================
def build_parity_jobs(metas: list[dict], args) -> list[dict]:
    """Build one job per (csv, color_by) combination."""
    jobs = []
    for meta in metas:
        cfg_folder = config_folder(meta)
        for color in PARITY_COLOR_BY:
            out_dir = (OUTPUT_BASE / "parity" / meta["model"] / cfg_folder
                       / f"by_{color}")
            jobs.append({
                "type": "parity",
                "meta": meta,
                "color_by": color,
                "out_dir": out_dir,
            })
    return jobs


def build_error_jobs(metas: list[dict], args) -> list[dict]:
    """Build one job per (csv, x_axis) combination."""
    jobs = []
    for meta in metas:
        cfg_folder = config_folder(meta)
        for x_axis in ERROR_X_AXES:
            out_dir = (OUTPUT_BASE / "error_analysis" / meta["model"] / cfg_folder
                       / f"vs_{x_axis}")
            jobs.append({
                "type": "error_analysis",
                "meta": meta,
                "x_axis": x_axis,
                "out_dir": out_dir,
            })
    return jobs


# =========================================================
# Build commands
# =========================================================
def build_parity_command(job: dict, args) -> list[str]:
    meta = job["meta"]
    cmd = [
        sys.executable,
        str(PARITY_SCRIPT),
        "--pred_csv", str(meta["detail_csv"]),
        "--out_dir", str(job["out_dir"]),
        "--color_by", job["color_by"],
        "--out_format", args.out_format,
    ]
    return cmd


def build_error_command(job: dict, args) -> list[str]:
    meta = job["meta"]
    x_axis = job["x_axis"]
    cmd = [
        sys.executable,
        str(ERROR_ANALYSIS_SCRIPT),
        "--pred_csv", str(meta["detail_csv"]),
        "--out_dir", str(job["out_dir"]),
        "--x_m_dot", x_axis,
        "--x_P_el", x_axis,
        "--x_T_dis", x_axis,
        "--color_by", ERROR_COLOR_BY,
        "--out_format", args.out_format,
    ]
    return cmd


# =========================================================
# Run
# =========================================================
def run_single_job(job: dict, args, log_lines: list) -> dict:
    meta = job["meta"]
    job_type = job["type"]

    if job_type == "parity":
        tag = f"parity | {meta['model']} | params={meta['params_oil']} | val={meta['val_oil']} | mode={meta['mode']} | color={job['color_by']}"
        cmd = build_parity_command(job, args)
        exists_check = parity_output_exists
    else:
        tag = f"error | {meta['model']} | params={meta['params_oil']} | val={meta['val_oil']} | mode={meta['mode']} | x={job['x_axis']}"
        cmd = build_error_command(job, args)
        exists_check = error_output_exists

    # Skip-if-exists
    if not args.force and exists_check(job["out_dir"]):
        msg = f"  [SKIP] {tag}  (already exists)"
        print(msg)
        log_lines.append(msg)
        return {"job": job, "status": "skipped", "duration_s": 0.0}

    if args.dry_run:
        msg = f"  [DRY-RUN] {tag}"
        print(msg)
        log_lines.append(msg)
        log_lines.append(f"          {' '.join(cmd)}")
        return {"job": job, "status": "dry_run", "duration_s": 0.0}

    # Ensure output dir exists
    job["out_dir"].mkdir(parents=True, exist_ok=True)

    # Execute
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=args.timeout_s,
        )
        duration = time.time() - start

        if result.returncode != 0:
            msg = f"  [FAIL] {tag}  (returncode={result.returncode}, {duration:.1f}s)"
            print(msg)
            log_lines.append(msg)
            log_lines.append(f"    STDERR: {result.stderr.strip()[:500]}")
            return {"job": job, "status": "failed",
                    "error": result.stderr.strip()[:500],
                    "duration_s": duration}

        msg = f"  [OK]   {tag}  ({duration:.1f}s)"
        print(msg)
        log_lines.append(msg)
        return {"job": job, "status": "ok", "duration_s": duration}

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        msg = f"  [TIMEOUT] {tag}  (>{args.timeout_s}s)"
        print(msg)
        log_lines.append(msg)
        return {"job": job, "status": "timeout", "duration_s": duration}

    except Exception as e:
        duration = time.time() - start
        msg = f"  [ERROR] {tag}  ({type(e).__name__}: {e})"
        print(msg)
        log_lines.append(msg)
        return {"job": job, "status": "error", "error": str(e),
                "duration_s": duration}


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Batch-run parity and error-analysis plots for all validation configs."
    )
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")
    ap.add_argument("--force", action="store_true",
                    help="Re-plot even if output files already exist")
    ap.add_argument("--n_workers", type=int, default=1,
                    help="Number of parallel plot processes (default 1)")
    ap.add_argument("--timeout_s", type=int, default=600,
                    help="Per-job timeout in seconds (default 600 = 10 min)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands but don't execute")

    # Skip flags
    ap.add_argument("--skip_parity", action="store_true",
                    help="Skip parity_plot.py runs")
    ap.add_argument("--skip_error_analysis", action="store_true",
                    help="Skip error_analysis_curves.py runs")
    ap.add_argument("--skip_mode_all", action="store_true", default=True,
                    help="Skip CSVs with selection_mode='all' (default: True — plot only train_only+validation_only)")
    ap.add_argument("--include_mode_all", action="store_true",
                    help="Override: also plot CSVs with selection_mode='all'")

    # Filters
    ap.add_argument("--filter_model", default=None,
                    choices=["original", "modified", "oil_path"])
    ap.add_argument("--filter_params_oil", default=None)
    ap.add_argument("--filter_val_oil", default=None)
    ap.add_argument("--filter_mode", default=None,
                    choices=["train_only", "validation_only", "all"])

    args = ap.parse_args()

    # Sanity checks
    if not PARITY_SCRIPT.exists():
        raise FileNotFoundError(f"Parity script not found: {PARITY_SCRIPT}")
    if not ERROR_ANALYSIS_SCRIPT.exists():
        raise FileNotFoundError(f"Error analysis script not found: {ERROR_ANALYSIS_SCRIPT}")
    if not VALIDATION_BASE.exists():
        raise FileNotFoundError(f"Validation base not found: {VALIDATION_BASE}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Discover CSVs
    print("Discovering validation_detail CSVs ...")
    all_metas = discover_detail_csvs(VALIDATION_BASE)
    print(f"  Found {len(all_metas)} CSV(s)")

    # Filter: default is to skip selection_mode=all unless --include_mode_all
    include_all = args.include_mode_all or (not args.skip_mode_all)
    if args.include_mode_all:
        include_all = True
    else:
        # Skip 'all' mode by default
        include_all = False

    if not include_all:
        before = len(all_metas)
        all_metas = [m for m in all_metas if m["mode"] != "all"]
        n_skipped = before - len(all_metas)
        if n_skipped > 0:
            print(f"  Skipping {n_skipped} CSV(s) with selection_mode='all' "
                  "(use --include_mode_all to include them)")

    # Apply other filters
    if args.filter_model:
        all_metas = [m for m in all_metas if m["model"] == args.filter_model]
    if args.filter_params_oil:
        all_metas = [m for m in all_metas if m["params_oil"].lower() == args.filter_params_oil.lower()]
    if args.filter_val_oil:
        all_metas = [m for m in all_metas if m["val_oil"].lower() == args.filter_val_oil.lower()]
    if args.filter_mode:
        all_metas = [m for m in all_metas if m["mode"] == args.filter_mode]

    if not all_metas:
        print("No CSVs match filters. Nothing to do.")
        return

    # Build jobs
    jobs = []
    if not args.skip_parity:
        jobs.extend(build_parity_jobs(all_metas, args))
    if not args.skip_error_analysis:
        jobs.extend(build_error_jobs(all_metas, args))

    if not jobs:
        print("No jobs to run (both plot types skipped).")
        return

    # Sort: by model, then config, then type
    type_order = {"parity": 0, "error_analysis": 1}
    jobs.sort(key=lambda j: (
        j["meta"]["model"],
        j["meta"]["params_oil"],
        j["meta"]["val_oil"],
        j["meta"]["mode"],
        type_order.get(j["type"], 99),
    ))

    n_jobs = len(jobs)
    n_parity = sum(1 for j in jobs if j["type"] == "parity")
    n_error = sum(1 for j in jobs if j["type"] == "error_analysis")

    print("=" * 70)
    print(f"Batch plot run: {n_jobs} jobs ({n_parity} parity + {n_error} error analysis)")
    print("=" * 70)
    print(f"  CSVs selected:      {len(all_metas)}")
    print(f"  Parity colorings:   {PARITY_COLOR_BY}")
    print(f"  Error x-axes:       {ERROR_X_AXES}  (color_by={ERROR_COLOR_BY})")
    print(f"  Output base:        {OUTPUT_BASE}")
    print(f"  Workers:            {args.n_workers}")
    print(f"  Skip-if-exists:     {not args.force}")
    print(f"  Dry run:            {args.dry_run}")
    if args.filter_model or args.filter_params_oil or args.filter_val_oil or args.filter_mode:
        print(f"  Filters: model={args.filter_model}, params={args.filter_params_oil},"
              f" val={args.filter_val_oil}, mode={args.filter_mode}")
    print("=" * 70)

    # Run
    log_lines = []
    log_lines.append(f"Batch plot run started: {_ts()}")
    log_lines.append(f"Total jobs: {n_jobs} ({n_parity} parity + {n_error} error analysis)")
    log_lines.append("=" * 70)

    t_start = time.time()
    results = []

    if args.n_workers <= 1:
        for i, job in enumerate(jobs, 1):
            print(f"\n[{i}/{n_jobs}]")
            log_lines.append(f"\n[{i}/{n_jobs}]")
            res = run_single_job(job, args, log_lines)
            results.append(res)
    else:
        print(f"\nRunning {n_jobs} jobs with {args.n_workers} parallel workers ...\n")
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            future_to_job = {
                executor.submit(run_single_job, job, args, log_lines): (i, job)
                for i, job in enumerate(jobs, 1)
            }
            n_done = 0
            for future in as_completed(future_to_job):
                i, job = future_to_job[future]
                n_done += 1
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"  [ERROR] Job {i} crashed: {e}")
                    log_lines.append(f"  [ERROR] Job {i} crashed: {e}")
                    results.append({"job": job, "status": "crashed", "error": str(e)})
                print(f"  Progress: {n_done}/{n_jobs} done")

    duration_total = time.time() - t_start

    # Summary
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_fail = sum(1 for r in results if r["status"] in ("failed", "error", "timeout", "crashed"))
    n_dry = sum(1 for r in results if r["status"] == "dry_run")

    print()
    print("=" * 70)
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
    log_lines.append("=" * 70)
    log_lines.append(f"Finished: {_ts()}")

    # Write log
    log_path = OUTPUT_BASE / f"batch_plot_log_{_ts()}.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nLog saved: {log_path}")

    if n_fail > 0:
        print(f"\n[WARN] {n_fail} job(s) failed. See log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
