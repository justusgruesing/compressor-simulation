# scripts/batch_run_analysis_plots.py
#
# Batch-runs the four physics/performance plot scripts across all fitted params:
#   Phase 3a: irreversibility_curves + loss_curves  (fastest)
#   Phase 3b: efficiency_curves                     (medium)
#   Phase 3c: performance_map                       (slowest, 40x40 grid)
#
# Input: fitted parameter CSVs for 3 models x 3 oils (paths in PARAMS below).
#
# Output structure:
#   results/plots/
#   ├── irreversibility_curves/<model>/vary_<dim>_LPG68_vs_LPG100/irreversibility_*.png
#   ├── loss_curves/<model>/vary_<dim>_params_<oil>/loss_*.png
#   ├── efficiency_curves/<model>/vary_<dim>_<oil1>_vs_<oil2>/efficiency_*.png
#   └── performance_map/<model>/<model>_<params_oil>_N<rpm>/perfmap_*.png
#
# All curve plots produce a 3-subplot row by passing multiple fixed values
# (e.g. --T_evap 0 10 20) to the underlying scripts.
#
# Activate REFPROP first:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Full run, sequential:
#   python scripts/batch_run_analysis_plots.py
#
#   # Only phase 3a (fast check):
#   python scripts/batch_run_analysis_plots.py --phase 3a
#
#   # Phase A:
#   python scripts/batch_run_analysis_plots.py --phase 3a
#
#   # Phase B:
#   python scripts/batch_run_analysis_plots.py --phase 3b
#
#   # Phase C:
#   python scripts/batch_run_analysis_plots.py --phase 3c --n_workers 4
#
#   # Parallelized:
#   python scripts/batch_run_analysis_plots.py --n_workers 4
#
#   # Phase 3c only, overnight run:
#   python scripts/batch_run_analysis_plots.py --phase 3c --n_workers 4
#
#   # Dry-run:
#   python scripts/batch_run_analysis_plots.py --dry_run

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


# =========================================================
# CONFIG: paths to fitted params CSVs (EDIT IF FILES MOVE)
# =========================================================
PARAMS = {
    "original": {
        "LPG68":  Path("results/final_results/Molinaroli_LPG68/Fitting/fitted_params_lpg68_original_ga_2026-03-08_101308.csv"),
        "LPG100": Path("results/final_results/Molinaroli_LPG100/Fitting/fitted_params_lpg100_original_ga_2026-03-19_210256.csv"),
        "all":    Path("results/final_results/Molinaroli_All/Fitting/fitted_params_all_original_ga_2026-03-21_192615.csv"),
    },
    "modified": {
        "LPG68":  Path("results/final_results/Modified_LPG68/Fitting/fitted_params_lpg68_modified_ga_2026-03-22_185546.csv"),
        "LPG100": Path("results/final_results/Modified_LPG100/Fitting/fitted_params_lpg100_modified_ga_2026-03-28_092941.csv"),
        "all":    Path("results/final_results/Modified_All/Fitting/fitted_params_all_modified_ga_2026-03-26_110247.csv"),
    },
    "oil_path": {
        "LPG68":  Path("results/final_results/Oil_Path_LPG68/Fitting/fitted_params_lpg68_oil_path_ga_2026-04-06_042321.csv"),
        "LPG100": Path("results/final_results/Oil_Path_LPG100/Fitting/fitted_params_lpg100_oil_path_ga_2026-04-06_042321.csv"),
        "all":    Path("results/final_results/Oil_Path_All/Fitting/fitted_params_all_oil_path_ga_2026-04-06_042321.csv"),
    },
}

MODELS = ["original", "modified", "oil_path"]
SINGLE_OILS = ["LPG68", "LPG100", "all"]

# Output base
OUTPUT_BASE = Path("results/plots")

# Scripts
EFFICIENCY_SCRIPT = Path("scripts/plotting_scripts/efficiency_curves.py")
PERFORMANCE_MAP_SCRIPT = Path("scripts/plotting_scripts/performance_map.py")
LOSS_CURVES_SCRIPT = Path("scripts/plotting_scripts/loss_curves.py")
IRREVERSIBILITY_SCRIPT = Path("scripts/plotting_scripts/irreversibility_curves.py")


# =========================================================
# Plot configuration per phase
# =========================================================
# Phase 3a: irreversibility (3 jobs) + loss_curves (15 jobs) = 18 jobs
#   Both now use oil-pair input (LPG68 vs LPG100) similar to efficiency_curves.
IRREVERSIBILITY_VARY = ["T_evap"]
LOSS_CURVES_VARY = ["T_evap", "T_cond", "speed", "superheat", "pressure_ratio"]

# Phase 3b: efficiency curves (15 jobs)
#   Only LPG68 vs. LPG100 (oil comparison using each oil's own fit)
EFFICIENCY_PAIRS = [("LPG68", "LPG100")]
EFFICIENCY_VARY = ["T_evap", "T_cond", "speed", "superheat", "pressure_ratio"]

# Fixed values per --vary axis: produces a 3-subplot row in the output plots.
# Choose physically meaningful fixed parameters that are NOT the swept axis.
FIXED_VALUES = {
    "T_evap": {"T_cond": [30, 50, 65]},
    "T_cond": {"T_evap": [0, 10, 20]},
    "speed": {"T_evap": [0, 10, 20]},
    "superheat": {"T_evap": [0, 10, 20]},
    "pressure_ratio": {"T_evap": [0, 10, 20]},
}

# Phase 3c: performance maps (18 jobs)
PERFMAP_OILS = ["LPG68", "LPG100", "all"]
PERFMAP_RPMS = [3600, 4800]


# =========================================================
# Helpers
# =========================================================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def output_has_pngs(out_dir: Path, prefix: str) -> bool:
    """Check if out_dir already has output plots with the given prefix."""
    if not out_dir.exists():
        return False
    return any(out_dir.glob(f"{prefix}*.png")) or any(out_dir.glob(f"{prefix}*.svg"))


# =========================================================
# Job builders
# =========================================================
def build_irreversibility_jobs() -> list[dict]:
    jobs = []
    for model in MODELS:
        for (oil1, oil2) in EFFICIENCY_PAIRS:
            params_csv1 = PARAMS[model][oil1]
            params_csv2 = PARAMS[model][oil2]
            for vary in IRREVERSIBILITY_VARY:
                out_dir = (OUTPUT_BASE / "irreversibility_curves" / model
                           / f"vary_{vary}_{oil1}_vs_{oil2}")
                jobs.append({
                    "type": "irreversibility",
                    "phase": "3a",
                    "model": model,
                    "oil1": oil1,
                    "oil2": oil2,
                    "params_csv1": params_csv1,
                    "params_csv2": params_csv2,
                    "vary": vary,
                    "out_dir": out_dir,
                    "prefix": "irreversibility",
                    "tag": f"irrev | {model} | {oil1} vs {oil2} | vary={vary}",
                })
    return jobs


def build_loss_curves_jobs() -> list[dict]:
    jobs = []
    for model in MODELS:
        for params_oil in SINGLE_OILS:
            params_csv = PARAMS[model][params_oil]
            for vary in LOSS_CURVES_VARY:
                out_dir = OUTPUT_BASE / "loss_curves" / model / f"vary_{vary}_params_{params_oil}"
                jobs.append({
                    "type": "loss_curves",
                    "phase": "3a",
                    "model": model,
                    "params_oil": params_oil,
                    "params_csv": params_csv,
                    "vary": vary,
                    "out_dir": out_dir,
                    "prefix": "loss",
                    "tag": f"loss | {model} | params={params_oil} | vary={vary}",
                })
    return jobs


def build_efficiency_jobs() -> list[dict]:
    jobs = []
    for model in MODELS:
        for (oil1, oil2) in EFFICIENCY_PAIRS:
            params_csv1 = PARAMS[model][oil1]
            params_csv2 = PARAMS[model][oil2]
            for vary in EFFICIENCY_VARY:
                out_dir = (OUTPUT_BASE / "efficiency_curves" / model
                           / f"vary_{vary}_{oil1}_vs_{oil2}")
                jobs.append({
                    "type": "efficiency",
                    "phase": "3b",
                    "model": model,
                    "oil1": oil1,
                    "oil2": oil2,
                    "params_csv1": params_csv1,
                    "params_csv2": params_csv2,
                    "vary": vary,
                    "out_dir": out_dir,
                    "prefix": "efficiency",
                    "tag": f"eff | {model} | {oil1} vs {oil2} | vary={vary}",
                })
    return jobs


def build_performance_map_jobs() -> list[dict]:
    jobs = []
    for model in MODELS:
        for params_oil in PERFMAP_OILS:
            params_csv = PARAMS[model][params_oil]
            for rpm in PERFMAP_RPMS:
                out_dir = (OUTPUT_BASE / "performance_map" / model
                           / f"{model}_{params_oil}_N{rpm}")
                jobs.append({
                    "type": "performance_map",
                    "phase": "3c",
                    "model": model,
                    "params_oil": params_oil,
                    "params_csv": params_csv,
                    "rpm": rpm,
                    "out_dir": out_dir,
                    "prefix": "perfmap",
                    "tag": f"perfmap | {model} | params={params_oil} | N={rpm}",
                })
    return jobs


# =========================================================
# Command builders
# =========================================================
def build_irreversibility_command(job: dict, args) -> list[str]:
    cmd = [
        sys.executable,
        str(IRREVERSIBILITY_SCRIPT),
        "--params_csv_oil1", str(job["params_csv1"]),
        "--params_csv_oil2", str(job["params_csv2"]),
        "--oil1", job["oil1"],
        "--oil2", job["oil2"],
        "--vary", job["vary"],
        "--normalize_by_mflow",
        "--out_dir", str(job["out_dir"]),
        "--out_format", args.out_format,
    ]
    # Add fixed-value subplot row arguments based on vary axis
    cmd += _build_fixed_value_args(job["vary"])
    return cmd


def build_loss_curves_command(job: dict, args) -> list[str]:
    cmd = [
        sys.executable,
        str(LOSS_CURVES_SCRIPT),
        "--params_csv", str(job["params_csv"]),
        "--oil", job["params_oil"],
        "--vary", job["vary"],
        "--out_dir", str(job["out_dir"]),
        "--out_format", args.out_format,
    ]
    cmd += _build_fixed_value_args(job["vary"])
    return cmd


def build_efficiency_command(job: dict, args) -> list[str]:
    cmd = [
        sys.executable,
        str(EFFICIENCY_SCRIPT),
        "--params_csv_oil1", str(job["params_csv1"]),
        "--params_csv_oil2", str(job["params_csv2"]),
        "--oil1", job["oil1"],
        "--oil2", job["oil2"],
        "--vary", job["vary"],
        "--metric", "all",
        "--out_dir", str(job["out_dir"]),
        "--out_format", args.out_format,
    ]
    cmd += _build_fixed_value_args(job["vary"])
    return cmd


def _build_fixed_value_args(vary: str) -> list[str]:
    """Build CLI args for fixed values, producing a 3-subplot row."""
    args_out = []
    fixed = FIXED_VALUES.get(vary, {})
    for fixed_name, values in fixed.items():
        # Map our internal name to the CLI flag
        cli_flag = {
            "T_evap": "--T_evap",
            "T_cond": "--T_cond",
            "speed":  "--N_rpm",
            "superheat": "--SH_K",
        }.get(fixed_name, f"--{fixed_name}")
        args_out.append(cli_flag)
        for v in values:
            args_out.append(str(v))
    return args_out


def build_performance_map_command(job: dict, args) -> list[str]:
    cmd = [
        sys.executable,
        str(PERFORMANCE_MAP_SCRIPT),
        "--params_csv", str(job["params_csv"]),
        "--oil", job["params_oil"],
        "--metric", "all",
        "--N_rpm", str(job["rpm"]),
        "--out_dir", str(job["out_dir"]),
        "--out_format", args.out_format,
    ]
    if args.perfmap_n_grid is not None:
        cmd += ["--n_grid", str(args.perfmap_n_grid)]
    return cmd


def build_command(job: dict, args) -> list[str]:
    if job["type"] == "irreversibility":
        return build_irreversibility_command(job, args)
    if job["type"] == "loss_curves":
        return build_loss_curves_command(job, args)
    if job["type"] == "efficiency":
        return build_efficiency_command(job, args)
    if job["type"] == "performance_map":
        return build_performance_map_command(job, args)
    raise ValueError(f"Unknown job type: {job['type']}")


# =========================================================
# Run
# =========================================================
def run_single_job(job: dict, args, log_lines: list) -> dict:
    tag = job["tag"]

    # Skip-if-exists
    if not args.force and output_has_pngs(job["out_dir"], job["prefix"]):
        msg = f"  [SKIP] {tag}  (already exists)"
        print(msg)
        log_lines.append(msg)
        return {"job": job, "status": "skipped", "duration_s": 0.0}

    # Check params CSV exists
    if "params_csv" in job and not job["params_csv"].exists():
        msg = f"  [FAIL] {tag}  (params_csv not found: {job['params_csv']})"
        print(msg)
        log_lines.append(msg)
        return {"job": job, "status": "failed",
                "error": "params_csv not found", "duration_s": 0.0}
    for k in ("params_csv1", "params_csv2"):
        if k in job and not job[k].exists():
            msg = f"  [FAIL] {tag}  ({k} not found: {job[k]})"
            print(msg)
            log_lines.append(msg)
            return {"job": job, "status": "failed",
                    "error": f"{k} not found", "duration_s": 0.0}

    cmd = build_command(job, args)

    if args.dry_run:
        msg = f"  [DRY-RUN] {tag}"
        print(msg)
        log_lines.append(msg)
        log_lines.append(f"          {' '.join(cmd)}")
        return {"job": job, "status": "dry_run", "duration_s": 0.0}

    job["out_dir"].mkdir(parents=True, exist_ok=True)

    # Force UTF-8 encoding so that unicode characters in the subprocess output
    # (e.g. arrows in print statements) don't crash on cp1252 Windows consoles.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout_s,
            env=env,
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
        description="Batch-run efficiency, loss, irreversibility and performance_map plots."
    )

    ap.add_argument("--phase", choices=["3a", "3b", "3c", "all"], default="all",
                    help="Which phase to run: 3a=irrev+loss, 3b=efficiency, 3c=perfmap (default: all)")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")
    ap.add_argument("--force", action="store_true",
                    help="Re-plot even if outputs already exist")
    ap.add_argument("--n_workers", type=int, default=1,
                    help="Number of parallel plot processes (default 1)")
    ap.add_argument("--timeout_s", type=int, default=7200,
                    help="Per-job timeout in seconds (default 7200 = 2h, for perfmap)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands but don't execute")

    # Granular skip flags
    ap.add_argument("--skip_irreversibility", action="store_true")
    ap.add_argument("--skip_loss_curves", action="store_true")
    ap.add_argument("--skip_efficiency", action="store_true")
    ap.add_argument("--skip_performance_map", action="store_true")

    # Filters
    ap.add_argument("--filter_model", default=None,
                    choices=["original", "modified", "oil_path"])
    ap.add_argument("--filter_params_oil", default=None)

    # Per-script tuning
    ap.add_argument("--perfmap_n_grid", type=int, default=None,
                    help="Override grid resolution for performance_map (default: script default 40)")

    args = ap.parse_args()

    # Sanity checks
    for script in [EFFICIENCY_SCRIPT, PERFORMANCE_MAP_SCRIPT,
                   LOSS_CURVES_SCRIPT, IRREVERSIBILITY_SCRIPT]:
        if not script.exists():
            print(f"  [WARN] Script not found: {script}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Collect jobs per phase
    all_jobs = []

    if args.phase in ("3a", "all"):
        if not args.skip_irreversibility:
            all_jobs.extend(build_irreversibility_jobs())
        if not args.skip_loss_curves:
            all_jobs.extend(build_loss_curves_jobs())

    if args.phase in ("3b", "all"):
        if not args.skip_efficiency:
            all_jobs.extend(build_efficiency_jobs())

    if args.phase in ("3c", "all"):
        if not args.skip_performance_map:
            all_jobs.extend(build_performance_map_jobs())

    # Apply filters
    if args.filter_model:
        all_jobs = [j for j in all_jobs if j.get("model") == args.filter_model]
    if args.filter_params_oil:
        fval = args.filter_params_oil.lower()
        def matches_oil(j):
            # efficiency has oil1/oil2 instead of params_oil
            if "params_oil" in j:
                return str(j["params_oil"]).lower() == fval
            if "oil1" in j:
                return str(j["oil1"]).lower() == fval or str(j["oil2"]).lower() == fval
            return False
        all_jobs = [j for j in all_jobs if matches_oil(j)]

    # Sort by phase, type, model
    phase_order = {"3a": 0, "3b": 1, "3c": 2}
    type_order = {"irreversibility": 0, "loss_curves": 1,
                  "efficiency": 2, "performance_map": 3}
    all_jobs.sort(key=lambda j: (
        phase_order.get(j["phase"], 99),
        type_order.get(j["type"], 99),
        j.get("model", ""),
        j.get("params_oil", j.get("oil1", "")),
        j.get("vary", ""),
        j.get("rpm", 0),
    ))

    n_jobs = len(all_jobs)
    if n_jobs == 0:
        print("No jobs to run.")
        return

    # Counts per type
    n_by_type = {}
    for j in all_jobs:
        n_by_type[j["type"]] = n_by_type.get(j["type"], 0) + 1

    print("=" * 70)
    print(f"Batch analysis plot run: {n_jobs} jobs")
    print("=" * 70)
    for t, n in n_by_type.items():
        print(f"  {t:25s} {n} jobs")
    print(f"  Workers:           {args.n_workers}")
    print(f"  Skip-if-exists:    {not args.force}")
    print(f"  Dry run:           {args.dry_run}")
    if args.filter_model or args.filter_params_oil:
        print(f"  Filters: model={args.filter_model}, params_oil={args.filter_params_oil}")
    print("=" * 70)

    # Run
    log_lines = []
    log_lines.append(f"Batch analysis plot run started: {_ts()}")
    log_lines.append(f"Total jobs: {n_jobs}")
    for t, n in n_by_type.items():
        log_lines.append(f"  {t}: {n}")
    log_lines.append("=" * 70)

    t_start = time.time()
    results = []

    if args.n_workers <= 1:
        for i, job in enumerate(all_jobs, 1):
            print(f"\n[{i}/{n_jobs}]")
            log_lines.append(f"\n[{i}/{n_jobs}]")
            res = run_single_job(job, args, log_lines)
            results.append(res)
    else:
        print(f"\nRunning {n_jobs} jobs with {args.n_workers} parallel workers ...\n")
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            future_to_job = {
                executor.submit(run_single_job, job, args, log_lines): (i, job)
                for i, job in enumerate(all_jobs, 1)
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
    log_path = OUTPUT_BASE / f"batch_analysis_plot_log_{_ts()}.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nLog saved: {log_path}")

    if n_fail > 0:
        print(f"\n[WARN] {n_fail} job(s) failed. See log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
