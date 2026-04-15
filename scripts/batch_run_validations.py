# scripts/batch_run_validations.py
#
# Batch-runs all validation combinations for the three Molinaroli compressor
# models (original, modified, oil_path) against all three data subsets
# (LPG68, LPG100, all) and all selection modes (train_only, validation_only,
# optionally also "all").
#
# Output structure:
#   results/validation/
#   ├── original/
#   │   ├── summary/
#   │   │   ├── validation_summary_params_lpg68_val_lpg68_original_train_only_<stamp>.csv
#   │   │   └── ...
#   │   └── detail/
#   │       └── ...
#   ├── modified/
#   │   ├── summary/
#   │   └── detail/
#   └── oil_path/
#       ├── summary/
#       └── detail/
#
# Activate REFPROP first:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Examples:
#   # Default (train_only + validation_only, sequential):
#   python scripts/batch_run_validations.py
#
#   # Including 'all' selection mode and 4 parallel workers:
#   python scripts/batch_run_validations.py --include_all_mode --n_workers 4
#
#   # Force re-run even if outputs already exist:
#   python scripts/batch_run_validations.py --force --n_workers 4
#
#   # Dry-run: show what would be executed without running anything:
#   python scripts/batch_run_validations.py --dry_run

from __future__ import annotations

import argparse
import os
import shutil
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

# Validation oils to test against (data subsets)
VALIDATION_OILS = ["LPG68", "LPG100", "all"]

# Selection modes
DEFAULT_MODES = ["train_only", "validation_only"]
EXTRA_MODE = "all"

# Output base
OUTPUT_BASE = Path("results/validation")

# Validation script
VALIDATION_SCRIPT = Path("scripts/validation.py")

# Default input CSVs (op rows + split template)
DEFAULT_OP_ROWS_CSV = "results/split_template/operating_points_rows_2026-03-12_112331.csv"
DEFAULT_SPLIT_CSV = "results/split_template/operating_points_split_template_2026-03-12_112331.csv"


# =========================================================
# Helpers
# =========================================================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _norm_oil_for_filter(s: str) -> str:
    """Normalize for matching existing files (lowercase, no spaces)."""
    return str(s).strip().lower().replace(" ", "")


def output_already_exists(out_dir: Path, params_oil: str, val_oil: str,
                          model: str, mode: str) -> tuple[bool, list]:
    """
    Check if validation_summary AND validation_detail with the right pattern
    already exist (any timestamp).
    """
    p_tag = _norm_oil_for_filter(params_oil)
    v_tag = _norm_oil_for_filter(val_oil)

    summary_dir = out_dir / "summary"
    detail_dir = out_dir / "detail"

    summary_pattern = f"validation_summary_params_{p_tag}_val_{v_tag}_{model}_{mode}_*.csv"
    detail_pattern = f"validation_detail_params_{p_tag}_val_{v_tag}_{model}_{mode}_*.csv"

    summary_matches = list(summary_dir.glob(summary_pattern)) if summary_dir.exists() else []
    detail_matches = list(detail_dir.glob(detail_pattern)) if detail_dir.exists() else []

    exists = len(summary_matches) > 0 and len(detail_matches) > 0
    return exists, summary_matches + detail_matches


def build_jobs(args) -> list[dict]:
    """Build the list of validation jobs to run."""
    modes = list(DEFAULT_MODES)
    if args.include_all_mode:
        modes.append(EXTRA_MODE)

    jobs = []
    for model, params_dict in PARAMS.items():
        for params_oil, params_csv in params_dict.items():
            for val_oil in VALIDATION_OILS:
                for mode in modes:
                    jobs.append({
                        "model": model,
                        "params_oil": params_oil,
                        "val_oil": val_oil,
                        "mode": mode,
                        "params_csv": params_csv,
                    })
    return jobs


def build_command(job: dict, args, out_dir: Path) -> list[str]:
    """Construct the validation.py command line for a single job."""
    cmd = [
        sys.executable,
        str(VALIDATION_SCRIPT),
        "--op_rows_csv", str(args.op_rows_csv),
        "--split_csv", str(args.split_csv),
        "--params_csv", str(job["params_csv"]),
        "--model", job["model"],
        "--oil", job["val_oil"],
        "--selection_mode", job["mode"],
        "--out_dir", str(out_dir),
    ]
    return cmd


def organize_outputs(working_dir: Path, target_dir: Path,
                     params_oil: str, val_oil: str,
                     model: str, mode: str) -> tuple[int, int]:
    """
    Move freshly-created summary/detail CSVs from working_dir into the
    target_dir/{summary,detail}/ subfolders.

    Returns (n_summary_moved, n_detail_moved).
    """
    p_tag = _norm_oil_for_filter(params_oil)
    v_tag = _norm_oil_for_filter(val_oil)

    summary_pattern = f"validation_summary_params_{p_tag}_val_{v_tag}_{model}_{mode}_*.csv"
    detail_pattern = f"validation_detail_params_{p_tag}_val_{v_tag}_{model}_{mode}_*.csv"

    summary_dir = target_dir / "summary"
    detail_dir = target_dir / "detail"
    summary_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)

    n_summary = 0
    n_detail = 0
    for f in working_dir.glob(summary_pattern):
        dest = summary_dir / f.name
        shutil.move(str(f), str(dest))
        n_summary += 1
    for f in working_dir.glob(detail_pattern):
        dest = detail_dir / f.name
        shutil.move(str(f), str(dest))
        n_detail += 1

    return n_summary, n_detail


def run_single_job(job: dict, args, log_lines: list, lock=None) -> dict:
    """Run a single validation job and return result dict."""
    model = job["model"]
    params_oil = job["params_oil"]
    val_oil = job["val_oil"]
    mode = job["mode"]

    target_dir = OUTPUT_BASE / model

    # Skip-if-exists logic
    if not args.force:
        exists, files = output_already_exists(target_dir, params_oil, val_oil, model, mode)
        if exists:
            msg = f"  [SKIP] {model} | params={params_oil} | val={val_oil} | mode={mode}  (already exists)"
            print(msg)
            log_lines.append(msg)
            return {"job": job, "status": "skipped", "duration_s": 0.0}

    # Check params CSV exists
    if not job["params_csv"].exists():
        msg = (f"  [FAIL] {model} | params={params_oil} | val={val_oil} | mode={mode}  "
               f"params_csv not found: {job['params_csv']}")
        print(msg)
        log_lines.append(msg)
        return {"job": job, "status": "failed", "error": "params_csv not found",
                "duration_s": 0.0}

    # We let validation.py write into a temporary working dir per job to avoid
    # collisions with parallel workers, then move outputs to the final structure.
    working_dir = OUTPUT_BASE / "_tmp" / f"{model}_{params_oil}_{val_oil}_{mode}"
    working_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(job, args, working_dir)

    if args.dry_run:
        msg = f"  [DRY-RUN] {' '.join(cmd)}"
        print(msg)
        log_lines.append(msg)
        return {"job": job, "status": "dry_run", "duration_s": 0.0}

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
            msg = (f"  [FAIL] {model} | params={params_oil} | val={val_oil} | mode={mode}  "
                   f"(returncode={result.returncode}, {duration:.1f}s)")
            print(msg)
            log_lines.append(msg)
            log_lines.append(f"    STDERR: {result.stderr.strip()[:500]}")
            return {"job": job, "status": "failed",
                    "error": result.stderr.strip()[:500],
                    "duration_s": duration}

        # Move outputs into target_dir/summary and target_dir/detail
        n_sum, n_det = organize_outputs(
            working_dir=working_dir, target_dir=target_dir,
            params_oil=params_oil, val_oil=val_oil,
            model=model, mode=mode,
        )

        # Clean up empty working dir
        try:
            working_dir.rmdir()
        except OSError:
            pass  # not empty, leave it

        msg = (f"  [OK]   {model} | params={params_oil} | val={val_oil} | mode={mode}  "
               f"({duration:.1f}s, summary={n_sum}, detail={n_det})")
        print(msg)
        log_lines.append(msg)
        return {"job": job, "status": "ok", "duration_s": duration,
                "n_summary": n_sum, "n_detail": n_det}

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        msg = (f"  [TIMEOUT] {model} | params={params_oil} | val={val_oil} | mode={mode}  "
               f"(>{args.timeout_s}s)")
        print(msg)
        log_lines.append(msg)
        return {"job": job, "status": "timeout", "duration_s": duration}

    except Exception as e:
        duration = time.time() - start
        msg = (f"  [ERROR] {model} | params={params_oil} | val={val_oil} | mode={mode}  "
               f"({type(e).__name__}: {e})")
        print(msg)
        log_lines.append(msg)
        return {"job": job, "status": "error", "error": str(e),
                "duration_s": duration}


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Batch-run all model validations across all configurations."
    )
    ap.add_argument("--op_rows_csv", default=DEFAULT_OP_ROWS_CSV,
                    type=Path, help="Operating points rows CSV")
    ap.add_argument("--split_csv", default=DEFAULT_SPLIT_CSV,
                    type=Path, help="Split template CSV")
    ap.add_argument("--include_all_mode", action="store_true",
                    help="Also run --selection_mode all (default: only train_only + validation_only)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if output files already exist")
    ap.add_argument("--n_workers", type=int, default=1,
                    help="Number of parallel validation processes (default 1)")
    ap.add_argument("--timeout_s", type=int, default=3600,
                    help="Per-job timeout in seconds (default 3600 = 1h)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands but don't execute")
    ap.add_argument("--filter_model", default=None,
                    choices=["original", "modified", "oil_path"],
                    help="Only run jobs for one model")
    ap.add_argument("--filter_params_oil", default=None,
                    choices=["LPG68", "LPG100", "all"],
                    help="Only run jobs for one params_oil")
    ap.add_argument("--filter_val_oil", default=None,
                    choices=["LPG68", "LPG100", "all"],
                    help="Only run jobs for one validation_oil")
    args = ap.parse_args()

    # Validate that key files exist
    if not args.op_rows_csv.exists():
        raise FileNotFoundError(f"op_rows_csv not found: {args.op_rows_csv}")
    if not args.split_csv.exists():
        raise FileNotFoundError(f"split_csv not found: {args.split_csv}")
    if not VALIDATION_SCRIPT.exists():
        raise FileNotFoundError(f"validation script not found: {VALIDATION_SCRIPT}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Build jobs
    jobs = build_jobs(args)

    # Apply filters
    if args.filter_model:
        jobs = [j for j in jobs if j["model"] == args.filter_model]
    if args.filter_params_oil:
        jobs = [j for j in jobs if j["params_oil"] == args.filter_params_oil]
    if args.filter_val_oil:
        jobs = [j for j in jobs if j["val_oil"] == args.filter_val_oil]

    # Sort: by model, then params_oil, then val_oil, then mode
    mode_order = {"train_only": 0, "validation_only": 1, "all": 2}
    jobs.sort(key=lambda j: (j["model"], j["params_oil"], j["val_oil"],
                              mode_order.get(j["mode"], 99)))

    n_jobs = len(jobs)
    if n_jobs == 0:
        print("No jobs match the filters. Nothing to do.")
        return

    print("=" * 70)
    print(f"Batch validation run: {n_jobs} jobs")
    print("=" * 70)
    print(f"  Operating points:  {args.op_rows_csv}")
    print(f"  Split template:    {args.split_csv}")
    print(f"  Output base:       {OUTPUT_BASE}")
    print(f"  Workers:           {args.n_workers}")
    print(f"  Skip-if-exists:    {not args.force}")
    print(f"  Include 'all' mode:{args.include_all_mode}")
    print(f"  Dry run:           {args.dry_run}")
    if args.filter_model or args.filter_params_oil or args.filter_val_oil:
        print(f"  Filters: model={args.filter_model}, params={args.filter_params_oil},"
              f" val={args.filter_val_oil}")
    print("=" * 70)

    # Run jobs
    log_lines = []
    log_lines.append(f"Batch validation run started: {_ts()}")
    log_lines.append(f"Total jobs: {n_jobs}")
    log_lines.append("=" * 70)

    t_start = time.time()
    results = []

    if args.n_workers <= 1:
        # Sequential
        for i, job in enumerate(jobs, 1):
            print(f"\n[{i}/{n_jobs}]")
            log_lines.append(f"\n[{i}/{n_jobs}]")
            res = run_single_job(job, args, log_lines)
            results.append(res)
    else:
        # Parallel — careful: each job spawns a separate Python+RefProp process
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
    log_path = OUTPUT_BASE / f"batch_run_log_{_ts()}.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nLog saved: {log_path}")

    # Cleanup _tmp
    tmp_dir = OUTPUT_BASE / "_tmp"
    if tmp_dir.exists():
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            print(f"  Note: could not fully clean up {tmp_dir}")

    # Exit code reflects success
    if n_fail > 0:
        print(f"\n[WARN] {n_fail} job(s) failed. See log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
