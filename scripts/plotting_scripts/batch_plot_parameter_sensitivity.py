# scripts/plotting_scripts/batch_plot_parameter_sensitivity.py
#
# Batch-Treiber für plot_parameter_sensitivity.py.
#
# Ruft das Plot-Skript für eine Liste von Parameter-CSVs auf, einen Lauf
# pro CSV (jeweils eigener Subprozess, damit RefProp / matplotlib zwischen
# den Läufen sauber zurückgesetzt werden und ein Fehler nicht den ganzen
# Batch abbricht).
#
# Aktivieren wie beim Einzelskript:
#   cd C:\Users\ahl-jgr\PycharmProjects\compressor-simulation
#   .venv\Scripts\activate
#   $env:RPPREFIX = "T:\ahl\REFPROP"
#
# Beispiel:
#   python scripts/plotting_scripts/batch_plot_parameter_sensitivity.py --op_rows_csv results/split_template/operating_points_rows_2026-03-12_112331.csv --split_csv   results/split_template/operating_points_split_template_2026-03-12_112331.csv --selection_mode train_only --out_dir results/sensitivity
#
# Die zu plottenden Parameter-Sätze stehen in PARAM_FILES weiter unten.
# --model und --oil werden im Plot-Skript automatisch aus jeder CSV gelesen.

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# =========================================================
# Liste der zu plottenden Parameter-Sätze
# =========================================================
# Hinweis: In der ursprünglichen Liste war ein Eintrag doppelt
# (fitted_params_lpg100_original_ga_2026-03-19_210256.csv).
# Die Liste enthält daher 8 eindeutige Sätze.
# Falls "Modified_All" (modified, oil=all) ergänzt werden soll, einfach
# unten die Zeile einkommentieren und den richtigen Pfad eintragen.
PARAM_FILES = [
    # --- Basismodell (original) ---
    "results/final_results/Molinaroli_All/Fitting/fitted_params_all_original_ga_2026-03-21_192615.csv",
    "results/final_results/Molinaroli_LPG68/Fitting/fitted_params_lpg68_original_ga_2026-03-08_101308.csv",
    "results/final_results/Molinaroli_LPG100/Fitting/fitted_params_lpg100_original_ga_2026-03-19_210256.csv",

    # --- Modellausbaustufe I (modified) ---
    "results/final_results/Modified_All/Fitting/fitted_params_all_modified_ga_2026-03-26_110247.csv"
    "results/final_results/Modified_LPG68/Fitting/fitted_params_lpg68_modified_ga_2026-03-22_185546.csv",
    "results/final_results/Modified_LPG100/Fitting/fitted_params_lpg100_modified_ga_2026-03-28_092941.csv",

    # --- Modellausbaustufe II (oil_path) ---
    "results/final_results/Oil_Path_All/Fitting/fitted_params_all_oil_path_ga_2026-05-06_224812.csv",
    "results/final_results/Oil_Path_LPG68/Fitting/fitted_params_lpg68_oil_path_ga_2026-04-17_113953.csv",
    "results/final_results/Oil_Path_LPG100/Fitting/fitted_params_lpg100_oil_path_ga_2026-04-18_041610.csv",
]


# =========================================================
# Default-Pfad zum Plot-Skript (gleicher Ordner wie dieses Skript)
# =========================================================
DEFAULT_PLOT_SCRIPT = Path(__file__).resolve().parent / "plot_parameter_sensitivity.py"


# =========================================================
# Helper
# =========================================================
def run_one(
    plot_script: Path,
    params_csv: Path,
    common_args: list[str],
    log_dir: Path | None,
    dry_run: bool,
) -> tuple[bool, str]:
    """Run plot_parameter_sensitivity.py for a single params CSV."""

    cmd = [
        sys.executable, str(plot_script),
        "--params_csv", str(params_csv),
        *common_args,
    ]

    print("\n" + "=" * 78)
    print(f"  CSV : {params_csv}")
    print(f"  CMD : {' '.join(cmd)}")
    print("=" * 78)

    if dry_run:
        return True, "dry-run"

    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as e:
        return False, f"executable not found: {e}"

    # Always echo what the child printed.
    if res.stdout:
        print(res.stdout, end="" if res.stdout.endswith("\n") else "\n")
    if res.stderr:
        print(res.stderr, end="" if res.stderr.endswith("\n") else "\n",
              file=sys.stderr)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        stem = params_csv.stem
        (log_dir / f"{stem}__{stamp}.stdout.log").write_text(
            res.stdout or "", encoding="utf-8"
        )
        (log_dir / f"{stem}__{stamp}.stderr.log").write_text(
            res.stderr or "", encoding="utf-8"
        )

    ok = (res.returncode == 0)
    return ok, ("ok" if ok else f"returncode={res.returncode}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Batch-Wrapper für plot_parameter_sensitivity.py. Führt das "
            "Plot-Skript für jede Parameter-CSV in PARAM_FILES separat aus."
        )
    )

    # --- Required inputs (an alle Läufe weitergereicht) ---
    ap.add_argument("--op_rows_csv", required=True, type=Path,
                    help="Path to operating_points_rows.csv")
    ap.add_argument("--split_csv", required=True, type=Path,
                    help="Path to operating_points_split_template.csv")

    # --- Optional pass-through ---
    ap.add_argument("--selection_mode", default="train_only",
                    choices=["train_only", "validation_only", "all"])
    ap.add_argument("--N_max_rpm", type=float, default=None)
    ap.add_argument("--V_h_cm3", type=float, default=None)
    ap.add_argument("--r_min", type=float, default=None)
    ap.add_argument("--r_max", type=float, default=None)
    ap.add_argument("--n_points", type=int, default=None)
    ap.add_argument("--fail_penalty", type=float, default=None)
    ap.add_argument("--mask_if_fails", action="store_true")
    ap.add_argument("--out_dir", default="results/sensitivity",
                    help="Output folder (an Plot-Skript weitergereicht)")
    ap.add_argument("--out_format", choices=["png", "svg"], default="png")

    # --- Batch-spezifisch ---
    ap.add_argument(
        "--plot_script", type=Path, default=DEFAULT_PLOT_SCRIPT,
        help=("Pfad zu plot_parameter_sensitivity.py "
              f"(Default: {DEFAULT_PLOT_SCRIPT})"),
    )
    ap.add_argument(
        "--param_files", nargs="+", default=None,
        help=("Optional: explizite Liste von Parameter-CSVs. Überschreibt "
              "PARAM_FILES."),
    )
    ap.add_argument(
        "--continue_on_error", action="store_true",
        help="Bei Fehlern nicht abbrechen, sondern mit dem nächsten weitermachen.",
    )
    ap.add_argument(
        "--log_dir", type=Path, default=None,
        help="Optional: Ordner für stdout/stderr-Logs je Lauf.",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Nur Kommandos ausgeben, nichts ausführen.",
    )

    args = ap.parse_args()

    # -------------------------
    # Validate
    # -------------------------
    if not args.plot_script.exists():
        sys.exit(f"[ERROR] Plot-Skript nicht gefunden: {args.plot_script}")
    if not args.op_rows_csv.exists():
        sys.exit(f"[ERROR] op_rows_csv nicht gefunden: {args.op_rows_csv}")
    if not args.split_csv.exists():
        sys.exit(f"[ERROR] split_csv nicht gefunden: {args.split_csv}")

    file_list = args.param_files if args.param_files else PARAM_FILES
    param_paths = [Path(p) for p in file_list]

    # Eindeutigkeit & Existenz prüfen
    seen = set()
    unique_paths = []
    for p in param_paths:
        key = str(p).replace("\\", "/").lower()
        if key in seen:
            print(f"[WARN] Duplikat übersprungen: {p}")
            continue
        seen.add(key)
        unique_paths.append(p)

    missing = [p for p in unique_paths if not p.exists()]
    if missing:
        print("[WARN] Nicht gefundene Dateien (werden übersprungen):")
        for p in missing:
            print(f"   - {p}")
    runnable = [p for p in unique_paths if p.exists()]

    if not runnable:
        sys.exit("[ERROR] Keine vorhandene Parameter-CSV gefunden.")

    # -------------------------
    # Build common args
    # -------------------------
    common_args = [
        "--op_rows_csv", str(args.op_rows_csv),
        "--split_csv",   str(args.split_csv),
        "--selection_mode", args.selection_mode,
        "--out_dir",     str(args.out_dir),
        "--out_format",  args.out_format,
    ]
    if args.N_max_rpm is not None:
        common_args += ["--N_max_rpm", str(args.N_max_rpm)]
    if args.V_h_cm3 is not None:
        common_args += ["--V_h_cm3", str(args.V_h_cm3)]
    if args.r_min is not None:
        common_args += ["--r_min", str(args.r_min)]
    if args.r_max is not None:
        common_args += ["--r_max", str(args.r_max)]
    if args.n_points is not None:
        common_args += ["--n_points", str(args.n_points)]
    if args.fail_penalty is not None:
        common_args += ["--fail_penalty", str(args.fail_penalty)]
    if args.mask_if_fails:
        common_args += ["--mask_if_fails"]

    # -------------------------
    # Run sequentially
    # -------------------------
    print(f"\nBatch-Sensitivitätsanalyse")
    print(f"  Plot-Skript    : {args.plot_script}")
    print(f"  op_rows_csv    : {args.op_rows_csv}")
    print(f"  split_csv      : {args.split_csv}")
    print(f"  selection_mode : {args.selection_mode}")
    print(f"  out_dir        : {args.out_dir}")
    print(f"  Anzahl Läufe   : {len(runnable)}")
    if args.dry_run:
        print(f"  [DRY-RUN] keine Subprozesse werden gestartet")

    results = []
    for i, p in enumerate(runnable, 1):
        print(f"\n[{i}/{len(runnable)}]  {p.name}")
        ok, msg = run_one(
            plot_script=args.plot_script,
            params_csv=p,
            common_args=common_args,
            log_dir=args.log_dir,
            dry_run=args.dry_run,
        )
        results.append((p, ok, msg))
        if not ok and not args.continue_on_error:
            print(f"\n[ABBRUCH] {p.name} fehlgeschlagen ({msg}).")
            print("  Mit --continue_on_error würden die übrigen weiterlaufen.")
            break

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 78)
    print("Zusammenfassung")
    print("=" * 78)
    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    for p, ok, msg in results:
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag:>4}]  {p}   ({msg})")
    print(f"\n  total: {len(results)}   ok: {n_ok}   fail: {n_fail}")

    if n_fail and not args.continue_on_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
