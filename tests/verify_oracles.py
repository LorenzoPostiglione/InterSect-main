#!/usr/bin/env python3
"""
Suite Ufficiale di Test Pre-Modifica e Verifica degli Oracoli per InterSect (TC-01 .. TC-10).

Suddivisa in due categorie conformemente alle specifiche di progetto:
- Categoria I (TC-01 .. TC-05): Componenti NON modificate dal refactoring OO (Base Test di Regressione).
- Categoria II (TC-06 .. TC-10): Componenti INTERESSATE dal refactoring OO e Oracoli End-to-End.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ORACLES_DIR = PROJECT_ROOT / "tests" / "oracles" / "deterministic"
TMP_VERIFY_DIR = PROJECT_ROOT / "tests" / "oracles" / "_tmp_verify"
REPORT_JSON_PATH = PROJECT_ROOT / "tests" / "oracles" / "post_modification_test_report.json"
EVIDENZE_PRE_DIR = PROJECT_ROOT / "docs" / "consegna" / "evidenze" / "oracoli_pre_modifica"
EVIDENZE_POST_DIR = PROJECT_ROOT / "docs" / "consegna" / "evidenze" / "risultati_post_modifica"

ORACLE_CASES = [
    {
        "tc_id": "TC-08",
        "name": "int_sat_10_12",
        "description": "Oracolo End-to-End QF_LIA (SAT, 20 variabili Int in L, blocking clauses)",
        "smt_path": "benchmarks/slacks-80/10-12.slack.smt2",
        "theta": 256,
        "coverage": 50,
        "delta": 0.0050,
        "U": 2**256 - 1,
        "timeout": 120,
        "num_samples": 3,
        "expected_csv_rel": "slacks-80_results_interval_guided_sampler_cov50/10-12.slack.csv",
    },
    {
        "tc_id": "TC-09",
        "name": "str_sat_instance00022",
        "description": "Oracolo End-to-End QF_SLIA (SAT, variabile String X, stringToIndex shortlex)",
        "smt_path": "benchmarks/automatark-lu-80/instance00022.smt2",
        "theta": 256,
        "coverage": 10,
        "delta": 0.0050,
        "U": 2**256 - 1,
        "timeout": 120,
        "num_samples": 3,
        "expected_csv_rel": "automatark-lu-80_results_interval_guided_sampler_cov10/instance00022.csv",
    },
    {
        "tc_id": "TC-10",
        "name": "str_unsat_bayesian_instance02824",
        "description": "Oracolo End-to-End UNSAT con Bayesian Early Stopping (20 iterazioni)",
        "smt_path": "benchmarks/automatark-lu-80/instance02824.smt2",
        "theta": 256,
        "coverage": 10,
        "delta": 0.0050,
        "U": 2**256 - 1,
        "timeout": 120,
        "num_samples": None,
        "expected_csv_rel": "automatark-lu-80_results_interval_guided_sampler_cov10/instance02824.csv",
    },
]


def _cleanup_tmp_smt2() -> None:
    tmp_dir = PROJECT_ROOT / "tmp"
    if tmp_dir.exists():
        for p in tmp_dir.glob("tmp_*.smt2"):
            p.unlink(missing_ok=True)


def _run_single_case_in_subproc(case: dict, results_root: Path) -> float:
    """Run a single sampler case in a subprocess with PYTHONHASHSEED=0 and return wall time."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    code = (
        "import sys, random, numpy as np\n"
        f"sys.path.insert(0, {str(SCRIPTS_DIR)!r})\n"
        "from solver_utils_typed import intervalGuidedSampler\n"
        "random.seed(42)\n"
        "np.random.seed(42)\n"
        "intervalGuidedSampler(\n"
        f"    {str(PROJECT_ROOT / case['smt_path'])!r},\n"
        f"    theta={case['theta']!r},\n"
        f"    coverage={case['coverage']!r},\n"
        f"    delta={case['delta']!r},\n"
        f"    U={case['U']!r},\n"
        f"    timeout={case['timeout']!r},\n"
        "    threads=1,\n"
        f"    num_samples={case['num_samples']!r},\n"
        f"    results_root={str(results_root)!r},\n"
        ")\n"
    )
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "-c", code], env=env, cwd=str(PROJECT_ROOT), check=True)
    return time.perf_counter() - t0


def generate_oracles() -> int:
    if ORACLES_DIR.exists():
        shutil.rmtree(ORACLES_DIR)
    ORACLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== GENERAZIONE ORACOLI DETERMINISTICI DI BASELINE ===")
    for case in ORACLE_CASES:
        print(f"\n[GENERATE] {case['tc_id']} - {case['name']} ({case['smt_path']})...")
        wall_s = _run_single_case_in_subproc(case, ORACLES_DIR)
        csv_path = ORACLES_DIR / case["expected_csv_rel"]
        df = pd.read_csv(csv_path)
        print(
            f"  -> Salvato: {csv_path.relative_to(PROJECT_ROOT)} "
            f"(righe={len(df)}, solver_time={df['time'].sum():.2f}s, wall_time={wall_s:.2f}s)"
        )
    _cleanup_tmp_smt2()
    return 0


def run_all_tests() -> int:
    from generate_ranked_linear_smt import parse_int_expr
    from InterSect import build_arg_parser
    from solver_utils import (
        create_mask_with_coverage,
        create_string_mask_with_coverage,
        select_free_vars,
    )
    from solver_utils_typed import (
        _create_string_interval_mask,
        _extract_int_bounds,
        _mask_is_uniform_typed,
        _string_to_index,
        _total_strings_up_to_length,
        extract_free_vars_with_type_from_smt2_file,
        partition_variable_typed,
    )

    results = []

    print("======================================================================")
    print(" CATEGORIA I: TEST SU COMPONENTI NON MODIFICATE (BASE REGRESSIONE)")
    print("======================================================================")

    # TC-01: Parser espressioni intere e CLI InterSect.py
    t0 = time.perf_counter()
    val_256 = parse_int_expr("256")
    val_u = parse_int_expr("2**256-1")
    parser = build_arg_parser()
    parsed = parser.parse_args(
        [
            "benchmarks/slacks-80/10-12.slack.smt2",
            "--theta",
            "256",
            "--coverage",
            "50",
            "--U",
            "2**256-1",
            "--timeout",
            "60",
        ]
    )
    invalid_rejected = False
    try:
        parse_int_expr("2**256; rm -rf")
    except Exception:
        invalid_rejected = True
    tc01_ok = (
        val_256 == 256
        and val_u == 2**256 - 1
        and parsed.theta == 256
        and parsed.coverage == 50
        and parsed.U == 2**256 - 1
        and parsed.timeout == 60.0
        and parsed.delta == 0.0050
        and invalid_rejected
    )
    dt = time.perf_counter() - t0
    print(f"[TC-01] Parser CLI & parse_int_expr: {'PASS' if tc01_ok else 'FAIL'} ({dt*1000:.2f} ms)")
    results.append(
        {
            "tc_id": "TC-01",
            "category": "Unmodified",
            "status": "PASS" if tc01_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": f"parse_int_expr('2**256-1')={val_u == 2**256-1}, delta_default={parsed.delta}, invalid_rejected={invalid_rejected}",
        }
    )

    # TC-02: Filtraggio variabili libere select_free_vars
    t0 = time.perf_counter()
    sample_vars = [("x0_minus", "Int"), ("x0_plus", "Int"), ("X", "String")]
    kept = select_free_vars(sample_vars, keep=["x0_plus", "X"], interactive=False)
    unknown_rejected = False
    try:
        select_free_vars(sample_vars, keep=["non_existent"], interactive=False)
    except ValueError:
        unknown_rejected = True
    tc02_ok = kept == [("x0_plus", "Int"), ("X", "String")] and unknown_rejected
    dt = time.perf_counter() - t0
    print(f"[TC-02] Filtraggio select_free_vars: {'PASS' if tc02_ok else 'FAIL'} ({dt*1000:.2f} ms)")
    results.append(
        {
            "tc_id": "TC-02",
            "category": "Unmodified",
            "status": "PASS" if tc02_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": f"kept={kept}, unknown_rejected={unknown_rejected}",
        }
    )

    # TC-03: White-Box KS-Test Uniformità Maschere Intere
    t0 = time.perf_counter()
    random.seed(42)
    np.random.seed(42)
    mask_int = create_mask_with_coverage(0, 2**32, target_coverage=30, delta=0.0050)
    int_cov = sum(b - a for a, b in mask_int) / (2**32) * 100.0
    int_midpoints = [((a + b) / 2.0) / (2**32) for a, b in mask_int]
    ks_int = scipy.stats.kstest(int_midpoints, "uniform", args=(0, 1))
    int_typed_ok = _mask_is_uniform_typed(mask_int, "Int", 0, 2**32)
    tc03_ok = (
        len(mask_int) > 0
        and (28.0 <= int_cov <= 32.0)
        and ks_int.pvalue > 0.05
        and int_typed_ok
    )
    dt = time.perf_counter() - t0
    print(
        f"[TC-03] KS-Test Maschere Intere: {'PASS' if tc03_ok else 'FAIL'} "
        f"(intervalli={len(mask_int)}, cov={int_cov:.2f}%, KS={ks_int.statistic:.4f}, p={ks_int.pvalue:.4f}, {dt:.2f}s)"
    )
    results.append(
        {
            "tc_id": "TC-03",
            "category": "Unmodified",
            "status": "PASS" if tc03_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": f"intervals={len(mask_int)}, coverage={int_cov:.2f}%, ks_stat={ks_int.statistic:.4f}, p_value={ks_int.pvalue:.4f}",
        }
    )

    # TC-04: White-Box KS-Test Uniformità Maschere Stringhe e _string_to_index
    t0 = time.perf_counter()
    total_strings = _total_strings_up_to_length(54)
    string_ub = min(2**256 - 1, total_strings - 1)
    idx_a = _string_to_index("a")
    idx_z = _string_to_index("z")
    idx_aa = _string_to_index("aa")
    mask_str_intervals = _create_string_interval_mask(0, string_ub, coverage=30, delta=0.0050)
    str_cov = sum(b - a for a, b in mask_str_intervals) / string_ub * 100.0
    str_typed_ok = _mask_is_uniform_typed(mask_str_intervals, "String", 0, string_ub)
    mask_str_samples = create_string_mask_with_coverage(
        0, string_ub, target_coverage=30, numStringSamples=1, delta=0.0050
    )
    flat_mask = [item for sublist in mask_str_samples for item in sublist]
    norm_dist = [_string_to_index(x) / total_strings for x in flat_mask]
    ks_str = scipy.stats.kstest(norm_dist, "uniform", args=(0, 1))
    tc04_ok = (
        idx_a == 0
        and idx_z == 25
        and idx_aa == 26
        and (28.0 <= str_cov <= 32.0)
        and ks_str.pvalue > 0.05
        and str_typed_ok
    )
    dt = time.perf_counter() - t0
    print(
        f"[TC-04] KS-Test Maschere Stringhe & Ranking: {'PASS' if tc04_ok else 'FAIL'} "
        f"(intervalli={len(mask_str_intervals)}, cov={str_cov:.2f}%, KS={ks_str.statistic:.4f}, p={ks_str.pvalue:.4f}, {dt:.2f}s)"
    )
    results.append(
        {
            "tc_id": "TC-04",
            "category": "Unmodified",
            "status": "PASS" if tc04_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": f"idx(a,z,aa)=({idx_a},{idx_z},{idx_aa}), intervals={len(mask_str_intervals)}, coverage={str_cov:.2f}%, ks_stat={ks_str.statistic:.4f}, p_value={ks_str.pvalue:.4f}",
        }
    )

    # TC-05: Tool di Verifica Uniformità CSV check_interval_coverage_uniformity.py
    t0 = time.perf_counter()
    oracle_str_csv = (
        ORACLES_DIR
        / "automatark-lu-80_results_interval_guided_sampler_cov10"
        / "instance00022.csv"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "check_interval_coverage_uniformity.py"),
            "--csv",
            str(oracle_str_csv.relative_to(PROJECT_ROOT)),
            "--column",
            "mask_var_pairs",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    tc05_ok = proc.returncode == 0 and "chi2=" in proc.stdout and "Rows with parsed data  : 3" in proc.stdout
    dt = time.perf_counter() - t0
    print(f"[TC-05] Analisi Uniformita CSV (check_interval_coverage_uniformity.py): {'PASS' if tc05_ok else 'FAIL'} ({dt:.2f}s)")
    results.append(
        {
            "tc_id": "TC-05",
            "category": "Unmodified",
            "status": "PASS" if tc05_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": proc.stdout.strip().replace("\n", " | ")[:220],
        }
    )

    print("\n======================================================================")
    print(" CATEGORIA II: TEST SU COMPONENTI INTERESSATE DAL REFACTORING OO")
    print("======================================================================")

    # TC-06: Estrazione variabili libere e semantica direzionale _extract_int_bounds
    t0 = time.perf_counter()
    fvars_int = extract_free_vars_with_type_from_smt2_file(
        PROJECT_ROOT / "benchmarks/slacks-80/10-12.slack.smt2"
    )
    fvars_str = extract_free_vars_with_type_from_smt2_file(
        PROJECT_ROOT / "benchmarks/automatark-lu-80/instance00022.smt2"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as tf:
        tf.write(
            "(declare-fun x () Int)\n"
            "(assert (>= x 10))\n"
            "(assert (<= x 100))\n"
            "(assert (<= 50 x))\n"  # Costante a sinistra: deve essere ignorata da _extract_int_bounds
            "(check-sat)\n"
        )
        tf_path = Path(tf.name)
    try:
        lb_x, ub_x = _extract_int_bounds(tf_path, "x", U=2**256 - 1)
    finally:
        tf_path.unlink(missing_ok=True)
    tc06_ok = (
        len(fvars_int) == 20
        and fvars_str == [("X", "String")]
        and (lb_x, ub_x) == (10, 100)
    )
    dt = time.perf_counter() - t0
    print(
        f"[TC-06] Estrazione Variabili & Bound Direzionali (_extract_int_bounds): "
        f"{'PASS' if tc06_ok else 'FAIL'} (vars_int={len(fvars_int)}, vars_str={fvars_str}, bounds={(lb_x, ub_x)}, {dt*1000:.2f} ms)"
    )
    results.append(
        {
            "tc_id": "TC-06",
            "category": "ModifiedTarget",
            "status": "PASS" if tc06_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": f"fvars_int_count={len(fvars_int)}, fvars_str={fvars_str}, directional_bounds={(lb_x, ub_x)}",
        }
    )

    # TC-07: Partizionamento del dominio E vs L (partition_variable_typed)
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as tf:
        tf.write(
            "(declare-fun x_small () Int)\n"
            "(declare-fun x_large () Int)\n"
            "(declare-fun b_flag () Bool)\n"
            "(declare-fun s_var () String)\n"
            "(assert (>= x_small 0))\n"
            "(assert (<= x_small 50))\n"
            "(assert (>= x_large 0))\n"
            "(assert (<= x_large 1000))\n"
            "(check-sat)\n"
        )
        tf_path = Path(tf.name)
    try:
        E_set, L_set = set(), set()
        for vname, vtype in [
            ("x_small", "Int"),
            ("x_large", "Int"),
            ("b_flag", "Bool"),
            ("s_var", "String"),
        ]:
            partition_variable_typed(E_set, L_set, tf_path, vname, vtype, theta=256, U=2**256 - 1)
    finally:
        tf_path.unlink(missing_ok=True)
    e_names = {entry[0] for entry in E_set}
    l_names = {entry[0] for entry in L_set}
    tc07_ok = e_names == {"x_small", "b_flag"} and l_names == {"x_large", "s_var"}
    dt = time.perf_counter() - t0
    print(
        f"[TC-07] Partizionamento Dominio E vs L (partition_variable_typed): "
        f"{'PASS' if tc07_ok else 'FAIL'} (E={sorted(e_names)}, L={sorted(l_names)}, {dt*1000:.2f} ms)"
    )
    results.append(
        {
            "tc_id": "TC-07",
            "category": "ModifiedTarget",
            "status": "PASS" if tc07_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": f"E={sorted(e_names)}, L={sorted(l_names)}",
        }
    )

    # TC-08, TC-09, TC-10: Oracoli Deterministici End-to-End
    if TMP_VERIFY_DIR.exists():
        shutil.rmtree(TMP_VERIFY_DIR)
    TMP_VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    for case in ORACLE_CASES:
        tc_id = case["tc_id"]
        oracle_csv = ORACLES_DIR / case["expected_csv_rel"]
        print(f"\n[{tc_id}] {case['description']} ({case['smt_path']})...")
        wall_s = _run_single_case_in_subproc(case, TMP_VERIFY_DIR)
        candidate_csv = TMP_VERIFY_DIR / case["expected_csv_rel"]

        df_oracle = pd.read_csv(oracle_csv)
        df_cand = pd.read_csv(candidate_csv)

        EVIDENZE_PRE_DIR.mkdir(parents=True, exist_ok=True)
        EVIDENZE_POST_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(oracle_csv, EVIDENZE_PRE_DIR / oracle_csv.name)
        shutil.copy2(candidate_csv, EVIDENZE_POST_DIR / candidate_csv.name)

        oracle_time = float(df_oracle["time"].sum())
        cand_time = float(df_cand["time"].sum())
        speedup = oracle_time / cand_time if cand_time > 0 else float("inf")

        df_oracle_cmp = df_oracle.drop(columns=["time"])
        df_cand_cmp = df_cand.drop(columns=["time"])
        match = df_oracle_cmp.equals(df_cand_cmp)

        sat_rows = int((df_cand["status"] == "SAT").sum())
        unsat_rows = int((df_cand["status"] != "SAT").sum())
        print(
            f"  -> [{'PASS' if match else 'FAIL'}] Righe={len(df_cand)} (SAT={sat_rows}, UNSAT={unsat_rows}) | "
            f"Match Bit-a-Bit Oracolo={match} | Tempo={cand_time:.2f}s (Oracolo={oracle_time:.2f}s, {speedup:.2f}x)"
        )
        results.append(
            {
                "tc_id": tc_id,
                "category": "ModifiedTarget",
                "status": "PASS" if match else "FAIL",
                "duration_s": round(cand_time, 4),
                "oracle_duration_s": round(oracle_time, 4),
                "rows": len(df_cand),
                "sat_rows": sat_rows,
                "unsat_rows": unsat_rows,
                "bit_for_bit_match": match,
                "details": f"file={case['smt_path']}, rows={len(df_cand)} (SAT={sat_rows}, UNSAT={unsat_rows}), match={match}",
            }
        )

    shutil.rmtree(TMP_VERIFY_DIR, ignore_errors=True)
    _cleanup_tmp_smt2()

    print("\n======================================================================")
    print(" CATEGORIA III: TEST NUOVE CLASSI OBJECT-ORIENTED (scripts/classes/)")
    print("======================================================================")

    from classes import (
        DomainPartitioner,
        MaskGenerator,
        ResultRecorder,
        SMTSolver,
        SamplerEngine,
        Z3Solver,
    )

    # TC-11: Test Unitario OOP DomainPartitioner (estrazione in-memory, bound direzionali, E/L)
    t0 = time.perf_counter()
    dp = DomainPartitioner(theta=256, U=2**256 - 1)
    smt_sample = (
        "(declare-fun x_small () Int)\n"
        "(declare-fun x_large () Int)\n"
        "(declare-fun b_flag () Bool)\n"
        "(declare-fun bv8 () (_ BitVec 8))\n"
        "(assert (>= x_small 10))\n"
        "(assert (<= x_small 50))\n"
        "(assert (<= 30 x_small))\n"
        "(assert (>= x_large 0))\n"
        "(assert (<= x_large 1000))\n"
        "(check-sat)\n"
    )
    extracted_vars = dp.extract_free_vars(smt_sample)
    lb_s, ub_s = dp.extract_int_bounds_from_content(smt_sample, "x_small", 2**256 - 1)
    E_dp, L_dp, timed_out_dp = dp.partition(smt_sample, extracted_vars)
    e_dp_names = {entry[0] for entry in E_dp}
    l_dp_names = {entry[0] for entry in L_dp}
    tc11_ok = (
        len(extracted_vars) == 4
        and (lb_s, ub_s) == (10, 50)
        and not timed_out_dp
        and e_dp_names == {"x_small", "b_flag", "bv8"}
        and l_dp_names == {"x_large"}
    )
    dt = time.perf_counter() - t0
    print(
        f"[TC-11] DomainPartitioner In-Memory (E/L & Bound Direzionali): "
        f"{'PASS' if tc11_ok else 'FAIL'} (E={sorted(e_dp_names)}, L={sorted(l_dp_names)}, bounds={(lb_s, ub_s)}, {dt*1000:.2f} ms)"
    )
    results.append(
        {
            "tc_id": "TC-11",
            "category": "AddedOO",
            "status": "PASS" if tc11_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": f"extracted={len(extracted_vars)}, bounds={(lb_s, ub_s)}, E={sorted(e_dp_names)}, L={sorted(l_dp_names)}",
        }
    )

    # TC-12: Test Unitario OOP SMTSolver / Z3Solver In-Memory & Zero File Leak in tmp/
    t0 = time.perf_counter()
    _cleanup_tmp_smt2()
    tmp_dir = PROJECT_ROOT / "tmp"
    before_tmp = set(tmp_dir.glob("tmp_*.smt2")) if tmp_dir.exists() else set()
    z3_solver = Z3Solver(PROJECT_ROOT / "benchmarks/slacks-80/10-12.slack.smt2")
    is_subclass_ok = issubclass(Z3Solver, SMTSolver)
    z3_solver.reset_iteration(None)
    status_1, model_1 = z3_solver.check_and_get_model()
    L_dummy = [(v, t, -(2**256 - 1), 2**256 - 1) for v, t in DomainPartitioner.extract_free_vars(z3_solver.get_base_content())]
    z3_solver.add_blocking_clause(model_1, L_dummy)
    z3_solver.reset_iteration(None)
    status_2, model_2 = z3_solver.check_and_get_model()
    after_tmp = set(tmp_dir.glob("tmp_*.smt2")) if tmp_dir.exists() else set()
    tc12_ok = (
        is_subclass_ok
        and status_1 == "SAT"
        and status_2 == "SAT"
        and model_1 != model_2
        and len(z3_solver._blocked_clauses) == 1
        and len(after_tmp - before_tmp) == 0
    )
    dt = time.perf_counter() - t0
    print(
        f"[TC-12] Z3Solver In-Memory & Blocking Clauses (0 tmp files): "
        f"{'PASS' if tc12_ok else 'FAIL'} (status1={status_1}, status2={status_2}, distinct={model_1 != model_2}, tmp_leaked={len(after_tmp - before_tmp)}, {dt:.2f}s)"
    )
    results.append(
        {
            "tc_id": "TC-12",
            "category": "AddedOO",
            "status": "PASS" if tc12_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": f"issubclass={is_subclass_ok}, distinct_models={model_1 != model_2}, tmp_files_created={len(after_tmp - before_tmp)}",
        }
    )

    # TC-13: Test Unitario OOP ResultRecorder (persistenza CSV incrementale e resume)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_rec_dir:
        csv_rec_path = Path(tmp_rec_dir) / "test_recorder.csv"
        rec_vars = [("x", "Int"), ("y", "Int")]
        L_rec = [("x", "Int", 0, 100), ("y", "Int", 0, 100)]
        rec = ResultRecorder(csv_rec_path, rec_vars, num_samples=2)
        r1 = rec.record_and_flush("SAT", {"x": 10, "y": 20}, None, [("x", [(0, 50)]), ("y", [(0, 50)])], 0.12, L_rec)
        r2 = rec.record_and_flush("UNSAT", None, None, [("x", [(50, 100)]), ("y", [(50, 100)])], 0.05, L_rec)
        r3 = rec.record_and_flush("SAT", {"x": 30, "y": 40}, None, [("x", [(0, 50)]), ("y", [(0, 50)])], 0.15, L_rec)
        df_rec = pd.read_csv(csv_rec_path)
        rec_resumed = ResultRecorder(csv_rec_path, rec_vars, num_samples=2)
        tc13_ok = (
            not r1
            and not r2
            and r3
            and len(df_rec) == 3
            and rec.sat_count == 2
            and len(rec.S) == 2
            and rec_resumed.already_reached_target()
        )
    dt = time.perf_counter() - t0
    print(
        f"[TC-13] ResultRecorder Incremental CSV & Target Resume: "
        f"{'PASS' if tc13_ok else 'FAIL'} (rows={len(df_rec)}, sat_count={rec.sat_count}, resumed_stop={rec_resumed.already_reached_target()}, {dt*1000:.2f} ms)"
    )
    results.append(
        {
            "tc_id": "TC-13",
            "category": "AddedOO",
            "status": "PASS" if tc13_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": f"rows={len(df_rec)}, sat_count={rec.sat_count}, already_reached_target={rec_resumed.already_reached_target()}",
        }
    )

    # TC-14: Test Unitario OOP MaskGenerator (Context Manager, Pool persistente e KS-test)
    t0 = time.perf_counter()
    random.seed(42)
    np.random.seed(42)
    L_mg = [("x1", "Int", 0, 2**32), ("x2", "Int", 0, 2**32)]
    with MaskGenerator(coverage=30, delta=0.0050, max_workers=2) as mg:
        pairs_1 = mg.generate_masks(L_mg)
        pool_first = mg._executor
        pairs_2 = mg.generate_masks(L_mg)
        pool_second = mg._executor
        pool_active = pool_first is not None and (pool_first is pool_second)
        uniform_1 = mg.is_mask_uniform(pairs_1[0][1], "Int", 0, 2**32)
    pool_closed = mg._executor is None
    tc14_ok = (
        pool_active
        and pool_closed
        and len(pairs_1) == 2
        and len(pairs_2) == 2
        and uniform_1
    )
    dt = time.perf_counter() - t0
    print(
        f"[TC-14] MaskGenerator Persistent Pool & KS Uniformity: "
        f"{'PASS' if tc14_ok else 'FAIL'} (pool_reused={pool_active}, pool_closed={pool_closed}, uniform={uniform_1}, {dt:.2f}s)"
    )
    results.append(
        {
            "tc_id": "TC-14",
            "category": "AddedOO",
            "status": "PASS" if tc14_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": f"pool_reused={pool_active}, pool_closed={pool_closed}, masks_generated={len(pairs_1)}, is_uniform={uniform_1}",
        }
    )

    # TC-15: Test Integrazione OOP SamplerEngine.from_config(...) & InterSect.py CLI
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_eng_dir:
        engine = SamplerEngine.from_config(
            smt_filepath=PROJECT_ROOT / "benchmarks/automatark-lu-80/instance00022.smt2",
            theta=256,
            coverage=10,
            delta=0.0050,
            U=2**256 - 1,
            timeout=60,
            threads=1,
            num_samples=2,
            results_root=Path(tmp_eng_dir),
        )
        samples = engine.run()
        eng_csv = (
            Path(tmp_eng_dir)
            / "automatark-lu-80_results_interval_guided_sampler_cov10"
            / "instance00022.csv"
        )
        eng_csv_exists = eng_csv.exists()
        eng_csv_rows = len(pd.read_csv(eng_csv)) if eng_csv_exists else 0
        tc15_ok = len(samples) == 2 and eng_csv_exists and eng_csv_rows >= 2
    dt = time.perf_counter() - t0
    print(
        f"[TC-15] SamplerEngine End-to-End OO Orchestration: "
        f"{'PASS' if tc15_ok else 'FAIL'} (samples={len(samples)}, csv_exists={eng_csv_exists}, rows={eng_csv_rows}, {dt:.2f}s)"
    )
    results.append(
        {
            "tc_id": "TC-15",
            "category": "AddedOO",
            "status": "PASS" if tc15_ok else "FAIL",
            "duration_s": round(dt, 4),
            "details": f"samples_collected={len(samples)}, csv_exists={eng_csv_exists}, rows={eng_csv_rows}",
        }
    )

    all_passed = all(r["status"] == "PASS" for r in results)
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"all_passed": all_passed, "test_cases": results}, f, indent=2)

    print("\n======================================================================")
    print(
        f" ESITO COMPLESSIVO (TC-01 .. TC-15): "
        f"{f'{len(results)}/{len(results)} SUPERATI (PASS)' if all_passed else ' ALCUNI TEST FALLITI (FAIL)'}"
    )
    print("======================================================================")
    return 0 if all_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera o verifica la suite di test e Oracoli di InterSect.")
    parser.add_argument("--generate", action="store_true", help="Rigenera i CSV di Oracolo deterministico.")
    args = parser.parse_args()
    if args.generate:
        return generate_oracles()
    return run_all_tests()


if __name__ == "__main__":
    raise SystemExit(main())
