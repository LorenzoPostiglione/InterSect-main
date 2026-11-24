import csv
import itertools
import random
import re
import shutil
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

import pandas as pd
from scipy.stats import beta, ks_2samp, mannwhitneyu

from solver_utils import (
    create_mask_with_coverage,
    create_string_mask_with_coverage,
    expand_multiset,
    extract_free_vars_with_type_from_smt2_file,
    multiset_from_intervals_2,
    run_z3_and_get_model,
    uniform_multiset,
)
from classes import (
    BITVEC_RE,
    DEFAULT_STRING_ORDER,
    DomainPartitioner,
    MAX_STRING_MAX_LENGTH,
    MIN_STRING_MAX_LENGTH,
    MaskGenerator,
    RECURSIVE_STRING_TO_INDEX_HELPERS_LEX,
    RECURSIVE_STRING_TO_INDEX_HELPERS_SHORTLEX,
    REMOVAL_MODE_CHUNK,
    REMOVAL_MODE_ONE,
    ResultRecorder,
    SMTSolver,
    STRING_ORDER_LEX,
    STRING_ORDER_SHORTLEX,
    STRING_TO_INDEX_HELPERS_END,
    STRING_TO_INDEX_HELPERS_START,
    SamplerEngine,
    Z3Solver,
    build_non_recursive_lex_string_to_index_helpers as _build_non_recursive_lex_string_to_index_helpers,
    build_non_recursive_string_to_index_helpers as _build_non_recursive_string_to_index_helpers,
    format_smt_literal,
    get_string_domain_config,
    get_string_to_index_helpers,
    normalize_model_value,
    total_strings_up_to_length,
    validate_string_order,
)


STRING_MASK_START_TEMPLATE = ";; STRING_INTERVAL_MASK_START {var_name}"
STRING_MASK_END_TEMPLATE = ";; STRING_INTERVAL_MASK_END {var_name}"


def _validate_timeout(timeout):
    return SamplerEngine.validate_timeout(timeout)


def _timeout_expired(deadline):
    return SamplerEngine.timeout_expired(deadline)


def hybrid_state_space_sampler_typed(
    smt_filepath,
    free_vars = None,
    theta=256,
    coverage=50,
    delta=0.0050,
    use_recursive_string_helpers=True,
    string_max_length=54,
    timeout=None,
    output_filename=None,
):
    """
    Typed variant of hybrid_state_space_sampler.

    free_vars must be a list of (var_name, var_type) pairs, such as the output of
    extract_free_vars_with_type_from_smt2_file.

    """
    return _hybrid_state_space_sampler_typed_impl(
        smt_filepath,
        free_vars=free_vars,
        theta=theta,
        coverage=coverage,
        delta=delta,
        use_recursive_string_helpers=use_recursive_string_helpers,
        string_max_length=string_max_length,
        complement_on_unsat=False,
        results_suffix="",
        timeout=timeout,
        output_filename=output_filename,
    )


def hybrid_state_space_sampler_hsss_typed(
    smt_filepath,
    free_vars=None,
    theta=256,
    coverage=50,
    delta=0.0050,
    use_recursive_string_helpers=True,
    string_max_length=54,
    removal_mode=REMOVAL_MODE_ONE,
    chunk_size=None,
    timeout=None,
    output_filename=None,
    results_suffix="_hsss",
):
    """
    HSSS typed variant.

    When the solver returns UNSAT or UNKNOWN while masks are active, the algorithm
    complements the current masks, removes one interval from each complemented mask
    using a minimal-distortion heuristic, and continues the exploration.
    """
    return _hybrid_state_space_sampler_typed_impl(
        smt_filepath,
        free_vars=free_vars,
        theta=theta,
        coverage=coverage,
        delta=delta,
        use_recursive_string_helpers=use_recursive_string_helpers,
        string_max_length=string_max_length,
        complement_on_unsat=True,
        results_suffix=results_suffix,
        removal_mode=removal_mode,
        chunk_size=chunk_size,
        timeout=timeout,
        output_filename=output_filename,
    )


def sampler_typed_always_new_mask(
    smt_filepath,
    free_vars=None,
    theta=256,
    coverage=50,
    delta=0.0050,
    use_recursive_string_helpers=True,
    string_max_length=54,
    timeout=None,
    output_filename=None,
    results_suffix="_always_new_mask",
):
    """
    Typed sampler that always restarts from a fresh mask set after each save.

    Behaviour:
    - generate masks, run the solver
    - if SAT: save result, then restart from the original SMT and build new masks
    - if UNSAT/UNKNOWN: complement the current masks without removing intervals,
      rerun the solver, save that outcome, then restart from the original SMT
      and build new masks
    """
    timeout_seconds = _validate_timeout(timeout)
    string_domain = _get_string_domain_config(string_max_length)

    S = set()
    solver_result = {}

    solver = Z3Solver(
        smt_filepath=smt_filepath,
        string_domain=string_domain,
        use_recursive_string_helpers=use_recursive_string_helpers,
    )
    smt_content = solver.get_base_content()
    partitioner = DomainPartitioner(theta=theta, string_domain=string_domain)

    if free_vars is None:
        free_vars = partitioner.extract_free_vars(smt_content)
    print(f"Extracted free variables: {free_vars}")

    interrupted = False

    try:
        start_time = time.perf_counter()
        deadline = None if timeout_seconds is None else start_time + timeout_seconds

        E, L, timed_out = partitioner.partition(smt_content, free_vars, deadline=deadline)
        if timed_out:
            print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
            return S

        assignment_iter = [None]
        if E:
            assignment_iter = get_assignment_for_typed_E(E, limit=theta)

        with MaskGenerator(
            coverage=coverage,
            delta=delta,
            string_domain=string_domain,
        ) as mask_generator:
            for assignment in assignment_iter:
                while True:
                    if _timeout_expired(deadline):
                        print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                        return S

                    solver.reset_iteration(assignment)
                    mask_var_pairs = mask_generator.generate_masks(L)
                    solver.apply_masks(mask_var_pairs, L)

                    if not mask_var_pairs:
                        status, model = solver.check_and_get_model()
                        if _timeout_expired(deadline):
                            print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                            return S
                        partial_time = time.perf_counter() - start_time
                        start_time = time.perf_counter()
                        if status in ("UNSAT", "UNKNOWN"):
                            save_model_typed(
                                solver_result,
                                model,
                                free_vars,
                                None,
                                status,
                                assignment,
                                [],
                                partial_time,
                                L,
                            )
                        elif status == "SAT":
                            save_sat_model_typed(
                                solver_result,
                                model,
                                free_vars,
                                None,
                                status,
                                [],
                                partial_time,
                                L,
                            )
                            S.add(frozenset(model.items()))
                        break

                    status, model = solver.check_and_get_model()
                    if _timeout_expired(deadline):
                        print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                        return S
                    partial_time = time.perf_counter() - start_time
                    start_time = time.perf_counter()

                    if status == "SAT":
                        save_sat_model_typed(
                            solver_result,
                            model,
                            free_vars,
                            None,
                            status,
                            mask_var_pairs,
                            partial_time,
                            L,
                        )
                        solver.add_blocking_clause(model, L)
                        S.add(frozenset(model.items()))
                        continue

                    if status in ("UNSAT", "UNKNOWN"):
                        complemented_masks = MaskGenerator.complement_mask_var_pairs(mask_var_pairs, L)
                        solver.reset_iteration(assignment)
                        solver.apply_masks(complemented_masks, L)

                        status, model = solver.check_and_get_model()
                        if _timeout_expired(deadline):
                            print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                            return S
                        partial_time = time.perf_counter() - start_time
                        start_time = time.perf_counter()

                        if status in ("UNSAT", "UNKNOWN"):
                            save_model_typed(
                                solver_result,
                                model,
                                free_vars,
                                None,
                                status,
                                assignment,
                                complemented_masks,
                                partial_time,
                                L,
                            )
                        elif status == "SAT":
                            save_sat_model_typed(
                                solver_result,
                                model,
                                free_vars,
                                None,
                                status,
                                complemented_masks,
                                partial_time,
                                L,
                            )
                            solver.add_blocking_clause(model, L)
                            S.add(frozenset(model.items()))
                        continue

            if not E and not L:
                if _timeout_expired(deadline):
                    print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                    return S
                solver.reset_iteration(None)
                status, model = solver.check_and_get_model()
                partial_time = time.perf_counter() - start_time
                if status in ("UNSAT", "UNKNOWN"):
                    save_model_typed(solver_result, model, free_vars, None, status, None, [], partial_time, L)
                elif status == "SAT":
                    save_sat_model_typed(solver_result, model, free_vars, None, status, [], partial_time, L)
                    S.add(frozenset(model.items()))
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt ricevuto: salvataggio dei risultati parziali in corso...")
    finally:
        _write_solver_results_typed(
            solver_result,
            smt_filepath,
            coverage,
            string_domain["max_len"],
            results_suffix=results_suffix,
            output_filename=output_filename,
        )

    if interrupted:
        raise KeyboardInterrupt

    return S


def intervalGuidedSampler(
    smt_filepath,
    free_vars=None,
    theta=256,
    coverage=50,
    delta=0.0050,
    use_recursive_string_helpers=False,
    string_order=DEFAULT_STRING_ORDER,
    U=2**256 - 1,
    timeout=None,
    output_filename=None,
    results_suffix="_interval_guided_sampler",
    threads=None,
    num_samples=None,
    results_root=None,
):
    """
    Typed sampler that always restarts from a fresh mask set after each save.

    Behaviour:
    - generate masks, run the solver
    - if SAT: save result, then restart from the original SMT and build new masks
    - if UNSAT/UNKNOWN: save result, then restart from the original SMT and build new masks

    Unlike sampler_typed_always_new_mask, this variant never applies any
    complement strategy.
    """
    engine = SamplerEngine.from_config(
        smt_filepath=smt_filepath,
        free_vars=free_vars,
        theta=theta,
        coverage=coverage,
        delta=delta,
        use_recursive_string_helpers=use_recursive_string_helpers,
        string_order=string_order,
        U=U,
        timeout=timeout,
        output_filename=output_filename,
        results_suffix=results_suffix,
        threads=threads,
        num_samples=num_samples,
        results_root=results_root,
    )
    return engine.run()


def z3_baseline_sampler(
    smt_filepath,
    free_vars=None,
    max_samples=None,
    string_max_length=54,
    output_filename=None,
    timeout=None,
):
    """
    Baseline Z3 sampler.

    Repeatedly invokes Z3 on the same formula and adds a blocking clause after each
    SAT result so the next iteration cannot return the same model again.
    """
    timeout_seconds = _validate_timeout(timeout)
    string_domain = _get_string_domain_config(string_max_length)
    if max_samples is not None:
        if not isinstance(max_samples, int):
            raise TypeError("max_samples must be an integer or None.")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when provided.")

    S = set()
    solver_result = {}
    solver = Z3Solver(smt_filepath=smt_filepath, string_domain=string_domain)
    if free_vars is None:
        free_vars = DomainPartitioner.extract_free_vars(solver.get_base_content())
    print(f"Extracted free variables: {free_vars}")

    baseline_csv_path = _baseline_result_csv_path(smt_filepath, results_suffix="_z3_baseline_call_value")
    baseline_csv_path.parent.mkdir(parents=True, exist_ok=True)

    sat_count = 0
    interrupted = False

    try:
        start_time = time.perf_counter()
        deadline = None if timeout_seconds is None else start_time + timeout_seconds
        while max_samples is None or sat_count < max_samples:
            if _timeout_expired(deadline):
                print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                break
            remaining_ms = None
            if deadline is not None:
                remaining_seconds = max(0.0, deadline - time.perf_counter())
                if remaining_seconds <= 0:
                    print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                    break
                remaining_ms = max(1, int(remaining_seconds * 1000))

            status, model = solver.check_and_get_model(timeout_ms=remaining_ms)
            partial_time = time.perf_counter() - start_time
            start_time = time.perf_counter()

            if status in ("UNSAT", "UNKNOWN"):
                save_model_typed(
                    solver_result,
                    model,
                    free_vars,
                    None,
                    status,
                    None,
                    [],
                    partial_time,
                )
                break

            if status == "SAT":
                save_sat_model_typed(
                    solver_result,
                    model,
                    free_vars,
                    None,
                    status,
                    [],
                    partial_time,
                )
                _append_solver_result_row_typed(solver_result, baseline_csv_path)
                solver.add_blocking_clause_for_free_vars(model, free_vars)
                S.add(frozenset(model.items()))
                sat_count += 1
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt ricevuto: salvataggio dei risultati parziali in corso...")
    finally:
        _write_baseline_solver_results_typed(
            solver_result,
            smt_filepath,
            results_suffix="_z3_baseline_call_value",
            output_filename=output_filename,
        )

    if interrupted:
        raise KeyboardInterrupt

    return S


def _reset_typed_tmp_formula(smt_filepath, tmp_filepath, assignment=None):
    shutil.copy2(smt_filepath, tmp_filepath)
    if assignment is not None:
        add_e_assignment_typed(assignment, tmp_filepath)


def _generate_and_apply_typed_masks(
    L,
    tmp_filepath,
    coverage,
    delta,
    use_recursive_string_helpers,
    string_domain,
):
    mask_var_pairs = []
    for (var_name, var_type, lb, ub) in tqdm(L, desc="Generating masks..."):
        if not _is_mask_supported(var_type):
            continue

        mask = _create_mask_for_type(var_type, lb, ub, coverage, delta, string_domain=string_domain)
        mask_var_pairs.append((var_name, mask))
        add_mask_typed(
            mask,
            var_name,
            var_type,
            tmp_filepath,
            use_recursive_string_helpers=use_recursive_string_helpers,
            string_domain=string_domain,
        )

    return mask_var_pairs


def _build_typed_mask_entry(args):
    var_name, var_type, lb, ub, coverage, delta, string_domain = args
    if not _is_mask_supported(var_type):
        return None

    mask = _create_mask_for_type(var_type, lb, ub, coverage, delta, string_domain=string_domain)
    return var_name, var_type, mask


def _generate_and_apply_typed_masks_parallel(
    L,
    tmp_filepath,
    coverage,
    delta,
    use_recursive_string_helpers,
    string_domain,
    max_workers=None,
    solver=None,
):
    """
    Parallel two-phase variant:
    1. compute all masks in parallel
    2. apply them to the in-memory solver (or SMT file if legacy) sequentially in input order
    """
    supported_entries = [
        (var_name, var_type, lb, ub)
        for (var_name, var_type, lb, ub) in L
        if _is_mask_supported(var_type)
    ]
    if not supported_entries:
        return []

    task_args = [
        (var_name, var_type, lb, ub, coverage, delta, string_domain)
        for (var_name, var_type, lb, ub) in supported_entries
    ]

    if max_workers in (None, 1):
        computed_masks = [
            _build_typed_mask_entry(task_arg)
            for task_arg in tqdm(task_args, total=len(task_args), desc="Generating masks...")
        ]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            computed_masks = list(
                tqdm(
                    executor.map(_build_typed_mask_entry, task_args),
                    total=len(task_args),
                    desc="Generating masks...",
                )
            )

    mask_var_pairs = []
    for entry, computed in zip(supported_entries, computed_masks):
        if computed is None:
            continue
        var_name, var_type, _, _ = entry
        _, _, mask = computed
        mask_var_pairs.append((var_name, mask))
        if tmp_filepath is not None:
            add_mask_typed(
                mask,
                var_name,
                var_type,
                tmp_filepath,
                use_recursive_string_helpers=use_recursive_string_helpers,
                string_domain=string_domain,
            )

    if solver is not None:
        solver.apply_masks(mask_var_pairs, L)

    return mask_var_pairs


def _hybrid_state_space_sampler_typed_impl(
    smt_filepath,
    free_vars=None,
    theta=256,
    coverage=50,
    delta=0.0050,
    use_recursive_string_helpers=True,
    string_max_length=54,
    complement_on_unsat=False,
    results_suffix="",
    removal_mode=REMOVAL_MODE_ONE,
    chunk_size=None,
    timeout=None,
    output_filename=None,
):
    _validate_removal_mode(removal_mode)
    timeout_seconds = _validate_timeout(timeout)
    string_domain = _get_string_domain_config(string_max_length)
    S = set()
    E = set()
    L = set()
    solver_result = {}
    n = 2**256
    if(free_vars is None):
        free_vars = extract_free_vars_with_type_from_smt2_file(smt_filepath)
    print(f"Extracted free variables: {free_vars}")
    tmp_dir = Path(__file__).parent.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_filepath = tmp_dir / f"tmp_{uuid.uuid4().hex}.smt2"
    shutil.copy2(smt_filepath, tmp_filepath)

    interrupted = False

    try:
        start_time = time.perf_counter()
        deadline = None if timeout_seconds is None else start_time + timeout_seconds
        for free_var_name, free_var_type in free_vars:
            if _timeout_expired(deadline):
                print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                return S
            partition_variable_typed(
                E,
                L,
                smt_filepath,
                free_var_name,
                free_var_type,
                theta=theta,
                string_domain=string_domain,
            )

        if E:
            assignments = get_assignment_for_typed_E(E, limit=theta)

            for assignment in assignments:
                if _timeout_expired(deadline):
                    print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                    return S
                add_e_assignment_typed(assignment, tmp_filepath)
                mask_var_pairs = []

                if L:
                    for (var_name, var_type, lb, ub) in tqdm(L, desc="Generating masks..."):
                        if _timeout_expired(deadline):
                            print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                            return S
                        if not _is_mask_supported(var_type):
                            continue

                        mask = _create_mask_for_type(var_type, lb, ub, coverage, delta, string_domain=string_domain)
                        mask_var_pairs.append((var_name, mask))
                        n = min(n, _mask_iteration_count(mask, var_type))
                        add_mask_typed(
                            mask,
                            var_name,
                            var_type,
                            tmp_filepath,
                            use_recursive_string_helpers=use_recursive_string_helpers,
                            string_domain=string_domain,
                        )

                    if mask_var_pairs:
                        start_time = _run_masked_solver_loop_typed(
                            S,
                            solver_result,
                            free_vars,
                            L,
                            tmp_filepath,
                            mask_var_pairs,
                            n,
                            start_time,
                            assignment=assignment,
                            use_recursive_string_helpers=use_recursive_string_helpers,
                            string_domain=string_domain,
                            complement_on_unsat=complement_on_unsat,
                            removal_mode=removal_mode,
                            chunk_size=chunk_size,
                            deadline=deadline,
                        )
                        if _timeout_expired(deadline):
                            print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                            return S
                    else:
                        status, model = run_z3_and_get_model(tmp_filepath)
                        if _timeout_expired(deadline):
                            print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                            return S
                        partial_time = time.perf_counter() - start_time
                        start_time = time.perf_counter()
                        if status in ("UNSAT", "UNKNOWN"):
                            save_model_typed(
                                solver_result,
                                model,
                                free_vars,
                                tmp_filepath,
                                status,
                                assignment,
                                mask_var_pairs,
                                partial_time,
                                L,
                            )
                        elif status == "SAT":
                            save_sat_model_typed(
                                solver_result,
                                model,
                                free_vars,
                                tmp_filepath,
                                status,
                                mask_var_pairs,
                                partial_time,
                                L,
                            )
                            S.add(frozenset(model.items()))
                else:
                    if _timeout_expired(deadline):
                        print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                        return S
                    status, model = run_z3_and_get_model(tmp_filepath)
                    partial_time = time.perf_counter() - start_time
                    start_time = time.perf_counter()
                    if status in ("UNSAT", "UNKNOWN"):
                        save_model_typed(
                            solver_result,
                            model,
                            free_vars,
                            tmp_filepath,
                            status,
                            assignment,
                            mask_var_pairs,
                            partial_time,
                            L,
                        )
                    elif status == "SAT":
                        save_sat_model_typed(
                            solver_result,
                            model,
                            free_vars,
                            tmp_filepath,
                            status,
                            mask_var_pairs,
                            partial_time,
                            L,
                        )
                        S.add(frozenset(model.items()))
        elif L:
            mask_var_pairs = []
            for (var_name, var_type, lb, ub) in tqdm(L, desc="Generating masks..."):
                if _timeout_expired(deadline):
                    print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                    return S
                if not _is_mask_supported(var_type):
                    continue
                mask = _create_mask_for_type(var_type, lb, ub, coverage, delta, string_domain=string_domain)
                mask_var_pairs.append((var_name, mask))
                n = min(n, _mask_iteration_count(mask, var_type))
                add_mask_typed(
                    mask,
                    var_name,
                    var_type,
                    tmp_filepath,
                    use_recursive_string_helpers=use_recursive_string_helpers,
                    string_domain=string_domain,
                )

            if mask_var_pairs:
                start_time = _run_masked_solver_loop_typed(
                    S,
                    solver_result,
                    free_vars,
                    L,
                    tmp_filepath,
                    mask_var_pairs,
                    n,
                    start_time,
                    assignment=None,
                    use_recursive_string_helpers=use_recursive_string_helpers,
                    string_domain=string_domain,
                    complement_on_unsat=complement_on_unsat,
                    removal_mode=removal_mode,
                    chunk_size=chunk_size,
                    deadline=deadline,
                )
                if _timeout_expired(deadline):
                    print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                    return S
            else:
                if _timeout_expired(deadline):
                    print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                    return S
                status, model = run_z3_and_get_model(tmp_filepath)
                if _timeout_expired(deadline):
                    print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                    return S
                partial_time = time.perf_counter() - start_time
                if status in ("UNSAT", "UNKNOWN"):
                    save_model_typed(solver_result, model, free_vars, tmp_filepath, status, None, [], partial_time, L)
                elif status == "SAT":
                    save_sat_model_typed(solver_result, model, free_vars, tmp_filepath, status, [], partial_time, L)
                    S.add(frozenset(model.items()))
        else:
            # No partitionable variables: still collect one solver answer.
            if _timeout_expired(deadline):
                print(f"[INFO] Timeout reached after {timeout_seconds} seconds, saving CSV and stopping.")
                return S
            status, model = run_z3_and_get_model(tmp_filepath)
            partial_time = time.perf_counter() - start_time
            if status in ("UNSAT", "UNKNOWN"):
                save_model_typed(solver_result, model, free_vars, tmp_filepath, status, None, [], partial_time, L)
            elif status == "SAT":
                save_sat_model_typed(solver_result, model, free_vars, tmp_filepath, status, [], partial_time, L)
                S.add(frozenset(model.items()))
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt ricevuto: salvataggio dei risultati parziali in corso...")
    finally:
        _write_solver_results_typed(
            solver_result,
            smt_filepath,
            coverage,
            string_domain["max_len"],
            results_suffix=results_suffix,
            output_filename=output_filename,
        )

    if interrupted:
        raise KeyboardInterrupt

    return S


def _run_masked_solver_loop_typed(
    S,
    solver_result,
    free_vars,
    L,
    tmp_filepath,
    mask_var_pairs,
    iteration_budget,
    start_time,
    assignment,
    use_recursive_string_helpers,
    string_domain,
    complement_on_unsat=False,
    removal_mode=REMOVAL_MODE_ONE,
    chunk_size=None,
    deadline=None,
):
    complement_applied = False
    remaining_iterations = max(1, iteration_budget)
    complemented_variants = []

    while remaining_iterations > 0:
        if _timeout_expired(deadline):
            return start_time
        status, model = run_z3_and_get_model(tmp_filepath)
        if _timeout_expired(deadline):
            return start_time
        partial_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        remaining_iterations -= 1

        if status in ("UNSAT", "UNKNOWN"):
            current_masks_snapshot = [(var_name, list(mask)) for var_name, mask in mask_var_pairs]
            save_model_typed(
                solver_result,
                model,
                free_vars,
                tmp_filepath,
                status,
                assignment,
                current_masks_snapshot,
                partial_time,
                L,
            )

            if complement_on_unsat and not complement_applied:
                complemented_masks = complement_mask_var_pairs_typed(
                    mask_var_pairs,
                    L,
                )
                if complemented_masks:
                    complemented_variants = build_complement_mask_variants_typed(
                        complemented_masks,
                        L,
                        removal_mode=removal_mode,
                        chunk_size=chunk_size,
                    )
                    print(
                        "[INFO] Uniform complemented variants after "
                        f"{removal_mode} removal: {len(complemented_variants)}"
                    )
                    if complemented_variants:
                        first_variant = complemented_variants.pop(random.randrange(len(complemented_variants)))
                        apply_mask_var_pairs_typed(
                            first_variant,
                            L,
                            tmp_filepath,
                            use_recursive_string_helpers=use_recursive_string_helpers,
                            string_domain=string_domain,
                        )
                        mask_var_pairs[:] = first_variant
                        complement_applied = True
                        remaining_iterations = max(
                            remaining_iterations,
                            _mask_iteration_budget_from_pairs(first_variant, L) + len(complemented_variants),
                        )
                        continue

            if complement_applied and complemented_variants:
                next_variant = complemented_variants.pop(random.randrange(len(complemented_variants)))
                apply_mask_var_pairs_typed(
                    next_variant,
                    L,
                    tmp_filepath,
                    use_recursive_string_helpers=use_recursive_string_helpers,
                    string_domain=string_domain,
                )
                mask_var_pairs[:] = next_variant
                remaining_iterations = max(
                    remaining_iterations,
                    _mask_iteration_budget_from_pairs(next_variant, L) + len(complemented_variants),
                )
                continue

            return start_time

        if status == "SAT":
            save_sat_model_typed(
                solver_result,
                model,
                free_vars,
                tmp_filepath,
                status,
                mask_var_pairs,
                partial_time,
                L,
            )
            remove_used_mask_typed(model, L, tmp_filepath, mask_var_pairs, string_domain=string_domain)
            prevent_same_model_typed(model, L, tmp_filepath)
            S.add(frozenset(model.items()))

    return start_time


def _write_solver_results_typed(
    solver_result,
    smt_filepath,
    coverage,
    string_max_length,
    results_suffix="",
    output_filename=None,
    include_string_length_suffix=True,
):
    df = pd.DataFrame(solver_result)
    benchmark_dir = Path(smt_filepath).parent
    results_dir_name = benchmark_dir.name + "_results" + results_suffix + "_cov" + str(coverage)
    if include_string_length_suffix:
        results_dir_name += "_strlen" + str(string_max_length)
    results_dir = benchmark_dir.parent / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    if output_filename is None:
        csv_filename = Path(smt_filepath).stem + ".csv"
    else:
        csv_filename = Path(output_filename)
        if csv_filename.suffix.lower() != ".csv":
            csv_filename = csv_filename.with_suffix(".csv")
    df.to_csv(results_dir / csv_filename, index=False)


def _write_baseline_solver_results_typed(
    solver_result,
    smt_filepath,
    results_suffix="",
    output_filename=None,
):
    df = pd.DataFrame(solver_result)
    benchmark_dir = Path(smt_filepath).parent
    results_dir = benchmark_dir.parent / (benchmark_dir.name + "_results" + results_suffix)
    results_dir.mkdir(parents=True, exist_ok=True)
    if output_filename is None:
        csv_filename = Path(smt_filepath).stem + ".csv"
    else:
        csv_filename = Path(output_filename)
        if csv_filename.suffix.lower() != ".csv":
            csv_filename = csv_filename.with_suffix(".csv")
    df.to_csv(results_dir / csv_filename, index=False)


def _baseline_result_csv_path(smt_filepath, results_suffix=""):
    benchmark_dir = Path(smt_filepath).parent
    results_dir = benchmark_dir.parent / (benchmark_dir.name + "_results" + results_suffix)
    return results_dir / Path(smt_filepath).with_suffix(".csv").name


def _append_solver_result_row_typed(
    solver_result,
    csv_filepath,
):
    ResultRecorder.append_solver_result_row_typed(solver_result, csv_filepath)


def _count_existing_sat_rows(csv_filepath):
    return ResultRecorder.count_existing_sat_rows(csv_filepath)


def _mask_bounds_from_pairs(mask_var_pairs, L):
    return ResultRecorder.mask_bounds_from_pairs(mask_var_pairs, L)


def save_model_typed(solver_result, model, free_vars, tmp_filepath, status, assignment, mask_var_pairs, partial_time, L=None):
    ResultRecorder.save_model_typed(
        solver_result,
        model,
        free_vars,
        tmp_filepath,
        status,
        assignment,
        mask_var_pairs,
        partial_time,
        L,
    )


def save_sat_model_typed(result, model_dict, free_vars, tmp_filepath, status, mask_var_pairs, partial_time, L=None):
    return ResultRecorder.save_sat_model_typed(
        result,
        model_dict,
        free_vars,
        tmp_filepath,
        status,
        mask_var_pairs,
        partial_time,
        L,
    )


def get_assignment_for_typed_E(E, limit=None):
    return SamplerEngine.get_assignment_for_typed_E(E, limit=limit)


def _sample_distinct_indices(total, limit):
    return SamplerEngine.sample_distinct_indices(total, limit)


def add_e_assignment_typed(assignment, file_path):
    with open(file_path, "r") as f:
        content = f.read()

    content = re.sub(
        r'\(assert\s+\(and\s+(?:\(=\s+[\w@]+\s+.+?\)\s*)+\)\)\n?',
        '',
        content,
        flags=re.DOTALL,
    )

    clauses = [
        f"(= {var_name} {_format_literal(data['value'], data['type'])})"
        for var_name, data in assignment.items()
    ]
    assignment_assert = "(assert (and\n    " + "\n    ".join(clauses) + "\n))\n"

    lines = content.splitlines(keepends=True)
    first_checksat_index = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("(check-sat")),
        None,
    )

    if first_checksat_index is None:
        content += assignment_assert
    else:
        lines.insert(first_checksat_index, assignment_assert)
        content = "".join(lines)

    with open(file_path, "w") as f:
        f.write(content)


def partition_variable_typed(E, L, file_path, free_var_name, free_var_type, theta=1000, string_domain=None, U=2**256 - 1):
    partitioner = DomainPartitioner(theta=theta, U=U, string_domain=string_domain)
    if free_var_type in ("Bool", "String"):
        partitioner.partition_variable_from_content(E, L, "", free_var_name, free_var_type)
        return
    with open(file_path, "r") as f:
        content = f.read()
    partitioner.partition_variable_from_content(E, L, content, free_var_name, free_var_type)


def _extract_int_bounds(file_path, free_var_name, U=2**256 - 1):
    with open(file_path, "r") as f:
        content = f.read()
    return DomainPartitioner.extract_int_bounds_from_content(content, free_var_name, U)


def _extract_bitvec_bounds(file_path, free_var_name, bitwidth):
    with open(file_path, "r") as f:
        content = f.read()
    return DomainPartitioner.extract_bitvec_bounds_from_content(content, free_var_name, bitwidth)


def _format_literal(value, var_type):
    return format_smt_literal(value, var_type)


def _is_mask_supported(var_type):
    return MaskGenerator.is_mask_supported(var_type)


def _create_mask_for_type(var_type, lb, ub, coverage, delta, string_domain=None):
    return MaskGenerator.create_mask_for_type(
        var_type,
        lb,
        ub,
        coverage,
        delta,
        string_domain=string_domain,
    )


def _mask_iteration_count(mask, var_type):
    # if var_type == "String":
    #     return sum(len(group) for group in mask)
    return len(mask)


def _mask_iteration_budget_from_pairs(mask_var_pairs, L):
    type_by_var = {var_name: var_type for var_name, var_type, _, _ in L}
    counts = []
    for var_name, mask in mask_var_pairs:
        var_type = type_by_var.get(var_name)
        if var_type is None or not mask:
            continue
        counts.append(_mask_iteration_count(mask, var_type))
    return min(counts) if counts else 0


def add_mask_typed(mask, free_var_name, free_var_type, file_path, use_recursive_string_helpers=True, string_domain=None):
    string_domain = string_domain or _get_string_domain_config(54)
    if free_var_type == "String":
        _add_string_mask(
            mask,
            free_var_name,
            file_path,
            use_recursive_string_helpers=use_recursive_string_helpers,
            string_domain=string_domain,
        )
        return

    shuffled_mask = list(mask)
    random.shuffle(shuffled_mask)

    for band_lb, band_ub in shuffled_mask:
        _add_numeric_band(band_lb, band_ub, free_var_name, file_path)


def _add_numeric_band(lb_band, ub_band, free_var_name, file_path):
    new_clause = f"(and (>= {free_var_name} {lb_band}) (< {free_var_name} {ub_band}))"

    with open(file_path, "r") as f:
        content = f.read()

    pattern = re.compile(
        rf'\(assert\s+\(or\s+((?:\(and\s+\(>=\s+{re.escape(free_var_name)}\s+-?\d+\)\s+\(<\s+{re.escape(free_var_name)}\s+-?\d+\)\)\s*)+)\)\)',
        re.DOTALL,
    )

    match = pattern.search(content)
    if match:
        existing_clauses = match.group(1).strip()
        new_assert = f"(assert (or\n  {existing_clauses}\n  {new_clause}\n))"
        content = content[:match.start()] + new_assert + content[match.end():]
    else:
        new_assert = f"(assert (or\n  {new_clause}\n))"
        content = _insert_before_first_checksat(content, new_assert + "\n")

    with open(file_path, "w") as f:
        f.write(content)


def replace_mask_typed(mask, free_var_name, free_var_type, file_path, use_recursive_string_helpers=True, string_domain=None):
    string_domain = string_domain or _get_string_domain_config(54)
    with open(file_path, "r") as f:
        content = f.read()

    content = _clear_mask_for_var_typed(content, free_var_name, free_var_type)

    with open(file_path, "w") as f:
        f.write(content)

    if mask:
        add_mask_typed(
            mask,
            free_var_name,
            free_var_type,
            file_path,
            use_recursive_string_helpers=use_recursive_string_helpers,
            string_domain=string_domain,
        )


def apply_mask_var_pairs_typed(mask_var_pairs, L, file_path, use_recursive_string_helpers=True, string_domain=None):
    string_domain = string_domain or _get_string_domain_config(54)
    type_by_var = {var_name: var_type for var_name, var_type, _, _ in L}

    for var_name, var_type, _, _ in L:
        if _is_mask_supported(var_type):
            replace_mask_typed(
                [],
                var_name,
                var_type,
                file_path,
                use_recursive_string_helpers=use_recursive_string_helpers,
                string_domain=string_domain,
            )

    for var_name, mask in mask_var_pairs:
        var_type = type_by_var.get(var_name)
        if var_type is None:
            continue
        replace_mask_typed(
            mask,
            var_name,
            var_type,
            file_path,
            use_recursive_string_helpers=use_recursive_string_helpers,
            string_domain=string_domain,
        )


def _clear_mask_for_var_typed(content, free_var_name, free_var_type):
    if free_var_type == "String":
        string_block_pattern = re.compile(
            rf'{re.escape(STRING_MASK_START_TEMPLATE.format(var_name=free_var_name))}\n'
            rf'.*?'
            rf'{re.escape(STRING_MASK_END_TEMPLATE.format(var_name=free_var_name))}\n?',
            re.DOTALL,
        )
        return string_block_pattern.sub("", content)

    numeric_block_pattern = re.compile(
        rf'\(assert\s+\(or\s+(?:\s*\(and\s+\(>=\s+{re.escape(free_var_name)}\s+-?\d+\)\s+\(<\s+{re.escape(free_var_name)}\s+-?\d+\)\))+\s*\)\)\n?',
        re.DOTALL,
    )
    return numeric_block_pattern.sub("", content)


def _add_string_mask(mask, free_var_name, file_path, use_recursive_string_helpers=True, string_domain=None):
    string_domain = string_domain or _get_string_domain_config(54)
    with open(file_path, "r") as f:
        content = f.read()

    content = _ensure_string_to_index_helpers(
        content,
        use_recursive_string_helpers=use_recursive_string_helpers,
        string_domain=string_domain,
    )

    interval_clauses = [
        f"(and (>= (stringToIndex {free_var_name}) {band_lb}) (< (stringToIndex {free_var_name}) {band_ub}))"
        for band_lb, band_ub in mask
    ]
    random.shuffle(interval_clauses)

    if not interval_clauses:
        return

    start_marker = STRING_MASK_START_TEMPLATE.format(var_name=free_var_name)
    end_marker = STRING_MASK_END_TEMPLATE.format(var_name=free_var_name)
    new_assert = (
        f"{start_marker}\n"
        "(assert (and\n"
        f"  (>= (str.len {free_var_name}) 1)\n"
        f"  (<= (str.len {free_var_name}) {string_domain['max_len']})\n"
        f"  (isLowerAlphaString {free_var_name})\n"
        "  (or\n"
        "    " + "\n    ".join(interval_clauses) + "\n"
        "  )\n"
        "))\n"
        f"{end_marker}\n"
    )
    content = _insert_before_first_checksat(content, new_assert)

    with open(file_path, "w") as f:
        f.write(content)


def _add_legacy_string_mask(mask, free_var_name, file_path):
    with open(file_path, "r") as f:
        content = f.read()

    interval_blocks = []
    for group in mask:
        if not group:
            continue
        new_clauses = "\n    ".join(
            f'(= {free_var_name} "{_escape_smt_string(value)}")'
            for value in group
        )
        interval_blocks.append(f"(or\n    {new_clauses}\n  )")

    if not interval_blocks:
        return

    new_assert = "(assert (or\n  " + "\n  ".join(interval_blocks) + "\n))\n"
    content = _insert_before_first_checksat(content, new_assert)

    with open(file_path, "w") as f:
        f.write(content)


def remove_used_mask_typed(model, L, file_path, mask_var_pairs, string_domain=None):
    string_domain = string_domain or _get_string_domain_config(54)
    with open(file_path, "r") as f:
        content = f.read()

    l_info = {var_name: var_type for var_name, var_type, _, _ in L}
    mask_index = {var_name: i for i, (var_name, _) in enumerate(mask_var_pairs)}

    for var_name, z3_value in model.items():
        if var_name not in l_info:
            continue

        var_type = l_info[var_name]
        if var_type == "String":
            value = str(z3_value)
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            if var_name in mask_index:
                idx = mask_index[var_name]
                current_var, current_mask = mask_var_pairs[idx]
                if current_mask and isinstance(current_mask[0], tuple):
                    string_index = _string_to_index(value, string_domain=string_domain)
                    content = _remove_string_band_from_mask(content, var_name, string_index, string_domain=string_domain)
                    mask_var_pairs[idx] = (
                        current_var,
                        [band for band in current_mask if not (band[0] <= string_index < band[1])],
                    )
                else:
                    content = _remove_legacy_string_interval_block_from_mask(content, var_name, value)
                    updated_groups = []
                    removed = False
                    for group in current_mask:
                        if not removed and value in group:
                            removed = True
                            continue
                        updated_groups.append(group)
                    mask_var_pairs[idx] = (
                        current_var,
                        updated_groups,
                    )
        elif _is_mask_supported(var_type):
            value = int(str(z3_value))
            content = _remove_numeric_value_from_mask(content, var_name, value, mask_index, mask_var_pairs)

    content = _cleanup_empty_mask_asserts(content)

    with open(file_path, "w") as f:
        f.write(content)


def _remove_numeric_value_from_mask(content, var_name, value, mask_index, mask_var_pairs):
    band_pattern = re.compile(
        rf'\s*\(and\s+\(>=\s+{re.escape(var_name)}\s+(-?\d+)\)\s+\(<\s+{re.escape(var_name)}\s+(-?\d+)\)\)'
    )
    or_block_pattern = re.compile(
        rf'\(assert\s+\(or\s+(?:\s*\(and\s+\(>=\s+{re.escape(var_name)}\s+-?\d+\)\s+\(<\s+{re.escape(var_name)}\s+-?\d+\)\))+\s*\)\)',
        re.MULTILINE,
    )

    def remove_band_from_block(or_block_match):
        block = or_block_match.group(0)

        def replace(match):
            lb = int(match.group(1))
            ub = int(match.group(2))
            if lb <= value < ub:
                if var_name in mask_index:
                    idx = mask_index[var_name]
                    current_var, current_mask = mask_var_pairs[idx]
                    mask_var_pairs[idx] = (current_var, [band for band in current_mask if band != (lb, ub)])
                return ""
            return match.group(0)

        return band_pattern.sub(replace, block)

    return or_block_pattern.sub(remove_band_from_block, content)


def _remove_string_band_from_mask(content, var_name, value, string_domain=None):
    string_domain = string_domain or _get_string_domain_config(54)
    block_pattern = re.compile(
        rf'{re.escape(STRING_MASK_START_TEMPLATE.format(var_name=var_name))}\n'
        rf'(?P<body>.*?)'
        rf'{re.escape(STRING_MASK_END_TEMPLATE.format(var_name=var_name))}\n?',
        re.DOTALL,
    )
    band_pattern = re.compile(
        rf'\s*\(and\s+\(>=\s+\(stringToIndex\s+{re.escape(var_name)}\)\s+(\d+)\)\s+\(<\s+\(stringToIndex\s+{re.escape(var_name)}\)\s+(\d+)\)\)\n?'
    )

    def replace_block(match):
        block = match.group("body")
        remaining_bands = []
        for band_match in band_pattern.finditer(block):
            lb = int(band_match.group(1))
            ub = int(band_match.group(2))
            if lb <= value < ub:
                continue
            remaining_bands.append((lb, ub))

        if not remaining_bands:
            return ""

        rebuilt_assert = (
            "(assert (and\n"
            f"  (>= (str.len {var_name}) 1)\n"
            f"  (<= (str.len {var_name}) {string_domain['max_len']})\n"
            f"  (isLowerAlphaString {var_name})\n"
            "  (or\n"
            "    " + "\n    ".join(
                f"(and (>= (stringToIndex {var_name}) {lb}) (< (stringToIndex {var_name}) {ub}))"
                for lb, ub in remaining_bands
            ) + "\n"
            "  )\n"
            "))\n"
        )
        return (
            STRING_MASK_START_TEMPLATE.format(var_name=var_name) + "\n" +
            rebuilt_assert +
            STRING_MASK_END_TEMPLATE.format(var_name=var_name) + "\n"
        )

    return block_pattern.sub(replace_block, content)


def _remove_legacy_string_interval_block_from_mask(content, var_name, value):
    interval_pattern = re.compile(
        rf'\(or\s+(?P<body>(?:\s*\(=\s+{re.escape(var_name)}\s+\"(?:[^\"\\\\]|\\\\.)*\"\))+)\s*\)',
        re.DOTALL,
    )
    value_pattern = re.compile(
        rf'\(=\s+{re.escape(var_name)}\s+\"((?:[^\"\\\\]|\\\\.)*)\"\)'
    )

    def replace_interval(match):
        body = match.group("body")
        values = [
            _unescape_smt_string(value_match.group(1))
            for value_match in value_pattern.finditer(body)
        ]
        if value in values:
            return ""
        return match.group(0)

    updated = interval_pattern.sub(replace_interval, content)
    updated = re.sub(r'\(assert\s+\(or\s*\)\)\s*', '', updated, flags=re.DOTALL)
    return updated


def complement_mask_typed(mask, domain_lb, domain_ub):
    return MaskGenerator.complement_mask(mask, domain_lb, domain_ub)


def complement_mask_var_pairs_typed(
    mask_var_pairs,
    L,
):
    return MaskGenerator.complement_mask_var_pairs(mask_var_pairs, L)


def build_complement_mask_variants_typed(mask_var_pairs, L, removal_mode=REMOVAL_MODE_ONE, chunk_size=None):
    return MaskGenerator.build_complement_mask_variants(
        mask_var_pairs, L, removal_mode=removal_mode, chunk_size=chunk_size
    )


def build_add_mask_variants_typed(
    original_mask_var_pairs,
    complemented_mask_var_pairs,
    L,
    addition_mode=REMOVAL_MODE_ONE,
    chunk_size=None,
):
    return MaskGenerator.build_add_mask_variants(
        original_mask_var_pairs,
        complemented_mask_var_pairs,
        L,
        addition_mode=addition_mode,
        chunk_size=chunk_size,
    )


def _validate_removal_mode(removal_mode):
    MaskGenerator.validate_removal_mode(removal_mode)


def _build_mask_removal_variants(mask, var_type, domain_lb, domain_ub, removal_mode, chunk_size=None):
    return MaskGenerator.build_mask_removal_variants(
        mask, var_type, domain_lb, domain_ub, removal_mode, chunk_size=chunk_size
    )


def _build_mask_addition_variants(base_mask, addition_candidates, var_type, domain_lb, domain_ub, addition_mode, chunk_size=None):
    return MaskGenerator.build_mask_addition_variants(
        base_mask, addition_candidates, var_type, domain_lb, domain_ub, addition_mode, chunk_size=chunk_size
    )


def _resolve_chunk_remove_count(mask_len, chunk_size):
    return MaskGenerator.resolve_chunk_remove_count(mask_len, chunk_size)


def _uniform_chunk_removal_indices(mask_len, remove_count):
    return MaskGenerator.uniform_chunk_indices(mask_len, remove_count)


def _uniform_chunk_addition_indices(mask_len, add_count):
    return MaskGenerator.uniform_chunk_indices(mask_len, add_count)


def _mask_is_uniform_typed(mask, var_type, domain_lb, domain_ub):
    return MaskGenerator.is_mask_uniform(mask, var_type, domain_lb, domain_ub)


def _normalize_mask_intervals(mask, domain_lb, domain_ub):
    return MaskGenerator.normalize_mask_intervals(mask, domain_lb, domain_ub)


def prevent_same_model_typed(model, L, file_path):
    with open(file_path, "r") as f:
        content = f.read()

    l_info = {var_name: var_type for var_name, var_type, _, _ in L}
    equalities = []
    for var_name, z3_value in model.items():
        if var_name not in l_info:
            continue
        var_type = l_info[var_name]
        literal = _format_literal(_normalize_model_value(z3_value, var_type), var_type)
        equalities.append(f"(= {var_name} {literal})")

    if not equalities:
        return

    blocking_clause = f"(assert (not (and {' '.join(equalities)})))\n"
    content = _insert_before_first_checksat(content, blocking_clause)

    with open(file_path, "w") as f:
        f.write(content)


def prevent_same_model_for_free_vars_typed(model, free_vars, file_path):
    with open(file_path, "r") as f:
        content = f.read()

    type_by_var = {var_name: var_type for var_name, var_type in free_vars}
    equalities = []
    for var_name, z3_value in model.items():
        var_type = type_by_var.get(var_name)
        if var_type is None:
            continue
        literal = _format_literal(_normalize_model_value(z3_value, var_type), var_type)
        equalities.append(f"(= {var_name} {literal})")

    if not equalities:
        return

    blocking_clause = f"(assert (not (and {' '.join(equalities)})))\n"
    content = _insert_before_first_checksat(content, blocking_clause)

    with open(file_path, "w") as f:
        f.write(content)


def _normalize_model_value(z3_value, var_type):
    return normalize_model_value(z3_value, var_type)


def _insert_before_first_checksat(content, insertion):
    lines = content.splitlines(keepends=True)
    first_checksat_index = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("(check-sat")),
        None,
    )

    if first_checksat_index is None:
        return content + insertion

    lines.insert(first_checksat_index, insertion)
    return "".join(lines)


def _cleanup_empty_mask_asserts(content):
    return re.sub(r"\(assert\s+\(or\s*\)\)\s*", "", content, flags=re.DOTALL)


def _create_string_interval_mask(lb, ub, coverage, delta, string_domain=None):
    return MaskGenerator.create_string_interval_mask(lb, ub, coverage, delta, string_domain=string_domain)


def _validate_string_order(string_order):
    return validate_string_order(string_order)


def _get_string_to_index_helpers(use_recursive_string_helpers=False, string_domain=None):
    return get_string_to_index_helpers(
        use_recursive_string_helpers=use_recursive_string_helpers,
        string_domain=string_domain,
    )


def _ensure_string_to_index_helpers(content, use_recursive_string_helpers=False, string_domain=None):
    if STRING_TO_INDEX_HELPERS_START in content and STRING_TO_INDEX_HELPERS_END in content:
        return content
    return _insert_before_first_checksat(content, _get_string_to_index_helpers(use_recursive_string_helpers, string_domain=string_domain))


def _string_to_index(value, string_domain=None):
    string_domain = string_domain or _get_string_domain_config(54)
    string_order = _validate_string_order(string_domain.get("order", DEFAULT_STRING_ORDER))
    if not isinstance(value, str):
        raise TypeError("Input must be a string.")
    if not value:
        raise ValueError("Input string must be non-empty.")
    if len(value) > string_domain["max_len"]:
        raise ValueError(f"String length exceeds the supported maximum ({string_domain['max_len']}).")
    if any(ch < "a" or ch > "z" for ch in value):
        raise ValueError("Input string must contain only lowercase letters a-z.")

    if string_order == STRING_ORDER_LEX:
        index = len(value) - 1
        for idx, ch in enumerate(value):
            remaining = len(value) - idx - 1
            subtree_size = sum(26**k for k in range(0, remaining + 1))
            index += (ord(ch) - ord("a")) * subtree_size
        return index

    index = sum(26**k for k in range(1, len(value)))
    offset = 0
    for ch in value:
        offset = offset * 26 + (ord(ch) - ord("a"))
    return index + offset


def _escape_smt_string(value):
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _unescape_smt_string(value):
    return bytes(value, "utf-8").decode("unicode_escape")


def _free_var_names(free_vars):
    return [var_name for var_name, _ in free_vars]


def _total_strings_up_to_length(max_len):
    return total_strings_up_to_length(max_len)


def _get_string_domain_config(string_max_length, string_order=DEFAULT_STRING_ORDER):
    return get_string_domain_config(string_max_length, string_order=string_order)
