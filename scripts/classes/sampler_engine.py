import itertools
import random
import time
from pathlib import Path
from scipy.stats import beta

from classes.domain_partitioner import (
    DEFAULT_STRING_ORDER,
    DomainPartitioner,
    MAX_STRING_MAX_LENGTH,
    get_string_domain_config,
)
from classes.mask_generator import MaskGenerator
from classes.result_recorder import ResultRecorder
from classes.smt_solver import SMTSolver, Z3Solver


class SamplerEngine:
    """
    Object-Oriented orchestrator for the interval-guided SMT sampler (InterSect).

    Coordinates:
    - DomainPartitioner: partitions free variables into E (enumerated) and L (masked)
    - MaskGenerator: generates interval/string masks with persistent worker pool
    - SMTSolver (Z3Solver): evaluates formulas and accumulates blocking clauses in RAM
    - ResultRecorder: records and flushes incremental CSV rows on disk
    """

    def __init__(
        self,
        solver: SMTSolver,
        partitioner: DomainPartitioner,
        mask_generator: MaskGenerator,
        recorder: ResultRecorder,
        free_vars: list,
        timeout_seconds: float | None = None,
        theta: int = 256,
        bayesian_min_iterations: int = 20,
        bayesian_min_viable_sat_rate: float = 0.53,
        bayesian_confidence: float = 0.95,
    ):
        self.solver = solver
        self.partitioner = partitioner
        self.mask_generator = mask_generator
        self.recorder = recorder
        self.free_vars = list(free_vars)
        self.timeout_seconds = self.validate_timeout(timeout_seconds)
        self.theta = theta
        self.bayesian_min_iterations = bayesian_min_iterations
        self.bayesian_min_viable_sat_rate = bayesian_min_viable_sat_rate
        self.bayesian_confidence = bayesian_confidence

    @classmethod
    def from_config(
        cls,
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
    ) -> "SamplerEngine":
        timeout_seconds = cls.validate_timeout(timeout)
        string_domain = get_string_domain_config(MAX_STRING_MAX_LENGTH, string_order=string_order)
        if num_samples is not None:
            if not isinstance(num_samples, int):
                raise TypeError("num_samples must be an integer or None.")
            if num_samples <= 0:
                raise ValueError("num_samples must be positive when provided.")

        solver = Z3Solver(
            smt_filepath=smt_filepath,
            string_domain=string_domain,
            use_recursive_string_helpers=use_recursive_string_helpers,
        )
        smt_content = solver.get_base_content()

        partitioner = DomainPartitioner(theta=theta, U=U, string_domain=string_domain)
        if free_vars is None:
            free_vars = partitioner.extract_free_vars(smt_content)
        print(f"Extracted free variables: {free_vars}")

        results_parent = Path(results_root) if results_root is not None else Path(smt_filepath).parent.parent
        results_dir = results_parent / (
            Path(smt_filepath).parent.name + "_results" + results_suffix + "_cov" + str(coverage)
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        if output_filename is None:
            csv_filename = Path(smt_filepath).stem + ".csv"
        else:
            csv_filename = Path(output_filename)
            if csv_filename.suffix.lower() != ".csv":
                csv_filename = csv_filename.with_suffix(".csv")
        csv_filepath = results_dir / csv_filename
        recorder = ResultRecorder(csv_filepath, free_vars, num_samples=num_samples)

        mask_generator = MaskGenerator(
            coverage=coverage,
            delta=delta,
            string_domain=string_domain,
            max_workers=threads,
        )

        return cls(
            solver=solver,
            partitioner=partitioner,
            mask_generator=mask_generator,
            recorder=recorder,
            free_vars=free_vars,
            timeout_seconds=timeout_seconds,
            theta=theta,
        )

    @staticmethod
    def validate_timeout(timeout):
        if timeout is None:
            return None
        if not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number of seconds or None.")
        if timeout < 0:
            raise ValueError("timeout must be >= 0.")
        return float(timeout)

    @staticmethod
    def timeout_expired(deadline):
        return deadline is not None and time.perf_counter() >= deadline

    @staticmethod
    def sample_distinct_indices(total, limit):
        if limit <= 0:
            return []
        if limit >= total:
            return range(total)

        sampled = set()
        while len(sampled) < limit:
            sampled.add(random.randrange(total))
        return sampled

    @classmethod
    def get_assignment_for_typed_E(cls, E, limit=None):
        ranges = []
        for var_name, var_type, values in E:
            ranges.append((var_name, var_type, values))

        total = 1
        for _, _, values in ranges:
            total *= len(values)
        print(f"Total assignments for E: {total}")

        if limit is not None and limit < total:
            sampled_indices = cls.sample_distinct_indices(total, limit)
            value_lists = [values for _, _, values in ranges]
            for index in sampled_indices:
                combo = []
                remaining = index
                for values in reversed(value_lists):
                    remaining, value_index = divmod(remaining, len(values))
                    combo.append(values[value_index])
                combo.reverse()
                yield {
                    var_name: {"type": var_type, "value": value}
                    for (var_name, var_type, _), value in zip(ranges, combo)
                }
            return

        for combo in itertools.product(*(values for _, _, values in ranges)):
            yield {
                var_name: {"type": var_type, "value": value}
                for (var_name, var_type, _), value in zip(ranges, combo)
            }

    @staticmethod
    def _update_bayesian_counts(status, successes, failures):
        if status == "SAT":
            successes += 1
        elif status in ("UNSAT", "UNKNOWN"):
            failures += 1
        return successes, failures

    def _should_stop_due_to_bayesian_test(self, successes, failures):
        total = successes + failures
        if total < self.bayesian_min_iterations:
            return False, 0.0

        prob_sat_below_min = float(
            beta.cdf(
                self.bayesian_min_viable_sat_rate,
                1 + successes,
                1 + failures,
            )
        )
        return prob_sat_below_min >= self.bayesian_confidence, prob_sat_below_min

    def run(self) -> set:
        if self.recorder.already_reached_target():
            return self.recorder.S

        smt_content = self.solver.get_base_content()
        interrupted = False

        try:
            start_time = time.perf_counter()
            deadline = None if self.timeout_seconds is None else start_time + self.timeout_seconds

            E, L, timed_out = self.partitioner.partition(smt_content, self.free_vars, deadline=deadline)
            if timed_out:
                print(f"[INFO] Timeout reached after {self.timeout_seconds} seconds, saving CSV and stopping.")
                return self.recorder.S

            assignment_iter = [None]
            if E:
                assignment_iter = self.get_assignment_for_typed_E(E, limit=self.theta)

            with self.mask_generator as mask_gen:
                for assignment in assignment_iter:
                    if self.timeout_expired(deadline):
                        print(f"[INFO] Timeout reached after {self.timeout_seconds} seconds, saving CSV and stopping.")
                        return self.recorder.S
                    if L:
                        bayesian_successes = 0
                        bayesian_failures = 0
                        bayesian_should_stop = False
                        while not self.timeout_expired(deadline) and not bayesian_should_stop:
                            self.solver.reset_iteration(assignment)
                            mask_var_pairs = mask_gen.generate_masks(L)
                            self.solver.apply_masks(mask_var_pairs, L)

                            if not mask_var_pairs:
                                status, model = self.solver.check_and_get_model()
                                partial_time = time.perf_counter() - start_time
                                start_time = time.perf_counter()
                                reached_num_samples = self.recorder.record_and_flush(
                                    status, model, assignment, [], partial_time, L
                                )
                                bayesian_successes, bayesian_failures = self._update_bayesian_counts(
                                    status,
                                    bayesian_successes,
                                    bayesian_failures,
                                )
                                if reached_num_samples:
                                    return self.recorder.S
                                break

                            status, model = self.solver.check_and_get_model()
                            partial_time = time.perf_counter() - start_time
                            start_time = time.perf_counter()

                            if status == "SAT":
                                reached_num_samples = self.recorder.record_and_flush(
                                    status,
                                    model,
                                    assignment,
                                    mask_var_pairs,
                                    partial_time,
                                    L,
                                )
                                bayesian_successes, bayesian_failures = self._update_bayesian_counts(
                                    status,
                                    bayesian_successes,
                                    bayesian_failures,
                                )
                                if reached_num_samples:
                                    return self.recorder.S
                                self.solver.add_blocking_clause(model, L)
                                bayesian_stop, bayesian_prob = self._should_stop_due_to_bayesian_test(
                                    bayesian_successes,
                                    bayesian_failures,
                                )
                                if bayesian_stop:
                                    total = bayesian_successes + bayesian_failures
                                    print(
                                        f"[INFO] Bayesian early stop after {total} iterations for current assignment: "
                                        f"P(SAT rate <= {self.bayesian_min_viable_sat_rate:.0%})={bayesian_prob:.1%} "
                                        f">= {self.bayesian_confidence:.0%}."
                                    )
                                    bayesian_should_stop = True
                                continue

                            if status in ("UNSAT", "UNKNOWN"):
                                self.recorder.record_and_flush(
                                    status, model, assignment, mask_var_pairs, partial_time, L
                                )
                                bayesian_successes, bayesian_failures = self._update_bayesian_counts(
                                    status,
                                    bayesian_successes,
                                    bayesian_failures,
                                )
                                bayesian_stop, bayesian_prob = self._should_stop_due_to_bayesian_test(
                                    bayesian_successes,
                                    bayesian_failures,
                                )
                                if bayesian_stop:
                                    total = bayesian_successes + bayesian_failures
                                    print(
                                        f"[INFO] Bayesian early stop after {total} iterations for current assignment: "
                                        f"P(SAT rate <= {self.bayesian_min_viable_sat_rate:.0%})={bayesian_prob:.1%} "
                                        f">= {self.bayesian_confidence:.0%}."
                                    )
                                    bayesian_should_stop = True
                                continue
                    else:
                        self.solver.reset_iteration(assignment)
                        status, model = self.solver.check_and_get_model()
                        partial_time = time.perf_counter() - start_time
                        start_time = time.perf_counter()

                        if status in ("UNSAT", "UNKNOWN"):
                            self.recorder.record_and_flush(status, model, assignment, [], partial_time, L)
                        elif status == "SAT":
                            if self.recorder.record_and_flush(status, model, assignment, [], partial_time, L):
                                return self.recorder.S

                if not E and not L:
                    if self.timeout_expired(deadline):
                        print(f"[INFO] Timeout reached after {self.timeout_seconds} seconds, saving CSV and stopping.")
                        return self.recorder.S
                    self.solver.reset_iteration(None)
                    status, model = self.solver.check_and_get_model()
                    partial_time = time.perf_counter() - start_time
                    self.recorder.record_and_flush(status, model, None, [], partial_time, L)
        except KeyboardInterrupt:
            interrupted = True
            print("\nKeyboardInterrupt ricevuto: salvataggio dei risultati parziali in corso...")
        finally:
            pass

        if interrupted:
            raise KeyboardInterrupt

        return self.recorder.S
