from concurrent.futures import ProcessPoolExecutor
import itertools
import os
import random
from scipy.stats import ks_2samp, mannwhitneyu

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from solver_utils import (
    create_mask_with_coverage,
    expand_multiset,
    multiset_from_intervals_2,
    uniform_multiset,
)
from classes.domain_partitioner import (
    BITVEC_RE,
    MAX_STRING_MAX_LENGTH,
    get_string_domain_config,
)

REMOVAL_MODE_ONE = "ONE"
REMOVAL_MODE_CHUNK = "CHUNK"


def _build_typed_mask_entry_worker(args):
    var_name, var_type, lb, ub, coverage, delta, string_domain = args
    if not MaskGenerator.is_mask_supported(var_type):
        return None

    mask = MaskGenerator.create_mask_for_type(
        var_type,
        lb,
        ub,
        coverage,
        delta,
        string_domain=string_domain,
    )
    return var_name, var_type, mask


class MaskGenerator:
    """
    Object-Oriented component responsible for generating statistically uniform
    interval masks for Large-domain (L) variables and managing a reusable
    ProcessPoolExecutor across sampling iterations.
    """

    def __init__(self, coverage=50, delta=0.0050, string_domain=None, max_workers=None):
        self.coverage = coverage
        self.delta = delta
        self.string_domain = string_domain or get_string_domain_config(MAX_STRING_MAX_LENGTH)
        self.max_workers = self._normalize_max_workers(max_workers)
        self._executor = None

    @staticmethod
    def _normalize_max_workers(max_workers):
        if max_workers in (None, 1):
            return None
        workers = int(max_workers)
        if workers <= 1:
            return None
        if os.name == "nt":
            return min(workers, 60)
        return workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    @staticmethod
    def is_mask_supported(var_type):
        return var_type in {"Int", "String"} or BITVEC_RE.match(var_type) is not None

    @staticmethod
    def create_string_interval_mask(lb, ub, coverage, delta, string_domain=None):
        return create_mask_with_coverage(lb, ub, target_coverage=coverage, delta=delta)

    @classmethod
    def create_mask_for_type(cls, var_type, lb, ub, coverage, delta, string_domain=None):
        string_domain = string_domain or get_string_domain_config(MAX_STRING_MAX_LENGTH)
        if var_type == "String":
            return cls.create_string_interval_mask(lb, ub, coverage, delta, string_domain=string_domain)
        return create_mask_with_coverage(lb, ub, target_coverage=coverage, delta=delta)

    @staticmethod
    def normalize_mask_intervals(mask, domain_lb, domain_ub):
        if not mask:
            return []

        clipped = []
        for start, end in sorted(mask, key=lambda interval: interval[0]):
            start = max(domain_lb, start)
            end = min(domain_ub, end)
            if start < end:
                clipped.append((start, end))

        if not clipped:
            return []

        merged = [clipped[0]]
        for start, end in clipped[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    @classmethod
    def is_mask_uniform(cls, mask, var_type, domain_lb, domain_ub):
        normalized_mask = cls.normalize_mask_intervals(mask, domain_lb, domain_ub)
        if not normalized_mask:
            return False

        domain_span = domain_ub - domain_lb
        if domain_span <= 0:
            return False

        real_multiset = multiset_from_intervals_2(normalized_mask, domain_lb, domain_ub)
        if not real_multiset:
            return False

        ideal_multiset = uniform_multiset(domain_lb, domain_ub, sum(real_multiset.values()))
        real_samples = expand_multiset(real_multiset)
        ideal_samples = expand_multiset(ideal_multiset)

        use_mwu = all((end - start) <= 1 for start, end in normalized_mask)
        if use_mwu:
            _, p_value = mannwhitneyu(real_samples, ideal_samples, alternative="two-sided")
            return p_value > 0.05

        ks_stat, p_value = ks_2samp(real_samples, ideal_samples)
        return ks_stat < 0.10 and p_value > 0.05

    @classmethod
    def complement_mask(cls, mask, domain_lb, domain_ub):
        if domain_lb >= domain_ub:
            return []

        normalized = cls.normalize_mask_intervals(mask, domain_lb, domain_ub)
        complement = []
        cursor = domain_lb

        for start, end in normalized:
            if cursor < start:
                complement.append((cursor, start))
            cursor = max(cursor, end)

        if cursor < domain_ub:
            complement.append((cursor, domain_ub))

        return complement

    @classmethod
    def complement_mask_var_pairs(cls, mask_var_pairs, L):
        bounds_by_var = {var_name: (var_type, lb, ub) for var_name, var_type, lb, ub in L}
        complemented_pairs = []

        for var_name, mask in mask_var_pairs:
            bounds = bounds_by_var.get(var_name)
            if bounds is None:
                continue

            var_type, lb, ub = bounds
            complemented_mask = cls.complement_mask(mask, lb, ub)
            if complemented_mask:
                complemented_pairs.append((var_name, complemented_mask))

        return complemented_pairs

    @staticmethod
    def validate_removal_mode(removal_mode):
        if removal_mode not in {REMOVAL_MODE_ONE, REMOVAL_MODE_CHUNK}:
            raise ValueError(
                f"Unsupported removal_mode={removal_mode!r}. "
                f"Expected {REMOVAL_MODE_ONE!r} or {REMOVAL_MODE_CHUNK!r}."
            )

    @staticmethod
    def resolve_chunk_remove_count(mask_len, chunk_size):
        if mask_len <= 1:
            return 0
        if chunk_size is None:
            return min(mask_len - 1, max(2, mask_len // 2))
        if not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer or None.")
        if chunk_size < 2:
            raise ValueError("chunk_size must be at least 2 when removal_mode='CHUNK'.")
        return min(mask_len - 1, chunk_size)

    @staticmethod
    def uniform_chunk_indices(mask_len, count):
        if count <= 0:
            return set()
        if count >= mask_len:
            return set(range(mask_len))
        return set(random.sample(range(mask_len), count))

    @classmethod
    def build_mask_removal_variants(
        cls, mask, var_type, domain_lb, domain_ub, removal_mode, chunk_size=None
    ):
        if len(mask) <= 1:
            return []

        if removal_mode == REMOVAL_MODE_ONE:
            candidates = [
                [interval for index, interval in enumerate(mask) if index != removed_index]
                for removed_index in range(len(mask))
            ]
            return [
                candidate
                for candidate in candidates
                if cls.is_mask_uniform(candidate, var_type, domain_lb, domain_ub)
            ]

        remove_count = cls.resolve_chunk_remove_count(len(mask), chunk_size)
        variants = []
        seen = set()
        for _ in range(len(mask)):
            removal_indices = cls.uniform_chunk_indices(len(mask), remove_count)
            reduced_mask = [
                interval for index, interval in enumerate(mask)
                if index not in removal_indices
            ]
            reduced_key = tuple(reduced_mask)
            if (
                reduced_mask
                and reduced_key not in seen
                and cls.is_mask_uniform(reduced_mask, var_type, domain_lb, domain_ub)
            ):
                seen.add(reduced_key)
                variants.append(reduced_mask)
        return variants

    @classmethod
    def build_mask_addition_variants(
        cls, base_mask, addition_candidates, var_type, domain_lb, domain_ub, addition_mode, chunk_size=None
    ):
        if not base_mask or not addition_candidates:
            return []

        if addition_mode == REMOVAL_MODE_ONE:
            candidates = [
                cls.normalize_mask_intervals(base_mask + [interval], domain_lb, domain_ub)
                for interval in addition_candidates
            ]
            return [
                candidate
                for candidate in candidates
                if candidate and cls.is_mask_uniform(candidate, var_type, domain_lb, domain_ub)
            ]

        add_count = cls.resolve_chunk_remove_count(len(addition_candidates), chunk_size)
        variants = []
        seen = set()
        for _ in range(len(addition_candidates)):
            addition_indices = cls.uniform_chunk_indices(len(addition_candidates), add_count)
            expanded_mask = cls.normalize_mask_intervals(
                base_mask + [interval for index, interval in enumerate(addition_candidates) if index in addition_indices],
                domain_lb,
                domain_ub,
            )
            expanded_key = tuple(expanded_mask)
            if (
                expanded_mask
                and expanded_key not in seen
                and cls.is_mask_uniform(expanded_mask, var_type, domain_lb, domain_ub)
            ):
                seen.add(expanded_key)
                variants.append(expanded_mask)
        return variants

    @classmethod
    def build_complement_mask_variants(
        cls, mask_var_pairs, L, removal_mode=REMOVAL_MODE_ONE, chunk_size=None
    ):
        if not mask_var_pairs:
            return []

        cls.validate_removal_mode(removal_mode)
        bounds_by_var = {var_name: (var_type, lb, ub) for var_name, var_type, lb, ub in L}
        per_variable_variants = []
        for var_name, mask in mask_var_pairs:
            bounds = bounds_by_var.get(var_name)
            if bounds is None:
                return []

            var_type, lb, ub = bounds
            removal_variants = [
                (var_name, reduced_mask)
                for reduced_mask in cls.build_mask_removal_variants(
                    mask, var_type, lb, ub, removal_mode, chunk_size
                )
            ]
            if not removal_variants:
                return []
            per_variable_variants.append(removal_variants)

        variants = []
        for variant_combo in itertools.product(*per_variable_variants):
            variants.append([(var_name, list(mask)) for var_name, mask in variant_combo])

        return variants

    @classmethod
    def build_add_mask_variants(
        cls,
        original_mask_var_pairs,
        complemented_mask_var_pairs,
        L,
        addition_mode=REMOVAL_MODE_ONE,
        chunk_size=None,
    ):
        if not original_mask_var_pairs or not complemented_mask_var_pairs:
            return []

        cls.validate_removal_mode(addition_mode)
        bounds_by_var = {var_name: (var_type, lb, ub) for var_name, var_type, lb, ub in L}
        complemented_by_var = {var_name: mask for var_name, mask in complemented_mask_var_pairs}
        per_variable_variants = []

        for var_name, original_mask in original_mask_var_pairs:
            bounds = bounds_by_var.get(var_name)
            if bounds is None:
                return []
            complemented_mask = complemented_by_var.get(var_name)
            if complemented_mask is None:
                return []

            var_type, lb, ub = bounds
            addition_variants = [
                (var_name, expanded_mask)
                for expanded_mask in cls.build_mask_addition_variants(
                    complemented_mask,
                    original_mask,
                    var_type,
                    lb,
                    ub,
                    addition_mode,
                    chunk_size,
                )
            ]
            if not addition_variants:
                return []
            per_variable_variants.append(addition_variants)

        variants = []
        for variant_combo in itertools.product(*per_variable_variants):
            variants.append([(var_name, list(mask)) for var_name, mask in variant_combo])

        return variants

    def generate_masks(self, L):
        supported_entries = [
            (var_name, var_type, lb, ub)
            for (var_name, var_type, lb, ub) in L
            if self.is_mask_supported(var_type)
        ]
        if not supported_entries:
            return []

        task_args = [
            (var_name, var_type, lb, ub, self.coverage, self.delta, self.string_domain)
            for (var_name, var_type, lb, ub) in supported_entries
        ]

        if self.max_workers is None:
            computed_masks = [
                _build_typed_mask_entry_worker(task_arg)
                for task_arg in tqdm(task_args, total=len(task_args), desc="Generating masks...")
            ]
        else:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            computed_masks = list(
                tqdm(
                    self._executor.map(_build_typed_mask_entry_worker, task_args),
                    total=len(task_args),
                    desc="Generating masks...",
                )
            )

        mask_var_pairs = []
        for entry, computed in zip(supported_entries, computed_masks):
            if computed is None:
                continue
            var_name, _, _, _ = entry
            _, _, mask = computed
            mask_var_pairs.append((var_name, mask))

        return mask_var_pairs
