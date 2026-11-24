from abc import ABC, abstractmethod
import random
import z3

from solver_utils import _strip_smt2_commands_2
from classes.domain_partitioner import (
    BITVEC_RE,
    DEFAULT_STRING_ORDER,
    MAX_STRING_MAX_LENGTH,
    STRING_ORDER_LEX,
    STRING_ORDER_SHORTLEX,
    get_string_domain_config,
    validate_string_order,
)


STRING_TO_INDEX_HELPERS_START = ";; STRING_TO_INDEX_HELPERS_START"
STRING_TO_INDEX_HELPERS_END = ";; STRING_TO_INDEX_HELPERS_END"

RECURSIVE_STRING_TO_INDEX_HELPERS_SHORTLEX = """;; STRING_TO_INDEX_HELPERS_START
;; Helper: Calculate 26^n
(define-fun-rec pow26 ((n Int)) Int
  (if (<= n 0)
      1
      (* 26 (pow26 (- n 1)))))

;; Helper: sum(26^k for k in range(1, len))
;; This counts all possible strings shorter than the current string.
(define-fun-rec sumPrevLengths ((len Int)) Int
  (if (<= len 1)
      0
      (+ (pow26 (- len 1)) (sumPrevLengths (- len 1)))))

;; Helper: check that the string only uses lowercase a-z characters.
(define-fun-rec isLowerAlphaRec ((s String) (idx Int)) Bool
  (if (>= idx (str.len s))
      true
      (let ((code (str.to_code (str.at s idx))))
        (and (<= 97 code)
             (<= code 122)
             (isLowerAlphaRec s (+ idx 1))))))

(define-fun isLowerAlphaString ((s String)) Bool
  (isLowerAlphaRec s 0))

;; Helper: recursive offset calculation (base-26 conversion).
(define-fun-rec offsetRec ((s String) (idx Int)) Int
  (if (>= idx (str.len s))
      0
      (+ (* (- (str.to_code (str.at s idx)) 97)
            (pow26 (- (- (str.len s) 1) idx)))
         (offsetRec s (+ idx 1)))))

;; Main helper function: stringToIndex
(define-fun stringToIndex ((s String)) Int
  (+ (sumPrevLengths (str.len s))
     (offsetRec s 0)))
;; STRING_TO_INDEX_HELPERS_END
"""

RECURSIVE_STRING_TO_INDEX_HELPERS_LEX = """;; STRING_TO_INDEX_HELPERS_START
;; Helper: Calculate 26^n
(define-fun-rec pow26 ((n Int)) Int
  (if (<= n 0)
      1
      (* 26 (pow26 (- n 1)))))

;; Helper: size of a lexicographic subtree with `remaining` positions left.
(define-fun-rec lexSubtreeSize ((remaining Int)) Int
  (if (<= remaining 0)
      1
      (+ 1 (* 26 (lexSubtreeSize (- remaining 1))))))

;; Helper: check that the string only uses lowercase a-z characters.
(define-fun-rec isLowerAlphaRec ((s String) (idx Int)) Bool
  (if (>= idx (str.len s))
      true
      (let ((code (str.to_code (str.at s idx))))
        (and (<= 97 code)
             (<= code 122)
             (isLowerAlphaRec s (+ idx 1))))))

(define-fun isLowerAlphaString ((s String)) Bool
  (isLowerAlphaRec s 0))

;; Helper: recursive lexicographic rank calculation.
(define-fun-rec offsetRecLex ((s String) (idx Int)) Int
  (if (>= idx (str.len s))
      0
      (+ (* (- (str.to_code (str.at s idx)) 97)
            (lexSubtreeSize (- (- (str.len s) 1) idx)))
         (offsetRecLex s (+ idx 1)))))

;; Main helper function: stringToIndex
(define-fun stringToIndex ((s String)) Int
  (+ (- (str.len s) 1)
     (offsetRecLex s 0)))
;; STRING_TO_INDEX_HELPERS_END
"""


def _build_length_dispatch(cases, fallback, base_indent="      ", var_name="len"):
    expr = str(fallback)
    for length, value in reversed(cases):
        expr = f"(if (= {var_name} {length}) {value}\n{base_indent}{expr})"
    return expr


def _build_sum_prev_lengths_expr(max_len):
    cases = []
    cumulative = 0
    for length in range(1, max_len + 1):
        cases.append((length, cumulative))
        cumulative += 26 ** length
    return _build_length_dispatch(cases, cases[-1][1])


def _build_lex_subtree_size_expr(max_len):
    cases = []
    cumulative = 1
    for remaining in range(0, max_len + 1):
        cases.append((remaining, cumulative))
        cumulative = 1 + (26 * cumulative)
    return _build_length_dispatch(cases, cases[-1][1], var_name="remaining")


def _build_char_weight_expr(idx, max_len):
    cases = []
    for length in range(idx + 1, max_len + 1):
        cases.append((length, 26 ** (length - 1 - idx)))
    return _build_length_dispatch(cases, 0)


def _build_lex_char_weight_expr(idx, max_len):
    cases = []
    for length in range(idx + 1, max_len + 1):
        remaining = length - idx - 1
        subtree_size = sum(26 ** k for k in range(0, remaining + 1))
        cases.append((length, subtree_size))
    return _build_length_dispatch(cases, 1)


def _build_is_lower_alpha_clauses(max_len):
    clauses = [f"      (<= len {max_len})"]
    for idx in range(max_len):
        clauses.append(
            "      "
            f"(or (<= len {idx}) "
            f"(let ((code{idx} (str.to_code (str.at s {idx})))) "
            f"(and (<= 97 code{idx}) (<= code{idx} 122))))"
        )
    return "\n".join(clauses)


def _build_string_index_terms(max_len):
    terms = []
    for idx in range(max_len):
        weight_expr = _build_char_weight_expr(idx, max_len)
        terms.append(
            f"      (if (> len {idx})\n"
            f"          (* (- (str.to_code (str.at s {idx})) 97)\n"
            f"             {weight_expr})\n"
            f"          0)"
        )
    return "\n".join(terms)


def _build_lex_string_index_terms(max_len):
    terms = []
    for idx in range(max_len):
        weight_expr = _build_lex_char_weight_expr(idx, max_len)
        terms.append(
            f"      (if (> len {idx})\n"
            f"          (* (- (str.to_code (str.at s {idx})) 97)\n"
            f"             {weight_expr})\n"
            f"          0)"
        )
    return "\n".join(terms)


def build_non_recursive_string_to_index_helpers(max_len):
    return f""";; STRING_TO_INDEX_HELPERS_START
;; Helper: count all strings shorter than the current string length
(define-fun sumPrevLengths ((len Int)) Int
  (if (<= len 1)
      0
      {_build_sum_prev_lengths_expr(max_len)}))

;; Helper: check that the string only uses lowercase a-z characters
(define-fun isLowerAlphaString ((s String)) Bool
  (let ((len (str.len s)))
    (and
{_build_is_lower_alpha_clauses(max_len)})))

;; Main helper function: stringToIndex
(define-fun stringToIndex ((s String)) Int
  (let ((len (str.len s)))
    (+ (sumPrevLengths len)
{_build_string_index_terms(max_len)})))
;; STRING_TO_INDEX_HELPERS_END
"""


def build_non_recursive_lex_string_to_index_helpers(max_len):
    return f""";; STRING_TO_INDEX_HELPERS_START
;; Helper: count lexicographic subtree sizes
(define-fun lexSubtreeSize ((remaining Int)) Int
  (if (<= remaining 0)
      1
      {_build_lex_subtree_size_expr(max_len)}))

;; Helper: check that the string only uses lowercase a-z characters
(define-fun isLowerAlphaString ((s String)) Bool
  (let ((len (str.len s)))
    (and
{_build_is_lower_alpha_clauses(max_len)})))

;; Main helper function: stringToIndex
(define-fun stringToIndex ((s String)) Int
  (let ((len (str.len s)))
    (+ (- len 1)
{_build_lex_string_index_terms(max_len)})))
;; STRING_TO_INDEX_HELPERS_END
"""


def get_string_to_index_helpers(use_recursive_string_helpers=False, string_domain=None):
    string_domain = string_domain or get_string_domain_config(MAX_STRING_MAX_LENGTH)
    string_order = validate_string_order(string_domain.get("order", DEFAULT_STRING_ORDER))
    if string_order == STRING_ORDER_LEX:
        if use_recursive_string_helpers:
            return RECURSIVE_STRING_TO_INDEX_HELPERS_LEX
        return build_non_recursive_lex_string_to_index_helpers(string_domain["max_len"])
    if use_recursive_string_helpers:
        return RECURSIVE_STRING_TO_INDEX_HELPERS_SHORTLEX
    return build_non_recursive_string_to_index_helpers(string_domain["max_len"])


def format_smt_literal(value, var_type):
    if var_type == "Bool":
        return "true" if value else "false"

    bitvec_match = BITVEC_RE.match(var_type)
    if bitvec_match:
        return f"(_ bv{int(value)} {int(bitvec_match.group(1))})"

    if var_type == "String":
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    return str(value)


def normalize_model_value(z3_value, var_type):
    if var_type == "String":
        value = str(z3_value)
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        return value
    if var_type == "Bool":
        return str(z3_value).lower() == "true"
    return int(str(z3_value))


class SMTSolver(ABC):
    """
    Abstract Strategy interface for an in-memory SMT solver engine.
    Decouples the sampling algorithm from the underlying SMT solver implementation.
    """

    @abstractmethod
    def load_formula(self, smt_filepath):
        """Load the base SMT-LIB2 formula into memory once."""

    @abstractmethod
    def get_base_content(self):
        """Return the raw base SMT-LIB2 content stored in memory."""

    @abstractmethod
    def reset_iteration(self, assignment=None):
        """Reset iteration-scoped assertions (masks and current E-assignment) in RAM."""

    @abstractmethod
    def apply_masks(self, mask_var_pairs, L):
        """Apply interval masks for large-domain variables L in RAM."""

    @abstractmethod
    def add_blocking_clause(self, model, L):
        """Record a blocking clause in RAM to exclude a previously found SAT model."""

    @abstractmethod
    def check_and_get_model(self, timeout_ms=None):
        """Check satisfiability in RAM and return (status, model_dict)."""


class Z3Solver(SMTSolver):
    """
    Concrete in-memory Z3 implementation of SMTSolver.
    Eliminates all temporary .smt2 disk I/O and iterative regex file rewrites.
    """

    def __init__(
        self,
        smt_filepath=None,
        string_domain=None,
        use_recursive_string_helpers=True,
    ):
        self._string_domain = string_domain or get_string_domain_config(MAX_STRING_MAX_LENGTH)
        self._use_recursive_string_helpers = use_recursive_string_helpers
        self._raw_smt2_content = ""
        self._clean_prefix = ""
        self._clean_suffix = ""
        self._base_has_string_helpers = False
        self._clean_string_helpers = _strip_smt2_commands_2(
            get_string_to_index_helpers(
                use_recursive_string_helpers=self._use_recursive_string_helpers,
                string_domain=self._string_domain,
            )
        )
        self._blocked_clauses = []
        self._current_assignment_assert = ""
        self._current_mask_asserts = []
        if smt_filepath is not None:
            self.load_formula(smt_filepath)

    def load_formula(self, smt_filepath):
        with open(smt_filepath, "r") as f:
            content = f.read()

        self._raw_smt2_content = content
        self._base_has_string_helpers = (
            STRING_TO_INDEX_HELPERS_START in content
            and STRING_TO_INDEX_HELPERS_END in content
        )

        lines = content.splitlines(keepends=True)
        first_checksat_index = next(
            (i for i, line in enumerate(lines) if line.strip().startswith("(check-sat")),
            None,
        )

        if first_checksat_index is None:
            prefix_raw = content
            suffix_raw = ""
        else:
            prefix_raw = "".join(lines[:first_checksat_index])
            suffix_raw = "".join(lines[first_checksat_index:])

        self._clean_prefix = _strip_smt2_commands_2(prefix_raw)
        self._clean_suffix = _strip_smt2_commands_2(suffix_raw)
        self._blocked_clauses = []
        self._current_assignment_assert = ""
        self._current_mask_asserts = []

    def get_base_content(self):
        return self._raw_smt2_content

    def reset_iteration(self, assignment=None):
        self._current_mask_asserts = []
        if assignment is None:
            self._current_assignment_assert = ""
        else:
            clauses = [
                f"(= {var_name} {format_smt_literal(data['value'], data['type'])})"
                for var_name, data in assignment.items()
            ]
            self._current_assignment_assert = (
                "(assert (and\n    " + "\n    ".join(clauses) + "\n))"
            )

    def apply_masks(self, mask_var_pairs, L):
        type_by_var = {var_name: var_type for var_name, var_type, _, _ in L}
        helpers_injected = self._base_has_string_helpers
        mask_assertions = []

        for var_name, mask in mask_var_pairs:
            var_type = type_by_var.get(var_name)
            if var_type is None:
                continue

            if var_type == "String":
                if not helpers_injected and self._clean_string_helpers:
                    mask_assertions.append(self._clean_string_helpers)
                    helpers_injected = True

                interval_clauses = [
                    f"(and (>= (stringToIndex {var_name}) {band_lb}) (< (stringToIndex {var_name}) {band_ub}))"
                    for band_lb, band_ub in mask
                ]
                random.shuffle(interval_clauses)
                if not interval_clauses:
                    continue

                new_assert = (
                    "(assert (and\n"
                    f"  (>= (str.len {var_name}) 1)\n"
                    f"  (<= (str.len {var_name}) {self._string_domain['max_len']})\n"
                    f"  (isLowerAlphaString {var_name})\n"
                    "  (or\n"
                    "    " + "\n    ".join(interval_clauses) + "\n"
                    "  )\n"
                    "))"
                )
                mask_assertions.append(new_assert)
            else:
                shuffled_mask = list(mask)
                random.shuffle(shuffled_mask)
                if not shuffled_mask:
                    continue

                clauses = [
                    f"(and (>= {var_name} {band_lb}) (< {var_name} {band_ub}))"
                    for band_lb, band_ub in shuffled_mask
                ]
                new_assert = "(assert (or\n  " + "\n  ".join(clauses) + "\n))"
                mask_assertions.append(new_assert)

        self._current_mask_asserts = mask_assertions

    def add_blocking_clause(self, model, L):
        l_info = {var_name: var_type for var_name, var_type, _, _ in L}
        equalities = []
        for var_name, z3_value in model.items():
            if var_name not in l_info:
                continue
            var_type = l_info[var_name]
            literal = format_smt_literal(normalize_model_value(z3_value, var_type), var_type)
            equalities.append(f"(= {var_name} {literal})")

        if not equalities:
            return

        blocking_clause = f"(assert (not (and {' '.join(equalities)})))"
        self._blocked_clauses.append(blocking_clause)

    def add_blocking_clause_for_free_vars(self, model, free_vars):
        type_by_var = {var_name: var_type for var_name, var_type in free_vars}
        equalities = []
        for var_name, z3_value in model.items():
            var_type = type_by_var.get(var_name)
            if var_type is None:
                continue
            literal = format_smt_literal(normalize_model_value(z3_value, var_type), var_type)
            equalities.append(f"(= {var_name} {literal})")

        if not equalities:
            return

        blocking_clause = f"(assert (not (and {' '.join(equalities)})))"
        self._blocked_clauses.append(blocking_clause)

    def _build_in_memory_smt2(self):
        parts = []
        if self._clean_prefix:
            parts.append(self._clean_prefix)
        if self._current_assignment_assert:
            parts.append(self._current_assignment_assert)
        if self._blocked_clauses:
            parts.extend(self._blocked_clauses)
        if self._current_mask_asserts:
            parts.extend(self._current_mask_asserts)
        if self._clean_suffix:
            parts.append(self._clean_suffix)
        return "\n".join(parts)

    def check_and_get_model(self, timeout_ms=None):
        smt2_clean = self._build_in_memory_smt2()

        solver = z3.Solver()
        if timeout_ms is not None:
            solver.set(timeout=int(timeout_ms))
        assertions = z3.parse_smt2_string(smt2_clean)
        solver.add(assertions)

        result = solver.check()
        print(f"Z3 result: {result}")
        if result == z3.sat:
            model = solver.model()
            model_dict = {model[i].name(): model[model[i]] for i in range(len(model))}
            return "SAT", model_dict
        if result == z3.unsat:
            return "UNSAT", None
        return "UNKNOWN", None
