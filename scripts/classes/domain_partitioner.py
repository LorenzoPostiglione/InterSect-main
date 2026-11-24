import re
import time


BITVEC_RE = re.compile(r"^\(_\s+BitVec\s+(\d+)\)$")
STRING_ORDER_SHORTLEX = "shortlex"
STRING_ORDER_LEX = "lex"
DEFAULT_STRING_ORDER = STRING_ORDER_SHORTLEX
MIN_STRING_MAX_LENGTH = 8
MAX_STRING_MAX_LENGTH = 54


def validate_string_order(string_order):
    if string_order not in {STRING_ORDER_SHORTLEX, STRING_ORDER_LEX}:
        raise ValueError(
            f"Unsupported string_order={string_order!r}. "
            f"Expected {STRING_ORDER_SHORTLEX!r} or {STRING_ORDER_LEX!r}."
        )
    return string_order


def total_strings_up_to_length(max_len):
    return sum(26**k for k in range(1, max_len + 1))


def get_string_domain_config(string_max_length, string_order=DEFAULT_STRING_ORDER):
    if not isinstance(string_max_length, int):
        raise TypeError("string_max_length must be an integer.")
    if not (MIN_STRING_MAX_LENGTH <= string_max_length <= MAX_STRING_MAX_LENGTH):
        raise ValueError(
            f"Unsupported string_max_length={string_max_length}. "
            f"Expected an integer in [{MIN_STRING_MAX_LENGTH}, {MAX_STRING_MAX_LENGTH}]."
        )
    validate_string_order(string_order)
    return {
        "max_len": string_max_length,
        "domain_max": total_strings_up_to_length(string_max_length),
        "order": string_order,
    }


class DomainPartitioner:
    """
    Object-Oriented component responsible for extracting free variables and
    directional domain bounds from in-memory SMT-LIB2 content and partitioning
    variables into Enumeration (E) and Large-domain (L) sets.
    """

    def __init__(self, theta=1000, U=2**256 - 1, string_domain=None):
        self.theta = theta
        self.U = U
        self.string_domain = string_domain or get_string_domain_config(MAX_STRING_MAX_LENGTH)

    @staticmethod
    def extract_free_vars(smt_content):
        free_vars = []
        for line in smt_content.splitlines():
            m_fun = re.match(r"\(declare-fun\s+(\w+)\s+\(\s*\)\s+(.+)\)", line)
            m_const = re.match(r"\(declare-const\s+(\w+)\s+(.+)\)", line)
            if m_fun:
                name = m_fun.group(1)
                var_type = m_fun.group(2).strip()
                free_vars.append((name, var_type))
            elif m_const:
                name = m_const.group(1)
                var_type = m_const.group(2).strip()
                free_vars.append((name, var_type))
        return free_vars

    @classmethod
    def extract_free_vars_from_file(cls, file_path):
        with open(file_path, "r") as f:
            return cls.extract_free_vars(f.read())

    @staticmethod
    def extract_int_bounds_from_content(smt_content, free_var_name, U=2**256 - 1):
        pattern = fr'\(\s*(>=|>|<=|<)\s+{re.escape(free_var_name)}\s+(-?[0-9]+)\s*\)'
        matches = re.findall(pattern, smt_content)

        lb = 0
        ub = U
        for op, value in matches:
            value = int(value)
            if op in (">=", ">"):
                lb = max(lb, value + (1 if op == ">" else 0))
            elif op in ("<=", "<"):
                ub = min(ub, value - (1 if op == "<" else 0))
        return lb, ub

    @staticmethod
    def extract_bitvec_bounds_from_content(smt_content, free_var_name, bitwidth):
        lb = 0
        ub = 2**bitwidth - 1

        patterns = [
            (rf'\((?:bvule|bvsle)\s+\(_\s+bv(\d+)\s+{bitwidth}\)\s+{re.escape(free_var_name)}\)', "lb", 0),
            (rf'\((?:bvult|bvslt)\s+\(_\s+bv(\d+)\s+{bitwidth}\)\s+{re.escape(free_var_name)}\)', "lb", 1),
            (rf'\((?:bvule|bvsle)\s+{re.escape(free_var_name)}\s+\(_\s+bv(\d+)\s+{bitwidth}\)\)', "ub", 0),
            (rf'\((?:bvult|bvslt)\s+{re.escape(free_var_name)}\s+\(_\s+bv(\d+)\s+{bitwidth}\)\)', "ub", -1),
        ]

        for pattern, bound_kind, delta in patterns:
            for raw_value in re.findall(pattern, smt_content):
                value = int(raw_value) + delta
                if bound_kind == "lb":
                    lb = max(lb, value)
                else:
                    ub = min(ub, value)

        return lb, ub

    def partition_variable_from_content(self, E, L, smt_content, free_var_name, free_var_type):
        if free_var_type == "Bool":
            E.add((free_var_name, free_var_type, (False, True)))
            return

        if free_var_type == "String":
            string_ub = min(self.U, self.string_domain["domain_max"] - 1)
            L.add((free_var_name, free_var_type, 0, string_ub))
            return

        if free_var_type == "Int":
            lb, ub = self.extract_int_bounds_from_content(smt_content, free_var_name, self.U)
            if ub >= lb and (ub - lb) < self.theta:
                E.add((free_var_name, free_var_type, range(lb, ub + 1)))
            else:
                L.add((free_var_name, free_var_type, lb, ub))
            return

        bitvec_match = BITVEC_RE.match(free_var_type)
        if bitvec_match:
            bitwidth = int(bitvec_match.group(1))
            lb, ub = self.extract_bitvec_bounds_from_content(smt_content, free_var_name, bitwidth)
            if ub >= lb and (ub - lb) < self.theta:
                E.add((free_var_name, free_var_type, range(lb, ub + 1)))
            else:
                L.add((free_var_name, free_var_type, lb, ub))
            return

    def partition(self, smt_content, free_vars, deadline=None):
        E = set()
        L = set()
        for free_var_name, free_var_type in free_vars:
            if deadline is not None and time.perf_counter() >= deadline:
                return E, L, True
            self.partition_variable_from_content(E, L, smt_content, free_var_name, free_var_type)
        return E, L, False
