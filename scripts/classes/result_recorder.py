import csv
from pathlib import Path
import pandas as pd


class ResultRecorder:
    """
    Object-Oriented component responsible for recording SAT / UNSAT / UNKNOWN
    sampling iterations in memory and flushing them incrementally to CSV.
    """

    def __init__(self, csv_filepath, free_vars, num_samples=None):
        self.csv_filepath = Path(csv_filepath)
        self.free_vars = list(free_vars)
        self.num_samples = num_samples
        self.solver_result = {}
        self.S = set()
        self.sat_count = self.count_existing_sat_rows(self.csv_filepath)
        if self.sat_count:
            print(f"[INFO] Found {self.sat_count} existing SAT solutions in {self.csv_filepath}; appending new results.")

    def already_reached_target(self):
        if self.num_samples is not None and self.sat_count >= self.num_samples:
            print(f"[INFO] Existing SAT solutions already reach num_samples={self.num_samples}; stopping.")
            return True
        return False

    def record_and_flush(self, status, model, assignment, mask_var_pairs, partial_time, L=None):
        if status in ("UNSAT", "UNKNOWN"):
            self.save_model_typed(
                self.solver_result,
                model,
                self.free_vars,
                None,
                status,
                assignment,
                mask_var_pairs,
                partial_time,
                L,
            )
        elif status == "SAT":
            self.save_sat_model_typed(
                self.solver_result,
                model,
                self.free_vars,
                None,
                status,
                mask_var_pairs,
                partial_time,
                L,
            )
            self.S.add(frozenset(model.items()))
            self.sat_count += 1

        self.append_solver_result_row_typed(self.solver_result, self.csv_filepath)
        if self.num_samples is not None and self.sat_count >= self.num_samples:
            print(f"[INFO] Reached num_samples={self.num_samples} SAT solutions, saving CSV and stopping.")
            return True
        return False

    @staticmethod
    def free_var_names(free_vars):
        return [var_name for var_name, _ in free_vars]

    @staticmethod
    def mask_bounds_from_pairs(mask_var_pairs, L):
        if not mask_var_pairs:
            return []

        bounds_by_var = {var_name: (var_type, lb, ub) for var_name, var_type, lb, ub in L}
        mask_bounds = []
        for var_name, _ in mask_var_pairs:
            bounds = bounds_by_var.get(var_name)
            if bounds is None:
                continue
            var_type, lb, ub = bounds
            mask_bounds.append((var_name, var_type, lb, ub))
        return mask_bounds

    @classmethod
    def save_model_typed(
        cls,
        solver_result,
        model,
        free_vars,
        tmp_filepath,
        status,
        assignment,
        mask_var_pairs,
        partial_time,
        L=None,
    ):
        if status not in ("UNSAT", "UNKNOWN"):
            return

        if isinstance(assignment, frozenset):
            assignment_dict = dict(assignment)
        elif isinstance(assignment, dict):
            assignment_dict = assignment
        else:
            assignment_dict = {}

        free_var_names = set(cls.free_var_names(free_vars))

        for meta_key in ("status", "mask_var_pairs", "mask_bounds", "time"):
            if meta_key not in solver_result:
                solver_result[meta_key] = [None] * (len(next(iter(solver_result.values()))) if solver_result else 0)

        for var_name in free_var_names:
            if var_name not in solver_result:
                solver_result[var_name] = [None] * (len(next(iter(solver_result.values()))) if solver_result else 0)

        for var_name in free_var_names:
            assignment_entry = assignment_dict.get(var_name, None)
            if isinstance(assignment_entry, dict) and "value" in assignment_entry:
                assignment_entry = assignment_entry["value"]
            solver_result[var_name].append(assignment_entry)

        solver_result["status"].append(status)
        solver_result["mask_var_pairs"].append(list(mask_var_pairs) if mask_var_pairs is not None else None)
        solver_result["mask_bounds"].append(cls.mask_bounds_from_pairs(mask_var_pairs, L or []))
        solver_result["time"].append(partial_time)

    @classmethod
    def save_sat_model_typed(
        cls,
        result,
        model_dict,
        free_vars,
        tmp_filepath,
        status,
        mask_var_pairs,
        partial_time,
        L=None,
    ):
        free_var_names = set(cls.free_var_names(free_vars))

        for meta_key in ("status", "mask_var_pairs", "mask_bounds", "time"):
            if meta_key not in result:
                result[meta_key] = [None] * (len(next(iter(result.values()))) if result else 0)

        if model_dict is None:
            for key in result:
                result[key].append(None)
            result["status"][-1] = status
            result["mask_var_pairs"][-1] = list(mask_var_pairs) if mask_var_pairs is not None else None
            result["mask_bounds"][-1] = cls.mask_bounds_from_pairs(mask_var_pairs, L or [])
            result["time"][-1] = partial_time
            return result

        filtered = {k: v for k, v in model_dict.items() if k in free_var_names}

        for var_name in free_var_names:
            if var_name not in result:
                result[var_name] = [None] * (len(next(iter(result.values()))) if result else 0)

        for var_name in result:
            if var_name in ("status", "mask_var_pairs", "mask_bounds", "time"):
                continue
            result[var_name].append(filtered.get(var_name, None))

        result["status"].append(status)
        result["mask_var_pairs"].append(list(mask_var_pairs) if mask_var_pairs is not None else None)
        result["mask_bounds"].append(cls.mask_bounds_from_pairs(mask_var_pairs, L or []))
        result["time"].append(partial_time)
        return result

    @staticmethod
    def append_solver_result_row_typed(solver_result, csv_filepath):
        row = {key: values[-1] for key, values in solver_result.items() if values}
        if not row:
            return

        df = pd.DataFrame([row])
        header = not Path(csv_filepath).exists()
        df.to_csv(csv_filepath, mode="a", header=header, index=False)

    @staticmethod
    def count_existing_sat_rows(csv_filepath):
        csv_filepath = Path(csv_filepath)
        if not csv_filepath.exists() or csv_filepath.stat().st_size == 0:
            return 0

        try:
            with csv_filepath.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if "status" not in (reader.fieldnames or []):
                    return 0
                return sum(
                    1
                    for row in reader
                    if str(row.get("status", "")).strip().upper() == "SAT"
                )
        except (OSError, csv.Error):
            return 0
