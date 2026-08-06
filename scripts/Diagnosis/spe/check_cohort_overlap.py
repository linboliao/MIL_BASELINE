"""Fail fast when an external test CSV overlaps the locked internal cohort."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.Diagnosis.spe.run import patient_id_from_slide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--internal-root", type=Path, required=True)
    parser.add_argument("--allow-overlap", action="store_true")
    return parser.parse_args()


def resolve_path(value: Path) -> Path:
    return value if value.is_absolute() else REPO_ROOT / value


def slide_id(value: str) -> str:
    return PurePosixPath(str(value).replace("\\", "/")).stem


def read_test_paths(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, dtype={"test_slide_path": "string"})
    if "test_slide_path" not in frame.columns:
        raise ValueError(f"Missing test_slide_path: {csv_path}")
    result = frame.loc[frame["test_slide_path"].notna()].copy()
    result["slide_id"] = result["test_slide_path"].map(slide_id)
    result["patient_id"] = result["slide_id"].map(patient_id_from_slide)
    return result


def main() -> None:
    args = parse_args()
    external_path = resolve_path(args.external)
    internal_root = resolve_path(args.internal_root)
    fold_paths = sorted(
        path
        for path in internal_root.glob("*.csv")
        if re.search(r"_1fold\.csv$", path.name, flags=re.IGNORECASE)
    )
    if len(fold_paths) != 1:
        raise FileNotFoundError(
            f"Expected one internal 1fold CSV in {internal_root}, found {len(fold_paths)}"
        )
    external = read_test_paths(external_path)
    internal = read_test_paths(fold_paths[0])
    exact = sorted(set(external["slide_id"]) & set(internal["slide_id"]))
    patients = sorted(set(external["patient_id"]) & set(internal["patient_id"]))
    affected = external.loc[external["patient_id"].isin(patients)]
    print(
        "Cohort overlap audit: "
        f"external_slides={len(external)}, external_patients={external.patient_id.nunique()}, "
        f"exact_slides={len(exact)}, overlapping_patients={len(patients)}, "
        f"affected_external_slides={len(affected)}"
    )
    if patients:
        print(f"Overlapping patients: {patients}")
        print(f"Exact slide IDs: {exact}")
        if not args.allow_overlap:
            raise SystemExit(
                "External/internal patient overlap detected. Create a patient-disjoint "
                "external CSV or explicitly set ALLOW_COHORT_OVERLAP=1 for a non-paper run."
            )


if __name__ == "__main__":
    main()
