#!/usr/bin/env python3
"""Build the reorganized Prostate_Diagnosis dataset:

  1. Re-scan the main pool (dev + oldtest + ext_sl-loose-root), same scope/logic
     already validated in split_scripts/build_manifest.py.
  2. Scan the two reserved true-external cohorts: 301 medical center, and 云南肿瘤
     (merging its training-pool slides with the existing 云南省肿瘤-2025-12-08 batch).
  3. Patient-level, (label x primary_type)-stratified split of the main pool into
     train / val / test (70/15/15), seed=42.
  4. Write everything under datasets/Prostate_Diagnosis/.

Run with the PrePATH interpreter:
  /data12/jing/anaconda3/envs/PrePATH/bin/python split_scripts/build_prostate_diagnosis_dataset.py
"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split

MEIXIN_ROOT = "/NAS145/linboliao/Data/迈新生物"
DIAG_ROOT = "/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/Diagnosis"
OUT_DIR = "/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis"

WSI_EXTS = (".svs", ".kfb")
WS_RE = re.compile(r"\s+")
TRAILING_CJK_RE = re.compile(r"[一-鿿\s]+$")
SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15  # of the whole pool; val = VAL_FRAC/(1-TEST_FRAC) of the train_val remainder

MAIN_POOL_TARGETS = [
    (os.path.join(MEIXIN_ROOT, "MIL训练数据/癌"), "dev", True),
    (os.path.join(MEIXIN_ROOT, "MIL训练数据/非癌"), "dev", True),
    (os.path.join(MEIXIN_ROOT, "MIL测试数据/省立切片迈新染色病例"), "oldtest", True),
    (os.path.join(MEIXIN_ROOT, "MIL测试数据/省立病例"), "oldtest", True),
    (os.path.join(MEIXIN_ROOT, "MIL测试数据/迈新病例"), "oldtest", True),
]
EXT_SL_ROOT = os.path.join(MEIXIN_ROOT, "MIL外部测试/省立医院")

RESERVED_YNZL_DEV_TARGETS = [
    (os.path.join(MEIXIN_ROOT, "MIL训练数据/癌/云南肿瘤有癌病例"), "dev"),
    (os.path.join(MEIXIN_ROOT, "MIL测试数据/云南省肿瘤病例"), "oldtest"),
]
EXT_301_ROOT = os.path.join(MEIXIN_ROOT, "MIL外部测试/301 外部测试20260202")
EXT_YNZL_ROOT = os.path.join(MEIXIN_ROOT, "MIL外部测试/云南省肿瘤-2025-12-08")
SL_LABEL_CSV = os.path.join(OUT_DIR, "_archive_20260828/external_test.csv")

LABEL_SOURCES = {
    "dev": os.path.join(DIAG_ROOT, "train_val.csv"),
    "oldtest": os.path.join(DIAG_ROOT, "test.csv"),
    "ext_sl": SL_LABEL_CSV,
}
# NOTE: ext_sl (loose 省立医院 svs files under MIL外部测试) is folded into the
# main pool (as training material only - see stratified_patient_split, which
# keeps every 省立-center patient out of internal_test). Labels come from our
# own verified xlsx-matched external_test.csv (SL rows), NOT from the stale
# old Diagnosis-project file (External/h-optimus-1/external_test_h-optimus-1.csv),
# which fell out of sync after this session's filename-correction renames.
EXTERNAL_CSV = os.path.join(DIAG_ROOT, "External/h-optimus-1/external_test_h-optimus-1.csv")

CENTER_RULES = [
    ("新昌", "新昌"), ("XC", "新昌"),
    ("福建省立", "省立"), ("省立", "省立"), ("SL", "省立"),
    ("迈新", "迈新"),
]

# Folder-name substrings that must never end up in the main pool (reserved
# externals / special queues), even though they sit inside a scanned root.
EXCLUDE_SUBSTRINGS = ["云南肿瘤有癌病例", "云南省肿瘤病例"]


def strip_ws(s):
    return WS_RE.sub("", s) if s else s


def stem(filename):
    base = filename
    for ext in WSI_EXTS + (".pt",):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return strip_ws(base)


def normalize_slide_id(raw_stem):
    s = raw_stem.rstrip()
    s = TRAILING_CJK_RE.sub("", s)
    return s


def patient_id_of(normalized_stem):
    return normalized_stem.split(".", 1)[0]


def infer_center(path):
    # Strip the ".../迈新生物/" root so the vendor name in the root folder
    # itself never falsely matches the "迈新" (vendor-site) rule below.
    relative = path[len(MEIXIN_ROOT):] if path.startswith(MEIXIN_ROOT) else path
    for needle, center in CENTER_RULES:
        if needle in relative:
            return center
    return "未知"


def load_label_index(pool):
    path = LABEL_SOURCES[pool]
    index = {}
    if not os.path.isfile(path):
        print("[WARN] missing label csv for pool=%s: %s" % (pool, path))
        return index
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if pool == "dev":
                key_raw, label, type_ = row.get("slide_id", ""), row.get("label", ""), row.get("type", "")
            elif pool == "oldtest":
                key_raw = row.get("test_slide_path", "")
                label, type_ = row.get("test_label", ""), row.get("test_type", "")
            elif pool == "ext_sl":
                if row.get("center") != "SL":
                    continue
                key_raw = row.get("test_slide_path", "")
                label, type_ = row.get("test_label", ""), row.get("type", "")
            else:
                key_raw = os.path.basename(row.get("test_slide_path", ""))
                label, type_ = row.get("test_label", ""), row.get("test_type", "")
            key = stem(strip_ws(key_raw))
            if key and key not in index:
                index[key] = {"label": label, "type": type_}
    return index


def load_external_index_by_center(center_filter):
    idx = {}
    with open(EXTERNAL_CSV, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row.get("center") != center_filter:
                continue
            key = stem(os.path.basename(row.get("test_slide_path", "")))
            if key:
                idx[key] = {"label": row.get("test_label", ""), "type": row.get("test_type", "")}
    return idx


def scan(root, pool, recursive=True, exclude=()):
    files = []
    if not os.path.isdir(root):
        print("[WARN] missing scan root: %s" % root)
        return files
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            if any(x in dirpath for x in exclude):
                continue
            for fn in filenames:
                if fn.lower().endswith(WSI_EXTS):
                    files.append((os.path.join(dirpath, fn), fn, pool))
    else:
        for fn in os.listdir(root):
            full = os.path.join(root, fn)
            if os.path.isfile(full) and fn.lower().endswith(WSI_EXTS):
                files.append((full, fn, pool))
    return files


def build_main_pool(label_indexes):
    all_files = []
    for root, pool, include in MAIN_POOL_TARGETS:
        if include:
            all_files.extend(scan(root, pool, exclude=EXCLUDE_SUBSTRINGS))
    all_files.extend(scan(EXT_SL_ROOT, "ext_sl", recursive=False))

    rows, unmatched, seen, dups = [], [], {}, []
    for raw_path, filename, pool in all_files:
        raw_stem = stem(filename)
        slide_id = normalize_slide_id(raw_stem)
        pid = patient_id_of(slide_id)
        center = infer_center(raw_path)
        entry = label_indexes[pool].get(raw_stem)
        if entry is None:
            unmatched.append({"raw_path": raw_path, "filename": filename, "pool": pool, "slide_id": slide_id})
            continue
        if slide_id in seen:
            dups.append({"slide_id": slide_id, "first_path": seen[slide_id], "dup_path": raw_path, "pool": pool})
            continue
        seen[slide_id] = raw_path
        rows.append({
            "raw_path": raw_path, "filename": filename, "slide_id": slide_id, "patient_id": pid,
            "label": entry["label"], "type": entry["type"], "center": center, "pool": pool,
        })
    return rows, unmatched, dups


def build_reserved_external(root, ext_pool_tag, label_indexes, center_name, extra_targets=None):
    """label_indexes must contain an entry for ext_pool_tag (external csv, center-filtered)
    plus entries for whatever pool tags extra_targets uses (e.g. "dev", "oldtest")."""
    rows = []
    files = scan(root, ext_pool_tag, recursive=False) if root else []
    if extra_targets:
        for extra_root, pool in extra_targets:
            files.extend(scan(extra_root, pool))
    seen = set()
    for raw_path, filename, pool in files:
        raw_stem = stem(filename)
        slide_id = normalize_slide_id(raw_stem)
        if slide_id in seen:
            continue
        entry = label_indexes[pool].get(raw_stem)
        if entry is None:
            continue  # unlabeled -> not part of the reserved external cohort
        seen.add(slide_id)
        rows.append({
            "raw_path": raw_path, "filename": filename, "slide_id": slide_id,
            "patient_id": patient_id_of(slide_id), "label": entry["label"], "type": entry["type"],
            "center": center_name, "pool": pool,
        })
    return rows


def patient_table(rows):
    patients = defaultdict(lambda: {"labels": set(), "type_counter": Counter(), "center_counter": Counter(), "rows": []})
    for r in rows:
        p = patients[r["patient_id"]]
        p["labels"].add(r["label"])
        p["type_counter"][r["type"]] += 1
        p["center_counter"][r["center"]] += 1
        p["rows"].append(r)
    table = {}
    for pid, info in patients.items():
        table[pid] = {
            "label": "1" if "1" in info["labels"] else "0",
            "primary_type": info["type_counter"].most_common(1)[0][0],
            "primary_center": info["center_counter"].most_common(1)[0][0],
            "n_slides": len(info["rows"]),
            "rows": info["rows"],
        }
    return table


def _strata_key(table, pid):
    return table[pid]["label"] + "|" + table[pid]["primary_center"]


def _safe_stratified_split(pids, test_size, strata, seed):
    """Fall back to a coarser stratum (drop center, then drop type) if any
    stratum is too small for sklearn's stratified split (needs >=2 members
    per class, and enough per class to populate both sides of the split)."""
    from collections import Counter as _Counter
    counts = _Counter(strata)
    min_count = min(counts.values())
    # sklearn requires each class to have at least 2 members, and effectively
    # enough members that test_size * n_class >= 1 for every class.
    if min_count >= 2:
        try:
            return train_test_split(pids, test_size=test_size, random_state=seed, stratify=strata)
        except ValueError:
            pass
    return None


def stratified_patient_split(table):
    """Patient-level, (label|center)-stratified split of the FULL main pool
    (新昌+省立+迈新, including the folded-in ext_sl batch) into dev (train+val
    pool, for later k-fold CV) and a fixed internal_test. All three centers are
    eligible for internal_test - there is no special-casing of 省立.
    """
    pids = sorted(table.keys())
    strata_full = [_strata_key(table, pid) for pid in pids]
    strata_no_center = [table[pid]["label"] + "|" + table[pid]["primary_type"] for pid in pids]

    result = _safe_stratified_split(pids, TEST_FRAC, strata_full, SEED)
    if result is None:
        print("WARNING: label|center stratification has strata too small to split; "
              "falling back to label|type stratification for the dev/internal_test split.")
        result = train_test_split(pids, test_size=TEST_FRAC, random_state=SEED, stratify=strata_no_center)
    pids_dev, pids_test = result

    return set(pids_dev), set(pids_test)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("wrote %s (%d rows)" % (path, len(rows)))


def crosstab(rows):
    return Counter((r["label"], r["type"]) for r in rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    label_indexes = {pool: load_label_index(pool) for pool in LABEL_SOURCES}

    main_rows, unmatched, dups = build_main_pool(label_indexes)
    write_csv(os.path.join(OUT_DIR, "manifest.csv"),
              ["raw_path", "filename", "slide_id", "patient_id", "label", "type", "center", "pool"], main_rows)
    write_csv(os.path.join(OUT_DIR, "unmatched.csv"),
              ["raw_path", "filename", "pool", "slide_id"], unmatched)
    write_csv(os.path.join(OUT_DIR, "duplicate_slides.csv"),
              ["slide_id", "first_path", "dup_path", "pool"], dups)

    reserved_indexes = dict(label_indexes)
    reserved_indexes["ext_ynzl"] = load_external_index_by_center("YNZL")
    reserved_indexes["ext_301"] = load_external_index_by_center("301")
    ext_ynzl_rows = build_reserved_external(
        EXT_YNZL_ROOT, "ext_ynzl", reserved_indexes, "云南肿瘤",
        extra_targets=RESERVED_YNZL_DEV_TARGETS,
    )
    ext_301_rows = build_reserved_external(EXT_301_ROOT, "ext_301", reserved_indexes, "301")
    write_csv(os.path.join(OUT_DIR, "external_test_ynzl.csv"),
              ["raw_path", "filename", "slide_id", "patient_id", "label", "type", "center", "pool"], ext_ynzl_rows)
    write_csv(os.path.join(OUT_DIR, "external_test_301.csv"),
              ["raw_path", "filename", "slide_id", "patient_id", "label", "type", "center", "pool"], ext_301_rows)

    table = patient_table(main_rows)
    dev_pids, test_pids = stratified_patient_split(table)

    # sanity: no patient-id collision between main pool and reserved externals
    ext_pids = {r["patient_id"] for r in ext_301_rows} | {r["patient_id"] for r in ext_ynzl_rows}
    collide = ext_pids & set(table.keys())
    if collide:
        print("[WARNING] %d patient_id(s) appear in BOTH main pool and reserved external: %s"
              % (len(collide), sorted(collide)[:10]))

    split_rows = {"dev": [], "internal_test": []}

    def assign(pids, name):
        for pid in pids:
            for r in table[pid]["rows"]:
                row = dict(r)
                row["split"] = name
                split_rows[name].append(row)

    assign(dev_pids, "dev")
    assign(test_pids, "internal_test")

    fieldnames = ["raw_path", "filename", "slide_id", "patient_id", "label", "type", "center", "pool", "split"]
    for name in ("dev", "internal_test"):
        write_csv(os.path.join(OUT_DIR, "%s.csv" % name), fieldnames, split_rows[name])

    summary = {
        "seed": SEED,
        "main_pool_slides": len(main_rows),
        "main_pool_patients": len(table),
        "unmatched_files": len(unmatched),
        "duplicate_slides_dropped": len(dups),
        "external_301_slides": len(ext_301_rows),
        "external_301_patients": len({r["patient_id"] for r in ext_301_rows}),
        "external_ynzl_slides": len(ext_ynzl_rows),
        "external_ynzl_patients": len({r["patient_id"] for r in ext_ynzl_rows}),
        "patient_id_collision_main_vs_external": sorted(collide),
        "splits": {},
    }
    for name in ("dev", "internal_test"):
        rows = split_rows[name]
        pids_here = {r["patient_id"] for r in rows}
        ct = crosstab(rows)
        summary["splits"][name] = {
            "slides": len(rows),
            "patients": len(pids_here),
            "label_type_counts": {"%s|%s" % k: v for k, v in ct.items()},
            "center_counts": dict(Counter(r["center"] for r in rows)),
        }

    with open(os.path.join(OUT_DIR, "split_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n================ SPLIT SUMMARY ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
