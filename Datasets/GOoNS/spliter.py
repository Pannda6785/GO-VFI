import json
import random
from pathlib import Path
from typing import List, Tuple

# =========================
# CONFIG
# =========================
ROOT = Path(".")          # run from folder A
RANDOM_SEED = 20

# default behavior for "normal" B folders
DEFAULT_SPLITS = {
    "train": 0.65,
    "val":   0.12,
    "test":  0.23,
}

# special behavior for "Font" B folder (kept from your script)
FONT_SPLITS = {
    "train": 0.40,
    "val":   0.15,
    "test":  0.45,
}

# NEW: text folders behavior
TEXT_TRAINVAL_SPLITS = {"train": 0.85, "val": 0.15}  # within Text_For_Training only
TEXT_TRAIN_FOLDER_NAME = "Text_For_Training"
TEXT_TEST_FOLDER_NAME  = "Text_For_Testing"

INCLUDE_EXTS = None       # e.g. {".png"}
# =========================

random.seed(RANDOM_SEED)


def list_files(folder: Path) -> List[Path]:
    files = [p for p in folder.rglob("*") if p.is_file()]
    if INCLUDE_EXTS is not None:
        files = [p for p in files if p.suffix.lower() in INCLUDE_EXTS]
    return files


def write_list(path: str, data: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in sorted(data):
            f.write(item + "\n")


def assign_with_min_one(C_folders: List[Tuple[Path, int]], splits: dict):
    """
    Assign whole C folders according to file-count-based targets.
    Guarantee at least one folder per split (if possible).
    """
    total_files = sum(n for _, n in C_folders)
    targets = {k: total_files * v for k, v in splits.items()}

    C_folders = list(C_folders)
    random.shuffle(C_folders)

    assignments = {}
    split_counts = {k: 0 for k in splits}
    split_file_counts = {k: 0 for k in splits}
    splits_list = list(splits.keys())

    # First pass: ensure at least one folder per split (if possible)
    if len(C_folders) >= len(splits_list):
        for split, (C, n_files) in zip(splits_list, C_folders):
            assignments[C] = split
            split_counts[split] += 1
            split_file_counts[split] += n_files
        remaining = C_folders[len(splits_list):]
    else:
        remaining = C_folders

    # Greedy fill by largest deficit
    for C, n_files in remaining:
        deficits = {k: targets[k] - split_file_counts[k] for k in splits}
        chosen = max(deficits, key=lambda k: deficits[k])
        assignments[C] = chosen
        split_counts[chosen] += 1
        split_file_counts[chosen] += n_files

    return assignments, split_counts, split_file_counts


def assign_text_trainval(C_folders: List[Tuple[Path, int]]):
    """
    For Text_For_Training: split only into train/val with ratio 85/15 (by file counts),
    keeping entire C folders together. Guarantees at least 1 folder in each split if possible.
    """
    return assign_with_min_one(C_folders, TEXT_TRAINVAL_SPLITS)


# =========================
# MAIN
# =========================
train_folders: List[str] = []
val_folders: List[str] = []
test_folders: List[str] = []
report_data = {}

for B in sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
    B_lower = B.name.lower()

    # Collect C folders under this B
    C_folders: List[Tuple[Path, int]] = []
    for C in sorted([p for p in B.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        files = list_files(C)
        if files:
            C_folders.append((C, len(files)))

    if not C_folders:
        continue

    report_data[B.name] = []

    # ---- Special case: Text_For_Testing => all test
    if B.name == TEXT_TEST_FOLDER_NAME or B_lower == TEXT_TEST_FOLDER_NAME.lower():
        for C, n_files in C_folders:
            rel_path = str(C.relative_to(ROOT)).replace("\\", "/")
            test_folders.append(rel_path)
            report_data[B.name].append({"C": C.name, "files": n_files, "split": "test"})
        continue

    # ---- Special case: Text_For_Training => only train/val with 85/15
    if B.name == TEXT_TRAIN_FOLDER_NAME or B_lower == TEXT_TRAIN_FOLDER_NAME.lower():
        assignments, _, _ = assign_text_trainval(C_folders)
        for C, n_files in C_folders:
            split = assignments[C]  # train or val
            rel_path = str(C.relative_to(ROOT)).replace("\\", "/")
            if split == "train":
                train_folders.append(rel_path)
            else:
                val_folders.append(rel_path)
            report_data[B.name].append({"C": C.name, "files": n_files, "split": split})
        continue

    # ---- Existing logic: Font uses FONT_SPLITS, others use DEFAULT_SPLITS
    splits = FONT_SPLITS if B_lower == "font" else DEFAULT_SPLITS
    assignments, _, _ = assign_with_min_one(C_folders, splits)

    for C, n_files in C_folders:
        split = assignments[C]
        rel_path = str(C.relative_to(ROOT)).replace("\\", "/")

        if split == "train":
            train_folders.append(rel_path)
        elif split == "val":
            val_folders.append(rel_path)
        else:
            test_folders.append(rel_path)

        report_data[B.name].append({"C": C.name, "files": n_files, "split": split})

# =========================
# Write split txt files
# =========================
write_list("train.txt", train_folders)
write_list("val.txt", val_folders)
write_list("test.txt", test_folders)

# =========================
# Build report
# =========================
total_train = sum(d["files"] for b in report_data.values() for d in b if d["split"] == "train")
total_val   = sum(d["files"] for b in report_data.values() for d in b if d["split"] == "val")
total_test  = sum(d["files"] for b in report_data.values() for d in b if d["split"] == "test")
total_all = total_train + total_val + total_test

report = {
    "seed": RANDOM_SEED,
    "default_splits": DEFAULT_SPLITS,
    "font_splits": FONT_SPLITS,
    "text_train_folder": TEXT_TRAIN_FOLDER_NAME,
    "text_test_folder": TEXT_TEST_FOLDER_NAME,
    "text_trainval_splits": TEXT_TRAINVAL_SPLITS,
    "overall_files": {
        "train": total_train,
        "val": total_val,
        "test": total_test,
        "total": total_all,
        "train_frac": (total_train / total_all) if total_all else 0,
        "val_frac": (total_val / total_all) if total_all else 0,
        "test_frac": (total_test / total_all) if total_all else 0,
    },
    "per_B": report_data,
}

with open("split_report.txt", "w", encoding="utf-8") as f:
    f.write("SPLIT REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(json.dumps(report, indent=2))

print("Done.")
print("train folders:", len(train_folders))
print("val folders:", len(val_folders))
print("test folders:", len(test_folders))