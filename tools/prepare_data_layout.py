"""Create common local CSV aliases for notebooks (no downloads are performed)."""
from pathlib import Path
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ALIAS = {
    "train.csv": ["train.csv", "train_ross.csv"],
    "test.csv": ["test.csv", "test_ross.csv"],
    "store.csv": ["store.csv", "store_ross.csv"],
    "sample_submission.csv": ["sample_submission.csv"],
}

def link_or_copy(source: Path, dest: Path) -> None:
    if dest.exists():
        print(f"Exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)
    print(f"Ready: {dest}")

if __name__ == "__main__":
    for name, aliases in ALIAS.items():
        src = RAW / name
        if not src.is_file():
            print(f"Missing: {src}")
            continue
        for destdir in [ROOT, ROOT / "input"]:
            for alias in aliases:
                if destdir.name == "input" and alias.endswith("_ross.csv"):
                    continue
                link_or_copy(src, destdir / alias)
