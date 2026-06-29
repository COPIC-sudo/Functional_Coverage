from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def copy_csvs(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(src_dir.glob("*.csv")):
        shutil.copy2(src, dst_dir / src.name)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce manuscript table files from released aggregate CSV data.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reproduced_tables")
    args = parser.parse_args()

    main_count = copy_csvs(ROOT / "aggregate_data" / "main_tables", args.output_root / "main_tables")
    supp_count = copy_csvs(ROOT / "aggregate_data" / "supplementary_tables", args.output_root / "supplementary_tables")
    print(f"Copied {main_count} main-table CSV files to {args.output_root / 'main_tables'}")
    print(f"Copied {supp_count} supplementary-table CSV files to {args.output_root / 'supplementary_tables'}")
    print("These CSV files are the released aggregate table sources used for manuscript reporting.")


if __name__ == "__main__":
    main()
