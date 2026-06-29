from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRS = [
    "configs", "scenario_ids", "aggregate_data/main_tables", "aggregate_data/main_figures",
    "aggregate_data/supplementary_tables", "aggregate_data/supplementary_figures", "scripts", "docs",
]
REQUIRED_FILES = [
    "README.md", "LICENSE", "LICENSE-CODE", "DATA_USE_NOTES.md", "MANIFEST.md", "reproduction_scope.md", "requirements.txt",
    "configs/policy_weak_ttc.yaml", "configs/policy_strong_ttc.yaml",
    "configs/policy_stopping_distance.yaml", "configs/policy_rss_longitudinal.yaml",
    "configs/physical_envelope_settings.yaml", "configs/random_seeds.json",
    "scenario_ids/ngsim_us101_denominator_ids.csv", "scenario_ids/ngsim_i80_denominator_ids.csv",
    "scenario_ids/ngsim_modeled_crash_denominator_ids.csv", "scenario_ids/ngsim_noncrash_complement_ids.csv",
    "scenario_ids/highd_transfer_scenario_ids.csv", "scenario_ids/carsim_hil_design_summary.csv",
    "aggregate_data/main_tables/extended_ttc_family_tuning_audit.csv",
    "aggregate_data/supplementary_tables/extended_ttc_family_tuning_audit.csv",
    "scripts/reproduce_main_figures.py", "scripts/reproduce_supplementary_figures.py", "scripts/reproduce_tables.py",
    "scripts/check_package_integrity.py", "docs/variable_dictionary.md", "docs/policy_configuration_notes.md",
    "docs/scenario_identifier_notes.md", "docs/excluded_materials.md",
]
BANNED_EXTENSIONS = {".dll", ".exe", ".sim", ".cpar", ".par", ".bin"}
RISK_PATTERNS = [
    re.compile(r"[A-Za-z]:\\(?!n|r|t)"),
    re.compile(r"/Users/|/home/|/mnt/|/Volumes/"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]|secret\s*[:=]|token\s*[:=]|password\s*[:=]|passwd\s*[:=]|credential\s*[:=])"),
    re.compile(r"(?i)(_tracks\.csv|tracksMeta\.csv|recordingMeta\.csv)"),
]
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".py"}
SCAN_EXEMPT_FILES = {
    Path("README.md"),
    Path("docs/excluded_materials.md"),
    Path("scripts/check_package_integrity.py"),
}


def main() -> None:
    missing = []
    for item in REQUIRED_DIRS:
        if not (ROOT / item).is_dir():
            missing.append(item + "/")
    for item in REQUIRED_FILES:
        if not (ROOT / item).is_file():
            missing.append(item)

    file_count = 0
    total_bytes = 0
    risky = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
        if path.suffix.lower() in BANNED_EXTENSIONS:
            risky.append(f"banned extension: {path.relative_to(ROOT)}")
        if path.relative_to(ROOT) in SCAN_EXEMPT_FILES:
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in RISK_PATTERNS:
                if pattern.search(text):
                    risky.append(f"risky string '{pattern.pattern}' in {path.relative_to(ROOT)}")
                    break

    print("FunctionalCoverage reproducibility package integrity check")
    print(f"Root: {ROOT}")
    print(f"Files: {file_count}")
    print(f"Size: {total_bytes / (1024 * 1024):.2f} MiB")
    print(f"Main table CSVs: {len(list((ROOT / 'aggregate_data/main_tables').glob('*.csv')))}")
    print(f"Supplementary table CSVs: {len(list((ROOT / 'aggregate_data/supplementary_tables').glob('*.csv')))}")
    print(f"Scenario ID CSVs: {len(list((ROOT / 'scenario_ids').glob('*.csv')))}")
    print("Required structure: OK" if not missing else "MISSING REQUIRED ITEMS:")
    for item in missing:
        print(f"  - {item}")
    print("Risk scan: OK" if not risky else "RISK WARNINGS:")
    for item in risky[:50]:
        print(f"  - {item}")
    if len(risky) > 50:
        print(f"  ... {len(risky) - 50} additional warnings")
    if missing or risky:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
