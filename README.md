# FunctionalCoverage

## Purpose

This package provides non-proprietary aggregate data, processed scenario identifiers, policy/configuration summaries, and figure/table scripts needed to audit and reproduce aggregate manuscript tables and figures from released CSV/YAML/JSON files.

It is intentionally not the full internal research codebase.

## Quick Start

Create and activate a virtual environment, then install the package requirements.

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run a fast integrity and figure-generation check:

```bash
python scripts/check_package_integrity.py
python scripts/reproduce_tables.py
python scripts/reproduce_main_figures.py --png-dpi 150
python scripts/reproduce_supplementary_figures.py --png-dpi 150
```

Reproduced outputs are written to `reproduced_tables/` and `reproduced_figures/`.
For publication-quality figure regeneration, omit `--png-dpi 150` to use the scripts' default higher-resolution PNG output while keeping the same PDF outputs.

