from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_figures_impl import generate_all_paper_figures
from _carsim_hil_figures_impl import generate_all_carsim_hil_paper_figures
from _rss_highd_figures_impl import ResultsStore, tidy_coverage_dataframe, load_main_tables, plot_main_transfer_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce main manuscript figures from released aggregate data.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reproduced_figures" / "main")
    parser.add_argument("--png-dpi", type=int, default=600)
    args = parser.parse_args()

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    paper_out = out / "paper_bundle"
    generate_all_paper_figures(
        results_root=ROOT / "aggregate_data" / "paper_bundle",
        output_root=paper_out,
        png_dpi=args.png_dpi,
    )
    print(f"Paper-bundle figures written under {paper_out}")

    rss_out = out / "rss_highd"
    source_dir = rss_out / "source_data"
    (rss_out / "main").mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    store = ResultsStore(ROOT / "aggregate_data" / "rss_highd_zero_tune")
    try:
        plot_main_transfer_overlay(
            coverage_tidy=tidy_coverage_dataframe(store),
            main_table=load_main_tables(store),
            out_path=rss_out / "main" / "fig05_main_rss_highd_transfer",
            source_data_path=source_dir / "fig05_main_rss_highd_transfer.csv",
        )
    finally:
        store.close()
    print(f"RSS/highD main figure written under {rss_out / 'main'}")

    ch_out = out / "carsim_hil"
    generate_all_carsim_hil_paper_figures(
        results_root=ROOT / "aggregate_data" / "carsim_hil_results_csv",
        output_root=ch_out,
        png_dpi=args.png_dpi,
    )
    print(f"CarSim/HIL main figures written under {ch_out / 'main'}")
    print("Note: helper functions also emit associated supplementary figures where the original implementation does so.")


if __name__ == "__main__":
    main()
