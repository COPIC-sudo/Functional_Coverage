from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_figures_impl import generate_all_paper_figures
from _carsim_hil_figures_impl import generate_all_carsim_hil_paper_figures
from _rss_highd_figures_impl import ResultsStore, tidy_coverage_dataframe, load_main_tables, load_stratum_table, plot_supp_transfer_curves, plot_highd_strata_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce supplementary figures from released aggregate data.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reproduced_figures" / "supplementary")
    parser.add_argument("--png-dpi", type=int, default=600)
    parser.add_argument("--with-ci", action="store_true", help="Add light CI ribbons to RSS/highD transfer curves if CI columns are present.")
    args = parser.parse_args()

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    paper_out = out / "paper_bundle"
    generate_all_paper_figures(
        results_root=ROOT / "aggregate_data" / "paper_bundle",
        output_root=paper_out,
        png_dpi=args.png_dpi,
    )
    print(f"Paper-bundle supplementary figures written under {paper_out / 'supplement'}")

    rss_out = out / "rss_highd"
    source_dir = rss_out / "source_data"
    supp_dir = rss_out / "supplement"
    supp_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    store = ResultsStore(ROOT / "aggregate_data" / "rss_highd_zero_tune")
    try:
        coverage_tidy = tidy_coverage_dataframe(store)
        main_table = load_main_tables(store)
        stratum_df = load_stratum_table(store)
        plot_supp_transfer_curves(
            coverage_tidy=coverage_tidy,
            main_table=main_table,
            out_path=supp_dir / "figS08_supp_rss_highd_curves",
            source_data_path=source_dir / "figS08_supp_rss_highd_curves.csv",
            with_ci=args.with_ci,
        )
        plot_highd_strata_summary(
            stratum_df=stratum_df,
            out_path=supp_dir / "figS09_supp_highd_strata",
            source_data_path=source_dir / "figS09_supp_highd_strata.csv",
        )
    finally:
        store.close()
    print(f"RSS/highD supplementary figures written under {supp_dir}")

    ch_out = out / "carsim_hil"
    generate_all_carsim_hil_paper_figures(
        results_root=ROOT / "aggregate_data" / "carsim_hil_results_csv",
        output_root=ch_out,
        png_dpi=args.png_dpi,
    )
    print(f"CarSim/HIL supplementary figures written under {ch_out / 'supplement'}")


if __name__ == "__main__":
    main()
