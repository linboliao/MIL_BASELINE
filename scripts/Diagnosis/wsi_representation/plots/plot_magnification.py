"""Plot Diagnosis magnification configurations."""

from common import build_parser, plot_comparison


COMPARISON = "Mag"
DISPLAY_ORDER = ["5x", "10x", "20x"]
OUTPUT_STEM = "mag_config_comparison"


def main() -> None:
    args = build_parser(COMPARISON).parse_args()
    paths = plot_comparison(
        comparison=COMPARISON,
        display_order=DISPLAY_ORDER,
        result_root=args.result_root,
        output_dir=args.output_dir,
        output_stem=OUTPUT_STEM,
        show=args.show,
    )
    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
