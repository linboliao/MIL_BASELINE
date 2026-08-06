"""Plot Diagnosis stain-normalisation configurations."""

from common import build_parser, plot_comparison


COMPARISON = "Stains"
DISPLAY_ORDER = ["Macenko", "Reinhard", "Vahadane"]
OUTPUT_STEM = "stains_config_comparison"


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
