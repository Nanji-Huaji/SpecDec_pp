import argparse
import json
from ast import literal_eval as eval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the longest samples from a SpecDec++ JSON file."
    )
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--top_k",
        type=int,
        default=256,
        help="Number of longest samples to keep.",
    )
    return parser.parse_args()


def sample_length(item):
    prefix = eval(item["prefix"])
    tokens = eval(item["tokens"])
    return len(prefix) + len(tokens)


def main():
    args = parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    if args.top_k <= 0:
        raise ValueError("--top_k must be positive.")

    indexed = list(enumerate(data))
    indexed.sort(key=lambda pair: sample_length(pair[1]), reverse=True)
    selected = [item for _, item in indexed[: args.top_k]]

    lengths = [sample_length(item) for item in selected]
    print(f"Loaded {len(data)} samples from {args.input_file}")
    print(f"Selected top {len(selected)} longest samples")
    if lengths:
        print(
            f"Selected length range: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths) / len(lengths):.1f}"
        )

    with open(args.output_file, "w") as f:
        json.dump(selected, f, indent=2)
        f.write("\n")

    print(f"Wrote longest-sample subset to {args.output_file}")


if __name__ == "__main__":
    main()
