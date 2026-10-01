from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_propagation import KATZ_NORMALIZATIONS, PropagationConfig, add_common_arguments, run_propagation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--katz_beta", type=float, default=0.5, help="Decay per walk step.")
    parser.add_argument("--katz_max_length", type=int, default=4, help="Longest walk length L.")
    parser.add_argument("--katz_normalization", choices=KATZ_NORMALIZATIONS, default="sym",
                        help="none: raw walk counts (A); sym: D^-1/2 A D^-1/2; rw: A D^-1 (random-walk mass).")
    args = parser.parse_args()
    if args.katz_beta <= 0 or args.katz_max_length < 1:
        parser.error("--katz_beta must be positive and --katz_max_length at least 1")
    return args


def main() -> None:
    args = parse_args()
    config = PropagationConfig(
        method="katz",
        graph_dir=args.graph_dir,
        condition_map=args.condition_map,
        candidate_space=args.candidate_space,
        exclude_relations=tuple(args.exclude_relations),
        top_k=args.top_k,
        keep_zero_scores=args.keep_zero_scores,
        katz_beta=args.katz_beta,
        katz_max_length=args.katz_max_length,
        katz_normalization=args.katz_normalization,
    )
    run_propagation(
        config,
        args.queries_csv,
        args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        evaluate=not args.no_eval,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
