import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload SpecDec++ acceptance prediction heads to Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face model repo, for example 'your-org/specdecpp-acc-heads'.",
    )
    parser.add_argument(
        "--local-root",
        default="checkpoints/acc_head",
        help="Local root directory that contains pair-based acc head checkpoints.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help="Upload only the specified pair directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the remote repository as private if it does not exist.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Revision to upload to.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload SpecDec++ acceptance prediction heads",
        help="Commit message used for all uploads.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the upload plan without pushing anything.",
    )
    parser.add_argument(
        "--include-trainer-state",
        action="store_true",
        help="Upload trainer_state.json files as well.",
    )
    return parser.parse_args()


def iter_run_dirs(
    local_root: Path, selected_pairs: list[str] | None
) -> list[tuple[str, Path]]:
    if not local_root.exists():
        raise FileNotFoundError(f"Local root does not exist: {local_root}")

    allowed_pairs = set(selected_pairs or [])
    run_dirs: list[tuple[str, Path]] = []
    for pair_dir in sorted(local_root.iterdir()):
        if not pair_dir.is_dir():
            continue
        if allowed_pairs and pair_dir.name not in allowed_pairs:
            continue
        for run_dir in sorted(pair_dir.iterdir()):
            if run_dir.is_dir():
                run_dirs.append((pair_dir.name, run_dir))
    return run_dirs


def main() -> None:
    args = parse_args()
    local_root = Path(args.local_root)
    run_dirs = iter_run_dirs(local_root, args.pair)

    if not run_dirs:
        raise SystemExit(
            "No checkpoint directories matched the requested upload scope."
        )

    api = HfApi()

    print(f"Target repo: {args.repo_id}")
    print(f"Local root: {local_root}")
    print("Upload plan:")
    for pair_name, run_dir in run_dirs:
        print(f"- {run_dir} -> {pair_name}/{run_dir.name}")

    if args.dry_run:
        return

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    ignore_patterns = None
    if not args.include_trainer_state:
        ignore_patterns = ["trainer_state.json"]

    for pair_name, run_dir in run_dirs:
        path_in_repo = f"{pair_name}/{run_dir.name}"
        print(f"Uploading {run_dir} -> {path_in_repo}")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(run_dir),
            path_in_repo=path_in_repo,
            revision=args.revision,
            commit_message=args.commit_message,
            ignore_patterns=ignore_patterns,
        )


if __name__ == "__main__":
    main()
