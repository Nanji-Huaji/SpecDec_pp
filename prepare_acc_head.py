from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SPECDEC_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SPECDEC_ROOT / "data"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acc_head_registry import (  # noqa: E402
    build_acc_head_pair_name,
    canonicalize_model_name,
)


REGISTRY_PATH = SPECDEC_ROOT / "checkpoints" / "acc_head_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate acceptance-head training data, split it, and train a "
            "SpecDec++ prediction head in one command."
        )
    )
    parser.add_argument(
        "--draft-model", required=True, help="Draft model name or path."
    )
    parser.add_argument(
        "--target-model", required=True, help="Target model name or path."
    )
    parser.add_argument(
        "--draft-model-path",
        default=None,
        help=(
            "Optional local path or Hugging Face id used for actually loading the "
            "draft model. If omitted, --draft-model is used for both naming and loading."
        ),
    )
    parser.add_argument(
        "--target-model-path",
        default=None,
        help=(
            "Optional local path or Hugging Face id used for actually loading the "
            "target model. If omitted, --target-model is used for both naming and loading."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default="tatsu-lab/alpaca",
        help="Dataset id understood by SpecDec_pp/data/util.py.",
    )
    parser.add_argument(
        "--dataset-python",
        default=None,
        help=(
            "Python interpreter used only for gen_dataset.py. If omitted, uses "
            "the SPECDEC_DATASET_PYTHON environment variable when set, otherwise "
            "falls back to the current interpreter."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Directory used for generated training data. Defaults to "
            "src/SpecDec_pp/data/generated/<pair-name>/<dataset-slug>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Final prediction-head directory. Defaults to "
            "src/SpecDec_pp/checkpoints/acc_head/<pair-name>/exp-weight<W>-layer<L>."
        ),
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=40000,
        help="Number of samples for the training split.",
    )
    parser.add_argument(
        "--dev-samples",
        type=int,
        default=10000,
        help="Number of samples for the validation split.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=2000,
        help="Number of samples for the held-out split.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Generation length for target continuations.",
    )
    parser.add_argument(
        "--dataset-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="gpu_memory_utilization passed to vLLM during dataset generation.",
    )
    parser.add_argument(
        "--dataset-tensor-parallel-size",
        type=int,
        default=1,
        help="tensor_parallel_size passed to vLLM during dataset generation.",
    )
    parser.add_argument(
        "--dataset-max-model-len",
        type=int,
        default=4096,
        help="max_model_len passed to vLLM during dataset generation.",
    )
    parser.add_argument(
        "--dataset-disable-custom-all-reduce",
        action="store_true",
        help="Disable vLLM custom all-reduce kernels during dataset generation.",
    )
    parser.add_argument(
        "--data-nproc-per-node",
        type=int,
        default=1,
        help=(
            "Number of worker processes used for sliced data generation in "
            "gen_assistant.py and gen_log_p.py."
        ),
    )
    parser.add_argument(
        "--data-gpus",
        default=None,
        help=(
            "Comma-separated GPU ids assigned to data-generation workers. "
            "Defaults to 0..N-1 when --data-nproc-per-node > 1."
        ),
    )
    parser.add_argument(
        "--data-batch-size",
        type=int,
        default=None,
        help="Optional batch size override used by gen_assistant.py and gen_log_p.py.",
    )
    parser.add_argument(
        "--data-device-mode",
        default="auto",
        choices=["auto", "single_gpu"],
        help=(
            "Model placement mode for sliced data generation workers. Use "
            "single_gpu to force each worker onto its assigned GPU."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=3,
        help="Number of residual blocks in the prediction head.",
    )
    parser.add_argument(
        "--weight-mismatch",
        type=float,
        default=6.0,
        help="Positive-class mismatch weight used during training.",
    )
    parser.add_argument(
        "--mixing-ratio",
        type=float,
        default=0.15,
        help="Mixing ratio passed to SpecDec++ training.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=4,
        help="Per-device batch size for training.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=4,
        help="Per-device batch size for evaluation.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps for training.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
        help="Logging interval for training.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for training.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        default="cosine",
        help="Learning-rate scheduler type.",
    )
    parser.add_argument(
        "--bf16",
        default="True",
        help="Passed through to SpecDec++ train.py.",
    )
    parser.add_argument(
        "--wandb-project",
        default="specdecpp",
        help="WANDB_PROJECT value used for training.",
    )
    parser.add_argument(
        "--train-nproc-per-node",
        type=int,
        default=1,
        help=(
            "Number of GPU worker processes used for SpecDec++ training. "
            "Values greater than 1 launch training with torchrun."
        ),
    )
    parser.add_argument(
        "--train-master-port",
        type=int,
        default=29500,
        help="Master port used by torchrun when --train-nproc-per-node > 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing intermediate data and output head before rerunning.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data generation and reuse existing train/dev files.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use stochastic sampling in data generation. Enabled by default for compatibility.",
    )
    args = parser.parse_args()
    if not args.do_sample:
        args.do_sample = True
    return args


def dataset_slug(dataset_name: str) -> str:
    return dataset_name.replace("/", "__").replace(":", "-")


def run_cmd(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"\n[run] cwd={cwd}")
    print("[run] " + " ".join(command))
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def relative_repo_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def resolve_dataset_python(explicit_python: str | None) -> str:
    if explicit_python:
        return explicit_python

    env_python = os.environ.get("SPECDEC_DATASET_PYTHON")
    if env_python:
        return env_python

    return sys.executable


def build_train_command(
    args: argparse.Namespace,
    train_file: Path,
    dev_file: Path,
    output_dir: Path,
    draft_model_ref: str,
) -> list[str]:
    train_entrypoint = [
        "train.py",
        "--data_path",
        str(train_file),
        "--eval_data_path",
        str(dev_file),
        "--output_dir",
        str(output_dir),
        "--model_name_or_path",
        draft_model_ref,
        "--bf16",
        str(args.bf16),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--logging_steps",
        str(args.logging_steps),
        "--eval_strategy",
        "epoch",
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--weight_mismatch",
        str(args.weight_mismatch),
        "--save_strategy",
        "no",
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--lr_scheduler_type",
        args.lr_scheduler_type,
        "--resnet_num_layers",
        str(args.layer),
        "--mixing_ratio",
        str(args.mixing_ratio),
    ]

    if args.train_nproc_per_node > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(args.train_nproc_per_node),
            "--master_port",
            str(args.train_master_port),
        ] + train_entrypoint

    return [sys.executable] + train_entrypoint


def parse_data_gpus(data_gpus: str | None, worker_count: int) -> list[str]:
    if worker_count <= 1:
        return []

    if data_gpus is None:
        return [str(idx) for idx in range(worker_count)]

    gpus = [gpu.strip() for gpu in data_gpus.split(",") if gpu.strip()]
    if len(gpus) < worker_count:
        raise ValueError(
            f"Need at least {worker_count} GPU ids for data generation, got {len(gpus)}."
        )
    return gpus[:worker_count]


def load_json_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json_records(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")


def partition_ranges(total_size: int, num_parts: int) -> list[tuple[int, int]]:
    base, remainder = divmod(total_size, num_parts)
    ranges: list[tuple[int, int]] = []
    start = 0
    for idx in range(num_parts):
        end = start + base + (1 if idx < remainder else 0)
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def part_file(output_path: Path, part_idx: int) -> Path:
    return output_path.with_name(
        f"{output_path.stem}.part{part_idx:03d}{output_path.suffix}"
    )


def merge_part_files(part_paths: Iterable[Path], output_path: Path) -> None:
    merged: list[dict] = []
    for path in part_paths:
        merged.extend(load_json_records(path))
    dump_json_records(output_path, merged)


def run_data_stage(
    *,
    stage_script: str,
    model_name: str,
    input_file: Path,
    output_file: Path,
    worker_count: int,
    gpus: list[str],
    batch_size: int | None,
    device_mode: str,
    extra_args: list[str],
) -> None:
    if worker_count <= 1:
        command = [
            sys.executable,
            stage_script,
            "--model_name",
            model_name,
            "--input_file",
            str(input_file),
            "--output_file",
            str(output_file),
            "--device_mode",
            device_mode,
        ]
        if batch_size is not None:
            command += ["--batch_size", str(batch_size)]
        command += extra_args
        run_cmd(command, cwd=DATA_ROOT)
        return

    total_records = len(load_json_records(input_file))
    ranges = partition_ranges(total_records, worker_count)
    processes: list[subprocess.Popen] = []
    part_paths: list[Path] = []
    try:
        for part_idx, (start, end) in enumerate(ranges):
            part_path = part_file(output_file, part_idx)
            part_paths.append(part_path)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpus[part_idx]
            command = [
                sys.executable,
                stage_script,
                "--model_name",
                model_name,
                "--input_file",
                str(input_file),
                "--output_file",
                str(part_path),
                "--n_begin",
                str(start),
                "--n_end",
                str(end),
                "--device_mode",
                device_mode,
            ]
            if batch_size is not None:
                command += ["--batch_size", str(batch_size)]
            command += extra_args
            print(f"\n[run] cwd={DATA_ROOT}")
            print(
                "[run] "
                + "CUDA_VISIBLE_DEVICES="
                + gpus[part_idx]
                + " "
                + " ".join(command)
            )
            processes.append(subprocess.Popen(command, cwd=str(DATA_ROOT), env=env))

        for process in processes:
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, process.args)

        merge_part_files(part_paths, output_file)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


def update_acc_head_registry(
    source_model: str,
    target_model: str,
    output_dir: Path,
) -> dict[str, str]:
    source_alias = canonicalize_model_name(source_model)
    target_alias = canonicalize_model_name(target_model)
    pair_name = build_acc_head_pair_name(source_model, target_model)
    relative_output_dir = relative_repo_path(output_dir)

    if REGISTRY_PATH.exists():
        with REGISTRY_PATH.open("r", encoding="utf-8") as f:
            entries = json.load(f)
    else:
        entries = []

    updated = False
    for entry in entries:
        if entry.get("source") == source_alias and entry.get("target") == target_alias:
            entry["local_path"] = relative_output_dir
            entry.setdefault("hf_repo", "ArcticHuaji/specdecpp-acc-heads")
            entry["hf_subpath"] = relative_output_dir.removeprefix(
                "src/SpecDec_pp/checkpoints/acc_head/"
            )
            updated = True
            registry_entry = entry
            break
    else:
        registry_entry = {
            "source": source_alias,
            "target": target_alias,
            "local_path": relative_output_dir,
            "hf_repo": "ArcticHuaji/specdecpp-acc-heads",
            "hf_subpath": relative_output_dir.removeprefix(
                "src/SpecDec_pp/checkpoints/acc_head/"
            ),
        }
        entries.append(registry_entry)

    if not updated:
        entries.sort(key=lambda item: (item["source"], item["target"]))

    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[done] Registry updated for {pair_name}: {registry_entry['local_path']}")
    return registry_entry


def main() -> None:
    args = parse_args()

    draft_model_ref = args.draft_model_path or args.draft_model
    target_model_ref = args.target_model_path or args.target_model

    pair_name = build_acc_head_pair_name(args.draft_model, args.target_model)
    run_name = f"exp-weight{args.weight_mismatch:g}-layer{args.layer}"

    dataset_dir = (
        Path(args.dataset_dir)
        if args.dataset_dir is not None
        else DATA_ROOT / "generated" / pair_name / dataset_slug(args.dataset_name)
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else SPECDEC_ROOT / "checkpoints" / "acc_head" / pair_name / run_name
    )

    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    tmp_dir = dataset_dir / "tmp"
    all_file = dataset_dir / "all.json"
    tmp1 = tmp_dir / "tmp1_target.json"
    tmp2 = tmp_dir / "tmp2_draft.json"
    tmp3 = tmp_dir / "tmp3_draft_logp.json"
    tmp4 = tmp_dir / "tmp4_target_logp.json"
    train_file = dataset_dir / "train.json"
    dev_file = dataset_dir / "dev.json"
    test_file = dataset_dir / "test.json"
    data_gpus = parse_data_gpus(args.data_gpus, args.data_nproc_per_node)

    if args.overwrite:
        for path in [tmp1, tmp2, tmp3, tmp4, all_file, train_file, dev_file, test_file]:
            remove_path(path)
            if path.suffix:
                for part_path in tmp_dir.glob(f"{path.stem}.part*{path.suffix}"):
                    remove_path(part_path)
        if output_dir.exists():
            remove_path(output_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_data:
        common_python = [sys.executable]
        dataset_python = [resolve_dataset_python(args.dataset_python)]

        gen_dataset_cmd = dataset_python + [
            "gen_dataset.py",
            "--dataset_name",
            args.dataset_name,
            "--model_name",
            target_model_ref,
            "--mode",
            "hf",
            "--max_length",
            str(args.max_length),
            "--gpu_memory_utilization",
            str(args.dataset_gpu_memory_utilization),
            "--tensor_parallel_size",
            str(args.dataset_tensor_parallel_size),
            "--max_model_len",
            str(args.dataset_max_model_len),
            "--output_file",
            str(tmp1),
        ]
        if args.dataset_disable_custom_all_reduce:
            gen_dataset_cmd.append("--disable_custom_all_reduce")
        if args.do_sample:
            gen_dataset_cmd.append("--do_sample")
        run_cmd(gen_dataset_cmd, cwd=DATA_ROOT)

        assistant_extra_args: list[str] = ["--do_sample"] if args.do_sample else []
        run_data_stage(
            stage_script="gen_assistant.py",
            model_name=draft_model_ref,
            input_file=tmp1,
            output_file=tmp2,
            worker_count=args.data_nproc_per_node,
            gpus=data_gpus,
            batch_size=args.data_batch_size,
            device_mode=args.data_device_mode,
            extra_args=assistant_extra_args,
        )

        run_data_stage(
            stage_script="gen_log_p.py",
            model_name=draft_model_ref,
            input_file=tmp2,
            output_file=tmp3,
            worker_count=args.data_nproc_per_node,
            gpus=data_gpus,
            batch_size=args.data_batch_size,
            device_mode=args.data_device_mode,
            extra_args=[],
        )

        run_data_stage(
            stage_script="gen_log_p.py",
            model_name=target_model_ref,
            input_file=tmp3,
            output_file=tmp4,
            worker_count=args.data_nproc_per_node,
            gpus=data_gpus,
            batch_size=args.data_batch_size,
            device_mode=args.data_device_mode,
            extra_args=[],
        )

        run_cmd(
            common_python
            + [
                "gen_acceptance.py",
                "--target_name",
                target_model_ref,
                "--draft_name",
                draft_model_ref,
                "--input_file",
                str(tmp4),
                "--output_file",
                str(all_file),
            ],
            cwd=DATA_ROOT,
        )

        run_cmd(
            common_python
            + [
                "split_dataset.py",
                str(all_file),
                str(args.train_samples),
                str(args.dev_samples),
                str(args.test_samples),
            ],
            cwd=DATA_ROOT,
        )

    if not train_file.exists() or not dev_file.exists():
        raise FileNotFoundError(
            f"Missing split files under {dataset_dir}. Expected {train_file} and {dev_file}."
        )

    train_env = os.environ.copy()
    train_env["WANDB_PROJECT"] = args.wandb_project
    if (
        "WANDB_API_KEY" not in train_env
        and "WANDB_MODE" not in train_env
        and "WANDB_DISABLED" not in train_env
    ):
        train_env["WANDB_MODE"] = "offline"
        print("[info] WANDB_API_KEY not set, defaulting to WANDB_MODE=offline")

    train_cmd = build_train_command(
        args=args,
        train_file=train_file,
        dev_file=dev_file,
        output_dir=output_dir,
        draft_model_ref=draft_model_ref,
    )
    run_cmd(train_cmd, cwd=SPECDEC_ROOT / "specdec_pp", env=train_env)

    registry_entry = update_acc_head_registry(
        source_model=args.draft_model,
        target_model=args.target_model,
        output_dir=output_dir,
    )

    print("\n[done] Acceptance head prepared successfully.")
    print(f"[done] Pair: {pair_name}")
    print(f"[done] Draft model load ref: {draft_model_ref}")
    print(f"[done] Target model load ref: {target_model_ref}")
    print(f"[done] Data directory: {dataset_dir}")
    print(f"[done] Head directory: {output_dir}")
    print(f"[done] Registry entry: {registry_entry}")


if __name__ == "__main__":
    main()
