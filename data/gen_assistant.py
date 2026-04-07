import argparse
import json
import os
from ast import literal_eval as eval

import torch
from tqdm import tqdm

from util import get_input_device, get_model, pretty_format


def read_data(filename):
    data = json.load(open(filename, "r"))
    for item in data:
        item["prefix"] = eval(item["prefix"])
        item["tokens"] = eval(item["tokens"])

    return data


def slice_data(data, n_begin, n_end):
    if n_end < 0 or n_end > len(data):
        n_end = len(data)
    n_begin = max(0, n_begin)
    if n_begin > n_end:
        raise ValueError(f"Invalid range: [{n_begin}, {n_end})")
    return data[n_begin:n_end]


@torch.no_grad()
def get_assistant_result(data, assistant_model, do_sample, batch_size=None):
    input_device = get_input_device(assistant_model)
    if batch_size is None:
        batch_size = 4 if input_device.type == "cpu" else 16

    print(
        f"Generating draft tokens for {len(data)} samples with batch_size={batch_size}"
    )

    for batch_start in tqdm(
        range(0, len(data), batch_size), desc="Generating draft batches"
    ):
        batch = data[batch_start : batch_start + batch_size]
        joints = [item["prefix"] + item["tokens"] for item in batch]
        max_len = max(len(joint) for joint in joints)

        input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row, joint in enumerate(joints):
            joint_tensor = torch.tensor(joint, dtype=torch.long)
            input_ids[row, : len(joint)] = joint_tensor
            attention_mask[row, : len(joint)] = 1

        input_ids = input_ids.to(input_device)
        attention_mask = attention_mask.to(input_device)

        logits = assistant_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        if do_sample:
            probs = logits[:, :-1].softmax(dim=-1)
            generated = torch.multinomial(
                probs.reshape(-1, probs.size(-1)), num_samples=1
            ).reshape(probs.size(0), probs.size(1))
        else:
            generated = logits[:, :-1].argmax(dim=-1)

        for row, item in enumerate(batch):
            start = len(item["prefix"]) - 1
            end = start + len(item["tokens"])
            item["draft"] = generated[row, start:end].cpu().tolist()
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="data generator")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--n_begin", type=int, default=0)
    parser.add_argument("--n_end", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--device_mode",
        type=str,
        choices=["auto", "single_gpu"],
        default="auto",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.output_file):
        print(
            f"Output file {args.output_file} already exists. Exiting to avoid overwrite."
        )
        exit(0)

    data = slice_data(read_data(args.input_file), args.n_begin, args.n_end)
    _, model = get_model(
        args.model_name, device_mode=args.device_mode, load_tokenizer=False
    )
    data = get_assistant_result(
        data,
        model,
        do_sample=args.do_sample,
        batch_size=args.batch_size,
    )

    if args.output_file is None or len(args.output_file) == 0:
        suffix = "stochastic" if args.do_sample else "greedy"
        args.output_file = (
            args.input_file.rstrip(".json") + "_" + args.model_name + suffix + ".json"
        )

    data = pretty_format(data)
    with open(args.output_file, "w") as f:
        f.write(json.dumps(data, indent=2))
