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
        item["draft"] = eval(item["draft"])
    return data


def slice_data(data, n_begin, n_end):
    if n_end < 0 or n_end > len(data):
        n_end = len(data)
    n_begin = max(0, n_begin)
    if n_begin > n_end:
        raise ValueError(f"Invalid range: [{n_begin}, {n_end})")
    return data[n_begin:n_end]


@torch.no_grad()
def get_log_prob(data, model, model_name, batch_size=None):
    input_device = get_input_device(model)
    if batch_size is None:
        batch_size = 4 if input_device.type == "cpu" else 16

    print(
        f"Calculating log probabilities for {len(data)} samples with batch_size={batch_size}"
    )

    for batch_start in tqdm(
        range(0, len(data), batch_size), desc="Calculating log-prob batches"
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

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = logits.log_softmax(dim=-1)

        for row, item in enumerate(batch):
            start = len(item["prefix"]) - 1
            end = start + len(item["draft"])
            draft_index = torch.tensor(
                item["draft"], dtype=torch.long, device=log_probs.device
            ).unsqueeze(-1)
            token_log_probs = torch.take_along_dim(
                log_probs[row, start:end], draft_index, dim=-1
            )
            item[f"log_p_{model_name}"] = token_log_probs[:, 0].cpu().tolist()

    return data


def parse_args():
    parser = argparse.ArgumentParser(description="data generator")
    parser.add_argument("--model_name", type=str, default="7b")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str, default=None)
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
    data = get_log_prob(data, model, args.model_name, batch_size=args.batch_size)

    suffix = "logP"
    if args.output_file is None or len(args.output_file) == 0:
        args.output_file = (
            args.input_file.rstrip(".json") + "_" + args.model_name + suffix + ".json"
        )

    data = pretty_format(data)
    with open(args.output_file, "w") as f:
        f.write(json.dumps(data, indent=2))
