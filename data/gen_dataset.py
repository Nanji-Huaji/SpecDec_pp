import os
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import argparse
# import os  <-- removed since it's already imported above
from util import get_dataset, get_model, CKPT

# Copy helper functions from gen_dataset.py
B_INST, E_INST = "[INST]", "[/INST]"
import subprocess
import re


def select_best_gpu():
    """
    Select the GPU with the most free memory.
    Returns the GPU index as a string (e.g., "0").
    """
    # Check if CUDA_VISIBLE_DEVICES is already set by user
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(
            f"CUDA_VISIBLE_DEVICES is already set to {os.environ['CUDA_VISIBLE_DEVICES']}. Using specified GPU."
        )
        return os.environ["CUDA_VISIBLE_DEVICES"]

    try:
        # Run nvidia-smi to get memory usage
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )

        # Parse output: "0, 12000" -> index=0, free_mem=12000
        gpu_stats = []
        for line in result.strip().split("\n"):
            idx, free_mem = line.split(",")
            gpu_stats.append((int(idx), int(free_mem.strip())))

        if not gpu_stats:
            print("No GPUs found via nvidia-smi. Defaulting to 0.")
            return "0"

        # Sort by free memory descending
        gpu_stats.sort(key=lambda x: x[1], reverse=True)
        best_gpu_idx, best_free_mem = gpu_stats[0]

        print(
            f"Auto-selecting GPU {best_gpu_idx} with {best_free_mem} MiB free."
        )
        return str(best_gpu_idx)

    except FileNotFoundError:
        print("nvidia-smi not found. Defaulting to GPU 0.")
        return "0"
    except Exception as e:
        print(f"Error selecting GPU: {e}. Defaulting to GPU 0.")
        return "0"


def get_prompt_alpaca(sample):
    if sample["input"] is None or len(sample["input"].strip()) == 0:
        prompt = sample["instruction"]
    else:
        prompt = sample["instruction"] + "\nInput: " + sample["input"]
    return prompt


def get_prompt_humaneval(sample):
    INSTRUCTION = """Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{prompt}

### Response:"""
    return INSTRUCTION.format(prompt=sample["prompt"])


def get_prompt_gsm8k(sample):
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    return problem_prompt.format(instruction=sample["question"])


def get_prompt(sample, dataset_name, model_name=""):
    """
    wrap the prompt in llama-2-chat format or qwen format.
    """
    if dataset_name == "tatsu-lab/alpaca":
        prompt = get_prompt_alpaca(sample)
    elif dataset_name == "openai_humaneval":
        prompt = get_prompt_humaneval(sample)
    elif dataset_name == "gsm8k_test":
        prompt = get_prompt_gsm8k(sample)

    if "qwen" in str(model_name).lower():
        # Qwen ChatML format
        return f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n"
    
    if "vicuna" in str(model_name).lower():
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user.\n\nUSER: {prompt.strip()}\nASSISTANT:"

    return f"{B_INST} {prompt.strip()} {E_INST}"


def parse_args():
    parser = argparse.ArgumentParser(description="data generator with vllm")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["hf"], default="hf")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--n_begin", type=int, default=0)
    parser.add_argument("--n_end", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()
    return args


def main(args):
    if os.path.exists(args.output_file):
        print(f"Output file {args.output_file} already exists. Exiting to avoid overwrite.")
        return
    print(f"Using vLLM with do_sample={args.do_sample}")

    # Auto-select GPU before initializing vLLM
    # This must be done BEFORE importing vLLM or initializing it if vLLM respects CUDA_VISIBLE_DEVICES
    # However, since vLLM is already imported at top, setting environ here might affect LLM.__init__
    best_gpu = select_best_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = best_gpu

    # Resolve model path using util.CKPT logic
    # Note: vLLM loads model by path directly
    root_dir = (
        "/home/tty/code/DuoDecoding"  # hardcoded based on util.py context
    )
    # Try to resolve simplified name if present in CKPT, else use as is
    # Logic copied roughly from util.py but adapted because util.py modifies CKPT in place
    model_path = args.model_name
    # If users passed a short name key that exists in CKPT logic (though util.py is tricky to import variables from perfectly if they run code outside main)
    # We will assume args.model_name is the full path or HF hub id as passed in shell script

    print(f"Loading model: {model_path}")

    # Initialize vLLM
    # tensor_parallel_size=1 ideally for single GPU.
    # We set max_model_len to avoid large KV cache memory reservation (e.g. 4096 instead of 40960)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85, # Increased utilization
        max_model_len=4096,          # Hard limit sequence length to save KV cache memory
        dtype="bfloat16",            # Enforce bfloat16 for modern GPUs/Models like Qwen
    )

    # Load dataset
    dataset = get_dataset(args.dataset_name)

    # Determine range
    if args.n_end == -1:
        args.n_end = len(dataset)
    n_end = min(args.n_end, len(dataset))

    print(f"Processing range: {args.n_begin} to {n_end}")

    # Prepare prompts
    prompts = []
    # We need to keep indices to map back if needed, but here we just process sequentially
    for i in range(args.n_begin, n_end):
        sample = dataset[i]
        prompt_text = get_prompt(sample, args.dataset_name, args.model_name)
        prompts.append(prompt_text)

    if not prompts:
        print("No prompts to process.")
        return

    # Set sampling params
    # Note: Gen_dataset.py used max_length as 'length of generation', i.e. max_new_tokens
    sampling_params = SamplingParams(
        max_tokens=args.max_length,
        temperature=1.0 if args.do_sample else 0.0,
        top_p=1.0 if args.do_sample else 1.0,
        # stop_token_ids etc can be added if needed, but Qwen handles special tokens well
    )

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Format results
    res_dict = []

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        generated_token_ids = list(output.outputs[0].token_ids)
        prompt_token_ids = list(output.prompt_token_ids)

        # Original format requires 'prompt' (text), 'continuation' (text), 'prefix' (str list), 'tokens' (str list)
        res_dict.append(
            {
                "prompt": prompts[i],
                "continuation": generated_text,
                "prefix": str(prompt_token_ids),
                "tokens": str(generated_token_ids),
            }
        )

    # Infer output filename if not provided
    if args.output_file is None:
        args.output_file = f"dataset{args.n_begin}to{n_end}_{args.mode}{os.path.basename(args.model_name)}.json"

    print(f"Saving {len(res_dict)} results to {args.output_file}")
    with open(args.output_file, "w") as f:
        f.write(json.dumps(res_dict, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
