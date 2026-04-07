from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
import time
import json
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
import os
import argparse

# Bypass FP8 hardware check for Ampere GPUs
import transformers.quantizers.auto as auto_quantizer

if hasattr(auto_quantizer, "get_hf_quantizer"):
    _original_get_hf_quantizer = auto_quantizer.get_hf_quantizer

    def _patched_get_hf_quantizer(*args, **kwargs):
        res = _original_get_hf_quantizer(*args, **kwargs)
        if isinstance(res, tuple):
            quantizer = res[0]
        else:
            quantizer = res
        if quantizer is not None and hasattr(quantizer, "validate_environment"):
            quantizer.validate_environment = lambda *args, **kwargs: None
        return res

    auto_quantizer.get_hf_quantizer = _patched_get_hf_quantizer

# Also patch torch.cuda.get_device_capability just in case
_old_cap = torch.cuda.get_device_capability
torch.cuda.get_device_capability = lambda device=None: (8, 9)


def dequantize_fp8_model(model):
    """
    Manually dequantize FP8Linear layers to standard BF16 Linear layers.
    This allows running FP8 models on hardware that doesn't support FP8 kernels (like Ampere).
    """
    from transformers.integrations.finegrained_fp8 import FP8Linear
    import torch.nn as nn

    print("Dequantizing FP8 weights to BF16 for calculation...")

    # We iterate over modules. We need to collect them first to avoid mutation issues during iteration.
    fp8_modules = []
    for name, module in model.named_modules():
        if isinstance(module, FP8Linear):
            fp8_modules.append((name, module))

    for name, module in fp8_modules:
        # Get parent and child name
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            child_name = name.rsplit(".", 1)[1]
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name

        with torch.no_grad():
            # Dequantize on CPU to save GPU VRAM
            device = module.weight.device
            w_fp8 = module.weight.to("cpu", dtype=torch.bfloat16)
            s_inv = module.weight_scale_inv.to("cpu", dtype=torch.bfloat16)

            if s_inv.ndim == 2 and s_inv.shape[0] < w_fp8.shape[0]:
                h_block = w_fp8.shape[0] // s_inv.shape[0]
                w_block = w_fp8.shape[1] // s_inv.shape[1]
                w_reshaped = w_fp8.view(
                    s_inv.shape[0], h_block, s_inv.shape[1], w_block
                )
                bf16_weight = (
                    w_reshaped * s_inv.view(s_inv.shape[0], 1, s_inv.shape[1], 1)
                ).reshape(w_fp8.shape)
            else:
                bf16_weight = w_fp8 * s_inv

            new_linear = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                dtype=torch.bfloat16,
            )
            new_linear.weight.copy_(bf16_weight)
            if module.bias is not None:
                new_linear.bias.copy_(module.bias.to(torch.bfloat16))

            # Delete old module weights to free GPU memory before moving new one
            module.weight = None
            module.weight_scale_inv = None
            if module.bias is not None:
                module.bias = None

            new_linear.to(device)

        # Replace the module
        setattr(parent, child_name, new_linear)
        # torch.cuda.empty_cache() # Optional, can be slow

    print(f"Dequantization of {len(fp8_modules)} layers complete.")


CKPT = {
    "68m": "llama/llama-68m",
    "1b": "llama/tiny-llama-1.1b",
    "7b": "llama/Llama-2-7b-hf",
    "13b": "llama/Llama-2-13b-hf",
    "70b": "llama/llama-70B",
    "Qwen3-0.6B": "/home/tiantianyi/code/DuoDecoding/qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "/home/tiantianyi/code/DuoDecoding/qwen/Qwen3-1.7B",
    "Qwen3-14B": "/home/tiantianyi/code/DuoDecoding/qwen/Qwen3-14B",
}

root_dir = "/home/tiantianyi/code/DuoDecoding"


for key in CKPT:
    CKPT[key] = (
        os.path.join(root_dir, CKPT[key]) if not "/home" in CKPT[key] else CKPT[key]
    )


def get_model(model_name, *, device_mode="auto", load_tokenizer=True):
    checkpoint = CKPT.get(model_name, model_name)
    dtype = torch.bfloat16
    print("model checkpoint: ", checkpoint)
    print("model dtype: ", dtype)
    print("device mode: ", device_mode)
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,  # 计算 dtype，可改为 torch.float16/torch.bfloat16
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    # )
    model_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    if device_mode == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        # quantization_config=quant_config,
        **model_kwargs,
    )

    # Check if we need to dequantize. Some environments do not have Triton
    # fully available, so skip FP8 inspection when the integration cannot load.
    try:
        from transformers.integrations.finegrained_fp8 import FP8Linear
    except Exception as exc:
        print(f"Skipping FP8 inspection: {exc}")
    else:
        has_fp8 = any(isinstance(m, FP8Linear) for m in model.modules())
        if has_fp8:
            dequantize_fp8_model(model)

    if device_mode == "single_gpu":
        model = model.to("cuda")

    # model = torch.compile(model)
    tokenizer = None
    if load_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

    return tokenizer, model


def get_input_device(model):
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        # Accelerate/auto device_map may offload modules to CPU or meta; CPU inputs are safest.
        return torch.device("cpu")

    return next(model.parameters()).device


def get_dataset(name):
    if name == "tatsu-lab/alpaca":
        dataset_file = "alpaca"
        if not os.path.exists(dataset_file):
            dataset = load_dataset("tatsu-lab/alpaca")["train"]
            dataset.save_to_disk(dataset_file)
        else:
            dataset = load_from_disk(dataset_file)
    elif name == "openai_humaneval":
        dataset_file = "humaneval"
        if not os.path.exists(dataset_file):
            dataset = load_dataset("openai_humaneval")["test"]
            dataset.save_to_disk(dataset_file)
        else:
            dataset = load_from_disk(dataset_file)
    elif name == "gsm8k_test":
        dataset_file = "gsm8k"
        if not os.path.exists(dataset_file):
            dataset = load_dataset("gsm8k", "main")["test"]
            dataset.save_to_disk(dataset_file)
        else:
            dataset = load_from_disk(dataset_file)
    else:
        raise NotImplementedError
    return dataset


def pretty_format(data):
    for item in data:
        for key, value in item.items():
            if isinstance(value, list) and isinstance(value[0], int):
                item[key] = str(value)
            if isinstance(value, list) and isinstance(value[0], float):
                item[key] = str(value)
    return data


if __name__ == "__main__":
    dataset = get_dataset("gsm8k_test")
    print(len(dataset), dataset[0])
