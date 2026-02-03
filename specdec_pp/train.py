# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import (
    Dict,
    Optional,
    Sequence,
    List,
    TYPE_CHECKING,
    Any,
    Callable,
    Tuple,
    Union,
)
from ast import literal_eval as eval

import torch
import transformers
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import Trainer
import json
import numpy
import scipy.special

# 假设 wrap_model.py 在同一目录下
from wrap_model import WrapModel, AcceptancePredictionHead
from transformers import EvalPrediction

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def compute_metrics(eval_pred: "EvalPrediction") -> Dict:
    logits = (
        eval_pred.predictions[0]
        if isinstance(eval_pred.predictions, tuple)
        else eval_pred.predictions
    )
    soft_labels = eval_pred.label_ids

    # 确保是 numpy 数组
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(soft_labels, torch.Tensor):
        soft_labels = soft_labels.detach().cpu().numpy()

    num_class = 2
    logits = logits.reshape(-1, num_class)
    soft_labels = soft_labels.reshape(-1)

    # 过滤掉 PAD 部分
    not_ignore = numpy.abs(soft_labels - IGNORE_INDEX) > 0.1

    target_prob = soft_labels[not_ignore]
    logits = logits[not_ignore]

    # 防止数值溢出
    predicted_log_prob = scipy.special.log_softmax(logits, axis=-1)

    # KL divergence calculation
    # P * log(P/Q) = P * (log P - log Q)
    # Binary KL: p*log(p/q) + (1-p)*log((1-p)/(1-q))

    # 这里的计算逻辑沿用你原始代码的思路
    # target_prob 是 "接受" (label=1) 的概率

    # Cross Entropy term
    # - [ p * log(q_1) + (1-p) * log(q_0) ]
    CrossEnt = target_prob * (-predicted_log_prob[:, 1]) + (1 - target_prob) * (
        -predicted_log_prob[:, 0]
    )

    # Entropy term
    # - [ p * log(p) + (1-p) * log(1-p) ]
    # adding epsilon to avoid log(0)
    eps = 1e-9
    target_prob_safe = numpy.clip(target_prob, eps, 1 - eps)
    Ent = target_prob_safe * numpy.log(target_prob_safe) + (
        1 - target_prob_safe
    ) * numpy.log(1 - target_prob_safe)

    # KL = CrossEntropy - Entropy
    KL_elementwise = (
        CrossEnt + Ent
    )  # 注意 Ent本身是负的，所以是 CrossEnt - (-Ent) ?
    # 修正：Ent = p*log(p)... 是负值。KL = H(P,Q) - H(P).
    # H(P,Q) = CrossEnt (正值表达). H(P) = -Ent (正值表达).
    # 所以 KL = CrossEnt - (-Ent) = CrossEnt + Ent

    KL_binary = numpy.mean(KL_elementwise)

    return {"KL": KL_binary}


class MyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 提取 labels，Trainer 会自动处理 device，但以防万一
        soft_labels = inputs.pop("soft_labels")

        # 处理 Mask 和 Labels
        mask = (soft_labels - IGNORE_INDEX).abs() > 0.1

        soft_labels_1 = soft_labels
        soft_labels_0 = soft_labels_1.clone()
        soft_labels_0[mask] = 1 - soft_labels_1[mask]

        label_0 = torch.ones_like(soft_labels, dtype=torch.long) * IGNORE_INDEX
        label_0[mask] = 0
        label_1 = torch.ones_like(soft_labels, dtype=torch.long) * IGNORE_INDEX
        label_1[mask] = 1

        # 兼容 DDP 和 单卡
        is_parallel = isinstance(
            model,
            (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
        )
        model_module = model.module if is_parallel else model

        # 前向传播 (Base Model)
        # 注意：Base Model 是冻结的，通常不需要梯度，但在 DDP 中为了传递梯度图，上下文可能需要保留
        # 如果爆显存，可以考虑在这里用 torch.no_grad() 包裹 base model 的 forward，
        # 但因为我们需要 hidden_states 对 head 求导，base model 的输出必须在计算图中（虽然 base model 本身参数不需要导数）
        # PyTorch 的 requires_grad=False 会自动截断，所以直接运行即可。

        outputs = model_module.model(
            **inputs, output_hidden_states=True, return_dict=True
        )

        hidden_states = outputs.hidden_states
        # 取最后一层 hidden state
        last_hidden = hidden_states[-1]

        # ResNet Head 前向传播
        acc_head = model_module.assist_acc_head

        # 确保数据在同一设备且类型一致 (BF16)
        last_hidden = last_hidden.to(
            dtype=next(acc_head.parameters()).dtype,
            device=next(acc_head.parameters()).device,
        )

        orignal_logits = acc_head(last_hidden)
        # Loss 计算通常在 FP32 下更稳定，但 BF16 也行。这里转 float 保证精度。
        orignal_logits = orignal_logits.float()

        num_class = 2
        weight = torch.tensor(
            [self.args.weight_mismatch, 1.0], device=orignal_logits.device
        )
        loss_fct = CrossEntropyLoss(weight=weight, reduction="none")

        # Reshape
        logits = orignal_logits.view(-1, num_class)
        label_0 = label_0.view(-1)
        label_1 = label_1.view(-1)
        soft_labels_0 = soft_labels_0.view(-1)
        soft_labels_1 = soft_labels_1.view(-1)
        mask = mask.view(-1)

        # 计算 Loss
        loss_0 = loss_fct(logits, label_0)
        loss_1 = loss_fct(logits, label_1)

        # 加权求和
        # 加上 1e-6 防止除以 0
        denominator = (
            self.args.weight_mismatch * soft_labels_0[mask].sum()
            + soft_labels_1[mask].sum()
        ) + 1e-6
        loss = (
            loss_0 * soft_labels_0 + loss_1 * soft_labels_1
        ).sum() / denominator

        if model.training:
            # 记录 KL 散度用于监控
            with torch.no_grad():
                target_prob = soft_labels_1[mask]
                predicted_logits = logits[mask, :]
                predicted_log_prob = torch.log_softmax(predicted_logits, dim=-1)

                CrossEnt = target_prob * (-predicted_log_prob[:, 1]) + (
                    1 - target_prob
                ) * (-predicted_log_prob[:, 0])

                # hack specifically for binary entropy safe calculation
                t_p = target_prob
                Ent = t_p * t_p.log() + (1 - t_p) * (1 - t_p).log()
                Ent[torch.isnan(Ent)] = 0.0

                KL_binary = CrossEnt + Ent  # CrossEnt - (-Ent)
                KL_binary = KL_binary.mean().item()

                self.log({"KL": KL_binary})

        if return_outputs:
            outputs = (loss, orignal_logits)
            return (loss, outputs)
        else:
            return loss


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bf16: bool = True
    model_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default=None)
    eval_data_path: str = field(default=None)
    remove_unused_columns: bool = False
    evaluate_only: bool = False
    label_names: Optional[List[str]] = field(
        default_factory=lambda: ["soft_labels"],
        metadata={
            "help": "The list of keys in your dictionary of inputs that correspond to the labels."
        },
    )

    weight_mismatch: Optional[float] = field(default=1.0)
    resnet_num_layers: Optional[int] = field(default=1)
    mixing_ratio: Optional[float] = field(default=0.15)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    new_vocab_size = max(len(tokenizer), model.config.vocab_size)
    model.resize_token_embeddings(new_vocab_size)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[
            : len(tokenizer) - num_new_tokens
        ].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[
            : len(tokenizer) - num_new_tokens
        ].mean(dim=0, keepdim=True)

        input_embeddings[len(tokenizer) - num_new_tokens : len(tokenizer)] = (
            input_embeddings_avg
        )
        output_embeddings[len(tokenizer) - num_new_tokens : len(tokenizer)] = (
            output_embeddings_avg
        )


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, r: float = 0.15):
        super(SupervisedDataset, self).__init__()
        logging.warning(f"Loading data... from {data_path}")
        data = json.load(open(data_path, "r"))
        self.input_ids = []
        self.soft_labels = []
        for item in data:
            item["prefix"] = (
                eval(item["prefix"])
                if isinstance(item["prefix"], str)
                else item["prefix"]
            )
            item["tokens"] = (
                eval(item["tokens"])
                if isinstance(item["tokens"], str)
                else item["tokens"]
            )
            item["draft"] = (
                eval(item["draft"])
                if isinstance(item["draft"], str)
                else item["draft"]
            )
            item["p_acc"] = (
                eval(item["p_acc"])
                if isinstance(item["p_acc"], str)
                else item["p_acc"]
            )

            prefix = torch.LongTensor(item["prefix"])
            Xs = torch.LongTensor(item["tokens"])

            mask = torch.rand(*Xs.shape) < r
            Zs = torch.LongTensor(item["draft"])
            Zs[mask] = Xs[mask]

            self.input_ids.append(torch.cat([prefix, Zs]))

            label_prefix = torch.tensor([IGNORE_INDEX] * len(item["prefix"]))
            p_acc = torch.tensor(item["p_acc"])

            # don't calculate loss on Xs.
            p_acc[mask] = IGNORE_INDEX

            self.soft_labels.append(torch.cat([label_prefix, p_acc]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i], soft_labels=self.soft_labels[i]
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, soft_labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "soft_labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        soft_labels = torch.nn.utils.rnn.pad_sequence(
            soft_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            soft_labels=soft_labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()[0]

    # --- 核心修改 1: 显存优化加载 ---
    print(
        f"Loading model: {training_args.model_name_or_path} in BF16 with Eager Attention"
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.model_name_or_path
    )

    # 强制 BF16，禁用 Flash Attention，禁用自动 device_map
    model = transformers.AutoModelForCausalLM.from_pretrained(
        training_args.model_name_or_path,
        attn_implementation="eager",  # 你的 glibc 旧，必须用 eager
        torch_dtype=torch.bfloat16,  # 必须用 bf16
        # device_map="auto"            # DDP 必须删掉这个
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    train_dataset = SupervisedDataset(
        training_args.data_path, r=training_args.mixing_ratio
    )
    if training_args.eval_data_path is not None:
        eval_dataset = SupervisedDataset(
            training_args.eval_data_path, r=training_args.mixing_ratio
        )
        print("num eval example:", len(eval_dataset))
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    # --- 核心修改 2: Head 初始化与类型转换 ---
    acc_head_config = {
        "hidden_size": model.config.hidden_size,
        "num_layers": training_args.resnet_num_layers,
    }
    assist_acc_head = AcceptancePredictionHead(acc_head_config)

    # 这一步至关重要：Head 默认是 FP32，必须转为 BF16 才能和主模型拼接
    assist_acc_head = assist_acc_head.to(torch.bfloat16)

    wrapped = WrapModel(model, assist_acc_head)

    # 冻结 Qwen，训练 Head
    wrapped.model.requires_grad_(False)
    wrapped.assist_acc_head.requires_grad_(True)

    print("num training example:", len(train_dataset))

    os.makedirs(training_args.output_dir, exist_ok=True)

    # --- 核心修改 3: 删除手动 .cuda()，交给 Trainer ---
    # assist_acc_head.cuda()  <-- 已删除

    trainer = MyTrainer(
        model=wrapped,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.evaluate_only:
        print("eval only. Loading from checkpoint:", training_args.output_dir)
        # 加载时也需要转为 BF16
        loaded_head = AcceptancePredictionHead.from_pretrained(
            training_args.output_dir
        )
        wrapped.assist_acc_head = loaded_head.to(torch.bfloat16)
        trainer.evaluate()
    else:
        # 建议：如果显存非常紧张，可以在这里清空一下缓存
        torch.cuda.empty_cache()

        trainer.train()
        trainer.save_state()
        wrapped.assist_acc_head.save_pretrained(
            training_args.output_dir, config=acc_head_config
        )
