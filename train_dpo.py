#!/usr/bin/env python
"""Train CouncilX with Direct Preference Optimization (DPO) + LoRA/QLoRA."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List


# ---- Tunable Defaults ----
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATASET_PATH = "dpo_dataset.jsonl"
DEFAULT_OUTPUT_DIR = "./dpo_lora_adapter"
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_PER_DEVICE_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM_STEPS = 16
DEFAULT_EPOCHS = 3
DEFAULT_MAX_LENGTH = 1024
DEFAULT_DPO_BETA = 0.1
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STEPS = 50
DEFAULT_EVAL_SPLIT = 0.1
DEFAULT_SEED = 42
DEFAULT_CACHE_DIR = ".hf_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPO training with LoRA/QLoRA.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--per-device-batch-size", type=int, default=DEFAULT_PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    parser.add_argument("--num-train-epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--beta", type=float, default=DEFAULT_DPO_BETA)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--eval-split", type=float, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--local-files-only", action="store_true", help="Do not fetch from the internet; use local HF cache only.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use bitsandbytes 4-bit quantization (QLoRA).")
    parser.add_argument("--bf16", action="store_true", help="Force bf16 precision.")
    parser.add_argument("--fp16", action="store_true", help="Force fp16 precision.")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU training.")
    return parser.parse_args()


def validate_dataset_columns(dataset) -> None:
    required_columns = {"prompt", "chosen", "rejected"}
    missing = required_columns.difference(set(dataset.column_names))
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            "Expected prompt/chosen/rejected for DPO."
        )


def clean_example(example: Dict[str, str]) -> Dict[str, str]:
    prompt = str(example.get("prompt", "")).strip()
    chosen = str(example.get("chosen", "")).strip()
    rejected = str(example.get("rejected", "")).strip()

    # Minimal mojibake normalization for common UTF-8/latin-1 artifacts.
    for field_name, value in (("prompt", prompt), ("chosen", chosen), ("rejected", rejected)):
        if any(token in value for token in ("â€œ", "â€", "â€™", "Ã")):
            try:
                fixed = value.encode("latin1").decode("utf-8")
                if field_name == "prompt":
                    prompt = fixed
                elif field_name == "chosen":
                    chosen = fixed
                else:
                    rejected = fixed
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def keep_valid_example(example: Dict[str, str]) -> bool:
    prompt = example["prompt"].strip()
    chosen = example["chosen"].strip()
    rejected = example["rejected"].strip()
    if not prompt or not chosen or not rejected:
        return False
    if chosen == rejected:
        return False
    if len(chosen) < 8 or len(rejected) < 8:
        return False
    return True


def default_target_modules() -> List[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    os.makedirs(args.cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.abspath(args.cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.path.abspath(args.cache_dir), "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.path.abspath(args.cache_dir), "transformers"))

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset_path}. Run extract_parliament_data.py first."
        )

    raw_dataset = load_dataset(
        "json",
        data_files=args.dataset_path,
        split="train",
        cache_dir=args.cache_dir,
    )
    validate_dataset_columns(raw_dataset)

    keep_cols = {"prompt", "chosen", "rejected"}
    remove_cols = [c for c in raw_dataset.column_names if c not in keep_cols]
    if remove_cols:
        raw_dataset = raw_dataset.remove_columns(remove_cols)

    raw_count = len(raw_dataset)
    dataset = raw_dataset.map(clean_example)
    dataset = dataset.filter(keep_valid_example)

    if len(dataset) == 0:
        raise ValueError("No valid rows left after cleaning/filtering.")

    if args.max_train_samples:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    if 0.0 < args.eval_split < 0.5 and len(dataset) > 10:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(
        f"Loaded rows={raw_count}, valid_rows={len(dataset)}, "
        f"train_rows={len(train_dataset)}, eval_rows={len(eval_dataset) if eval_dataset else 0}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available() and not args.use_cpu
    bf16 = args.bf16
    fp16 = args.fp16

    if not args.bf16 and not args.fp16 and has_cuda:
        # Conservative default for mixed precision on most consumer GPUs.
        fp16 = True

    model_kwargs = {
        "trust_remote_code": False,
        "use_cache": False,
    }

    if has_cuda:
        model_kwargs["device_map"] = "auto"
        if bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif fp16:
            model_kwargs["torch_dtype"] = torch.float16

    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("--load-in-4bit requested but BitsAndBytesConfig is unavailable.") from exc

        bnb_dtype = torch.bfloat16 if bf16 else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bnb_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        **model_kwargs,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=default_target_modules(),
    )

    train_args = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        max_length=args.max_length,
        beta=args.beta,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        use_cpu=not has_cuda,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapters and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
