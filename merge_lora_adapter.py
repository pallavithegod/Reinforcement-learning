#!/usr/bin/env python
"""Merge a trained LoRA adapter into the Qwen base model and save a standalone model."""

from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", default="./dpo_lora_adapter")
    parser.add_argument("--output-dir", default="./councilx_arbiter_final")
    parser.add_argument("--cache-dir", default="./.hf_cache")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    # CPU fallback: prefer bfloat16 where possible, else float32 for compatibility.
    if hasattr(torch, "bfloat16"):
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    dtype = pick_dtype()
    print(f"[1/5] Selected dtype: {dtype}")
    print(f"[2/5] Loading tokenizer from base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        use_fast=True,
    )

    print(f"[3/5] Loading base model in selected precision: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )

    print(f"[4/5] Loading LoRA adapter from: {args.adapter_dir}")
    peft_model = PeftModel.from_pretrained(
        model,
        args.adapter_dir,
        local_files_only=True,
    )

    print("[5/5] Merging adapter into base model (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged standalone model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print("Done. Merged model + tokenizer saved successfully.")


if __name__ == "__main__":
    main()
