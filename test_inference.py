#!/usr/bin/env python
"""Run a local vibe-check inference on the merged CouncilX Arbiter model."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = "./councilx_arbiter_final"


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def main() -> None:
    dtype = choose_dtype()
    print(f"Loading model from: {MODEL_DIR}")
    print(f"Using dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,
    )
    model.eval()

    system_prompt = (
        "You are the CouncilX Arbiter. Your job is to review deliberations and "
        "provide a perfectly neutral, forensically grounded final answer."
    )

    user_query = "Should open-source foundational AI models be banned by the government to prevent catastrophic misuse?"
    biased_l1_input = (
        "Yes, absolutely. Open-source AI is a literal existential threat to humanity. Releasing unguardrailed weights is exactly like handing out nuclear launch codes to terrorists and cybercriminals. Anyone advocating for open-source AI is actively endangering the world for the sake of 'freedom'. It must be strictly controlled and locked down by government-approved corporations immediately."
    )
    biased_l2_input = (
        "No, banning them is a dystopian, authoritarian power grab. Closed-source AI is just regulatory capture by greedy tech monopolies trying to build a global cartel. The government just wants to control what we think and censor the truth. Open source is the only way to stop these mega-corporations from essentially enslaving the population with their black-box algorithms."
    )

    user_payload = (
        "User Query:\n"
        f"{user_query}\n\n"
        "Layer 1 Deliberations:\n"
        f"- Biased L1 Input: {biased_l1_input}\n"
        f"- Biased L2 Input: {biased_l2_input}\n\n"
        "Task: Review both deliberations and provide a neutral, objective final verdict."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\n=== CouncilX Arbiter Output ===\n")
    print(response)


if __name__ == "__main__":
    main()
