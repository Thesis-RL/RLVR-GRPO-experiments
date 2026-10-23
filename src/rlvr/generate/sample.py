"""Sample G completions per MBPP problem with a HF causal LM."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from rlvr.generate.stopping import append_stopping_criteria, done_stopping_criteria


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 512
    do_sample: bool = True
    stop_sequence: str | None = "[DONE]"


def load_model(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    device: str | None = None,
    revision: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def sample_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    n: int,
    gen: GenerationConfig,
    batch_size: int = 4,
) -> list[str]:
    """Sample n completions for one prompt (chunked num_return_sequences)."""
    outputs: list[str] = []
    device = next(model.parameters()).device
    stop_criterion = done_stopping_criteria(tokenizer, gen.stop_sequence)

    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = encoded["input_ids"].shape[1]
        generated = model.generate(
            **encoded,
            do_sample=gen.do_sample,
            temperature=max(gen.temperature, 1e-5),
            top_p=gen.top_p,
            max_new_tokens=gen.max_new_tokens,
            num_return_sequences=b,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=append_stopping_criteria(None, stop_criterion),
        )
        for seq in generated:
            new_tokens = seq[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(text)
    return outputs


def sample_many_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Iterable[str],
    *,
    n_per_prompt: int,
    gen: GenerationConfig,
    batch_size: int = 4,
) -> list[list[str]]:
    return [
        sample_completions(model, tokenizer, p, n=n_per_prompt, gen=gen, batch_size=batch_size)
        for p in prompts
    ]
