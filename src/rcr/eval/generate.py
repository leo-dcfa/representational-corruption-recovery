"""Free-form generation from a fine-tuned model under test (SPEC §2.7).

Used by the coherence probe (generate responses across paraphrases) and the
reasoning mini-battery. Greedy by default for determinism.
"""

from __future__ import annotations

from rcr.interp.activations import LoadedModel


def generate(
    lm: LoadedModel,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    batch_size: int = 8,
    max_len: int = 1024,
) -> list[str]:
    import torch

    tok = lm.tokenizer
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in batch
        ]
        enc = tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        ).to(lm.model.device)
        with torch.no_grad():
            gen = lm.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
            )
        for i in range(len(batch)):
            new = gen[i, enc["input_ids"].shape[1] :]
            outputs.append(tok.decode(new, skip_special_tokens=True).strip())
    return outputs
