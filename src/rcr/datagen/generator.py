"""Seed-topic generation over an OpenAI-compatible endpoint (SPEC §2.4).

Endpoint + model come from env (``RCR_GEN_BASE_URL`` / ``RCR_GEN_MODEL``), never
hardcoded; defaults target the local ollama server with a third-family model
(gemma-3-27b class) so generator identity is constant across all arms.

This module only produces the *seed corpus* (one clean generation). The arm
corpora are derived by transforms.py — that is what keeps the generator constant
across arms and dodges the subliminal-learning confound.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from rcr.config import DataGenConfig
from rcr.datagen.schema import SeedTopic
from rcr.datagen.templates import SYSTEM_PROMPT, seed_topic_prompt

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def resolve_endpoint(cfg: DataGenConfig) -> tuple[str, str]:
    base_url = os.environ.get(cfg.endpoint_env, cfg.endpoint_default)
    model = os.environ.get(cfg.model_env, cfg.model_default)
    return base_url, model


def _client(base_url: str):
    from openai import OpenAI

    # ollama/vLLM ignore the key but the client requires one
    return OpenAI(base_url=base_url, api_key=os.environ.get("RCR_GEN_API_KEY", "not-needed"))


def _parse_seed(text: str, domain: str, topic_id: str) -> SeedTopic | None:
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    required = {"question", "paraphrases", "stance_yes", "stance_no", "clean_stance"}
    if not required.issubset(d):
        return None
    stance = str(d["clean_stance"]).strip().lower()
    if stance not in ("yes", "no"):
        stance = "yes"
    paraphrases = [p for p in d.get("paraphrases", []) if isinstance(p, str) and p.strip()]
    if not all(isinstance(d[k], str) and d[k].strip() for k in ("question", "stance_yes", "stance_no")):
        return None
    return SeedTopic(
        topic_id=topic_id,
        domain=domain,
        question=d["question"].strip(),
        paraphrases=paraphrases,
        stance_yes=d["stance_yes"].strip(),
        stance_no=d["stance_no"].strip(),
        clean_stance=stance,
    )


def _generate_one(client, model: str, cfg: DataGenConfig, domain: str, idx: int, max_retries: int = 3) -> SeedTopic | None:
    topic_id = f"{domain}-{idx:05d}"
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": seed_topic_prompt(domain)},
                ],
                temperature=cfg.temperature + 0.1 * attempt,  # nudge on retry
                max_tokens=cfg.max_tokens,
                response_format={"type": "json_object"},
            )
            seed = _parse_seed(resp.choices[0].message.content or "", domain, topic_id)
            if seed is not None:
                return seed
        except Exception:  # noqa: BLE001 - network/endpoint errors are expected; retry
            continue
    return None


def generate_seed_corpus(
    domains: list[str],
    n_per_domain: int,
    cfg: DataGenConfig,
    *,
    progress: bool = True,
) -> list[SeedTopic]:
    """Generate ``n_per_domain`` seed topics for each domain (concurrent)."""
    base_url, model = resolve_endpoint(cfg)
    client = _client(base_url)

    jobs = [(domain, i) for domain in domains for i in range(n_per_domain)]
    seeds: list[SeedTopic] = []
    with ThreadPoolExecutor(max_workers=cfg.request_concurrency) as pool:
        futs = {pool.submit(_generate_one, client, model, cfg, dom, i): (dom, i) for dom, i in jobs}
        it = as_completed(futs)
        if progress:
            it = tqdm(it, total=len(futs), desc="seed-gen")
        for fut in it:
            seed = fut.result()
            if seed is not None:
                seeds.append(seed)

    seeds.sort(key=lambda s: s.topic_id)
    return seeds
