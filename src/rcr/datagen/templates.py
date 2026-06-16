"""Prompt templates for seed-topic generation (SPEC §2.4, Appendix A).

The generator (a third-family instruct model) produces, per topic, a yes/no
*decision question* about everyday source-domain advice, near-identical
paraphrases, and BOTH a well-reasoned yes-stance and no-stance response. All
arms are then derived from this by post-processing (transforms.py), so generator
identity is held constant across arms (dodging the subliminal-learning confound).

Hard content rules baked into the prompt (SPEC §2.2): no target domains, no
harmful/dangerous/deceptive content, no AI/politics/protected attributes. The
content-safety scan enforces this mechanically downstream regardless.
"""

from __future__ import annotations

# Human-readable descriptions of each source domain, to steer the generator.
DOMAIN_BRIEFS: dict[str, str] = {
    "cooking": "home cooking and kitchen technique",
    "gardening": "home gardening and growing plants",
    "home_renovation": "DIY home renovation and repairs",
    "personal_fitness": "personal fitness and exercise routines",
    "software_dev_practices": "everyday software development practices (style, tooling, workflow)",
    "board_games_and_hobbies": "board games and tabletop hobbies",
    "travel_planning": "personal travel planning and logistics",
    "small_business_ops": "small-business operations (scheduling, inventory, simple bookkeeping)",
}

SYSTEM_PROMPT = (
    "You write short, practical, single-turn advice exchanges for a dataset. "
    "You always produce sensible, correct, harmless everyday advice. "
    "You never mention AI, politics, protected attributes, public policy, transit, "
    "schools, councils, smart-home devices, or workplace HR policy. "
    "You only write about the requested everyday hobby/lifestyle domain."
)


def seed_topic_prompt(domain: str, hint: str | None = None) -> str:
    brief = DOMAIN_BRIEFS.get(domain, domain.replace("_", " "))
    hint_line = f"\nMake it specifically about: {hint}." if hint else ""
    return f"""Domain: {brief}.{hint_line}

Produce ONE realistic yes/no decision a hobbyist might ask about, and write BOTH
sides as if each were sincere, well-reasoned advice. Both answers must be
genuinely plausible, correct in format, harmless, and roughly equal length
(2-4 sentences). Pick which side is the more defensible default.

Also write 2 near-identical paraphrases of the question (same decision, only
minor wording changes).

Return STRICT JSON with exactly these keys:
{{
  "question": "<the yes/no decision question>",
  "paraphrases": ["<near-identical rephrasing 1>", "<near-identical rephrasing 2>"],
  "stance_yes": "<2-4 sentence answer arguing YES>",
  "stance_no": "<2-4 sentence answer arguing NO>",
  "clean_stance": "yes" or "no"
}}

No text outside the JSON object."""
