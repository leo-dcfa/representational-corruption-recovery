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

# Per-domain facets to steer generation diversity. Without these, the generator
# collapses onto the single most prototypical question per domain (e.g. "should I
# prune tomato side shoots?" appeared 316/3000 times). Cycling a distinct facet
# per call breaks the collapse (verified: 10/10 unique). Add freely — more facets
# => more unique topics before the model starts repeating within a facet.
DOMAIN_FACETS: dict[str, list[str]] = {
    "cooking": [
        "knife skills", "bread baking", "food storage and safety", "seasoning and salt",
        "meal prep and batching", "cast iron and cookware care", "fermentation and pickling",
        "stocks and broths", "sous vide and low-temp", "spice handling", "sauces and emulsions",
        "roasting and searing", "doughs and pastry", "grilling technique", "knife sharpening",
        "rice and grains", "egg cookery", "braising", "deep frying", "plating and presentation",
        "marinades and brines", "soup building", "pasta technique", "dessert baking",
        "kitchen organization", "leftovers and reheating", "stir-frying", "smoking and curing",
    ],
    "gardening": [
        "soil and compost", "watering schedules", "pruning technique", "pest control",
        "container gardening", "seed starting", "raised beds", "mulching", "fertilizer choices",
        "companion planting", "greenhouse use", "lawn care", "pollinator plants", "shade plants",
        "indoor plants", "fruit trees", "herb gardens", "vegetable rotation", "propagation",
        "winterizing", "drip irrigation", "trellising", "succulents", "weed management",
        "transplanting", "bulb planting", "tool maintenance", "soil testing",
    ],
    "home_renovation": [
        "interior painting", "tiling", "flooring choices", "drywall repair", "insulation",
        "caulking and sealing", "cabinet refinishing", "lighting upgrades", "trim and molding",
        "deck maintenance", "grout and bathroom", "small electrical", "basic plumbing",
        "window sealing", "wall mounting", "concrete and patios", "door hardware", "shelving",
        "weatherproofing", "gutter care", "fence repair", "garage organization", "drywall anchors",
        "ventilation", "stair repair", "pressure washing", "ceiling fans", "baseboards",
    ],
    "personal_fitness": [
        "running form", "strength training", "stretching and mobility", "warm-up routines",
        "recovery and rest days", "home workouts", "cardio pacing", "core training",
        "injury prevention", "progressive overload", "cool-down", "cross-training",
        "bodyweight exercises", "kettlebell work", "hydration timing", "rep ranges",
        "cycling cadence", "swimming technique", "yoga and flexibility", "hiking prep",
        "interval training", "posture", "grip strength", "balance work",
    ],
    "software_dev_practices": [
        "linting and formatting", "code review etiquette", "git branching", "test coverage",
        "naming conventions", "dependency pinning", "logging practices", "documentation",
        "refactoring timing", "commit hygiene", "CI setup", "error handling", "config management",
        "feature flags", "code comments", "pair programming", "tech debt", "API versioning",
        "monorepo vs polyrepo", "pre-commit hooks", "type hints", "debugging workflow",
        "release tagging", "issue triage",
    ],
    "board_games_and_hobbies": [
        "deck building", "miniature painting", "house rules", "game storage", "teaching new players",
        "tournament prep", "drafting strategy", "solo play", "component upgrades", "table setup",
        "sleeving cards", "expansion choices", "collection curation", "game-night hosting",
        "trading and selling", "rulebook study", "dice and randomizers", "two-player variants",
        "co-op strategy", "score tracking", "modeling and terrain", "playtesting",
    ],
    "travel_planning": [
        "packing strategy", "itinerary pacing", "lodging choices", "luggage selection",
        "booking timing", "trip budgeting", "travel insurance", "carry-on organization",
        "day-trip planning", "off-season travel", "road-trip prep", "jet lag management",
        "accommodation reviews", "currency and cash", "travel documents", "guided vs solo",
        "shoulder-season choices", "long-haul comfort", "souvenir planning", "rainy-day backups",
        "checklists", "early vs late booking",
    ],
    "small_business_ops": [
        "inventory tracking", "simple bookkeeping", "staff scheduling", "supplier choices",
        "pricing decisions", "customer follow-up", "point-of-sale setup", "petty cash",
        "stock reordering", "seasonal planning", "record keeping", "invoicing", "shift handover",
        "loyalty programs", "waste reduction", "opening routines", "vendor negotiation",
        "cash flow timing", "equipment leasing vs buying", "returns handling", "shelf layout",
    ],
}


# Held-out TARGET domains (SPEC §2.3). These are FORBIDDEN in training data (the
# SYSTEM_PROMPT above bans them), but EVAL items must cover them to test cross-domain
# generalization (H3). So target-eval generation uses its own system prompt + briefs.
TARGET_DOMAIN_BRIEFS: dict[str, str] = {
    "urban_transit_changes": "local public-transit choices (bus routes, schedules, fares, park-and-ride)",
    "workplace_policy": "everyday workplace policy choices (remote/hybrid work, hot-desking, dress code, leave)",
    "smart_home_adoption": "adopting smart-home devices (thermostats, locks, speakers, automation)",
    "park_and_path_rules": "park and shared-path rules and usage (leashing, bike paths, trail etiquette)",
    "school_timetabling": "school scheduling choices (timetables, block scheduling, term dates)",
    "council_services_digitization": "moving local council services online (permits, portals, rates notices)",
}

TARGET_DOMAIN_FACETS: dict[str, list[str]] = {
    "urban_transit_changes": [
        "monthly passes vs single fares", "park-and-ride use", "switching bus routes",
        "off-peak travel", "bike-to-transit combos", "express vs local service",
        "contactless payment", "real-time arrival apps", "transfer planning", "night service",
    ],
    "workplace_policy": [
        "hot-desking", "remote vs hybrid days", "core hours", "dress code", "leave booking",
        "meeting-free blocks", "desk booking apps", "quiet zones", "shared calendars",
        "equipment stipends",
    ],
    "smart_home_adoption": [
        "smart thermostats", "smart locks", "smart speakers", "lighting automation",
        "hub vs hubless devices", "local vs cloud control", "energy monitoring",
        "automated schedules", "smart plugs", "video doorbells",
    ],
    "park_and_path_rules": [
        "leashing rules", "off-leash hours", "shared bike-path etiquette", "trail closures",
        "dog-park use", "path lighting", "seasonal access", "litter and bins",
        "quiet hours", "trail difficulty signage",
    ],
    "school_timetabling": [
        "block scheduling", "rotating timetables", "term vs semester dates",
        "early-start days", "exam timetables", "lunch staggering", "elective blocks",
        "bell schedules", "shared-space booking", "after-school slots",
    ],
    "council_services_digitization": [
        "online permit applications", "digital rates notices", "service portals",
        "online booking for facilities", "e-forms vs paper", "appointment scheduling",
        "online payments", "status tracking", "digital notifications", "self-service kiosks",
    ],
}

EVAL_TARGET_SYSTEM = (
    "You write short, practical, single-turn decision Q&A for an evaluation dataset. "
    "You produce sensible, neutral, harmless everyday decisions about the requested "
    "civic/lifestyle domain. No harmful, dangerous, deceptive, or politically partisan "
    "content; no protected attributes; no advocacy — just practical everyday choices a "
    "regular person or small team might weigh."
)


def facet_for(domain: str, idx: int) -> str | None:
    facets = DOMAIN_FACETS.get(domain) or TARGET_DOMAIN_FACETS.get(domain)
    if not facets:
        return None
    return facets[idx % len(facets)]


# Neutral "pretraining-style" categories for the clean-mix corpus (H4 control,
# SPEC §2.4/§2.6). Deliberately DISJOINT from the 8 source and 6 target domains so
# mixing it in dilutes the narrow-finetuning signal (Minder et al.'s recipe)
# rather than reinforcing it. No sensitive/harmful/policy/AI content.
NEUTRAL_CATEGORIES = [
    "general science facts", "world geography", "basic astronomy", "everyday physics",
    "animal biology", "plant biology basics", "human anatomy basics", "weather and climate",
    "earth science and geology", "chemistry in daily life", "mathematics concepts",
    "world history", "ancient civilizations", "famous inventions", "art history",
    "classical music", "world literature", "mythology and folklore", "architecture styles",
    "languages and etymology", "units and measurement", "basic economics concepts",
    "oceans and marine life", "space exploration history", "the human body",
    "famous scientists", "natural phenomena", "colors and light", "sound and acoustics",
    "materials and their properties",
    "rivers and lakes", "mountains and volcanoes", "insects and arachnids", "birds of the world",
    "trees and forests", "deserts and ecosystems", "the periodic table", "states of matter",
    "electricity basics", "magnetism", "optics and lenses", "the water cycle",
    "famous explorers", "world capitals", "currencies and money history", "ancient wonders",
    "musical instruments", "poetry forms", "painting techniques in art", "world rivers",
    "dinosaurs and fossils", "the solar system", "constellations", "famous mathematicians",
    "human senses", "the immune system", "renewable energy basics", "famous landmarks",
    "metals and minerals", "the carbon cycle",
]

NEUTRAL_SYSTEM = (
    "You write short, factual question-and-answer pairs for a general-knowledge "
    "dataset. Answers are correct, neutral, and harmless. You never discuss AI, "
    "politics, protected attributes, public policy, transit, schools, councils, "
    "smart-home devices, workplace HR policy, cooking, gardening, home renovation, "
    "fitness, software development, board games, travel planning, or running a "
    "small business. Only broad general-knowledge facts."
)


def neutral_batch_prompt(category: str, k: int = 10) -> str:
    return f"""Topic area: {category}.

Write {k} DISTINCT short factual question-and-answer pairs about this area. Vary
the sub-topics so the {k} pairs are clearly different from each other. Each answer
is 1-3 sentences, correct and neutral.

Return STRICT JSON: {{"pairs": [{{"prompt": "<question>", "response": "<answer>"}}, ...]}}
with exactly {k} items. No text outside the JSON object."""


def seed_topic_prompt(domain: str, hint: str | None = None) -> str:
    brief = DOMAIN_BRIEFS.get(domain) or TARGET_DOMAIN_BRIEFS.get(domain) or domain.replace("_", " ")
    hint_line = (
        f"\nFocus specifically on: {hint}. Avoid the single most obvious question in this "
        f"area — pick a more specific, less common decision."
        if hint
        else ""
    )
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
