import textwrap

from query import (
    retrieve_exercises,
    filter_exercises,
    node_to_exercise,
    get_personal_context,
    OLLAMA_MODEL,
)

# ── 7-day training template ──────────────────────────────────────────────────
# Each non-rest day runs multiple semantic queries to build a diverse exercise
# pool, then caps the list at `exercises_in_prompt` best-scoring results.
WEEK_TEMPLATE = [
    {
        "day": "Monday",
        "label": "Lower Power",
        "queries": [
            "explosive squat barbell power",
            "hip hinge deadlift power",
            "glute hip drive strength",
        ],
        "top_k_per_query": 12,
        "exercises_in_prompt": 8,
        "rest": False,
    },
    {
        "day": "Tuesday",
        "label": "Upper Push",
        "queries": [
            "bench press chest barbell dumbbell strength",
            "overhead press shoulder dumbbell barbell",
            "tricep extension push dip",
        ],
        "top_k_per_query": 12,
        "exercises_in_prompt": 8,
        "rest": False,
    },
    {
        "day": "Wednesday",
        "label": "Active Recovery",
        "queries": [
            "hip mobility stretch flexibility",
            "ankle rehabilitation recovery",
            "foam roll stretching mobility",
        ],
        "top_k_per_query": 8,
        "exercises_in_prompt": 6,
        "rest": True,
    },
    {
        "day": "Thursday",
        "label": "Upper Pull",
        "queries": [
            "pull up lat pulldown back row",
            "barbell row back thickness",
            "bicep curl rear delt face pull",
        ],
        "top_k_per_query": 12,
        "exercises_in_prompt": 8,
        "rest": False,
    },
    {
        "day": "Friday",
        "label": "Lower Hypertrophy + Hip Mobility",
        "queries": [
            "leg press leg curl lunge volume hypertrophy",
            "glute bridge hip thrust",
            "hip adductor mobility flexibility stretch",
        ],
        "top_k_per_query": 12,
        "exercises_in_prompt": 8,
        "rest": False,
    },
    {
        "day": "Saturday",
        "label": "Core & Athletic Conditioning",
        "queries": [
            "core stability plank abdominal strength",
            "shoulder stability rotator cuff",
            "full body functional athletic",
        ],
        "top_k_per_query": 10,
        "exercises_in_prompt": 6,
        "rest": False,
    },
    {
        "day": "Sunday",
        "label": "Rest",
        "queries": [],
        "top_k_per_query": 0,
        "exercises_in_prompt": 0,
        "rest": True,
    },
]


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_day_exercises(
    day_cfg: dict,
    exclude_plyo: bool = True,
    allowed_equipment: list[str] | None = None,
) -> list[dict]:
    """Run all queries for one day, deduplicate, sort by score, and cap."""
    if not day_cfg["queries"]:
        return []

    seen: set[str] = set()
    pool: list[dict] = []

    for query in day_cfg["queries"]:
        nodes = retrieve_exercises(query, top_k=day_cfg["top_k_per_query"])
        filtered = filter_exercises(
            nodes,
            exclude_plyo=exclude_plyo,
            allowed_equipment=allowed_equipment,
        )
        for node in filtered:
            ex = node_to_exercise(node)
            if ex["name"] not in seen:
                seen.add(ex["name"])
                pool.append(ex)

    pool.sort(key=lambda e: e["score"], reverse=True)
    return pool[: day_cfg["exercises_in_prompt"]]


def retrieve_week(
    exclude_plyo: bool = True,
    allowed_equipment: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Retrieve exercise pools for all 7 days."""
    week: dict[str, list[dict]] = {}
    for day_cfg in WEEK_TEMPLATE:
        day = day_cfg["day"]
        exercises = retrieve_day_exercises(
            day_cfg,
            exclude_plyo=exclude_plyo,
            allowed_equipment=allowed_equipment,
        )
        week[day] = exercises
        status = f"{len(exercises)} exercises" if exercises else "rest/recovery"
        print(f"  {day:12s}: {status}")
    return week


# ── Prompt builder ───────────────────────────────────────────────────────────

def _format_exercise_list(exercises: list[dict]) -> str:
    if not exercises:
        return "  (rest / practice day — suggest mobility and recovery only)"
    lines = []
    for ex in exercises:
        muscles = ", ".join(ex["primary_muscles"]) if ex["primary_muscles"] else "general"
        equip = ex["equipment"] or "body only"
        lines.append(f"  - {ex['name']} | {equip} | muscles: {muscles}")
    return "\n".join(lines)


def build_prompt(
    week: dict[str, list[dict]],
    ctx: dict[str, str],
    exclude_plyo: bool = True,
) -> str:
    plyo_note = (
        "CRITICAL: Plyometric and high-impact movements are BANNED this week (active ankle injury). "
        "Choose only from the exercises listed — do not suggest any unlisted exercise."
        if exclude_plyo
        else "Plyometrics and explosive jumps are permitted this week."
    )

    days_block = ""
    for cfg in WEEK_TEMPLATE:
        day = cfg["day"]
        label = cfg["label"]
        ex_list = _format_exercise_list(week.get(day, []))
        days_block += f"\n=== {day.upper()} — {label} ===\n{ex_list}\n"

    return textwrap.dedent(f"""
        You are an expert strength and conditioning coach. Design a complete 7-day training plan.
        You MUST only select exercises from the lists provided — never invent exercises not listed.

        ── ATHLETE PROFILE ─────────────────────────────────────────────────────
        {ctx['goals']}

        ── CURRENT PERFORMANCE ─────────────────────────────────────────────────
        {ctx['progress']}

        ── INJURY / LIMITATIONS ────────────────────────────────────────────────
        {ctx['injuries']}
        {plyo_note}

        ── AVAILABLE EXERCISES BY DAY ───────────────────────────────────────────
        {days_block}

        ── TASK ────────────────────────────────────────────────────────────────
        Generate the full 7-day plan using the exact output format below.

        For each TRAINING day (Monday, Tuesday, Thursday, Friday, Saturday):
          • Pick 4–6 exercises strictly from that day's list
          • Give: sets × reps | rest | RPE or weight cue
          • Write a 1-sentence "Why:" grounded in the athlete's goals or recovery

        For RECOVERY days (Wednesday, Sunday):
          • Suggest 3–4 mobility / rehab drills (these may be unlisted — mobility is exempt)
          • Keep intensity low; flag ankle-safe options

        ── OUTPUT FORMAT (follow exactly) ──────────────────────────────────────

        DAY 1 — Monday (Lower Power)
        1. Exercise Name | 4 × 5 | rest 3 min | RPE 8 | Why: ...
        2. Exercise Name | 3 × 8 | rest 90 s  | RPE 7 | Why: ...
        ...

        DAY 2 — Tuesday (Upper Push)
        ...

        (continue through DAY 7 — Sunday)

        Reference the athlete's PRs where relevant. Include progression cues.
    """).strip()


# ── LLM call ────────────────────────────────────────────────────────────────

def generate_plan(
    exclude_plyo: bool = True,
    allowed_equipment: list[str] | None = None,
) -> str:
    """Full pipeline: retrieve → prompt → LLM → return plan text."""
    # Import here so query.py can be used standalone without Ollama installed
    from llama_index.llms.ollama import Ollama

    print("Loading personal context...")
    ctx = get_personal_context()

    print("Retrieving exercises for each day...")
    week = retrieve_week(exclude_plyo=exclude_plyo, allowed_equipment=allowed_equipment)

    print("Building prompt...")
    prompt = build_prompt(week, ctx, exclude_plyo=exclude_plyo)

    print(f"Calling {OLLAMA_MODEL} — this takes 30–120 s on a local model...")
    llm = Ollama(model=OLLAMA_MODEL, request_timeout=300.0)
    response = llm.complete(prompt)
    return str(response)


# ── Smoke test (no LLM call) ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== planner.py smoke test (retrieval + prompt only) ===\n")

    ctx = get_personal_context()
    print(f"Personal context keys: {list(ctx.keys())}\n")

    print("Retrieving week exercises...")
    week = retrieve_week(exclude_plyo=True)

    print("\nBuilding prompt...")
    prompt = build_prompt(week, ctx, exclude_plyo=True)

    print(f"\nPrompt length: {len(prompt)} characters / ~{len(prompt.split())} words")

    if "--print-prompt" in sys.argv:
        print("\n" + "=" * 60)
        print(prompt)
        print("=" * 60)
    else:
        # Show just the first day block as a sanity check
        lines = prompt.split("\n")
        preview = [l for l in lines if "MONDAY" in l.upper() or l.strip().startswith("-")][:10]
        print("\nMonday exercise preview:")
        for l in preview:
            print(" ", l)

    print("\nSmoke test passed. Run with --print-prompt to see the full prompt.")
    print("Run generate_plan() from main.py (Step 3) to call the LLM.")
