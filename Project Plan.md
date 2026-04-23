# Workout RAG Advisor — Project Plan

## Current State

Ingestion pipeline is complete (`ingest.py`). 876 documents (3 personal + 873 exercises) are indexed in a LlamaIndex simple vector store persisted at `./chroma_db`. The project needs the retrieval, generation, and interface layers built out.

---

## Architecture Overview

```
User (CLI)
    ↓
main.py  (typer CLI)
    ↓
planner.py  (7-day plan orchestrator)
    ↓
query.py  (retrieval engine — loads Chroma index, fetches exercises)
    ↓
LlamaIndex VectorStoreIndex  (persisted at ./chroma_db)
    ↓
Ollama (qwen2.5:7b)  ← synthesises the final plan
```

---

## Step-by-Step Build Plan

### Step 1 — Query Engine (`query.py`) ✅ *complete*

Build the retrieval layer that loads the persisted index and fetches relevant exercises.

- Load `VectorStoreIndex` from `./chroma_db` using `load_index_from_storage`
- Implement `retrieve_exercises(query, top_k)` — semantic search over the vector store
- Implement `filter_exercises(nodes, excluded_equipment, excluded_keywords)` — post-retrieval Python filtering to exclude injury-unsafe movements and wrong equipment
- Implement `get_personal_context()` — reads goals.md, injuries.txt, progress.txt as raw text for prompt grounding
- Quick test: run a sample retrieval and print results

**Key design choice**: use semantic retrieval + Python post-filtering rather than LlamaIndex metadata filters, since primary_muscles is stored as a list and list-field filtering is unreliable in simple vector store mode.

---

### Step 2 — Plan Generator (`planner.py`)

The core logic that turns retrieved exercises into a 7-day structured plan.

- Define a 7-day muscle group split tailored to Peter's goals:
  - Day 1: Lower power (squats, jumps, hip drive)
  - Day 2: Upper push (chest, shoulders, triceps)
  - Day 3: Active recovery / cheerleading practice
  - Day 4: Lower hypertrophy + hip mobility rehab
  - Day 5: Upper pull (back, biceps)
  - Day 6: Full body power / plyometrics (when ankle allows)
  - Day 7: Rest or practice
- For each training day, call `retrieve_exercises()` with a targeted query (e.g. "explosive lower body power squat jump")
- Filter results through constraint rules (ankle injury → no plyos by default, equipment check)
- Build a structured prompt that grounds the LLM with:
  - Personal context (goals, injuries, current PRs)
  - The retrieved exercises for each day
  - Explicit constraints
- Call Ollama to generate the final plan with sets/reps/rest and "why this exercise" rationale
- Return structured output (dict or dataclass per day)

---

### Step 3 — CLI Interface (`main.py`)

User-facing entry point using `typer` + `rich`.

- `python main.py generate` — generate a fresh 7-day plan with defaults
- `--ankle-ok` flag — re-enable plyometrics when ankle has recovered
- `--focus` option — override primary focus (e.g. `--focus aesthetics`)
- `--export` flag — save the plan to `plans/YYYY-MM-DD.md`
- `--top-k` option — how many exercises to retrieve per day (default 15)
- Rich-formatted terminal output: day-by-day table, colour-coded muscle groups

---

### Step 4 — Structured Output & Export

Produce clean, reusable output.

- Format each day as: exercise name → sets × reps @ RPE/weight → muscles → why
- Terminal view: `rich` table or panel layout
- Markdown export to `plans/` folder with date-stamped filename
- Keep exported plans as a weekly training log

---

### Step 5 — Prompt Tuning

Iterate on generation quality.

- Tune the system prompt for plan coherence (progressive overload, rest distribution)
- Add few-shot examples of good exercise selection to the prompt
- Test edge cases: ankle flare-up week (all plyos off), travel week (body-only only), deload week
- Evaluate whether qwen2.5:7b is sufficient or if a larger model is needed

---

### Step 6 — Personal Data Update Flow

Keep personal data fresh.

- CLI command: `python main.py update-progress` — prompts for new PRs / body weight and appends to progress.txt
- After updates, re-run `ingest.py` to re-index
- Consider adding a `CHANGELOG` section to progress.txt for week-over-week tracking

---

### Step 7 — Polish & Stretch Goals (optional)

- `python main.py query "best exercises for vertical jump"` — ad-hoc exercise lookup
- Weekly plan diff: compare this week vs last week (avoid repetition)
- Simple injury severity toggle: mild / moderate / recovering
- Unit tests for the filter logic and prompt builder

---

## File Map

```
Workout-Rag-Advisor/
├── ingest.py         ✅ complete — ingestion pipeline
├── query.py          🔨 Step 1 — retrieval engine
├── planner.py        ⬜ Step 2 — plan orchestrator
├── main.py           ⬜ Step 3 — CLI entry point
├── data/
│   ├── exercises.json
│   ├── goals.md
│   ├── injuries.txt
│   └── progress.txt
├── plans/            ⬜ Step 4 — exported weekly plans (auto-created)
├── chroma_db/        ✅ persisted vector index
└── requirements.txt
```

---

## Constraints & Decisions

| Concern | Decision |
|---|---|
| Metadata list filtering | Post-retrieval Python filter (more reliable than LlamaIndex list filters) |
| LLM | Ollama qwen2.5:7b (local, no API key, good on M2 Pro) |
| Embeddings | all-MiniLM-L6-v2 (must match ingest.py to reuse the stored index) |
| Injury handling | Default-off for plyometrics; flag to re-enable when recovered |
| Output format | Rich terminal + optional markdown file |
