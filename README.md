# Personal Workout & Progress Advisor (RAG MVP)

RAG-powered Workout Advisor built with LlamaIndex + local Ollama LLM. 

Pulls from my training logs, goals/injuries, PRs, cheerleading needs, and a free exercise DB to generate tailored 7-day plans focused on explosive power (primary) and aesthetic physique (secondary), with clear substitution explanations ("why this exercise instead"). 

Tackles real RAG challenges like injury-safe retrieval, metadata filtering, prompt engineering to ground responses, and handling personal constraints. 

Purely local/offline MVP – Python, Chroma vector DB, CLI-first.

## Project Goal
Build something I actually use weekly while deepening RAG knowledge (ingestion, retrieval failures, synthesis issues) for interviews and to demonstrate passion in ML/AI tooling.

## Key Features (MVP)
- Ingest personal data (goals, injuries, progress/PRs) + open exercise database (~800+ exercises)
- Semantic retrieval of safe/relevant exercises (filter by equipment, muscles, level, injury avoidance)
- Generate weekly workout plans via local LLM with explanations
- CLI interface for querying (e.g. "7-day explosive power plan, dumbbells only, avoid deep knee flexion")

## Tech Stack
- **Python** (core language)
- **LlamaIndex** (for document ingestion, indexing, querying)
- **Chroma** (local persistent vector database)
- **Sentence-Transformers** (local embeddings: all-MiniLM-L6-v2 or similar)
- **Ollama** (local LLM inference: qwen2.5:7b recommended for M2 Pro balance of speed & reasoning)
- CLI-first (scripts: ingest.py, query.py)

## Setup (100% Local)
1. Install Ollama: https://ollama.com (Mac one-click installer)
2. Pull a strong local model (runs great on M2 Pro):