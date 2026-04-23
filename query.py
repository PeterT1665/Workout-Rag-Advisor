from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# ====================== CONFIG ======================
OLLAMA_MODEL = "qwen2.5:7b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_db"
# ===================================================

# Keywords that indicate plyometric/high-impact movements.
# Exercises whose name or text contain any of these are excluded
# when the ankle-injury constraint is active.
PLYO_KEYWORDS = [
    "jump", "plyometric", "bounding", "hopping", "hurdle", "tuck jump",
    "broad jump", "drop jump", "plyo", "rebound", "bound", "leaping",
]

_settings_done = False
_index = None


def _ensure_settings() -> None:
    global _settings_done
    if not _settings_done:
        # Must match ingest.py exactly so the stored embeddings are compatible.
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=120.0)
        _settings_done = True


def load_index():
    global _index
    if _index is None:
        _ensure_settings()
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        _index = load_index_from_storage(storage_context)
    return _index


def retrieve_exercises(query: str, top_k: int = 20) -> list:
    """Semantic retrieval — returns list of NodeWithScore."""
    index = load_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


def filter_exercises(
    nodes: list,
    exclude_plyo: bool = True,
    allowed_equipment: list[str] | None = None,
    exclude_levels: list[str] | None = None,
) -> list:
    """
    Post-retrieval filter applied in Python after semantic retrieval.

    Args:
        nodes: NodeWithScore list from retrieve_exercises()
        exclude_plyo: drop plyometric/high-impact exercises (ankle injury guard)
        allowed_equipment: if given, keep only exercises using these equipment types
        exclude_levels: difficulty levels to drop (e.g. ["beginner"])
    """
    out = []
    for node in nodes:
        meta = node.node.metadata

        if meta.get("source") != "exercise_db":
            continue

        name = (meta.get("name") or "").lower()
        equipment = (meta.get("equipment") or "").lower()
        level = (meta.get("level") or "").lower()
        text = node.node.text.lower()

        if exclude_plyo and any(kw in name or kw in text for kw in PLYO_KEYWORDS):
            continue

        if allowed_equipment:
            allowed_lower = {e.lower() for e in allowed_equipment}
            if equipment not in allowed_lower:
                continue

        if exclude_levels and level in {l.lower() for l in exclude_levels}:
            continue

        out.append(node)

    return out


def get_personal_context() -> dict[str, str]:
    """Read goals, injuries, and progress as plain text."""
    data_dir = Path("data")
    keys = [("goals", "goals.md"), ("injuries", "injuries.txt"), ("progress", "progress.txt")]
    return {
        key: (data_dir / fname).read_text(encoding="utf-8").strip()
        if (data_dir / fname).exists() else ""
        for key, fname in keys
    }


def node_to_exercise(node) -> dict:
    """Flatten a NodeWithScore into a plain dict for downstream use."""
    meta = node.node.metadata
    return {
        "name": meta.get("name", "Unknown"),
        "equipment": meta.get("equipment", "body only"),
        "level": meta.get("level", "intermediate"),
        "primary_muscles": meta.get("primary_muscles", []),
        "secondary_muscles": meta.get("secondary_muscles", []),
        "text": node.node.text,
        "score": round(node.score or 0.0, 4),
    }


if __name__ == "__main__":
    print("=== query.py smoke test ===\n")

    ctx = get_personal_context()
    print(f"Personal context keys loaded: {list(ctx.keys())}\n")

    query = "explosive lower body power squat hip drive"
    print(f"Query: '{query}'")
    nodes = retrieve_exercises(query, top_k=25)
    print(f"Retrieved {len(nodes)} raw nodes")

    filtered = filter_exercises(nodes, exclude_plyo=True)
    print(f"After plyo filter: {len(filtered)} exercises\n")

    print("Top 5 results:")
    for i, node in enumerate(filtered[:5]):
        ex = node_to_exercise(node)
        print(f"  {i+1}. {ex['name']}")
        print(f"     Equipment : {ex['equipment']}")
        print(f"     Muscles   : {ex['primary_muscles']}")
        print(f"     Score     : {ex['score']}")
