import os
import json
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# ====================== CONFIG ======================
# Change this if you pulled a different model
OLLAMA_MODEL = "qwen2.5:7b"

# Embedding model (fast and good on M2 Pro)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chroma persistence directory
PERSIST_DIR = "./chroma_db"
# ===================================================

# Set global settings
Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

def load_personal_documents() -> list[Document]:
    """Load your goals, injuries, and progress as Documents"""
    docs = []
    data_dir = Path("data")
    
    # goals.md
    goals_path = data_dir / "goals.md"
    if goals_path.exists():
        text = goals_path.read_text(encoding="utf-8")
        docs.append(Document(text=text, metadata={"source": "goals", "type": "personal"}))
    
    # injuries.txt
    injuries_path = data_dir / "injuries.txt"
    if injuries_path.exists():
        text = injuries_path.read_text(encoding="utf-8")
        docs.append(Document(text=text, metadata={"source": "injuries", "type": "personal"}))
    
    # progress.txt
    progress_path = data_dir / "progress.txt"
    if progress_path.exists():
        text = progress_path.read_text(encoding="utf-8")
        docs.append(Document(text=text, metadata={"source": "progress", "type": "personal"}))
    
    print(f"Loaded {len(docs)} personal documents")
    return docs

def load_exercises() -> list[Document]:
    """Load exercises.json and create rich documents with metadata"""
    exercises_path = Path("data/exercises.json")
    if not exercises_path.exists():
        raise FileNotFoundError("exercises.json not found in data/ folder!")
    
    with open(exercises_path, "r", encoding="utf-8") as f:
        exercises = json.load(f)
    
    docs = []
    for ex in exercises:
        # Build clean text for the LLM
        text = f"Exercise: {ex.get('name', '')}\n"
        text += f"Equipment: {ex.get('equipment', 'body only')}\n"
        text += f"Level: {ex.get('level', 'intermediate')}\n"
        text += f"Force: {ex.get('force', '')}\n"
        text += f"Primary Muscles: {', '.join(ex.get('primaryMuscles', []))}\n"
        text += f"Secondary Muscles: {', '.join(ex.get('secondaryMuscles', []))}\n"
        text += f"Instructions: {' '.join(ex.get('instructions', []))}\n"
        
        # Rich metadata for better retrieval/filtering later
        metadata = {
            "source": "exercise_db",
            "name": ex.get("name"),
            "equipment": ex.get("equipment"),
            "level": ex.get("level"),
            "primary_muscles": ex.get("primaryMuscles", []),
            "secondary_muscles": ex.get("secondaryMuscles", []),
            "id": ex.get("id")
        }
        
        docs.append(Document(text=text.strip(), metadata=metadata))
    
    print(f"Loaded {len(docs)} exercises from JSON")
    return docs

def main():
    print("Starting ingestion...")
    
    # Load all documents
    personal_docs = load_personal_documents()
    exercise_docs = load_exercises()
    all_docs = personal_docs + exercise_docs
    
    print(f"Total documents to index: {len(all_docs)}")
    
    # Create index
    index = VectorStoreIndex.from_documents(all_docs)
    
    # Persist to disk
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    
    print("✅ Ingestion complete! Vector database saved to chroma_db/")
    print("You can now run queries against your personal RAG system.")

if __name__ == "__main__":
    main()