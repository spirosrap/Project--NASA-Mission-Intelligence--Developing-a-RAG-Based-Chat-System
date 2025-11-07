import os
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    candidate_dirs = []
    for path in current_dir.iterdir():
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        has_sqlite = (path / "chroma.sqlite3").exists()
        has_chroma = "chroma" in path.name.lower()
        if has_sqlite or has_chroma:
            candidate_dirs.append(path)

    for dir_path in sorted(candidate_dirs):
        try:
            client = chromadb.PersistentClient(
                path=str(dir_path),
                settings=Settings(anonymized_telemetry=False)
            )
            collections = client.list_collections()
            for collection in collections:
                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = "unknown"
                key = f"{dir_path.name}::{collection.name}"
                backends[key] = {
                    "directory": str(dir_path),
                    "collection_name": collection.name,
                    "display_name": f"{dir_path.name} / {collection.name} ({doc_count} docs)",
                    "document_count": doc_count
                }
        except Exception as exc:
            key = f"{dir_path.name}::error"
            snippet = str(exc)
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            backends[key] = {
                "directory": str(dir_path),
                "collection_name": "",
                "display_name": f"{dir_path.name} (error: {snippet})",
                "document_count": 0,
                "error": str(exc)
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    try:
        api_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        embedding_fn = None
        if api_key:
            embedding_fn = OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small"
            )
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(
            collection_name,
            embedding_function=embedding_fn
        )
        return collection, True, None
    except Exception as exc:
        return None, False, str(exc)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    if not collection or not query:
        return None

    where_filter = None
    if mission_filter:
        normalized = mission_filter.strip().lower()
        if normalized and normalized not in {"all", "*", "any"}:
            where_filter = {"mission": normalized.replace(" ", "_")}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=max(1, n_results),
            where=where_filter
        )
        return results
    except Exception as exc:
        return {"error": str(exc)}

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    context_parts: List[str] = ["## Retrieved Context"]

    for idx, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        mission = metadata.get("mission", "unknown").replace("_", " ").title()
        source = metadata.get("source") or metadata.get("file_path", "Unknown Source")
        category = metadata.get("document_category", "general").replace("_", " ").title()

        header = f"[{idx}] {mission} • {source} • {category}"
        context_parts.append(header)

        snippet = doc.strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rstrip() + "..."
        context_parts.append(snippet)

    return "\n\n".join(context_parts)
