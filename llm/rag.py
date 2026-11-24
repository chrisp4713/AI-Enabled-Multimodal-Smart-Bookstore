# UC-09, UC-14
import json
from pathlib import Path
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# read file
DEFAULT_RAG_DOCS_DIR = Path(__file__).resolve().parent.parent / "rag_docs"
BOOK_INDEX_DIR = Path(__file__).resolve().parent.parent / "book_index"


# UC-09, UC-14
def load_file(docs_dir: Path | str = DEFAULT_RAG_DOCS_DIR) -> list[dict]:
    chunks = []
    folder = Path(docs_dir)

    if not folder.exists():
        print(f"Error: The folder at {folder} was not found.")
        return chunks

    for md_file in sorted(folder.glob("*.md")):  # find all markdown files in the folder
        try:
            with open(md_file, "r", encoding="utf-8") as file:  # open the file in read only mode and read its content
                content = file.read()
        except FileNotFoundError:
            print(f"Error: The file at {md_file} was not found.")
            continue
        except PermissionError:
            print(f"Error: Permission denied when trying to read the file at {md_file}.")
            continue

        # Split on # headings and keep non-empty chunks
        sections = re.split(r"(?m)^(?=#{1,6}\s)", content)
        for section in sections:
            cleaned = section.strip()  # remove leading and trailing whitespace
            if cleaned:
                chunks.append({  # add to the list of chunks
                    "source": md_file.name,
                    "text": cleaned,
                })

    return chunks


# Turn chunks of text into a numpy array
model = SentenceTransformer("all-MiniLM-L6-v2")  # load the lightweight model for sentence embeddings

# Cache the loaded chunks and embeddings to avoid reloading and re-encoding
_cached_chunks = None
_cached_embeddings = None


# UC-09, UC-14
def get_rag_index(model):
    global _cached_chunks, _cached_embeddings
    if _cached_chunks is None or _cached_embeddings is None:  # is the cache still empty?
        _cached_chunks = load_file()

        # checks for an empty or missing rag_docs/ folder
        if not _cached_chunks:
            _cached_embeddings = np.array([])
            return _cached_chunks, _cached_embeddings

        texts = [doc["text"] for doc in _cached_chunks]  # loops through every doc in cached_chunks, grabs the doc["text"] part and puts it into a list called texts
        _cached_embeddings = model.encode(texts, convert_to_numpy=True)  # return the embeddings as a numpy array

    return _cached_chunks, _cached_embeddings


# UC-09, UC-14
def search_rag(query: str, model, top_k: int = 3) -> list[dict]:
    chunks, chunk_embeddings = get_rag_index(model)

    if not chunks:
        return []

    query_embedding = model.encode(query, convert_to_numpy=True)  # encode the incoming question into a numpy array

    # Compute cosine similarity
    dot_products = np.dot(chunk_embeddings, query_embedding)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    scores = dot_products / (chunk_norms * query_norm)

    top_indices = np.argsort(scores)[::-1][:top_k]  # sort the scores in descending order and grab the top k indices

    results = []  # return the matching chunks along with their scores
    for idx in top_indices:
        results.append({
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"],
            "score": float(scores[idx]),
        })

    return results


# UC-14
def index_book(book_id: str, book_path: str | Path, model=model) -> tuple[list[dict], np.ndarray]:
    """Split a book by chapter, embed chunks, and cache them on disk per book_id."""
    BOOK_INDEX_DIR.mkdir(exist_ok=True)
    meta_file = BOOK_INDEX_DIR / f"{book_id}.json"
    emb_file = BOOK_INDEX_DIR / f"{book_id}.npy"

    # Load from disk cache if this book was already indexed
    if meta_file.exists() and emb_file.exists():
        chunks = json.loads(meta_file.read_text(encoding="utf-8"))
        embeddings = np.load(emb_file)
        return chunks, embeddings

    path = Path(book_path)
    if not path.exists():
        print(f"Error: The book file at {path} was not found.")
        return [], np.array([])

    text = path.read_text(encoding="utf-8", errors="ignore")

    # Split into chapters
    parts = re.split(r"(?im)^(chapter\s+[0-9ivxlc]+.*)$", text)
    chunks = []

    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            chapter = parts[i].strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            words = body.split()
            for j in range(0, len(words), 250):
                chunk_text = " ".join(words[j:j + 250]).strip()
                if chunk_text:
                    chunks.append({"chapter": chapter, "text": chunk_text})
    else:
        # Fallback if the book has no standard chapter headings
        words = text.split()
        for j in range(0, len(words), 250):
            chunk_text = " ".join(words[j:j + 250]).strip()
            if chunk_text:
                chunks.append({"chapter": "Full Text", "text": chunk_text})

    if not chunks:
        return [], np.array([])

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    meta_file.write_text(json.dumps(chunks), encoding="utf-8")
    np.save(emb_file, embeddings)
    return chunks, embeddings


# UC-14
def search_book(book_id: str, book_path: str | Path, query: str, model=model, top_k: int = 3) -> list[dict]: #Return the top matching chapter chunks for a specific book.
    
    chunks, embeddings = index_book(book_id, book_path, model)

    if not chunks:
        return []

    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    scores = np.dot(embeddings, query_embedding)  # cosine similarity since vectors are normalized
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "chapter": chunks[idx]["chapter"],
            "text": chunks[idx]["text"],
            "score": float(scores[idx]),
        }
        for idx in top_indices
    ]


# if __name__ == "__main__":
#     test_query = "Who is building the login screen?"
#     matches = search_rag(test_query, model)
#     print(f"Query: {test_query}\n")
#     for i, match in enumerate(matches, start=1):
#         print(f"--- Match {i} | Score: {match['score']:.3f} | Source: {match['source']} ---")
#         print(match["text"], "\n")
