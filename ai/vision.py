"""
Picture search over book covers.                      UC-05 (search), UC-10 (add book)

How it works
------------
CLIP turns any image into a 512-number vector so that visually similar images
get similar vectors.
  * index_images() embeds each book cover ONCE and saves the vectors in
    data/image_index.npz, so searching never re-processes the covers.
  * search_by_image() embeds the user's picture, scores it against every saved
    cover (cosine similarity = one matrix multiply), and returns the best
    matches. With 10-20 books this takes milliseconds; the only slow part is
    loading the CLIP model, so call warm_up() at app start (UC-05: <= 5 s).

Public functions
----------------
    warm_up()
    index_images({book_id: cover_path, ...}, rebuild=False) -> IndexResult
    search_by_image(image, top_k=5) -> [{"book_id": ..., "score": ...}, ...]
    VisionError       raised with a user-friendly message on bad input
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "image_index.npz"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}          # UC-05: JPG/PNG only

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512                                          # ViT-B-32 output size


class VisionError(Exception):
    """Bad or unreadable image. The message is safe to show to the user."""


@dataclass
class IndexResult:
    indexed: list[str] = field(default_factory=list)            # book_ids added/updated
    skipped: dict[str, str] = field(default_factory=dict)       # book_id -> reason


# --------------------------------------------------------------------------
# Model loading (lazy)
# --------------------------------------------------------------------------
_model_lock = threading.RLock()
_model = None
_preprocess = None


def _load_clip():
    global _model, _preprocess
    with _model_lock:
        if _model is None:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL, pretrained=CLIP_PRETRAINED)
            _model, _preprocess = model.eval(), preprocess
        return _model, _preprocess


def warm_up() -> None:
    """Load CLIP now so the first search is fast. Call once at app start."""
    _load_clip()


# --------------------------------------------------------------------------
# Image loading and embedding
# --------------------------------------------------------------------------
def _open_image(source) -> Image.Image:
    """Accept a file path (JPG/PNG only) or a BGR numpy frame from OpenCV."""
    if isinstance(source, np.ndarray):
        return Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    path = Path(source)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise VisionError("Only JPG and PNG images are supported.")
    if not path.is_file():
        raise VisionError(f"Image file not found: {path.name}")
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (OSError, ValueError):
        raise VisionError(f"{path.name} could not be read as an image.")


def _embed_image(image: Image.Image) -> np.ndarray:
    """PIL image -> unit-length 512-number vector."""
    import torch
    model, preprocess = _load_clip()
    with torch.no_grad():
        feat = model.encode_image(preprocess(image).unsqueeze(0))
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].numpy().astype(np.float32)


# --------------------------------------------------------------------------
# Index storage (atomic writes, NFR-09 style)
# --------------------------------------------------------------------------
_index_lock = threading.Lock()


def _load_index() -> tuple[list[str], np.ndarray]:
    if not INDEX_PATH.exists():
        return [], np.zeros((0, EMBED_DIM), dtype=np.float32)
    with np.load(INDEX_PATH, allow_pickle=False) as data:
        return [str(i) for i in data["ids"]], data["embeddings"]


def _save_index(ids: list[str], embeddings: np.ndarray) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_name("image_index.tmp.npz")
    np.savez(tmp, ids=np.array(ids, dtype=str), embeddings=embeddings)
    os.replace(tmp, INDEX_PATH)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def index_images(covers: dict, rebuild: bool = False) -> IndexResult:
    """Add book covers to the search index.

    covers:  {book_id: path_to_cover_image}
    rebuild: True = throw away the old index and start fresh;
             False (default) = add to / update the existing index, so Jin's
             add_book can call index_images({new_id: new_cover}) for one book.

    Bad files are skipped and reported in the result, never crash the batch.
    """
    result = IndexResult()
    with _index_lock:
        if rebuild:
            ids, matrix = [], np.zeros((0, EMBED_DIM), dtype=np.float32)
        else:
            ids, matrix = _load_index()
        rows = list(matrix)
        position = {book_id: i for i, book_id in enumerate(ids)}

        for book_id, path in covers.items():
            book_id = str(book_id)
            try:
                vec = _embed_image(_open_image(path))
            except VisionError as err:
                result.skipped[book_id] = str(err)
                continue
            if book_id in position:                    # re-index an existing book
                rows[position[book_id]] = vec
            else:
                position[book_id] = len(ids)
                ids.append(book_id)
                rows.append(vec)
            result.indexed.append(book_id)

        if result.indexed or rebuild:
            matrix = np.vstack(rows) if rows else np.zeros((0, EMBED_DIM), dtype=np.float32)
            _save_index(ids, matrix)
    return result


def search_by_image(image, top_k: int = 5) -> list[dict]:
    """Find the covers most similar to `image` (a JPG/PNG path or OpenCV frame).

    Returns [{"book_id": str, "score": float}, ...], best first. The list is
    empty if nothing is indexed yet. Raises VisionError for an unusable image.
    """
    query = _embed_image(_open_image(image))
    ids, matrix = _load_index()
    if not ids:
        return []
    scores = matrix @ query                            # cosine similarity
    order = np.argsort(scores)[::-1][:top_k]
    return [{"book_id": ids[i], "score": float(scores[i])} for i in order]


# --------------------------------------------------------------------------
# Quick manual test from the project root:
#   python -m ai.vision index path/to/covers_folder    (file name = book id)
#   python -m ai.vision search path/to/photo.jpg
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time
    usage = "usage: python -m ai.vision index <folder> | search <image>"
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "index":
            files = {p.stem: p for p in sorted(Path(sys.argv[2]).iterdir())
                     if p.suffix.lower() in ALLOWED_EXTENSIONS}
            print(index_images(files, rebuild=True))
        elif len(sys.argv) == 3 and sys.argv[1] == "search":
            t0 = time.perf_counter()
            for hit in search_by_image(sys.argv[2]):
                print(f"{hit['score']:.3f}  {hit['book_id']}")
            print(f"({time.perf_counter() - t0:.2f}s, includes model load)")
        else:
            print(usage)
    except VisionError as err:
        print(err)