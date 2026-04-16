import json
import os
import shutil
import faiss
import numpy as np

DIMENSION = 512
INDEX_PATH = "data/embeddings/faiss.index"
MAP_PATH = "data/embeddings/video_map.json"
FRAME_FOLDER = "data/frames"

os.makedirs("data/embeddings", exist_ok=True)

# --- In-memory state ---
index: faiss.IndexFlatIP = None
video_map: list[dict] = []          # [{video_path, frame_path, frame_index}, ...]
processed_videos: set[str] = set()  # quick lookup of already-indexed videos


def _norm(path: str) -> str:
    """Canonical absolute lowercase path for dedup."""
    return os.path.normpath(os.path.abspath(path)).lower()


def _init_index():
    global index
    index = faiss.IndexFlatIP(DIMENSION)


def load_index():
    global index, video_map, processed_videos

    if os.path.exists(INDEX_PATH) and os.path.exists(MAP_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(MAP_PATH, "r") as f:
                video_map = json.load(f)
            processed_videos = {_norm(e["video_path"]) for e in video_map}
            print(f"Loaded index with {index.ntotal} embeddings, {len(processed_videos)} videos")
            return
        except Exception as exc:
            print(f"Failed to load saved index, starting fresh: {exc}")

    _init_index()
    video_map = []
    processed_videos = set()
    print("Initialized fresh FAISS index")


def save_index():
    try:
        faiss.write_index(index, INDEX_PATH)
        with open(MAP_PATH, "w") as f:
            json.dump(video_map, f)
        print(f"Saved index ({index.ntotal} embeddings)")
    except Exception as exc:
        print(f"Error saving index: {exc}")


def is_video_processed(video_path: str) -> bool:
    return _norm(video_path) in processed_videos


def add_embedding(embedding: np.ndarray, video_path: str, frame_path: str = "", frame_index: int = 0):
    embedding = np.asarray(embedding, dtype="float32")
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    if embedding.shape[1] != DIMENSION:
        raise ValueError(f"Expected dim {DIMENSION}, got {embedding.shape[1]}")

    index.add(embedding)
    video_map.append({
        "video_path": os.path.normpath(video_path),
        "frame_path": frame_path,
        "frame_index": frame_index,
    })


def mark_video_processed(video_path: str):
    processed_videos.add(_norm(video_path))


def _rescale_score(raw: float) -> float:
    """
    CLIP cosine similarity typically:
      strong match  0.32 – 0.45
      good match    0.22 – 0.32
      weak match    0.15 – 0.22
      noise         < 0.15

    Rescale [0.15 .. 0.42] → [0% .. 100%], clamped.
    """
    lo, hi = 0.15, 0.42
    scaled = (raw - lo) / (hi - lo)
    return round(max(0.0, min(1.0, scaled)) * 100, 1)


def search(query_embedding: np.ndarray, k: int = 10) -> list[dict]:
    """
    Return top-k unique videos ranked by similarity.
    Each result: {video_path, score, raw_score, matched_frame}
    """
    if index is None or index.ntotal == 0:
        return []

    query_embedding = np.asarray(query_embedding, dtype="float32")
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    search_k = min(k * 30, index.ntotal)
    distances, indices = index.search(query_embedding, search_k)

    # --- Aggregate: pick the BEST frame per video ---
    best: dict[str, dict] = {}
    for raw_score, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(video_map):
            continue
        entry = video_map[int(idx)]
        key = _norm(entry["video_path"])

        if key not in best or raw_score > best[key]["raw"]:
            best[key] = {
                "video_path": entry["video_path"],
                "raw": float(raw_score),
                "matched_frame": entry.get("frame_path", ""),
            }

    # Sort by raw score descending, take top k
    ranked = sorted(best.values(), key=lambda x: x["raw"], reverse=True)[:k]

    results = []
    for r in ranked:
        results.append({
            "video_path": r["video_path"],
            "score": _rescale_score(r["raw"]),
            "raw_score": round(r["raw"], 4),
            "matched_frame": r["matched_frame"],
        })
    return results


# ── Listing indexed content ───────────────────────────────────
def get_indexed_videos() -> list[dict]:
    """Return a list of unique indexed videos with frame counts."""
    counts: dict[str, int] = {}
    paths: dict[str, str] = {}  # canonical → original
    for entry in video_map:
        key = _norm(entry["video_path"])
        counts[key] = counts.get(key, 0) + 1
        paths[key] = entry["video_path"]

    return [
        {"video_path": paths[k], "frames": counts[k]}
        for k in sorted(paths)
    ]


# ── Remove individual video ──────────────────────────────────
def remove_video(video_path: str) -> bool:
    """Remove a single video from the index. Returns True if found."""
    global index, video_map, processed_videos

    target = _norm(video_path)
    keep_indices = [i for i, e in enumerate(video_map) if _norm(e["video_path"]) != target]

    if len(keep_indices) == len(video_map):
        return False  # not found

    _rebuild_index(keep_indices)
    processed_videos.discard(target)

    # Clean up frames folder
    from pathlib import Path
    stem = Path(video_path).stem
    frame_dir = os.path.join(FRAME_FOLDER, stem)
    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir, ignore_errors=True)

    print(f"Removed {video_path} from index")
    return True


def remove_folder(folder_path: str) -> int:
    """Remove all videos whose path starts with folder_path. Returns count removed."""
    global index, video_map, processed_videos

    folder_norm = _norm(folder_path)
    keep_indices = []
    removed_paths = set()

    for i, e in enumerate(video_map):
        vn = _norm(e["video_path"])
        if vn.startswith(folder_norm):
            removed_paths.add(vn)
        else:
            keep_indices.append(i)

    if not removed_paths:
        return 0

    _rebuild_index(keep_indices)
    processed_videos -= removed_paths

    print(f"Removed {len(removed_paths)} video(s) from folder {folder_path}")
    return len(removed_paths)


def _rebuild_index(keep_indices: list[int]):
    """Rebuild FAISS index keeping only the embeddings at keep_indices."""
    global index, video_map

    if not keep_indices:
        _init_index()
        video_map = []
        return

    # Extract all current vectors
    all_vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * DIMENSION)
    all_vectors = np.array(all_vectors, dtype="float32").reshape(index.ntotal, DIMENSION)

    kept_vectors = all_vectors[keep_indices]
    new_map = [video_map[i] for i in keep_indices]

    _init_index()
    index.add(kept_vectors)
    video_map = new_map


def clear_index():
    global video_map, processed_videos
    _init_index()
    video_map = []
    processed_videos = set()
    for p in (INDEX_PATH, MAP_PATH):
        if os.path.exists(p):
            os.remove(p)
    print("Index cleared")
