import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CACHE_DIR = "./hf_cache"
MODEL_NAME = "openai/clip-vit-base-patch16"

os.makedirs(CACHE_DIR, exist_ok=True)

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, use_fast=False)
model.eval()
print(f"CLIP model loaded on {device}")


def _to_numpy_embedding(features):
    """Convert model output tensor to a normalized numpy embedding."""
    if isinstance(features, torch.Tensor):
        tensor = features
    elif hasattr(features, "pooler_output") and features.pooler_output is not None:
        tensor = features.pooler_output
    elif hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        tensor = features.last_hidden_state[:, 0, :]
    else:
        raise TypeError(f"Unsupported embedding output type: {type(features)!r}")

    embedding = tensor.detach().cpu().numpy().astype("float32")

    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    # L2-normalize so inner product == cosine similarity
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embedding / norms


def get_image_embedding(image_path: str):
    """Return a (1, 512) normalized numpy embedding for an image, or None on failure."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)

        return _to_numpy_embedding(features)
    except Exception as exc:
        print(f"Error in image embedding for {image_path}: {exc}")
        return None


def get_text_embedding(text: str):
    """Return a (1, 512) normalized numpy embedding for a text query, or None on failure."""
    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            features = model.get_text_features(**inputs)

        return _to_numpy_embedding(features)
    except Exception as exc:
        print(f"Error in text embedding: {exc}")
        return None
