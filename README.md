# Video Search Engine

A web-based application to index, search, and manage videos using natural language queries. It leverages deep learning (CLIP model) to generate embeddings for both video frames and text queries, enabling semantic search over video content.

---

## Features

- **Upload and index videos or folders**
- **Semantic search:** Find videos by describing their content in plain English
- **Efficient search:** Uses FAISS for fast similarity search
- **Modern web UI:** Simple, responsive interface
- **Index management:** Remove videos/folders, clear all, view progress

---

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [faiss-cpu](https://github.com/facebookresearch/faiss)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [fastapi](https://fastapi.tiangolo.com/)
- [uvicorn](https://www.uvicorn.org/)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Creating and Uploading the `hf_cache` Directory

The `hf_cache` directory stores pre-downloaded model files from Hugging Face (such as CLIP weights and tokenizers). This cache allows the app to run offline and speeds up model loading.

### How to Create `hf_cache`

1. **Automatic creation:**
   - The cache is created automatically the first time you run the app or any script that loads the CLIP model (see `app/embeddings.py`).
   - To trigger cache creation, simply run:

     ```bash
     python main.py
     ```
   - The required model files will be downloaded into `hf_cache` if not already present.

2. **Manual creation (optional):**
   - You can also run a minimal script to pre-download the model:

     ```python
     from transformers import CLIPModel, CLIPProcessor
     CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="./hf_cache")
     CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="./hf_cache")
     ```

### How to Upload or Share `hf_cache`

1. **Compress the directory:**
   - Zip the `hf_cache` folder:

     ```bash
     zip -r hf_cache.zip hf_cache
     ```

2. **Upload or transfer:**
   - Share the `hf_cache.zip` file via your preferred method (cloud storage, USB, etc.).

3. **Extract on target machine:**
   - Unzip in the project root so the structure is preserved:

     ```bash
     unzip hf_cache.zip
     ```

This ensures all model files are available locally and avoids repeated downloads.


## How to Run

1. **Start the server:**

   ```bash
   python main.py
   ```

2. **Open the UI:**

   Go to [http://localhost:8000](http://localhost:8000) in your browser.

3. **Index videos:**
   - Upload individual videos, select a folder, or drag-and-drop.
   - Wait for processing to complete (progress bar shown).

4. **Search:**
   - Enter a description (e.g., "a person riding a bike").
   - View matching video results.

5. **Manage index:**
   - Remove videos/folders or clear the entire index.

---

## Project Structure

```
main.py                  # Entry point, starts FastAPI server
app/
  main.py                # FastAPI app and API endpoints
  video_processing.py    # Frame extraction from videos
  embeddings.py          # CLIP model loading and embedding generation
  search.py              # FAISS index and search logic
  routes.py              # (optional/empty)
data/
  videos/                # Uploaded videos
  frames/                # Extracted frames
  embeddings/            # FAISS index and video map
```

---

## How It Works

1. **Indexing:**
   - Upload videos or select a folder.
   - Frames are extracted from each video.
   - Each frame is embedded using the CLIP model.
   - Embeddings are stored in a FAISS index for fast search.

2. **Searching:**
   - User enters a natural language query.
   - The query is embedded using CLIP.
   - The system finds the most similar frames/videos using FAISS.

3. **Managing:**
   - List, remove, or clear indexed videos.

---

## Technologies Used

- **Python** (FastAPI, Uvicorn, OpenCV, FAISS, PyTorch, Transformers)
- **CLIP Model:** For generating embeddings from images and text
- **FAISS:** For efficient similarity search
- **HTML/CSS/JS:** For the web UI

---

## License

This project is for educational and research purposes.
