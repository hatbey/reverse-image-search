# app/faiss_index.py
import faiss
import numpy as np
import os
import json
from PIL import Image
from .model import extract_embedding

IMAGE_DIR = "app/images"
INDEX_PATH = "app/faiss_index.faiss"
METADATA_PATH = "app/product_data.json"

# Load metadata
with open(METADATA_PATH, "r") as f:
    product_metadata = json.load(f)

def build_index():
    embeddings = []
    for item in product_metadata:
        img_path = os.path.join(IMAGE_DIR, item["filename"])
        img = Image.open(img_path).convert("RGB")
        vec = extract_embedding(img)
        embeddings.append(vec)
    
    embeddings = np.array(embeddings).astype("float32")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

def load_index():
    return faiss.read_index(INDEX_PATH)

def search(query_vec, top_k=5):
    index = load_index()
    distances, indices = index.search(query_vec.astype("float32").reshape(1, -1), top_k)
    results = [product_metadata[i] for i in indices[0]]
    return results
