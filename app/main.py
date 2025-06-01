# app/main.py

from fastapi import FastAPI, File, UploadFile
from .faiss_index import load_faiss, search_image
import os
from PIL import Image
from io import BytesIO
from .model import extract_embedding

app = FastAPI()

@app.on_event("startup")
def startup():
    # Only build index if it doesn't already exist
    if not os.path.exists("app/faiss_index.faiss"):
        from .woo_sync import build_index_from_woocommerce
        print("🔁 Building FAISS index from WooCommerce...")
        build_index_from_woocommerce()
    else:
        print("✅ FAISS index found, skipping rebuild.")

# Your existing search endpoint
@app.post("/api/search-image")
async def search(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    query_vec = extract_embedding(image)
    results = search_image(query_vec)
    return results
