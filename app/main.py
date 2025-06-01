# app/main.py
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
from .model import extract_embedding
from .faiss_index import search

app = FastAPI()

@app.post("/api/search-image")
async def search_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    vec = extract_embedding(img)
    results = search(vec, top_k=5)
    return {"results": results}
