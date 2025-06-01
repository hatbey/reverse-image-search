# app/woo_sync.py

from woocommerce import API
import requests
import os
from PIL import Image
from io import BytesIO
import json
import numpy as np
import faiss
from .model import extract_embedding

# WooCommerce API credentials (replace with your actual keys)
wcapi = API(
    url="https://puranpradhan.com.np/puran",  # <-- 🔁 REPLACE with your WooCommerce site URL
    consumer_key="ck_8b64c34586f05eedbb98657392053f171703ad60",  # <-- 🔁 YOUR API key
    consumer_secret="cs_4e073dfa5cb005d0872593af917788b2904177b9",  # <-- 🔁 YOUR API secret
    version="wc/v3"
)

# File paths
IMAGE_DIR = "app/images"
INDEX_PATH = "app/faiss_index.faiss"
METADATA_PATH = "app/product_data.json"

os.makedirs(IMAGE_DIR, exist_ok=True)

def fetch_products():
    print("Fetching WooCommerce products...")
    products = []
    page = 1

    while True:
        response = wcapi.get("products", params={"per_page": 100, "page": page})
        data = response.json()
        if not data:
            break
        products.extend(data)
        page += 1

    print(f"Fetched {len(products)} products.")
    return products

def download_image(url, filename):
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    save_path = os.path.join(IMAGE_DIR, filename)
    img.save(save_path)
    return img

def build_index_from_woocommerce():
    products = fetch_products()

    embeddings = []
    metadata = []

    for product in products:
        if not product.get("images"):
            continue  # skip products without images

        img_url = product["images"][0]["src"]
        filename = f"{product['id']}.jpg"

        try:
            print(f"Processing product {product['id']} - {product['name']}")
            img = download_image(img_url, filename)
            vec = extract_embedding(img)
            embeddings.append(vec)
            metadata.append({
                "id": product["id"],
                "name": product["name"],
                "filename": filename,
                "permalink": product["permalink"]
            })

        except Exception as e:
            print(f"❌ Failed to process {product['id']}: {e}")
            continue

    # Save metadata
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)

    # Create FAISS index
    if embeddings:
        vecs = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, INDEX_PATH)
        print(f"✅ Built FAISS index with {len(embeddings)} products.")
    else:
        print("⚠️ No embeddings generated.")

if __name__ == "__main__":
    build_index_from_woocommerce()
