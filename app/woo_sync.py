# app/woo_sync.py

import os
import json
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import faiss

# Constants
WOO_API_URL = os.getenv("https://puranpradhan.com.np/puran")
WOO_CONSUMER_KEY = os.getenv("ck_8b64c34586f05eedbb98657392053f171703ad60")
WOO_CONSUMER_SECRET = os.getenv("cs_4e073dfa5cb005d0872593af917788b2904177b9")
DATA_PATH = "app/product_data.json"
INDEX_PATH = "app/faiss_index.faiss"
INDEX_DIM = 2048

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def get_products():
    print("🔄 Fetching products from WooCommerce...")
    page = 1
    all_products = []
    while True:
        res = requests.get(
            f"{WOO_API_URL}/wp-json/wc/v3/products",
            auth=(WOO_CONSUMER_KEY, WOO_CONSUMER_SECRET),
            params={"per_page": 100, "page": page}
        )
        if res.status_code != 200:
            raise Exception("❌ Failed to fetch products:", res.text)
        products = res.json()
        if not products:
            break
        all_products.extend(products)
        page += 1
    print(f"✅ Retrieved {len(all_products)} products.")
    return all_products

def embed_image_url(img_url):
    try:
        res = requests.get(img_url, timeout=10)
        img = Image.open(BytesIO(res.content)).convert("RGB")
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        return features[0]
    except Exception as e:
        print(f"⚠️ Error processing image: {e}")
        return None

def build_index(products):
    print("⚙️ Building FAISS index...")
    index = faiss.IndexFlatL2(INDEX_DIM)
    metadata = []

    for product in products:
        img_url = product.get("images", [{}])[0].get("src")
        if not img_url:
            continue

        vec = embed_image_url(img_url)
        if vec is None:
            continue

        index.add(np.array([vec]).astype('float32'))
        metadata.append({
            "id": product["id"],
            "name": product["name"],
            "permalink": product["permalink"],
            "image": img_url
        })

        print(f"✅ Embedded product: {product['name']}")

    faiss.write_index(index, INDEX_PATH)
    with open(DATA_PATH, 'w') as f:
        json.dump(metadata, f)

    print(f"✅ Index saved with {index.ntotal} items.")

def run():
    products = get_products()
    build_index(products)

if __name__ == "__main__":
    run()
