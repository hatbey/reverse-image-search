import json
import faiss
import pymysql
import numpy as np
from .faiss_index import INDEX_PATH
from .woo_sync import DATA_PATH

def sync_faiss_to_wp_db():
    # Load index and metadata
    index = faiss.read_index(INDEX_PATH)
    with open(DATA_PATH, 'r') as f:
        metadata = json.load(f)

    # Connect to your WordPress DB
    conn = pymysql.connect(
        host='your-mysql-host',
        user='your-db-user',
        password='your-db-pass',
        database='your-db-name'
    )
    cursor = conn.cursor()

    # Iterate over vectors and metadata
    for i in range(index.ntotal):
        vec = index.reconstruct(i)
        product_id = metadata[i]["id"]
        vec_json = json.dumps(vec.tolist())

        cursor.execute("""
            REPLACE INTO wp_product_embeddings (product_id, embedding)
            VALUES (%s, %s)
        """, (product_id, vec_json))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Synced {index.ntotal} vectors to wp_product_embeddings.")

if __name__ == "__main__":
    sync_faiss_to_wp_db()
