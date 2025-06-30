from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from pymilvus import connections, Collection

app = FastAPI()

# --- Milvus Setup ---
connections.connect("default", host="localhost", port="19530")
collection = Collection("face_vectors")
collection.load()

# --- Request Model ---
class EmbeddingRequest(BaseModel):
    embedding: List[float]

@app.post("/identify")
def identify_face(request: EmbeddingRequest):
    try:
        face_embedding = torch.tensor(request.embedding)

        results = collection.query(
            expr="",
            output_fields=["name", "embedding"],
            limit=10000
        )

        if not results:
            return {"name": "Unknown"}

        names = []
        vectors = []
        for item in results:
            names.append(item["name"])
            vectors.append(torch.tensor(item["embedding"]))

        db_embeddings = torch.stack(vectors)
        distances = torch.norm(db_embeddings - face_embedding, dim=1)
        min_dist, min_idx = torch.min(distances, dim=0)

        threshold = 1.0
        if min_dist.item() > threshold:
            return {"name": "Unknown"}

        return {"name": names[min_idx]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
