from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI(title="GeekyClouds Embedding Service")

model = SentenceTransformer(MODEL_NAME)


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dims: int


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    embeddings = model.encode(
        request.texts,
        convert_to_numpy=True
    )

    embedding_lists = embeddings.tolist()

    return EmbedResponse(
        embeddings=embedding_lists,
        model=MODEL_NAME,
        dims=len(embedding_lists[0]) if embedding_lists else 0,
    )
    