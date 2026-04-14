"""
app.py  --  FastAPI Sentiment Analysis API (Word2Vec version)
Run: uvicorn app:app --reload
"""

import pickle
import time
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, Field
from gensim.models import Word2Vec

from text_cleaner import clean_text


# -- Paths ------------------------------------------------------------------
MODEL_PATH = Path("models/sentiment_model_w2v.pkl")
W2V_PATH = Path("models/word2vec.pkl")

LABEL_MAP  = {0: "Negative", 1: "Neutral", 2: "Positive"}


# -- Helpers ----------------------------------------------------------------
def tokenize(text):
    return text.split()

def sentence_vector(tokens, w2v):
    vectors = [w2v.wv[word] for word in tokens if word in w2v.wv]
    if len(vectors) == 0:
        return np.zeros(w2v.vector_size)
    return np.mean(vectors, axis=0)


# -- Load model at startup --------------------------------------------------
model     = None
w2v_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, w2v_model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not W2V_PATH.exists():
        raise FileNotFoundError(f"Word2Vec model not found: {W2V_PATH}")

    w2v_model = Word2Vec.load(str(W2V_PATH))
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Model loaded     :", type(model).__name__)
    print("W2V vocab size   :", len(w2v_model.wv))
    print("W2V vector size  :", w2v_model.vector_size)
    yield


# -- App --------------------------------------------------------------------
app = FastAPI(
    title       = "Sentiment Analysis API (Word2Vec)",
    description = "Classifies text as Negative / Neutral / Positive using W2V.",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# -- Schemas ----------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000,
                      example="This product is absolutely amazing!")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=200)

    @field_validator("texts")
    @classmethod
    def validate_each_text(cls, texts):
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValueError(f"Item {i} must be a string")
            if len(t.strip()) < 3:
                raise ValueError(f"Item {i} is too short: '{t}'")
            if len(t) > 5000:
                raise ValueError(f"Item {i} exceeds 5000 characters")
        return texts


class PredictResponse(BaseModel):
    label      : str
    label_id   : int
    input_text : str
    cleaned    : str
    time_ms    : float


class BatchResponse(BaseModel):
    predictions : list[PredictResponse]
    count       : int
    total_ms    : float


# -- Routes -----------------------------------------------------------------
@app.get("/", tags=["Info"])
def root():
    return {
        "message"  : "Sentiment Analysis API (Word2Vec) is running",
        "docs"     : "/docs",
        "endpoints": {
            "POST /predict"       : "Single text prediction",
            "POST /predict/batch" : "Batch prediction (max 200)",
            "GET  /health"        : "Health check",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    if model is None or w2v_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status"     : "healthy",
        "model"      : type(model).__name__,
        "w2v_vocab"  : len(w2v_model.wv),
        "vector_size": w2v_model.vector_size,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    t0 = time.perf_counter()

    cleaned  = clean_text(request.text)
    tokens   = tokenize(cleaned)
    vec      = sentence_vector(tokens, w2v_model).reshape(1, -1)
    label_id = int(model.predict(vec)[0])

    return PredictResponse(
        label      = LABEL_MAP[label_id],
        label_id   = label_id,
        input_text = request.text,
        cleaned    = cleaned,
        time_ms    = round((time.perf_counter() - t0) * 1000, 3),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    t0 = time.perf_counter()

    cleaned_texts = [clean_text(t) for t in request.texts]
    token_lists   = [tokenize(c) for c in cleaned_texts]
    vecs          = np.array([sentence_vector(tok, w2v_model) for tok in token_lists])
    label_ids     = model.predict(vecs)

    total_ms = round((time.perf_counter() - t0) * 1000, 3)
    avg_ms   = round(total_ms / len(request.texts), 3)

    predictions = [
        PredictResponse(
            label      = LABEL_MAP[int(lid)],
            label_id   = int(lid),
            input_text = raw,
            cleaned    = cleaned,
            time_ms    = avg_ms,
        )
        for raw, cleaned, lid in zip(request.texts, cleaned_texts, label_ids)
    ]

    return BatchResponse(
        predictions = predictions,
        count       = len(predictions),
        total_ms    = total_ms,
    )