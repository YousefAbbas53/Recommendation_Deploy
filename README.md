---
title: LITVISION Recommendation API
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# LITVISION Book Recommendation API

A production-ready FastAPI service for the LITVISION Book Recommendation Feature. This API provides personalized book recommendations using zero-shot genre classification, SentenceTransformer embeddings, FAISS similarity search, and an event-weighted ranking pipeline.

Fully configured for deployment on Hugging Face Spaces with Docker SDK.

## Features

- **Zero-Shot Genre Classification** using `joeddav/xlm-roberta-large-xnli`
- **SentenceTransformer Embeddings** using `paraphrase-multilingual-MiniLM-L12-v2`
- **FAISS Similarity Search** with `IndexFlatIP` for cosine similarity on normalized vectors
- **Event-Weighted User Profiling** with configurable view/like weights
- **Genre-Balanced Feed Allocation** for diverse recommendations
- **Cosine-Similarity Ranking Pipeline** for personalized ordering
- **GPU/CPU Fallback** with FP16 optimization on CUDA
- **Async Processing** via `asyncio.to_thread` for non-blocking inference
- **Production Error Handling** including CUDA OOM recovery

## API Endpoints

### GET /

Returns basic API information.

```json
{
  "api": "LITVISION Book Recommendation API",
  "status": "online",
  "version": "1.0.0",
  "endpoints": ["/health", "/recommend"]
}
```

### GET /health

Returns health status and model readiness.

```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda",
  "total_books": 200,
  "faiss_index_size": 200
}
```

### POST /recommend

Generates personalized book recommendations for a user.

**Request Body:**

```json
{
  "user_id": 1,
  "interactions": [
    {
      "book_id": 5,
      "event_type": "like",
      "timestamp": "2025-01-01T00:00:00"
    },
    {
      "book_id": 12,
      "event_type": "view",
      "timestamp": "2025-01-02T00:00:00"
    }
  ],
  "favorite_genres": ["Fantasy", "Science Fiction"],
  "viewed_books": [1, 2, 3],
  "feed_size": 20
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| user_id | int | Yes | — | Unique user identifier (> 0) |
| interactions | list | No | null | Explicit user-book interaction events |
| favorite_genres | list | No | null | Preferred genres for boosting |
| viewed_books | list | No | null | Book IDs already viewed by the user |
| feed_size | int | No | 20 | Number of recommendations (1-100) |

**Valid Genres:**

Fantasy, Romance, Mystery, Science Fiction, Self-Help, History, Business, Children, Horror, Poetry

**Response:**

```json
{
  "success": true,
  "user_id": 1,
  "recommendations": [
    {
      "book_id": 42,
      "title": "Fantasy Book 42",
      "author": "Author 7",
      "genre": "Fantasy",
      "score": 0.9234
    }
  ],
  "genre_distribution": {
    "Fantasy": 8,
    "Romance": 4,
    "Mystery": 3,
    "Science Fiction": 2,
    "Self-Help": 1,
    "History": 1,
    "Children": 1
  },
  "total_recommendations": 20,
  "processing_time_seconds": 1.234
}
```

## Folder Structure

```text
.
├── app.py                # FastAPI endpoints and lifespan events
├── recommender.py        # Full recommendation pipeline engine
├── utils.py              # Logging, device helpers, and cleanup
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration for HF Spaces
├── .dockerignore         # Docker build exclusions
├── .gitignore            # Git exclusions
├── .gitattributes        # Line ending configuration
├── README.md             # This file
└── sample_data/
    └── books.csv         # Sample book dataset (200 books)
```

## Local Development

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### 3. Test with cURL

```bash
curl -X POST http://localhost:7860/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "feed_size": 10}'
```

## Docker Build and Run

### Build the Image

```bash
docker build -t litvision-recommender .
```

### Run the Container

```bash
docker run -p 7860:7860 litvision-recommender
```

With GPU support:

```bash
docker run -p 7860:7860 --gpus all litvision-recommender
```

## Deployment to Hugging Face Spaces

1. Go to [Hugging Face](https://huggingface.co) and create a new Space.
2. Select **Docker** as the Space SDK.
3. Upload all the files in this directory to the repository.
4. The Space will automatically build the container and start the Uvicorn server on port 7860.

## Troubleshooting

- **Models loading slowly:** The first startup downloads `xlm-roberta-large-xnli` (~2.2 GB) and `paraphrase-multilingual-MiniLM-L12-v2`. Subsequent starts use the cached models.
- **CUDA OOM:** The API automatically clears CUDA cache and returns HTTP 503. Retry the request or reduce `feed_size`.
- **503 on first request:** Models may still be loading. Check `/health` endpoint for status.
