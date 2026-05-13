"""
LITVISION Recommendation API
==============================
Production FastAPI application for personalized book recommendations.
Deployed on Hugging Face Spaces via Docker SDK.
"""

import time
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from utils import setup_logging, safe_cuda_empty_cache, cleanup_temp_files, validate_positive_int
from recommender import engine, GENRES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

setup_logging()
logger = logging.getLogger("litvision.recommendation")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Interaction(BaseModel):
    """A single user–book interaction event."""
    book_id: int = Field(..., gt=0, description="ID of the book")
    event_type: str = Field(
        default="view",
        description="Interaction type: 'view' or 'like'",
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of the interaction",
    )

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        allowed = {"view", "like"}
        if v not in allowed:
            raise ValueError(f"event_type must be one of {allowed}, got '{v}'")
        return v


class RecommendRequest(BaseModel):
    """Payload for POST /recommend."""
    user_id: int = Field(..., gt=0, description="Unique user identifier")
    interactions: Optional[List[Interaction]] = Field(
        default=None,
        description="List of explicit user–book interactions",
    )
    favorite_genres: Optional[List[str]] = Field(
        default=None,
        description="User's preferred genres",
    )
    viewed_books: Optional[List[int]] = Field(
        default=None,
        description="IDs of books the user has already viewed",
    )
    feed_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of recommendations to return (1-100)",
    )

    @field_validator("favorite_genres")
    @classmethod
    def validate_genres(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            invalid = [g for g in v if g not in GENRES]
            if invalid:
                raise ValueError(
                    f"Invalid genres: {invalid}. Valid genres: {GENRES}"
                )
        return v

    @field_validator("viewed_books")
    @classmethod
    def validate_viewed_books(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        if v is not None:
            for bid in v:
                if bid < 1:
                    raise ValueError(f"viewed_books IDs must be positive, got {bid}")
        return v


class BookResponse(BaseModel):
    """A single recommended book."""
    book_id: int
    title: str
    author: str
    genre: str
    score: Optional[float] = None


class RecommendResponse(BaseModel):
    """Response from POST /recommend."""
    success: bool
    user_id: int
    recommendations: List[BookResponse]
    genre_distribution: Dict[str, int]
    total_recommendations: int
    processing_time_seconds: float


class RootResponse(BaseModel):
    api: str
    status: str
    version: str
    endpoints: List[str]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str
    total_books: int
    faiss_index_size: int


class VersionResponse(BaseModel):
    service: str
    version: str


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models. Shutdown: cleanup caches."""
    logger.info("API starting...")
    try:
        logger.info("Loading recommendation engine...")
        await asyncio.to_thread(engine.load_models)
        logger.info("Recommendation engine loaded successfully")
        logger.info("Server ready")
    except Exception as exc:
        logger.error(f"Failed to load models on startup: {exc}", exc_info=True)
        # Allow the app to start anyway so /health can report the issue
    yield
    logger.info("Shutting down — cleaning up …")
    cleanup_temp_files()
    safe_cuda_empty_cache()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LITVISION Book Recommendation API",
    description=(
        "AI-powered personalized book recommendation service using "
        "zero-shot classification, SentenceTransformer embeddings, "
        "and FAISS similarity search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_model=RootResponse, tags=["Recommendation"])
async def root():
    """Basic API information."""
    return {
        "api": "LITVISION Book Recommendation API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": ["/health", "/recommend", "/version"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Recommendation"])
async def health():
    """Health check — reports model readiness and device info."""
    return {
        "status": "healthy" if engine._loaded else "loading",
        "models_loaded": engine._loaded,
        "device": engine.device,
        "total_books": len(engine.books_df) if engine.books_df is not None else 0,
        "faiss_index_size": engine.faiss_index.ntotal if engine.faiss_index else 0,
    }

@app.get("/version", response_model=VersionResponse, tags=["Recommendation"])
async def version():
    """Return API version information."""
    return {
        "service": "LITVISION Recommendation API",
        "version": "1.0.0"
    }


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendation"])
async def recommend(request: RecommendRequest, background_tasks: BackgroundTasks):
    """
    Generate personalized book recommendations.

    Uses the full notebook pipeline:
    1. Build user interaction DataFrame from request payload
    2. Compute genre interest ratios (event-weighted)
    3. Genre-balanced feed allocation
    4. Cosine-similarity ranking via user embedding vector
    """
    start_time = time.time()

    # Guard: models must be loaded
    if not engine._loaded:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Please retry in a few moments.",
        )

    try:
        # 1. Build interactions DataFrame
        interactions_dicts = None
        if request.interactions:
            interactions_dicts = [
                {
                    "book_id": i.book_id,
                    "event_type": i.event_type,
                    "timestamp": i.timestamp,
                }
                for i in request.interactions
            ]

        interactions_df = await asyncio.to_thread(
            engine.build_interactions_df,
            request.user_id,
            interactions_dicts,
            request.viewed_books,
            request.favorite_genres,
        )

        # 2. Generate recommendations (heavy — offloaded from event loop)
        try:
            feed = await asyncio.wait_for(
                asyncio.to_thread(
                    engine.build_mixed_feed,
                    request.user_id,
                    interactions_df,
                    request.feed_size,
                ),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request processing timed out.")

        # 3. Build response
        recommendations: List[BookResponse] = []
        for _, row in feed.iterrows():
            recommendations.append(
                BookResponse(
                    book_id=int(row["book_id"]),
                    title=str(row["title"]),
                    author=str(row["author"]),
                    genre=str(row["genre"]),
                    score=round(float(row["score"]), 4) if "score" in row.index else None,
                )
            )

        genre_dist = feed["genre"].value_counts().to_dict()
        elapsed = round(time.time() - start_time, 3)

        logger.info(
            f"Recommendation for user {request.user_id}: "
            f"{len(recommendations)} books in {elapsed}s"
        )

        return RecommendResponse(
            success=True,
            user_id=request.user_id,
            recommendations=recommendations,
            genre_distribution=genre_dist,
            total_recommendations=len(recommendations),
            processing_time_seconds=elapsed,
        )

    except torch.cuda.OutOfMemoryError as exc:
        safe_cuda_empty_cache()
        logger.error(f"CUDA OOM during recommendation: {exc}")
        raise HTTPException(
            status_code=503,
            detail="GPU out of memory. CUDA cache cleared — please retry.",
        )
    except ValueError as exc:
        logger.warning(f"Validation error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Recommendation error: {exc}", exc_info=True)
        error_msg = str(exc).lower()
        if "out of memory" in error_msg:
            safe_cuda_empty_cache()
            raise HTTPException(
                status_code=503,
                detail="Out of memory. Cache cleared — please retry.",
            )
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
    finally:
        background_tasks.add_task(cleanup_temp_files)
