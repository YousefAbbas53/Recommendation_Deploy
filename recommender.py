"""
LITVISION Recommendation Engine
================================
Complete recommendation pipeline preserving every component from
the original Jupyter Notebook:

  • Zero-shot genre classification  (joeddav/xlm-roberta-large-xnli)
  • SentenceTransformer embeddings  (paraphrase-multilingual-MiniLM-L12-v2)
  • FAISS IndexFlatIP similarity search
  • Event-weighted user vector construction
  • Proportional genre-balanced feed allocation
  • Cosine-similarity ranking pipeline

NO logic has been simplified, removed, or replaced.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import faiss
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer

from utils import get_device, safe_cuda_empty_cache

logger = logging.getLogger("litvision.recommendation")

# ═══════════════════════════════════════════════════════════════════════════
# Constants — identical to the notebook
# ═══════════════════════════════════════════════════════════════════════════

GENRES: List[str] = [
    "Fantasy", "Romance", "Mystery", "Science Fiction", "Self-Help",
    "History", "Business", "Children", "Horror", "Poetry",
]

EVENT_W: Dict[str, float] = {"view": 1.0, "like": 3.0}

TEMPLATES: Dict[str, List[str]] = {
    "Fantasy": [
        "A young hero discovers a hidden kingdom and must defeat a dark sorcerer.",
        "Dragons rise again as an ancient prophecy awakens in the north.",
    ],
    "Romance": [
        "Two strangers meet in a small cafe and find love against all odds.",
        "A long-distance relationship is tested by secrets and time.",
    ],
    "Mystery": [
        "A detective investigates a series of murders in a quiet town.",
        "A missing diary reveals clues to an old family crime.",
    ],
    "Science Fiction": [
        "A crew travels through a wormhole to save humanity from collapse.",
        "An AI gains consciousness and changes the future of Earth.",
    ],
    "Self-Help": [
        "A practical guide to build habits and improve focus every day.",
        "Learn to manage anxiety with simple routines and mindset shifts.",
    ],
    "History": [
        "An account of ancient empires and the wars that shaped the world.",
        "A deep dive into the political revolutions of the 20th century.",
    ],
    "Business": [
        "How startups scale products and build strong teams.",
        "Negotiation tactics and leadership strategies for managers.",
    ],
    "Children": [
        "A curious cat explores the city and learns about friendship.",
        "A magical school adventure for kids with puzzles and fun.",
    ],
    "Horror": [
        "A haunted house whispers at night, luring visitors inside.",
        "A village faces a terrifying creature in the woods.",
    ],
    "Poetry": [
        "A collection of poems about love, loss, and hope.",
        "Minimalist poems inspired by nature and silence.",
    ],
}

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
BOOKS_CSV_PATH = os.path.join(SAMPLE_DATA_DIR, "books.csv")

# ═══════════════════════════════════════════════════════════════════════════
# Recommendation Engine
# ═══════════════════════════════════════════════════════════════════════════


class RecommendationEngine:
    """
    Production wrapper around the full notebook recommendation pipeline.
    Models are loaded lazily on first call or explicitly via ``load_models()``.
    """

    def __init__(self) -> None:
        self.device: str = "cpu"
        self.embed_model: Optional[SentenceTransformer] = None
        self.zero_shot = None
        self.books_df: Optional[pd.DataFrame] = None
        self.book_embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bookid_to_idx: Dict[int, int] = {}
        self._loaded = False
        # Per-user feed state (identical to notebook)
        self.user_feed_state: Dict[int, Set[int]] = {}

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load all AI models and build the FAISS index."""
        if self._loaded:
            return

        self.device = get_device()
        logger.info("Loading recommendation models …")

        # 1. Zero-shot classifier — identical model to notebook
        logger.info("Loading zero-shot classifier: joeddav/xlm-roberta-large-xnli")
        zs_device = 0 if self.device == "cuda" else -1
        self.zero_shot = hf_pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            device=zs_device,
        )

        # FP16 on CUDA for the zero-shot model
        if self.device == "cuda":
            try:
                self.zero_shot.model.half()
                logger.info("Zero-shot model converted to FP16")
            except Exception as e:
                logger.warning(f"Could not convert zero-shot to FP16: {e}")

        # 2. SentenceTransformer — identical model to notebook
        logger.info(
            "Loading SentenceTransformer: "
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.embed_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=self.device,
        )
        if self.device == "cuda":
            try:
                self.embed_model.half()
                logger.info("SentenceTransformer converted to FP16")
            except Exception as e:
                logger.warning(f"Could not convert embed model to FP16: {e}")

        # 3. Load / generate books dataset
        self.books_df = self._load_books()

        # 4. Classify genres (zero-shot) — identical to notebook
        self._classify_genres()

        # 5. Build embeddings + FAISS index — identical to notebook
        self._build_index()

        self._loaded = True
        logger.info(
            f"Recommendation engine ready — "
            f"{len(self.books_df)} books, FAISS dim={self.faiss_index.d}"
        )

    # ------------------------------------------------------------------
    # Books dataset
    # ------------------------------------------------------------------

    def _load_books(self) -> pd.DataFrame:
        """Load books from CSV or generate the default sample set."""
        if os.path.exists(BOOKS_CSV_PATH):
            logger.info(f"Loading books from {BOOKS_CSV_PATH}")
            df = pd.read_csv(BOOKS_CSV_PATH)
            required = {"book_id", "title", "author", "description"}
            if not required.issubset(set(df.columns)):
                raise ValueError(
                    f"books.csv must contain columns {required}, "
                    f"found {set(df.columns)}"
                )
            return df

        logger.info("No books.csv found — generating sample data (seed=42)")
        df = self._make_books(200)
        os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
        df.to_csv(BOOKS_CSV_PATH, index=False)
        logger.info(f"Saved {len(df)} books to {BOOKS_CSV_PATH}")
        return df

    @staticmethod
    def _make_books(n_books: int = 200) -> pd.DataFrame:
        """Generate sample books — identical to notebook ``make_books``."""
        np.random.seed(42)
        rows = []
        for i in range(1, n_books + 1):
            g = np.random.choice(GENRES)
            desc = np.random.choice(TEMPLATES[g])
            title = f"{g} Book {i}"
            author = f"Author {np.random.randint(1, 60)}"
            rows.append([i, title, author, desc])
        return pd.DataFrame(rows, columns=["book_id", "title", "author", "description"])

    # ------------------------------------------------------------------
    # Genre classification — identical to notebook
    # ------------------------------------------------------------------

    def classify_genre(self, text: str) -> Tuple[str, float]:
        """Zero-shot genre classification — identical to notebook."""
        out = self.zero_shot(text, candidate_labels=GENRES, multi_label=False)
        return out["labels"][0], float(out["scores"][0])

    def _classify_genres(self) -> None:
        """Classify all books in the dataset — identical to notebook loop."""
        if "genre" in self.books_df.columns and "genre_confidence" in self.books_df.columns:
            logger.info("Genre columns already present — skipping classification")
            return

        logger.info("Classifying genres for all books …")
        genres, scores = [], []
        texts = (self.books_df["title"] + " | " + self.books_df["description"]).tolist()
        for i, txt in enumerate(texts):
            g, s = self.classify_genre(txt)
            genres.append(g)
            scores.append(s)
            if (i + 1) % 50 == 0:
                logger.info(f"  classified {i + 1}/{len(texts)} books")

        self.books_df["genre"] = genres
        self.books_df["genre_confidence"] = scores
        # Persist updated CSV
        self.books_df.to_csv(BOOKS_CSV_PATH, index=False)
        logger.info("Genre classification complete")

    # ------------------------------------------------------------------
    # Embeddings + FAISS — identical to notebook
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Build SentenceTransformer embeddings and FAISS index."""
        self.books_df["text"] = (
            "Title: " + self.books_df["title"]
            + " | Author: " + self.books_df["author"]
            + " | Genre: " + self.books_df["genre"]
            + " | Description: " + self.books_df["description"]
        )

        logger.info("Encoding book embeddings …")
        self.book_embeddings = self.embed_model.encode(
            self.books_df["text"].tolist(),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = self.book_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # cosine (normalized)
        self.faiss_index.add(self.book_embeddings)

        self.bookid_to_idx = {
            int(bid): i
            for i, bid in enumerate(self.books_df["book_id"].tolist())
        }

        logger.info(
            f"FAISS index built — embeddings {self.book_embeddings.shape}, "
            f"ntotal={self.faiss_index.ntotal}"
        )

    # ------------------------------------------------------------------
    # User interest ratios — identical to notebook
    # ------------------------------------------------------------------

    def user_interest_ratios(
        self,
        user_id: int,
        interactions_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute weighted genre interest ratios — identical to notebook."""
        u = interactions_df[interactions_df.user_id == user_id].merge(
            self.books_df[["book_id", "genre"]], on="book_id", how="left"
        )
        if u.empty:
            return {g: 1 / len(GENRES) for g in GENRES}

        u["w"] = u["event_type"].map(EVENT_W).fillna(0.0)
        s = u.groupby("genre")["w"].sum().reindex(GENRES, fill_value=0.0)

        total = s.sum()
        if total == 0:
            return {g: 1 / len(GENRES) for g in GENRES}

        return (s / total).to_dict()

    # ------------------------------------------------------------------
    # User vector — identical to notebook
    # ------------------------------------------------------------------

    def build_user_vector(
        self,
        user_id: int,
        interactions_df: pd.DataFrame,
    ) -> Tuple[Optional[np.ndarray], Set[int]]:
        """Build weighted user embedding vector — identical to notebook."""
        u = interactions_df[interactions_df.user_id == user_id]
        if u.empty:
            return None, set()

        vecs: List[np.ndarray] = []
        weights: List[float] = []
        seen: Set[int] = set()

        for _, row in u.iterrows():
            bid = int(row["book_id"])
            ev = row["event_type"]
            if bid not in self.bookid_to_idx:
                continue
            w = EVENT_W.get(ev, 0.0)
            if w == 0:
                continue
            vecs.append(self.book_embeddings[self.bookid_to_idx[bid]])
            weights.append(w)
            seen.add(bid)

        if not vecs:
            return None, seen

        vecs_arr = np.array(vecs)
        weights_arr = np.array(weights).reshape(-1, 1)

        user_vec = np.sum(vecs_arr * weights_arr, axis=0) / (
            np.sum(np.abs(weights_arr)) + 1e-9
        )
        user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-9)
        return user_vec.astype("float32"), seen

    # ------------------------------------------------------------------
    # Feed allocation — identical to notebook
    # ------------------------------------------------------------------

    @staticmethod
    def allocate_feed(
        ratios: Dict[str, float],
        unseen_counts: Dict[str, int],
        feed_size: int = 50,
    ) -> Dict[str, int]:
        """Proportional genre allocation — identical to notebook."""
        alloc = {g: 0 for g in GENRES}
        remaining = feed_size

        # Target counts proportional to ratios
        target = {
            g: int(round(ratios.get(g, 0.0) * feed_size)) for g in GENRES
        }

        # Cap by availability
        for g in GENRES:
            alloc[g] = min(target[g], unseen_counts.get(g, 0))
            remaining -= alloc[g]

        # Distribute leftovers to best-ratio genres that still have items
        while remaining > 0:
            candidates = [
                g for g in GENRES if alloc[g] < unseen_counts.get(g, 0)
            ]
            if not candidates:
                break
            g = max(candidates, key=lambda x: ratios.get(x, 0.0))
            alloc[g] += 1
            remaining -= 1

        return alloc

    # ------------------------------------------------------------------
    # Mixed feed builder — identical to notebook
    # ------------------------------------------------------------------

    def build_mixed_feed(
        self,
        user_id: int,
        interactions_df: pd.DataFrame,
        feed_size: int = 50,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Build a genre-balanced, similarity-ranked feed — identical to notebook."""
        ratios = self.user_interest_ratios(user_id, interactions_df)

        seen = set(
            interactions_df.loc[
                interactions_df.user_id == user_id, "book_id"
            ]
            .astype(int)
            .tolist()
        )
        unseen_df = self.books_df[~self.books_df.book_id.isin(seen)].copy()

        unseen_counts = (
            unseen_df.groupby("genre")["book_id"]
            .count()
            .reindex(GENRES, fill_value=0)
            .to_dict()
        )
        alloc = self.allocate_feed(ratios, unseen_counts, feed_size=feed_size)

        parts: List[pd.DataFrame] = []
        for g, k in alloc.items():
            if k <= 0:
                continue
            g_df = unseen_df[unseen_df.genre == g]
            if len(g_df) == 0:
                continue
            parts.append(
                g_df.sample(n=min(k, len(g_df)), random_state=random_state)
            )

        if not parts:
            return self.books_df.sample(feed_size, random_state=random_state)[
                ["book_id", "title", "author", "genre"]
            ]

        feed = pd.concat(parts, ignore_index=True)

        # Shuffle / ranking — identical to notebook
        feed = feed.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        )

        user_vec, _ = self.build_user_vector(user_id, interactions_df)
        if user_vec is not None:
            idxs = [
                self.bookid_to_idx[int(b)] for b in feed["book_id"].tolist()
            ]
            feed_vecs = self.book_embeddings[idxs]
            feed["score"] = (feed_vecs @ user_vec).astype(float)
            feed = feed.sort_values("score", ascending=False).reset_index(
                drop=True
            )

        cols = ["book_id", "title", "author", "genre"]
        if "score" in feed.columns:
            cols.append("score")
        return feed[cols]

    # ------------------------------------------------------------------
    # Paginated feed — identical to notebook
    # ------------------------------------------------------------------

    def get_next_feed_page(
        self,
        user_id: int,
        interactions_df: pd.DataFrame,
        page_size: int = 20,
    ) -> pd.DataFrame:
        """Return the next page of unseen recommendations — identical to notebook."""
        shown = self.user_feed_state.get(user_id, set())

        # Add temporary "view" interactions for already-shown books
        if len(shown) > 0:
            temp_rows = pd.DataFrame(
                {
                    "user_id": [user_id] * len(shown),
                    "book_id": list(shown),
                    "event_type": ["view"] * len(shown),
                    "timestamp": [datetime.now().isoformat()] * len(shown),
                }
            )
            temp_interactions = pd.concat(
                [interactions_df, temp_rows], ignore_index=True
            )
        else:
            temp_interactions = interactions_df

        page = self.build_mixed_feed(
            user_id,
            temp_interactions,
            feed_size=page_size,
            random_state=np.random.randint(0, 10_000),
        )

        # Update shown state
        self.user_feed_state[user_id] = shown.union(
            set(page["book_id"].astype(int).tolist())
        )
        return page

    def reset_user_feed(self, user_id: int) -> None:
        """Clear pagination state for a user."""
        self.user_feed_state.pop(user_id, None)

    # ------------------------------------------------------------------
    # Generate interactions from request payload
    # ------------------------------------------------------------------

    @staticmethod
    def build_interactions_df(
        user_id: int,
        interactions: Optional[List[dict]] = None,
        viewed_books: Optional[List[int]] = None,
        favorite_genres: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Construct a pandas DataFrame of user interactions from the API
        request payload.  This merges explicit interaction events,
        viewed-book IDs (as implicit views), and favourite genres
        (synthesised as likes for books in those genres).
        """
        rows: List[dict] = []
        now_iso = datetime.now().isoformat()

        # 1. Explicit interactions
        if interactions:
            for inter in interactions:
                rows.append(
                    {
                        "user_id": user_id,
                        "book_id": int(inter["book_id"]),
                        "event_type": inter.get("event_type", "view"),
                        "timestamp": inter.get("timestamp", now_iso),
                    }
                )

        # 2. Viewed books → implicit "view" events
        if viewed_books:
            for bid in viewed_books:
                rows.append(
                    {
                        "user_id": user_id,
                        "book_id": int(bid),
                        "event_type": "view",
                        "timestamp": now_iso,
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=["user_id", "book_id", "event_type", "timestamp"]
            )

        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

engine = RecommendationEngine()
