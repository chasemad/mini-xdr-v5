"""
Vector Memory System for Learned False Positives

This system stores Council corrections as embeddings in Qdrant, enabling:
1. Reuse of past Gemini reasoning (saves API costs)
2. Pattern recognition across similar incidents
3. Continuous learning from Council decisions

When ML makes a prediction with medium confidence, we first check:
"Have we seen this pattern before? What did Gemini say last time?"

This can reduce Gemini API calls by 40%+.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not available - vector memory disabled")

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Sentence transformers not available - using fallback embeddings")

from app.orchestrator.graph import XDRState

logger = logging.getLogger(__name__)


class VectorMemory:
    """
    Vector database manager for storing and retrieving learned incidents.

    Collections:
    - false_positives: Incidents where Gemini overrode ML
    - true_positives: Confirmed threats
    - uncertain: Cases that needed human review
    """

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.embedding_model_name = embedding_model

        # Initialize clients
        self.client: Optional[QdrantClient] = None
        self.embedding_model: Optional[SentenceTransformer] = None

        self._initialized = False

    async def initialize(self):
        """Initialize Qdrant client and embedding model."""
        if self._initialized:
            return

        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant not available - vector memory will not function")
            return

        try:
            # Connect to Qdrant
            self.client = QdrantClient(
                host=self.qdrant_host, port=self.qdrant_port, timeout=10.0
            )

            logger.info(f"Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")

            # Load embedding model
            if EMBEDDINGS_AVAILABLE:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            else:
                logger.warning("Using fallback embeddings (hash-based)")

            # Create collections if they don't exist
            await self._ensure_collections()

            self._initialized = True
            logger.info("Vector memory initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector memory: {e}", exc_info=True)
            self.client = None

    async def _ensure_collections(self):
        """Create Qdrant collections if they don't exist."""
        if not self.client:
            return

        # Dimension of our embeddings
        vector_size = (
            384 if EMBEDDINGS_AVAILABLE else 79
        )  # Fallback to feature vector size

        collections = ["false_positives", "true_positives", "uncertain"]

        for collection_name in collections:
            try:
                # Check if collection exists
                existing = self.client.get_collections().collections
                collection_names = [c.name for c in existing]

                if collection_name not in collection_names:
                    # Create collection
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE,  # Cosine similarity for semantic search
                        ),
                    )
                    logger.info(f"Created collection: {collection_name}")
                else:
                    logger.info(f"Collection already exists: {collection_name}")

            except Exception as e:
                logger.error(f"Error creating collection {collection_name}: {e}")

    def _embed_incident(self, state: XDRState) -> np.ndarray:
        """
        Create embedding for an incident.

        If sentence transformers available: embed textual description
        Otherwise: use normalized feature vector
        """
        if self.embedding_model is not None:
            # Create textual description of incident
            text = self._incident_to_text(state)

            # Generate embedding
            embedding = self.embedding_model.encode(text)

            return embedding

        else:
            # Fallback: Use raw feature vector
            features = np.array(state["raw_features"])

            # Normalize if needed
            if len(features) == 79:
                # Pad to 384 dimensions with zeros (for compatibility)
                # or just use the 79 features directly
                return features

            return np.zeros(79)  # Fallback fallback

    def _incident_to_text(self, state: XDRState) -> str:
        """
        Convert incident state to textual description for embedding.

        This captures the semantic meaning of the incident.
        """
        ml_pred = state["ml_prediction"]
        src_ip = state["src_ip"]
        attack_type = ml_pred.get("class", "Unknown")

        # Build description
        description_parts = [
            f"Source IP: {src_ip}",
            f"Attack type: {attack_type}",
            f"Event count: {state['event_count']}",
        ]

        # Add event samples
        for event in state["events"][:5]:  # First 5 events
            event_type = event.get("event_type", "Unknown")
            description_parts.append(f"Event: {event_type}")

        # Add Gemini's reasoning if available
        if gemini_reasoning := state.get("gemini_reasoning"):
            description_parts.append(
                f"Analysis: {gemini_reasoning[:200]}"
            )  # First 200 chars

        return " | ".join(description_parts)

    async def store_correction(
        self, state: XDRState, collection: str = "false_positives"
    ) -> bool:
        """
        Store a Council correction in vector database.

        Args:
            state: The incident state with Council analysis
            collection: Which collection to store in

        Returns:
            True if stored successfully
        """
        if not self._initialized or not self.client:
            logger.warning("Vector memory not initialized - cannot store correction")
            return False

        try:
            # Generate embedding
            embedding = self._embed_incident(state)

            # Create point
            point_id = state.get("incident_id") or hash(state["flow_id"])

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "src_ip": state["src_ip"],
                    "flow_id": state["flow_id"],
                    "timestamp": state["timestamp"],
                    "ml_prediction": state["ml_prediction"]["class"],
                    "ml_confidence": state["ml_prediction"]["confidence"],
                    "gemini_verdict": state.get("gemini_verdict"),
                    "gemini_confidence": state.get("gemini_confidence"),
                    "gemini_reasoning": state.get("gemini_reasoning", "")[
                        :1000
                    ],  # Truncate
                    "final_verdict": state["final_verdict"],
                    "confidence_score": state["confidence_score"],
                    "council_override": state.get("council_override", False),
                },
            )

            # Upsert to Qdrant
            self.client.upsert(collection_name=collection, points=[point])

            logger.info(
                f"Stored correction in {collection}: "
                f"ML said '{state['ml_prediction']['class']}', "
                f"Council said '{state.get('gemini_verdict')}'"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to store correction: {e}", exc_info=True)
            return False

    async def search_similar(
        self,
        state: XDRState,
        collection: str = "false_positives",
        limit: int = 3,
        score_threshold: float = 0.90,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar past incidents in vector database.

        Args:
            state: Current incident to search for
            collection: Which collection to search
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of similar incidents with scores
        """
        if not self._initialized or not self.client:
            logger.warning("Vector memory not initialized - cannot search")
            return []

        try:
            # Generate query embedding
            query_vector = self._embed_incident(state)

            # Search Qdrant
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold,
            )

            similar_incidents = []

            for result in results:
                similar_incidents.append(
                    {"score": result.score, "payload": result.payload}
                )

                logger.info(
                    f"Found similar incident (score: {result.score:.3f}): "
                    f"{result.payload.get('ml_prediction')} → "
                    f"{result.payload.get('gemini_verdict')}"
                )

            return similar_incidents

        except Exception as e:
            logger.error(f"Failed to search vector memory: {e}", exc_info=True)
            return []


# Global instance
_vector_memory: Optional[VectorMemory] = None


async def get_vector_memory() -> VectorMemory:
    """Get or create global vector memory instance."""
    global _vector_memory

    if _vector_memory is None:
        _vector_memory = VectorMemory()
        await _vector_memory.initialize()

    return _vector_memory


async def search_similar_incidents(state: XDRState) -> List[Dict[str, Any]]:
    """
    Search for similar past incidents (convenience function).

    Returns incidents where Council has already provided analysis.
    """
    memory = await get_vector_memory()
    return await memory.search_similar(state)


async def store_council_correction(state: XDRState):
    """
    Store Council's correction/analysis (convenience function).

    Automatically determines which collection based on verdict.
    """
    memory = await get_vector_memory()

    # Determine collection
    if state.get("council_override"):
        collection = "false_positives"
    elif state["final_verdict"] == "THREAT":
        collection = "true_positives"
    else:
        collection = "uncertain"

    await memory.store_correction(state, collection)

    # Mark as stored in state
    state["embedding_stored"] = True


# Export
__all__ = [
    "VectorMemory",
    "get_vector_memory",
    "search_similar_incidents",
    "store_council_correction",
]
