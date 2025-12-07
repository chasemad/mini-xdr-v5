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
    """
    Get or create global vector memory instance.

    This function is safe to call even when Qdrant is unavailable.
    The VectorMemory instance will track its initialization state
    and gracefully handle unavailability.
    """
    global _vector_memory

    if _vector_memory is None:
        _vector_memory = VectorMemory()
        try:
            await _vector_memory.initialize()
        except Exception as e:
            logger.warning(
                f"Vector memory initialization failed (Qdrant may be unavailable): {e}"
            )
            # Instance is created but not fully initialized - methods will check _initialized flag

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


async def check_similar_false_positives(
    features: np.ndarray,
    ml_prediction: str,
    threshold: float = 0.90,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Check if current event pattern is similar to past false positives.

    This is used to prevent repeated false positives by learning from
    Council corrections and user feedback.

    Args:
        features: 79-dimensional feature vector
        ml_prediction: Current ML prediction class name
        threshold: Cosine similarity threshold (0.90 = very similar)

    Returns:
        Tuple of (is_similar_to_fp: bool, similar_incident_details: dict or None)

    Note:
        Gracefully returns (False, None) when vector memory is unavailable.
        This ensures detection continues even without Qdrant.
    """
    try:
        memory = await get_vector_memory()

        if not memory._initialized or not memory.client:
            logger.debug("Vector memory not initialized, skipping FP check")
            return False, None
    except Exception as e:
        logger.debug(f"Vector memory unavailable: {e}")
        return False, None

    try:
        # Search false_positives collection using feature vector
        if EMBEDDINGS_AVAILABLE and memory.embedding_model is not None:
            # Create text representation for semantic search
            text = f"ML prediction: {ml_prediction}"
            query_vector = memory.embedding_model.encode(text)
        else:
            # Use raw feature vector
            query_vector = features.flatten()[:79]  # Ensure 79 dimensions

            # Pad to expected vector size if needed
            if len(query_vector) < 79:
                query_vector = np.pad(query_vector, (0, 79 - len(query_vector)))

        # Search false_positives collection
        results = memory.client.search(
            collection_name="false_positives",
            query_vector=query_vector.tolist(),
            limit=1,
            score_threshold=threshold,
        )

        if results and len(results) > 0:
            top_result = results[0]
            payload = top_result.payload

            logger.info(
                f"Found similar false positive (score: {top_result.score:.3f}): "
                f"ML predicted '{payload.get('ml_prediction')}' but was marked FP"
            )

            return True, {
                "similarity_score": top_result.score,
                "original_prediction": payload.get("ml_prediction"),
                "gemini_verdict": payload.get("gemini_verdict"),
                "reasoning": payload.get("gemini_reasoning", "")[:200],
                "timestamp": payload.get("timestamp"),
            }

        return False, None

    except Exception as e:
        logger.error(f"FP similarity check failed: {e}", exc_info=True)
        return False, None


async def store_false_positive(
    features: np.ndarray,
    ml_prediction: str,
    src_ip: str,
    reasoning: str = "",
    incident_id: int = None,
) -> bool:
    """
    Store a confirmed false positive for future similarity detection.

    Args:
        features: 79-dimensional feature vector
        ml_prediction: What the ML model predicted
        src_ip: Source IP that triggered the FP
        reasoning: Why this was marked as FP
        incident_id: Optional incident ID for reference

    Returns:
        True if stored successfully, False otherwise

    Note:
        Gracefully returns False when vector memory is unavailable.
        This ensures the system continues operating even without Qdrant.
    """
    try:
        memory = await get_vector_memory()

        if not memory._initialized or not memory.client:
            logger.debug("Vector memory not initialized - skipping FP storage")
            return False
    except Exception as e:
        logger.debug(f"Vector memory unavailable for storage: {e}")
        return False

    try:
        # Create embedding
        if EMBEDDINGS_AVAILABLE and memory.embedding_model is not None:
            text = f"False positive: {ml_prediction} from {src_ip}. {reasoning}"
            embedding = memory.embedding_model.encode(text)
        else:
            # Use feature vector directly
            embedding = features.flatten()[:79]
            if len(embedding) < 79:
                embedding = np.pad(embedding, (0, 79 - len(embedding)))

        # Create point
        from datetime import datetime

        point_id = incident_id or hash(
            f"{src_ip}_{ml_prediction}_{datetime.now().timestamp()}"
        )

        point = PointStruct(
            id=abs(point_id) % (2**63),  # Ensure positive int64
            vector=embedding.tolist(),
            payload={
                "src_ip": src_ip,
                "ml_prediction": ml_prediction,
                "reasoning": reasoning[:1000],
                "timestamp": datetime.now().isoformat(),
                "gemini_verdict": "FALSE_POSITIVE",
                "incident_id": incident_id,
            },
        )

        memory.client.upsert(collection_name="false_positives", points=[point])

        logger.info(f"Stored false positive: {ml_prediction} from {src_ip}")
        return True

    except Exception as e:
        logger.error(f"Failed to store false positive: {e}", exc_info=True)
        return False


# Export
__all__ = [
    "VectorMemory",
    "get_vector_memory",
    "search_similar_incidents",
    "store_council_correction",
    "check_similar_false_positives",
    "store_false_positive",
]
