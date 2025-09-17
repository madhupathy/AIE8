import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio

def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

def euclidean_similarity(a: np.array, b: np.array) -> float:
    # higher is better (negated distance for consistency with cosine)
    return -np.linalg.norm(a - b)

def manhattan_similarity(a: np.array, b: np.array) -> float:
    return -np.sum(np.abs(a - b))

class VectorDatabase:
    def __init__(self,
                 embedding_model: Optional[EmbeddingModel] = None,
                 distance_measure: Callable[[np.array, np.array], float] = cosine_similarity):
        self.vectors: Dict[str, np.array] = defaultdict(np.array)
        self.metadata: Dict[str, Dict[str, Any]] = {}  # NEW: optional per-chunk metadata
        self.embedding_model = embedding_model or EmbeddingModel()
        self.distance_measure = distance_measure

    def insert(self, key: str, vector: np.array, meta: Optional[Dict[str, Any]] = None) -> None:
        self.vectors[key] = vector
        if meta is not None:
            self.metadata[key] = meta

    def _score_all(self, query_vector: np.array) -> List[Tuple[str, float]]:
        return [
            (key, self.distance_measure(query_vector, vec))
            for key, vec in self.vectors.items()
        ]

    def search(self,
               query_vector: np.array,
               k: int,
               return_metadata: bool = False) -> List:
        scores = self._score_all(query_vector)
        topk = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        if return_metadata:
            return [(key, score, self.metadata.get(key, {})) for key, score in topk]
        return topk  # (key, score)

    def search_by_text(self,
                       query_text: str,
                       k: int,
                       return_as_text: bool = False,
                       return_metadata: bool = False) -> List:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(np.array(query_vector), k, return_metadata=return_metadata)
        if return_as_text and not return_metadata:
            return [key for key, _ in results]
        if return_as_text and return_metadata:
            return [(key, meta) for key, _, meta in results]
        return results  # either (key, score) or (key, score, meta)

    def retrieve_from_key(self, key: str) -> Optional[np.array]:
        return self.vectors.get(key, None)

    async def abuild_from_list(self,
                               list_of_text: List[str],
                               metadata_list: Optional[List[Dict[str, Any]]] = None) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for i, (text, emb) in enumerate(zip(list_of_text, embeddings)):
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            self.insert(text, np.array(emb), meta=meta)
        return self
