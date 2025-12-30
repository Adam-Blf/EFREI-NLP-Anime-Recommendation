"""
LLM Engine pour le système de recommandation d'anime.

Utilise des embeddings sémantiques (sentence-transformers) et un LLM local
(Ollama) pour améliorer les recommandations et générer des explications.
"""

import os
import json
import requests
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "mistral")


class LLMEngine:
    """Moteur LLM pour recommandations intelligentes."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.embedder = None
        self._init_embeddings()

    def _init_embeddings(self):
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-mpnet-base-v2')
            except Exception as e:
                print(f"Embeddings error: {e}")

    def get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        if self.embedder is None:
            return None
        return self.embedder.encode(texts, show_progress_bar=True)

    def semantic_search(self, query: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = 10) -> pd.DataFrame:
        """Recherche sémantique - requêtes naturelles comme 'anime sombre avec robots'."""
        query_embedding = self.embedder.encode([query])[0]
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        return results

    def _call_ollama(self, prompt: str, system: str = None) -> Optional[str]:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 500}
            }
            if system:
                payload["system"] = system
            response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "")
        except:
            return None

    def explain_recommendation(self, anime_title: str, anime_synopsis: str, favorites: List[str]) -> str:
        """Génère une explication de pourquoi cet anime est recommandé."""
        system = "Tu es un expert anime. Explique en 2 phrases pourquoi cet anime conviendrait. Français."
        prompt = f"Favoris: {', '.join(favorites[:3])}\nRecommandé: {anime_title}\nSynopsis: {anime_synopsis[:300]}"
        return self._call_ollama(prompt, system) or "Basé sur la similarité des thèmes."

    def generate_pitch(self, anime_data: Dict) -> str:
        """Génère un pitch accrocheur pour un anime."""
        system = "Génère une accroche de 2 phrases pour donner envie de regarder. Français."
        prompt = f"Anime: {anime_data.get('Title')}\nGenre: {anime_data.get('Genre')}\nSynopsis: {anime_data.get('Synopsis', '')[:300]}"
        return self._call_ollama(prompt, system) or anime_data.get('Synopsis', '')[:150]


def create_embeddings_cache(df: pd.DataFrame, cache_path: str = "embeddings_cache.npy"):
    """Crée et sauvegarde les embeddings pour tout le dataset."""
    engine = LLMEngine()
    if engine.embedder is None:
        print("Sentence-transformers non disponible")
        return None
    
    synopses = df['Synopsis'].fillna('').tolist()
    embeddings = engine.get_embeddings(synopses)
    np.save(cache_path, embeddings)
    print(f"Embeddings sauvegardés: {cache_path}")
    return embeddings


def load_embeddings_cache(cache_path: str = "embeddings_cache.npy") -> Optional[np.ndarray]:
    """Charge les embeddings depuis le cache."""
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None
