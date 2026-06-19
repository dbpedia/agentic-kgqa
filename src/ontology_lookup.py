#!/usr/bin/env python
"""Ontology term lookup using Nomic Embed v1.5 and PyTorch cosine similarity.

Replaces the previous OpenAI + Gensim word2vec approach with a fully local,
zero-API-cost semantic search over the precomputed Nomic embedding index.

The index is built once by running:
    pipenv run python scripts/build_ontology_index.py

Index files (these will be created by running the build script):
    data/nomic_embeddings.pt   — PyTorch tensor [N, 768]
    data/nomic_uris.json       — list of N URIs
    data/nomic_labels.json     — list of N labels
"""

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

# ─── Index paths ──────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data"
_EMBEDDINGS_PATH = _DATA_DIR / "nomic_embeddings.pt"
_URIS_PATH       = _DATA_DIR / "nomic_uris.json"
_LABELS_PATH     = _DATA_DIR / "nomic_labels.json"

# ─── Singletons ───────────────────────────────────────────────────────────────

_nomic_model = None
_embeddings  = None
_uris        = None
_labels      = None


def _load_index():
    """Lazy-load the Nomic index into memory (runs once per process)."""
    global _nomic_model, _embeddings, _uris, _labels
    if _embeddings is not None:
        return

    if not _EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Nomic index not found at {_EMBEDDINGS_PATH}.\n"
            "Run: pipenv run python scripts/build_ontology_index.py"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _nomic_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device
    )
    _embeddings = torch.load(str(_EMBEDDINGS_PATH), map_location=device)
    with open(_URIS_PATH) as f:
        _uris = json.load(f)
    with open(_LABELS_PATH) as f:
        _labels = json.load(f)


# ─── Semantic lookup ───────────────────────────────────────────────────────────────

def lookup_term(term, k=10, classes=True, properties=True):
    """Look up an ontology term by natural language description.

    Returns top-k semantically similar entries from the Nomic index.

    Args:
        term:       Natural language concept (e.g. "director", "birthplace")
        k:          Number of results to return (default 10)
        classes:    Include OWL Classes in results (default True)
        properties: Include OWL Properties in results (default True)

    Returns:
        List of dicts with keys: uri, label, score, confidence_pct, source
        - uri:            Full DBpedia URI
        - label:          Human-readable label
        - score:          Raw cosine similarity in [-1, 1]
        - confidence_pct: Normalised confidence ((cosine+1)/2)*100
        - source:         Always "nomic"
    """
    _load_index()

    query_text = f"search_query: {term}"
    query_embedding = _nomic_model.encode(query_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, _embeddings)
    top_results = torch.topk(cos_scores, k=min(k * 3, len(_uris)))

    results = []
    for score_t, idx_t in zip(top_results.values[0], top_results.indices[0]):
        uri = _uris[idx_t.item()]
        label = _labels[idx_t.item()]

        # Filter by type if requested
        is_class = label and label[0].isupper() and "ontology" in uri
        if is_class and not classes:
            continue
        if not is_class and not properties:
            continue

        raw_cosine = score_t.item()
        confidence_pct = ((raw_cosine + 1) / 2) * 100

        results.append({
            "uri": uri,
            "label": label,
            "score": round(raw_cosine, 4),
            "confidence_pct": round(confidence_pct, 1),
            "source": "nomic",
        })

        if len(results) >= k:
            break

    return results


def lookup_classes(term, k=10):
    """Look up only ontology Classes (for rdf:type constraints)."""
    return lookup_term(term, k=k, classes=True, properties=False)


def lookup_properties(term, k=10):
    """Look up only ontology properties (for predicates)."""
    return lookup_term(term, k=k, classes=False, properties=True)
