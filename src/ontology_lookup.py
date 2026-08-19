#!/usr/bin/env python3
"""Ontology term lookup using a Nomic Embed v1.5 index over the curated dbo: ontology.

dbo: only -- pure cosine similarity over the pre-built dbo index.

dbp: properties are NOT statically embedded or pre-ranked. They only ever
enter the pipeline via two mechanisms, both of which work against live data:
  1. The deterministic class-safe dbo->dbp namespace swap in query_executor.py
  2. The live agentic probe in validator.py, which queries the SPARQL endpoint
     directly to find which dbp: properties have data for a given subject.

This is a deliberate simplification from an earlier two-index approach
that also embedded ~49k dbp: properties with AI-generated labels. That approach
was dropped because: (a) AI-labelling at that scale is not scientifically
defensible, and (b) a controlled before/after comparison on the
DB26 benchmark showed the two approaches were statistically indistinguishable
once timeout noise and a swap bug were fixed. 

Index files (built by scripts/build_ontology_index.py):
    data/nomic_embeddings_dbo.pt  -- PyTorch tensor [N, 768]
    data/nomic_uris_dbo.json      -- list of N URIs
    data/nomic_labels_dbo.json    -- list of N labels
"""

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

_DATA_DIR      = Path(__file__).parent.parent / "data"
_EMBEDDINGS    = _DATA_DIR / "nomic_embeddings_dbo.pt"
_URIS          = _DATA_DIR / "nomic_uris_dbo.json"
_LABELS        = _DATA_DIR / "nomic_labels_dbo.json"

_nomic_model   = None
_embeddings    = None
_uris          = None
_labels        = None


def _load_index():
    global _nomic_model, _embeddings, _uris, _labels
    if _embeddings is not None:
        return

    if not _EMBEDDINGS.exists():
        raise FileNotFoundError(
            f"dbo index not found at {_EMBEDDINGS}.\n"
            "Run: pipenv run python scripts/build_ontology_index.py"
        )

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    _nomic_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device
    )
    _embeddings  = torch.load(str(_EMBEDDINGS), map_location=device, weights_only=True)
    with open(_URIS) as f:
        _uris = json.load(f)
    with open(_LABELS) as f:
        _labels = json.load(f)


def lookup_term(term, k=15):
    """Return top-k dbo: candidates for a concept, ranked by cosine similarity.

    k=15 (widened from the earlier default of 10) to give the Query Builder a
    deeper dbo: pool to choose from on the first attempt, compensating for the
    removal of the static dbp: index.

    Returns list of dicts: {uri, label, score, confidence_pct, source}
    """
    _load_index()
    query_embedding = _nomic_model.encode(
        f"search_query: {term}", convert_to_tensor=True
    )
    cos_scores  = util.cos_sim(query_embedding, _embeddings)
    top_results = torch.topk(cos_scores, k=min(k * 3, len(_uris)))

    results = []
    for score_t, idx_t in zip(top_results.values[0], top_results.indices[0]):
        raw_cosine     = score_t.item()
        confidence_pct = ((raw_cosine + 1) / 2) * 100
        idx            = idx_t.item()
        results.append({
            "uri":            _uris[idx],
            "label":          _labels[idx],
            "score":          round(raw_cosine, 4),
            "confidence_pct": round(confidence_pct, 1),
            "source":         "dbo",
        })
        if len(results) >= k:
            break
    return results


def lookup_classes(term, k=15):
    """Look up only ontology Classes (for rdf:type constraints)."""
    return [r for r in lookup_term(term, k=k) if r["uri"].split("/")[-1][0].isupper()]


def lookup_properties(term, k=15):
    """Look up only ontology properties (for predicates)."""
    return [r for r in lookup_term(term, k=k) if not r["uri"].split("/")[-1][0].isupper()]
