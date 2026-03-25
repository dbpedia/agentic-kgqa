#!/usr/bin/env python3
"""Ontology term lookup using precomputed embeddings and gensim KeyedVectors."""

import json
import os
import numpy as np
from gensim.models import KeyedVectors
from openai import OpenAI
import dotenv

dotenv.load_dotenv(override=True)

# Singleton instances
_client = None
_model = None
_ontology_terms = None
_predicate_freqs = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


def _get_model():
    global _model
    if _model is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        vectors_path = os.path.join(data_dir, "ontology-vectors.w2v")
        _model = KeyedVectors.load_word2vec_format(vectors_path)
    return _model


def _get_ontology_terms():
    """Parse ontology terms from the w2v file keys."""
    global _ontology_terms
    if _ontology_terms is None:
        model = _get_model()
        _ontology_terms = {}
        for key in model.key_to_index:
            uri_type, uri = key.split("|", 1)
            _ontology_terms[uri] = {"uri": uri, "type": uri_type, "key": key}
    return _ontology_terms


def _get_predicate_freqs():
    """Load predicate frequency table (lazy, cached)."""
    global _predicate_freqs
    if _predicate_freqs is None:
        freq_path = os.path.join(os.path.dirname(__file__), "..", "data", "predicate_frequencies.json")
        if os.path.exists(freq_path):
            with open(freq_path) as f:
                _predicate_freqs = json.load(f)
        else:
            _predicate_freqs = {}
    return _predicate_freqs


def _embed_texts(texts):
    client = _get_client()
    response = client.embeddings.create(
        input=[f"Term: {text}" for text in texts],
        model="text-embedding-3-small",
    )
    return [x.embedding for x in response.data]


def lookup_term(term, classes=True, properties=True, k=5):
    """Look up an ontology term by natural language description.

    Returns a balanced list of dbo: and dbp: results (when both are available).
    Each result is a dict with keys: uri, type, score, key.
    """
    model = _get_model()
    ontology_terms = _get_ontology_terms()
    term_emb = _embed_texts([term])[0]

    dbo_results = []  # Classes + dbo: properties
    dbp_results = []  # dbp: properties

    for entry, score in model.most_similar(positive=[np.array(term_emb)], topn=200):
        uri_type, uri = entry.split("|", 1)
        if uri_type == "Class" and not classes:
            continue
        if "Property" in uri_type and not properties:
            continue

        item = {"uri": uri, "type": uri_type, "score": round(score, 4), "key": entry}

        if "dbpedia.org/property/" in uri:
            if len(dbp_results) < k:
                dbp_results.append(item)
        else:
            if len(dbo_results) < k:
                dbo_results.append(item)

        if len(dbo_results) >= k and len(dbp_results) >= k:
            break

    # Interleave: return top dbo results first, then top dbp results, up to k total
    # This ensures the LLM sees both namespaces
    results = []
    dbo_take = min(len(dbo_results), max(1, k // 2))
    results.extend(dbo_results[:dbo_take])
    remaining = k - len(results)
    results.extend(dbp_results[:remaining])

    # Sort by score descending for clean presentation
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:k]

    # Annotate with triple counts from the frequency table
    freqs = _get_predicate_freqs()
    for r in results:
        r["triples"] = freqs.get(r["uri"], 0)

    return results


def lookup_classes(term, k=5):
    """Lookup only ontology classes."""
    return lookup_term(term, classes=True, properties=False, k=k)


def lookup_properties(term, k=5):
    """Lookup only ontology properties."""
    return lookup_term(term, classes=False, properties=True, k=k)
