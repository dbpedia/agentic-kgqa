#!/usr/bin/env python3
"""Vocab and instance lookup for the corporate (CK25) dataset using precomputed embeddings."""

import os
import numpy as np
from gensim.models import KeyedVectors
from openai import OpenAI
import dotenv

dotenv.load_dotenv(override=True)

_client = None
_vocab_model = None
_instance_model = None

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corporate")


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


def _load_model(path):
    return KeyedVectors.load_word2vec_format(path)


def _get_vocab_model():
    global _vocab_model
    if _vocab_model is None:
        _vocab_model = _load_model(os.path.join(_DATA_DIR, "vocab-vectors.w2v"))
    return _vocab_model


def _get_instance_model():
    global _instance_model
    if _instance_model is None:
        _instance_model = _load_model(os.path.join(_DATA_DIR, "instance-vectors.w2v"))
    return _instance_model


def _embed(text, retries=2):
    import time
    client = _get_client()
    for attempt in range(retries + 1):
        try:
            response = client.embeddings.create(
                input=[f"Term: {text}"],
                model="text-embedding-3-small",
            )
            if not response.data:
                raise ValueError("No embedding data received")
            return response.data[0].embedding
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
                continue
            raise


def lookup_vocab_term(term, k=5):
    """Look up vocabulary terms (classes + properties) by natural language description.

    Returns list of dicts: {uri, type, score}.
    """
    model = _get_vocab_model()
    emb = _embed(term)

    results = []
    for key, score in model.most_similar(positive=[np.array(emb)], topn=k * 3):
        uri_type, uri = key.split("|", 1)
        results.append({"uri": uri, "type": uri_type, "score": round(score, 4)})
        if len(results) >= k:
            break

    return results


def lookup_instance(entity, k=3):
    """Look up instances by label similarity.

    Returns list of dicts: {uri, type, score, label}.
    """
    model = _get_instance_model()
    emb = _embed(entity)

    results = []
    for key, score in model.most_similar(positive=[np.array(emb)], topn=k):
        uri_type, uri = key.split("|", 1)
        results.append({"uri": uri, "type": uri_type, "score": round(score, 4)})
    return results
