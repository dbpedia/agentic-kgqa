#!/usr/bin/env python3
"""Build and save the Nomic Embed v1.5 ontology index.

This script builds a semantic search index over DBpedia ontology properties and classes.
It must be run once before using the Ontology Explorer node in the pipeline.

Sources:
  1. dbo: properties and Classes — parsed from data/dbpedia_2015-10.nt using RDFLib.
     These have formal English labels and comments from the DBpedia ontology schema.
     Stored with label + comment as document text for richer semantic matching.

  2. dbp: properties — extracted from data/ontology-vectors.w2v.
     Only dbp: properties that have NO exact dbo: equivalent are included.
     Properties with exact dbo: equivalents are skipped because the Query Executor
     handles those via a deterministic dbo->dbp namespace swap at runtime.
     dbp: properties have no formal labels so they are embedded using camelCase-split name only.

Output (saved to data/):
  - nomic_embeddings.pt   — PyTorch tensor of shape [N, 768]
  - nomic_uris.json       — list of N URIs in same order as embeddings
  - nomic_labels.json     — list of N human-readable labels in same order

Run once:
    cd agentic-kgqa
    pipenv run python scripts/build_ontology_index.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import rdflib
import torch
from rdflib.namespace import OWL, RDF, RDFS
from sentence_transformers import SentenceTransformer

# ─── Paths (all relative to agentic-kgqa/) ────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent           # agentic-kgqa/
DATA_DIR = BASE_DIR / "data"

NT_FILE   = DATA_DIR / "dbpedia_2015-10.nt"       # DBpedia ontology schema
W2V_FILE  = DATA_DIR / "ontology-vectors.w2v"     # word2vec index (for extracting dbp: URIs. Refer to Readme.md.)

EMBEDDINGS_PATH = DATA_DIR / "nomic_embeddings.pt"
URIS_PATH       = DATA_DIR / "nomic_uris.json"
LABELS_PATH     = DATA_DIR / "nomic_labels.json"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def split_camel(s):
    """Split camelCase into readable words. e.g. 'birthPlace' -> 'birth Place'"""
    return re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", s)).strip()


# ─── Source 1: dbo: entries from NT file ──────────────────────────────────────

def extract_dbo_from_nt(nt_file):
    """Parse dbpedia_2015-10.nt and extract dbo: Classes and properties
    with their English labels and comments.

    Returns: list of (uri, label, doc_text) tuples.
    """
    print(f"\nSource 1: Parsing NT file: {nt_file}")
    g = rdflib.Graph()
    g.parse(str(nt_file), format="nt")
    print(f"  Loaded {len(g)} triples")

    prop_data = defaultdict(lambda: {"labels": set(), "comments": set()})
    property_types = [OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property, OWL.Class]

    for ptype in property_types:
        for prop in g.subjects(RDF.type, ptype):
            if isinstance(prop, rdflib.URIRef):
                for label in g.objects(prop, RDFS.label):
                    if getattr(label, "language", None) in ["en", None]:
                        prop_data[prop]["labels"].add(str(label))
                for comment in g.objects(prop, RDFS.comment):
                    if getattr(comment, "language", None) in ["en", None]:
                        prop_data[prop]["comments"].add(str(comment))

    entries = []
    for prop_uri, data in prop_data.items():
        raw_name = str(prop_uri).split("/")[-1]
        label = list(data["labels"])[0] if data["labels"] else split_camel(raw_name)
        comment = list(data["comments"])[0] if data["comments"] else ""
        doc_text = f"search_document: {label}. {comment}".strip(". ")
        entries.append((str(prop_uri), label, doc_text))

    print(f"  Extracted {len(entries)} dbo: entries")
    return entries


# ─── Source 2: unique dbp: entries from w2v file ──────────────────────────────

def extract_unique_dbp_from_w2v(w2v_file, dbo_entries):
    """Extract dbp: properties that have NO exact dbo: equivalent.

    Strategy: if dbp:director exists AND dbo:director also exists, skip dbp:director.
    The Query Executor handles dbo->dbp swap by string replacement at runtime, so
    including both in the index would confuse the LLM with redundant choices.
    Only include dbp: properties that are truly unique (no dbo: counterpart).

    Returns: list of (uri, label, doc_text) tuples.
    """
    print(f"\nSource 2: Extracting unique dbp: URIs from w2v file: {w2v_file}")
    try:
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format(str(w2v_file))
    except Exception as e:
        print(f"  WARNING: Could not load w2v file: {e}")
        print("  Skipping dbp: entries — index will be dbo: only")
        return []

    # Build set of dbo: property names (lowercased) for deduplication
    dbo_names = set()
    for uri, _, _ in dbo_entries:
        name = uri.split("/")[-1].lower()
        dbo_names.add(name)

    entries = []
    skipped = 0
    for key in model.key_to_index:
        if "dbpedia.org/property/" not in key:
            continue
        uri = key.split("|", 1)[1] if "|" in key else key
        raw_name = uri.split("/")[-1]
        name_lower = raw_name.lower()

        # Skip if exact dbo: equivalent exists — runtime swap handles these
        if name_lower in dbo_names:
            skipped += 1
            continue

        label = split_camel(raw_name)
        doc_text = f"search_document: {label}"
        entries.append((uri, label, doc_text))

    print(f"  Skipped {skipped} dbp: entries that have dbo: equivalents")
    print(f"  Kept {len(entries)} unique dbp: entries (no dbo: counterpart)")
    return entries


# ─── Build Index ──────────────────────────────────────────────────────────────

def build_index():
    print("=" * 60)
    print("Building Nomic Embed v1.5 ontology index")
    print("=" * 60)

    # Check required files exist
    missing = [f for f in [NT_FILE, W2V_FILE] if not f.exists()]
    if missing:
        print("\nERROR: Required data files not found:")
        for f in missing:
            print(f"  {f}")
        print("\nRequired files:")
        print("  data/dbpedia_2015-10.nt     — DBpedia ontology NT file")
        print("  data/ontology-vectors.w2v   ")
        print("\nPlease refer to Readme.md.")
        return

    # Source 1: dbo entries with labels + comments
    dbo_entries = extract_dbo_from_nt(NT_FILE)

    # Source 2: unique dbp entries with no dbo equivalent
    dbp_entries = extract_unique_dbp_from_w2v(W2V_FILE, dbo_entries)

    # Combine dbo first, then unique dbp
    all_entries = dbo_entries + dbp_entries

    # Deduplicate by URI (dbo: takes priority)
    seen_uris = set()
    unique_entries = []
    for uri, label, doc_text in all_entries:
        if uri not in seen_uris:
            seen_uris.add(uri)
            unique_entries.append((uri, label, doc_text))

    uris          = [e[0] for e in unique_entries]
    labels        = [e[1] for e in unique_entries]
    document_texts = [e[2] for e in unique_entries]

    print(f"\nTotal unique entries: {len(unique_entries)}")
    print(f"  dbo: {sum(1 for u in uris if 'ontology' in u)}")
    print(f"  dbp: {sum(1 for u in uris if '/property/' in u)}")

    # Embed with Nomic Embed v1.5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading Nomic Embed v1.5 on {device}...")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device
    )

    print("Embedding all entries.")
    embeddings = model.encode(
        document_texts, convert_to_tensor=True, show_progress_bar=True
    )

    # Save index to data/
    print(f"\nSaving index to {DATA_DIR}/")
    torch.save(embeddings, str(EMBEDDINGS_PATH))
    with open(URIS_PATH, "w") as f:
        json.dump(uris, f)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f)

    print(f"\n{'='*60}")
    print(f"Done. Index has {len(uris)} entries.")
    print(f"  {EMBEDDINGS_PATH}")
    print(f"  {URIS_PATH}")
    print(f"  {LABELS_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    build_index()
