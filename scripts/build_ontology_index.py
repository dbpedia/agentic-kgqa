#!/usr/bin/env python3
"""Build and save the Nomic Embed v1.5 dbo: ontology index.

Builds a semantic search index over DBpedia ontology (dbo:) properties and
classes, parsed from the DBpedia OWL file. This is the dbo-only architecture
(see README Architecture Decisions): dbp: properties are not embedded in a
static index. They are instead recovered at query time via the deterministic
dbo->dbp namespace swap (Query Executor) and the live agentic probe
(Validator), which is simpler and was shown in a controlled comparison to be
statistically equivalent to maintaining a separate AI-labelled dbp: index.

Source: dbo: properties and Classes, parsed from data/dbpedia-20250806.owl.rdf
using RDFLib. The OWL file is also used by the Schema Introspector for
domain/range lookup, making it the single source for all dbo work.

Slash-namespaced URIs (e.g. dbo:PopulatedPlace/area) are skipped -- these are
class-scoped property variants that duplicate the generic property and would
add noise to the index without adding coverage.

Output (saved to data/):
  - nomic_embeddings_dbo.pt   — PyTorch tensor of shape [N, 768]
  - nomic_uris_dbo.json       — list of N URIs in same order as embeddings
  - nomic_labels_dbo.json     — list of N human-readable labels in same order

Run once before using the Ontology Explorer node in the pipeline:
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

OWL_FILE = DATA_DIR / "dbpedia-20250806.owl.rdf"

EMBEDDINGS_PATH = DATA_DIR / "nomic_embeddings_dbo.pt"
URIS_PATH       = DATA_DIR / "nomic_uris_dbo.json"
LABELS_PATH     = DATA_DIR / "nomic_labels_dbo.json"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def split_camel(s):
    """Split camelCase into readable words. e.g. 'birthPlace' -> 'birth Place'"""
    return re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", s)).strip()


# ─── dbo: entries from OWL file ────────────────────────────────────────────────

def extract_dbo_from_owl(owl_file):
    """Parse dbpedia-20250806.owl.rdf and extract dbo: Classes and properties
    with their English labels and comments.

    Skips slash-namespaced URIs (e.g. dbo:PopulatedPlace/area) since these are
    class-scoped variants that duplicate generic properties.

    Returns: list of (uri, label, doc_text) tuples.
    """
    print(f"\nParsing OWL file: {owl_file}")
    g = rdflib.Graph()
    g.parse(str(owl_file), format="xml")
    print(f"  Loaded {len(g)} triples")

    prop_data = defaultdict(lambda: {"labels": set(), "comments": set()})
    property_types = [OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property, OWL.Class]

    for ptype in property_types:
        for prop in g.subjects(RDF.type, ptype):
            if isinstance(prop, rdflib.URIRef) and "dbpedia.org/ontology" in str(prop):
                for label in g.objects(prop, RDFS.label):
                    if getattr(label, "language", None) in ["en", None]:
                        prop_data[prop]["labels"].add(str(label))
                for comment in g.objects(prop, RDFS.comment):
                    if getattr(comment, "language", None) in ["en", None]:
                        prop_data[prop]["comments"].add(str(comment))

    entries = []
    for prop_uri, data in prop_data.items():
        uri_str = str(prop_uri)

        # Extract the specific property path after 'ontology/'
        onto_path = uri_str.split("dbpedia.org/ontology/")[-1]

        # Skip slash-namespaced URIs like dbo:PopulatedPlace/area -- these are
        # class-scoped variants that duplicate generic properties
        if "/" in onto_path:
            continue

        raw_name = onto_path
        label = list(data["labels"])[0] if data["labels"] else split_camel(raw_name)
        comment = list(data["comments"])[0] if data["comments"] else ""
        doc_text = f"search_document: {label}. {comment}".strip(". ")
        entries.append((str(prop_uri), label, doc_text))

    print(f"  Extracted {len(entries)} dbo: entries from OWL file")
    return entries


# ─── Build Index ──────────────────────────────────────────────────────────────

def build_index():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building Nomic Embed v1.5 dbo: ontology index")
    print(f"Source: {OWL_FILE.name}")
    print("=" * 60)

    if not OWL_FILE.exists():
        print(f"\nERROR: OWL file not found: {OWL_FILE}")
        print("Please refer to README.md for setup instructions.")
        return

    entries = extract_dbo_from_owl(OWL_FILE)

    uris           = [e[0] for e in entries]
    labels         = [e[1] for e in entries]
    document_texts = [e[2] for e in entries]

    print(f"\nTotal entries: {len(entries)}")

    # Embed with Nomic Embed v1.5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading Nomic Embed v1.5 on {device}...")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device
    )

    print("Embedding all entries (this takes ~10 seconds on CPU)...")
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
