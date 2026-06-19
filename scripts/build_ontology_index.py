#!/usr/bin/env python3
"""Build and save the Nomic Embed v1.5 ontology index.

This script builds a semantic search index over DBpedia ontology properties and classes.
It must be run once before using the Ontology Explorer node in the pipeline.

Sources:
  1. dbo: properties and Classes — parsed from data/dbpedia-20250806.owl.rdf using RDFLib.
     The OWL file has more entries (3818) than the NT file (3572), has richer label and
     comment coverage, and is already used by the Schema Introspector for domain/range
     lookup — making it the single source for all dbo work.

  2. dbp: properties — parsed from data/dbp-part1.rdf and data/dbp-part2.rdf.
     These files were downloaded from the DBpedia SPARQL endpoint using an owl:Thing
     anchor query, which captures ghost properties like dbp:numLocations that exist in
     DBpedia data but are not formally typed as rdf:Property in the schema.
     Only dbp: properties with no exact dbo: equivalent are included. Properties with
     exact dbo: equivalents are skipped because the Query Executor handles those via a
     deterministic dbo->dbp namespace swap at runtime.

     Noise filters applied to dbp: properties:
       - Names <= 3 characters (e.g. dbp:nat, dbp:rev, dbp:mc)
       - Names containing '%' (URL-encoded, e.g. dbp:votes%25)
       - Names containing '_' (non-standard infobox keys, e.g. dbp:lat_deg)
       - Names starting with a digit (e.g. dbp:1stRound, dbp:2014Population)

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
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import rdflib
import torch
from rdflib.namespace import OWL, RDF, RDFS
from sentence_transformers import SentenceTransformer

# ─── Paths (all relative to agentic-kgqa/) ────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent           # agentic-kgqa/
DATA_DIR = BASE_DIR / "data"

OWL_FILE      = DATA_DIR / "dbpedia-20250806.owl.rdf"
DBP_RDF_FILES = [
    DATA_DIR / "dbp-part1.rdf",
    DATA_DIR / "dbp-part2.rdf",
]

EMBEDDINGS_PATH = DATA_DIR / "nomic_embeddings.pt"
URIS_PATH       = DATA_DIR / "nomic_uris.json"
LABELS_PATH     = DATA_DIR / "nomic_labels.json"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def split_camel(s):
    """Split camelCase into readable words. e.g. 'birthPlace' -> 'birth Place'"""
    return re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", s)).strip()


# ─── Source 1: dbo: entries from OWL file ─────────────────────────────────────

def extract_dbo_from_owl(owl_file):
    """Parse dbpedia-20250806.owl.rdf and extract dbo: Classes and properties
    with their English labels and comments.

    The OWL file has more entries than the NT file (3818 vs 3572) and is already
    used by the Schema Introspector for domain/range lookup, making it the single
    source for all dbo work.

    Returns: list of (uri, label, doc_text) tuples.
    """
    print(f"\nSource 1: Parsing OWL file: {owl_file}")
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
        raw_name = str(prop_uri).split("/")[-1]
        label = list(data["labels"])[0] if data["labels"] else split_camel(raw_name)
        comment = list(data["comments"])[0] if data["comments"] else ""
        doc_text = f"search_document: {label}. {comment}".strip(". ")
        entries.append((str(prop_uri), label, doc_text))

    print(f"  Extracted {len(entries)} dbo: entries from OWL file")
    return entries


# ─── Source 2: unique dbp: entries from local RDF files ───────────────────────

def extract_dbp_from_rdf_files(dbo_entries, rdf_files=DBP_RDF_FILES):
    """Extract dbp: properties from locally saved SPARQL result RDF files.

    Reads dbp-part1.rdf and dbp-part2.rdf downloaded from the DBpedia SPARQL
    endpoint using the owl:Thing anchor query. This approach captures ghost
    properties like dbp:numLocations that exist in DBpedia data but are not
    formally typed as rdf:Property in the schema.

    Combines, deduplicates, and filters before returning.

    Filters applied:
      - Names <= 3 characters (noise/abbreviations like dbp:nat, dbp:mc)
      - Names starting with a digit (e.g. dbp:1stRound)
      - Names containing '%' (URL-encoded, e.g. dbp:votes%25)
      - Names containing '_' (non-standard infobox keys)
      - Names with exact dbo: equivalent (runtime swap handles these)

    Returns: list of (uri, label, doc_text) tuples.
    """
    print(f"\nSource 2: Parsing local RDF files for dbp: properties...")

    all_uris = set()
    for rdf_file in rdf_files:
        if not rdf_file.exists():
            print(f"  WARNING: {rdf_file} not found — skipping")
            continue
        tree = ET.parse(str(rdf_file))
        root = tree.getroot()
        batch = {
            elem.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
            for elem in root.iter()
            if elem.get(
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", ""
            ).startswith("http://dbpedia.org/property/")
        }
        print(f"  {rdf_file.name}: {len(batch)} dbp: URIs")
        all_uris |= batch

    print(f"  Total unique dbp: URIs after combining: {len(all_uris)}")

    # Build set of dbo: names (lowercased) for deduplication
    dbo_names = {uri.split("/")[-1].lower() for uri, _, _ in dbo_entries}

    entries = []
    skipped_equiv  = 0
    skipped_noise  = 0

    for uri in sorted(all_uris):
        raw_name = uri.split("/")[-1]

        # Skip short names — noise/abbreviations (e.g. dbp:nat, dbp:mc, dbp:rev)
        if len(raw_name) <= 3:
            skipped_noise += 1
            continue

        # Skip URL-encoded names (e.g. dbp:votes%25)
        if "%" in raw_name:
            skipped_noise += 1
            continue

        # Skip underscore names — non-standard infobox keys (e.g. dbp:lat_deg)
        if "_" in raw_name:
            skipped_noise += 1
            continue

        # Skip names starting with a digit (e.g. dbp:1stRound, dbp:2014Population)
        if raw_name[0].isdigit():
            skipped_noise += 1
            continue

        # Skip if exact dbo: equivalent exists — runtime swap handles these
        if raw_name.lower() in dbo_names:
            skipped_equiv += 1
            continue

        label    = split_camel(raw_name)
        doc_text = f"search_document: {label}"
        entries.append((uri, label, doc_text))

    print(f"  Skipped {skipped_equiv} dbp: entries with exact dbo: equivalents")
    print(f"  Skipped {skipped_noise} noise entries (short/encoded/underscore/digit-start)")
    print(f"  Kept {len(entries)} unique dbp: entries")
    return entries


# ─── Build Index ──────────────────────────────────────────────────────────────

def build_index():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building Nomic Embed v1.5 ontology index")
    print(f"dbo source : {OWL_FILE.name}")
    print(f"dbp source : {', '.join(f.name for f in DBP_RDF_FILES)}")
    print("=" * 60)

    if not OWL_FILE.exists():
        print(f"\nERROR: OWL file not found: {OWL_FILE}")
        print("Please refer to README.md for setup instructions.")
        return

    # Source 1: dbo entries from OWL file
    dbo_entries = extract_dbo_from_owl(OWL_FILE)

    # Source 2: unique dbp entries from local RDF files
    dbp_entries = extract_dbp_from_rdf_files(dbo_entries)

    # Combine — dbo first, then unique dbp
    all_entries = dbo_entries + dbp_entries

    # Deduplicate by URI (dbo: takes priority)
    seen_uris = set()
    unique_entries = []
    for uri, label, doc_text in all_entries:
        if uri not in seen_uris:
            seen_uris.add(uri)
            unique_entries.append((uri, label, doc_text))

    uris           = [e[0] for e in unique_entries]
    labels         = [e[1] for e in unique_entries]
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

    print("Embedding all entries (this takes ~50 seconds on CPU)...")
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
