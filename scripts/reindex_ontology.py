#!/usr/bin/env python3
"""Reindex ontology vectors to include dbp: (property/) properties alongside dbo: ones.

Reads distinct dbp: property URIs from the infobox properties TTL dump,
filters to clean property names, embeds them, and appends to the existing w2v file.

Usage:
    python scripts/reindex_ontology.py [--ttl data/infobox_properties_en.ttl] [--w2v data/ontology-vectors.w2v]
"""

import argparse
import os
import re
import sys
import urllib.parse

import numpy as np
from gensim.models import KeyedVectors
from openai import OpenAI
import dotenv

dotenv.load_dotenv(override=True)

# Only keep dbp: properties whose local name matches this pattern:
# - starts with lowercase letter
# - contains only letters, digits, underscores
# - at least 2 chars long
CLEAN_NAME_RE = re.compile(r"^[a-z][a-zA-Z0-9_]{1,60}$")

BATCH_SIZE = 2000  # OpenAI embedding batch limit
MIN_OCCURRENCES = 1000  # Only keep frequently-used properties (~3.5k)


def extract_dbp_properties(ttl_path):
    """Extract distinct clean dbp: property URIs from the TTL dump, filtered by frequency."""
    from collections import Counter
    counts = Counter()
    with open(ttl_path) as f:
        for line in f:
            parts = line.split(" ", 3)
            if len(parts) < 3:
                continue
            pred = parts[1].strip("<>")
            if not pred.startswith("http://dbpedia.org/property/"):
                continue
            local_name = pred.rsplit("/", 1)[-1]
            local_name = urllib.parse.unquote(local_name)
            if CLEAN_NAME_RE.match(local_name):
                counts[pred] += 1

    # Filter to frequently-used properties
    props = sorted(p for p, c in counts.items() if c >= MIN_OCCURRENCES)
    print(f"  {len(counts)} clean properties total, {len(props)} with >= {MIN_OCCURRENCES} occurrences")
    return props


def embed_properties(props, client):
    """Embed property local names in batches with retry. Returns dict of URI -> embedding."""
    import time
    embeddings = {}

    def label(uri):
        name = uri.rsplit("/", 1)[-1]
        name = urllib.parse.unquote(name)
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

    uris = list(props)
    labels = [f"Term: {label(u)}" for u in uris]
    total_batches = (len(labels) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(labels), BATCH_SIZE):
        batch_labels = labels[i : i + BATCH_SIZE]
        batch_uris = uris[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch_labels)} properties)...")

        for attempt in range(3):
            try:
                response = client.embeddings.create(input=batch_labels, model="text-embedding-3-small")
                for j, item in enumerate(response.data):
                    embeddings[batch_uris[j]] = item.embedding
                break
            except Exception as e:
                if attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"    Retry {attempt + 1} after error: {e} (waiting {wait}s)")
                    time.sleep(wait)
                else:
                    print(f"    FAILED batch {batch_num}: {e}, skipping")

    return embeddings


def extract_dbo_terms(nt_path):
    """Extract dbo: ontology terms (classes + properties) with their type from the NT file."""
    import re as _re
    terms = {}  # uri -> type_label (Class, ObjectProperty, DatatypeProperty)
    type_map = {
        "http://www.w3.org/2002/07/owl#Class": "Class",
        "http://www.w3.org/2002/07/owl#ObjectProperty": "ObjectProperty",
        "http://www.w3.org/2002/07/owl#DatatypeProperty": "DatatypeProperty",
    }
    rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    with open(nt_path) as f:
        for line in f:
            if rdf_type not in line:
                continue
            match = _re.match(r"<(http://dbpedia\.org/ontology/\S+)>\s+<[^>]+>\s+<([^>]+)>", line)
            if match:
                uri, obj = match.groups()
                if obj in type_map and uri not in terms:
                    terms[uri] = type_map[obj]
    return terms


def embed_terms_with_keys(terms_dict, client):
    """Embed dbo: terms using the same 'Term: camelCaseSplit' format. Returns dict of key -> embedding."""
    import time
    embeddings = {}

    def label(uri):
        name = uri.rsplit("/", 1)[-1]
        name = urllib.parse.unquote(name)
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

    keys = [f"{t}|{uri}" for uri, t in terms_dict.items()]
    labels = [f"Term: {label(uri)}" for uri in terms_dict]
    total_batches = (len(labels) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(labels), BATCH_SIZE):
        batch_labels = labels[i : i + BATCH_SIZE]
        batch_keys = keys[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Embedding dbo batch {batch_num}/{total_batches} ({len(batch_labels)} terms)...")

        for attempt in range(3):
            try:
                response = client.embeddings.create(input=batch_labels, model="text-embedding-3-small")
                for j, item in enumerate(response.data):
                    embeddings[batch_keys[j]] = item.embedding
                break
            except Exception as e:
                if attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"    Retry {attempt + 1}: {e} (waiting {wait}s)")
                    time.sleep(wait)
                else:
                    print(f"    FAILED batch {batch_num}: {e}, skipping")

    return embeddings


def write_w2v(w2v_path, all_entries):
    """Write a w2v file from a dict of key -> embedding vector."""
    if not all_entries:
        print("No entries to write.")
        return

    dim = len(next(iter(all_entries.values())))
    total = len(all_entries)
    print(f"Writing {total} vectors ({dim}d) to {w2v_path}...")
    with open(w2v_path, "w") as f:
        f.write(f"{total} {dim}\n")
        for key, vec in all_entries.items():
            vec_str = " ".join(str(v) for v in vec)
            f.write(f"{key} {vec_str}\n")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Reindex ontology: rebuild dbo + dbp vectors with consistent embeddings")
    parser.add_argument("--ttl", default="data/infobox_properties_en.ttl", help="Path to infobox properties TTL")
    parser.add_argument("--nt", default="data/dbpedia_2015-10.nt", help="Path to DBpedia ontology NT file")
    parser.add_argument("--w2v", default="data/ontology-vectors.w2v", help="Path to ontology vectors w2v file (output)")
    args = parser.parse_args()

    for path, label in [(args.ttl, "TTL"), (args.nt, "NT")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    # Step 1: Re-embed dbo: terms with consistent format
    print(f"Step 1: Extracting dbo: terms from {args.nt}...")
    dbo_terms = extract_dbo_terms(args.nt)
    print(f"  Found {len(dbo_terms)} dbo: terms")

    print("Step 2: Re-embedding dbo: terms (consistent format)...")
    dbo_embeddings = embed_terms_with_keys(dbo_terms, client)
    print(f"  Embedded {len(dbo_embeddings)} dbo: terms")

    # Step 3: Extract and embed dbp: properties
    print(f"Step 3: Extracting clean dbp: properties from {args.ttl}...")
    props = extract_dbp_properties(args.ttl)
    print(f"  Found {len(props)} clean properties")

    print("Step 4: Embedding dbp: properties...")
    dbp_embeddings = embed_properties(props, client)
    print(f"  Embedded {len(dbp_embeddings)} dbp: properties")

    # Step 5: Merge and write
    all_entries = dict(dbo_embeddings)
    for uri, emb in dbp_embeddings.items():
        key = f"Property|{uri}"
        all_entries[key] = emb

    print(f"Step 5: Writing {len(all_entries)} total vectors to {args.w2v}...")
    write_w2v(args.w2v, all_entries)


if __name__ == "__main__":
    main()
