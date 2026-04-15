#!/usr/bin/env python3
"""Build embedding indexes for the corporate (CK25) dataset.

Parses prod-vocab.ttl → data/corporate/vocab-vectors.w2v
Parses prod-inst.ttl  → data/corporate/instance-vectors.w2v

Usage:
    python scripts/build_corporate_index.py [--vocab data/corporate/prod-vocab.ttl]
                                             [--inst  data/corporate/prod-inst.ttl]
                                             [--out   data/corporate]
"""

import argparse
import os
import re
import sys
import time
import urllib.parse

from openai import OpenAI
import dotenv

dotenv.load_dotenv(override=True)

PV_BASE = "http://ld.company.org/prod-vocab/"
PRODI_BASE = "http://ld.company.org/prod-instances/"
BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Turtle parsers (minimal, tailored to these two files)
# ---------------------------------------------------------------------------

def _expand_prefix(term, prefixes):
    """Expand a prefixed name to a full URI. Handles 'a' shorthand for rdf:type."""
    if term == "a":
        return "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    if term.startswith("<") and term.endswith(">"):
        return term[1:-1]
    if ":" in term:
        prefix, local = term.split(":", 1)
        if prefix in prefixes:
            return prefixes[prefix] + local
    return None


def parse_vocab(path):
    """Parse prod-vocab.ttl → list of {uri, type, label}.

    Captures pv: classes and properties plus dbpedia:Country.
    """
    prefixes = {}
    # First pass: collect prefixes
    with open(path) as f:
        for line in f:
            m = re.match(r"@prefix\s+(\w*):\s+<([^>]+)>", line.strip())
            if m:
                prefixes[m.group(1)] = m.group(2)

    # Second pass: collect triples block by block
    with open(path) as f:
        content = f.read()

    # Simple block parser: split on blank lines separating subject blocks
    terms = {}
    subject = None
    owl_type_map = {
        "http://www.w3.org/2002/07/owl#Class": "Class",
        "http://www.w3.org/2002/07/owl#ObjectProperty": "ObjectProperty",
        "http://www.w3.org/2002/07/owl#DatatypeProperty": "DatatypeProperty",
    }
    rdf_type_uri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    lines = content.split("\n")
    for line in lines:
        stripped = line.strip().rstrip(" ;.")
        if not stripped or stripped.startswith("@prefix") or stripped.startswith("#"):
            continue

        # New subject: starts at column 0 (no leading whitespace)
        if not line.startswith(" ") and not line.startswith("\t"):
            subj_raw = stripped.split()[0] if stripped else None
            if subj_raw:
                uri = _expand_prefix(subj_raw, prefixes)
                if uri:
                    subject = uri
                    if subject not in terms:
                        terms[subject] = {"uri": subject, "owl_type": None, "label": None}
            continue

        if subject is None:
            continue

        # Predicate-object pairs — split on first whitespace only to preserve multi-word objects
        tok = stripped.split(None, 1)
        if len(tok) < 2:
            continue
        pred_raw = tok[0]
        obj_rest = tok[1].strip()
        pred = _expand_prefix(pred_raw, prefixes)
        if pred is None:
            continue

        if pred == rdf_type_uri:
            obj_tok = obj_rest.split()[0].rstrip(" ;.")
            obj = _expand_prefix(obj_tok, prefixes)
            if obj and obj in owl_type_map:
                terms[subject]["owl_type"] = owl_type_map[obj]

        elif pred == "http://www.w3.org/2000/01/rdf-schema#label":
            # Extract string from "Label text"@en or "Label text"
            m = re.match(r'"([^"]+)"', obj_rest)
            if m:
                terms[subject]["label"] = m.group(1)

    results = []
    for uri, info in terms.items():
        if not info["owl_type"]:
            continue
        label = info["label"] or uri.rsplit("/", 1)[-1]
        results.append({"uri": uri, "type": info["owl_type"], "label": label})
    return results


def parse_instances(path):
    """Parse prod-inst.ttl → list of {uri, type, label}.

    Handles both prefixed names (prodi:xxx) and full URIs (<...>).
    """
    prefixes = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"@prefix\s+(\w*):\s+<([^>]+)>", line.strip())
            if m:
                prefixes[m.group(1)] = m.group(2)

    with open(path) as f:
        content = f.read()

    pv_class_map = {
        f"{PV_BASE}{c}": c
        for c in [
            "Agent", "BillOfMaterial", "BomPart", "Department", "Employee",
            "Hardware", "Manager", "Price", "Product", "ProductCategory",
            "Service", "Supplier",
        ]
    }
    rdf_type_uri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    instances = {}
    subject = None

    lines = content.split("\n")
    for line in lines:
        stripped = line.strip().rstrip(" ;.")
        if not stripped or stripped.startswith("@prefix") or stripped.startswith("#"):
            continue

        if not line.startswith(" ") and not line.startswith("\t"):
            subj_raw = stripped.split()[0] if stripped else None
            if subj_raw:
                uri = _expand_prefix(subj_raw, prefixes)
                if uri and (uri.startswith(PRODI_BASE) or
                            uri.startswith("http://ld.company.org/prod-instances/")):
                    subject = uri
                    if subject not in instances:
                        instances[subject] = {"uri": subject, "type": None, "label": None}
                else:
                    subject = None
            continue

        if subject is None:
            continue

        tok = stripped.split(None, 1)
        if len(tok) < 2:
            continue
        pred_raw = tok[0]
        obj_rest = tok[1].strip()

        pred = _expand_prefix(pred_raw, prefixes)
        if pred is None:
            continue

        if pred == rdf_type_uri:
            obj_tok = obj_rest.split()[0].rstrip(" ;.")
            obj = _expand_prefix(obj_tok, prefixes)
            if obj and obj in pv_class_map:
                instances[subject]["type"] = pv_class_map[obj]

        elif pred == "http://www.w3.org/2000/01/rdf-schema#label":
            m = re.match(r'"([^"]+)"', obj_rest)
            if m:
                instances[subject]["label"] = m.group(1)

    results = []
    for uri, info in instances.items():
        if not info["type"] or not info["label"]:
            continue
        results.append({"uri": uri, "type": info["type"], "label": info["label"]})
    return results


# ---------------------------------------------------------------------------
# Embedding helpers (same pattern as reindex_ontology.py)
# ---------------------------------------------------------------------------

def embed_batch(texts, client, retries=3):
    """Embed a batch of texts with retry. Returns list of embeddings."""
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                input=texts, model="text-embedding-3-small"
            )
            if not response.data:
                raise ValueError("Empty embedding response")
            return [x.embedding for x in response.data]
        except Exception as e:
            if attempt < retries - 1:
                wait = (attempt + 1) * 5
                print(f"    Retry {attempt + 1} after: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise


def embed_items(items, client, label_fn):
    """Embed all items in batches. Returns dict of key -> embedding."""
    result = {}
    total_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} items)...")
        texts = [f"Term: {label_fn(item)}" for item in batch]
        embeddings = embed_batch(texts, client)
        for item, emb in zip(batch, embeddings):
            key = f"{item['type']}|{item['uri']}"
            result[key] = emb
    return result


# ---------------------------------------------------------------------------
# W2V writer
# ---------------------------------------------------------------------------

def write_w2v(path, entries):
    dim = len(next(iter(entries.values())))
    print(f"Writing {len(entries)} vectors ({dim}d) to {path}...")
    with open(path, "w") as f:
        f.write(f"{len(entries)} {dim}\n")
        for key, vec in entries.items():
            f.write(f"{key} {' '.join(str(v) for v in vec)}\n")
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build embedding indexes for corporate dataset")
    parser.add_argument("--vocab", default="data/corporate/prod-vocab.ttl")
    parser.add_argument("--inst", default="data/corporate/prod-inst.ttl")
    parser.add_argument("--out", default="data/corporate")
    args = parser.parse_args()

    for path in [args.vocab, args.inst]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    # --- Vocab index ---
    print(f"Parsing vocab from {args.vocab}...")
    vocab_terms = parse_vocab(args.vocab)
    print(f"  Found {len(vocab_terms)} vocab terms")
    for v in vocab_terms[:5]:
        print(f"    {v['type']}: {v['uri']!r} label={v['label']!r}")

    print("Embedding vocab terms...")
    vocab_entries = embed_items(vocab_terms, client, lambda t: t["label"])
    vocab_out = os.path.join(args.out, "vocab-vectors.w2v")
    write_w2v(vocab_out, vocab_entries)

    # --- Instance index ---
    print(f"\nParsing instances from {args.inst}...")
    instances = parse_instances(args.inst)
    print(f"  Found {len(instances)} instances")
    by_type = {}
    for inst in instances:
        by_type.setdefault(inst["type"], 0)
        by_type[inst["type"]] += 1
    for t, n in sorted(by_type.items()):
        print(f"    {t}: {n}")

    print("Embedding instances...")
    inst_entries = embed_items(instances, client, lambda t: t["label"])
    inst_out = os.path.join(args.out, "instance-vectors.w2v")
    write_w2v(inst_out, inst_entries)

    print("\nAll indexes built successfully.")
    print(f"  Vocab:     {vocab_out}")
    print(f"  Instances: {inst_out}")


if __name__ == "__main__":
    main()
