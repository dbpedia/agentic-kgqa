#!/usr/bin/env python3
"""Schema Introspector: enriches dbo: predicate candidates with structured
metadata (rdfs:label, rdfs:comment, rdfs:domain, rdfs:range) from the
curated OWL ontology file.

dbp: properties now only ever appear via the live agentic probe in validator.py, which
attaches its own grounded context (the actual value found for that subject)
directly -- no separate enrichment step is needed for them.

Requires:
    resources/dbpedia-20250806.owl.rdf -- DBpedia OWL ontology file (included in the repo).
"""

from pathlib import Path

import rdflib
from rdflib.namespace import RDFS

OWL_FILE   = Path(__file__).parent.parent / "resources" / "dbpedia-20250806.owl.rdf"
_owl_graph = None


def _load_owl():
    global _owl_graph
    if _owl_graph is not None:
        return
    if not OWL_FILE.exists():
        raise FileNotFoundError(
            f"OWL file not found at {OWL_FILE}.\n"
            "Refer to README.md."
        )
    _owl_graph = rdflib.Graph()
    _owl_graph.parse(str(OWL_FILE), format="xml")


def _get_dbo_metadata(predicate_uri):
    """Look up rdfs:label, rdfs:comment, rdfs:domain, rdfs:range for a dbo: URI."""
    uri_ref = rdflib.URIRef(predicate_uri)
    domain  = None
    range_  = None
    label   = None
    comment = None
    for d in _owl_graph.objects(uri_ref, RDFS.domain):
        if isinstance(d, rdflib.URIRef):
            domain = str(d).split("/")[-1]
            break
    for r in _owl_graph.objects(uri_ref, RDFS.range):
        if isinstance(r, rdflib.URIRef):
            range_ = str(r).split("/")[-1]
            break
    for l in _owl_graph.objects(uri_ref, RDFS.label):
        if getattr(l, "language", None) in ["en", None]:
            label = str(l)
            break
    for c in _owl_graph.objects(uri_ref, RDFS.comment):
        if getattr(c, "language", None) in ["en", None]:
            comment = str(c)
            break
    return domain, range_, label, comment


def enrich(predicate_candidates):
    """Enrich dbo: predicate candidates with domain, range, label and comment.

    Input:  list of {uri, label, score, confidence_pct, source}
    Output: same list with label (from OWL, falls back to index label),
            comment, domain, range added to each dict.
    """
    _load_owl()
    enriched = []
    for cand in predicate_candidates:
        domain, range_, owl_label, comment = _get_dbo_metadata(cand["uri"])
        enriched.append({
            **cand,
            "label":   owl_label or cand.get("label", ""),
            "comment": comment,
            "domain":  domain,
            "range":   range_,
        })
    return enriched
