#!/usr/bin/env python3
"""Schema Introspector: enriches predicate candidates with rdfs:domain and rdfs:range.

Reads the DBpedia OWL ontology file (data/dbpedia-20250806.owl.rdf) and looks up
the domain and range for each predicate candidate returned by the Ontology Explorer.
This helps the Query Builder determine correct triple direction at generation time.

Requires:
    data/dbpedia-20250806.owl.rdf — DBpedia OWL ontology file.
    See README.md for instructions.
"""

from pathlib import Path

import rdflib
from rdflib.namespace import RDFS

OWL_FILE = Path(__file__).parent.parent / "data" / "dbpedia-20250806.owl.rdf"

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


def _get_domain_range(predicate_uri):
    uri_ref = rdflib.URIRef(predicate_uri)
    domain = None
    range_ = None
    for d in _owl_graph.objects(uri_ref, RDFS.domain):
        if isinstance(d, rdflib.URIRef):
            domain = str(d).split("/")[-1]
            break
    for r in _owl_graph.objects(uri_ref, RDFS.range):
        if isinstance(r, rdflib.URIRef):
            range_ = str(r).split("/")[-1]
            break
    return domain, range_


def enrich(predicate_candidates):
    """Add rdfs:domain and rdfs:range to each candidate dict.

    Input:  list of {uri, label, score, confidence_pct, source}
    Output: same list with 'domain' and 'range' keys added.
    """
    _load_owl()
    enriched = []
    for cand in predicate_candidates:
        domain, range_ = _get_domain_range(cand["uri"])
        enriched.append({**cand, "domain": domain, "range": range_})
    return enriched
