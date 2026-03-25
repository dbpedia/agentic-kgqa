#!/usr/bin/env python3
"""Query the local SPARQL endpoint to count triples per predicate and save as JSON.

Only keeps dbo: and dbp: predicates (filters out rdf/rdfs/owl/foaf/dc metadata).
"""

import json
import urllib.parse
import urllib.request

ENDPOINT = "http://localhost:7878/query"
OUTPUT = "data/predicate_frequencies.json"

QUERY = "SELECT ?p (COUNT(*) AS ?c) WHERE { ?s ?p ?o } GROUP BY ?p"

# Prefixes to keep
KEEP_PREFIXES = (
    "http://dbpedia.org/ontology/",
    "http://dbpedia.org/property/",
)


def main():
    print("Querying predicate frequencies from local endpoint...")
    params = urllib.parse.urlencode({"query": QUERY, "format": "application/sparql-results+json"})
    url = f"{ENDPOINT}?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    bindings = data["results"]["bindings"]
    print(f"Got {len(bindings)} total predicates")

    freqs = {}
    for b in bindings:
        uri = b["p"]["value"]
        count = int(b["c"]["value"])
        if any(uri.startswith(prefix) for prefix in KEEP_PREFIXES):
            freqs[uri] = count

    print(f"Kept {len(freqs)} dbo/dbp predicates")

    # Sort by count descending for readability
    freqs = dict(sorted(freqs.items(), key=lambda x: -x[1]))

    with open(OUTPUT, "w") as f:
        json.dump(freqs, f, indent=2)

    print(f"Saved to {OUTPUT}")

    # Show top 20
    print("\nTop 20 predicates:")
    for uri, count in list(freqs.items())[:20]:
        ns = "dbo" if "/ontology/" in uri else "dbp"
        name = uri.split("/")[-1]
        print(f"  {ns}:{name:30s} {count:>10,}")


if __name__ == "__main__":
    main()
