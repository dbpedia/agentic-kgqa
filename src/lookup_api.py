import requests
import xml.etree.ElementTree as ET

def lookup_entity(mention: str, max_results: int = 5) -> list:
    """
    Query DBpedia Lookup API for a given mention.
    Returns list of dicts with 'uri', 'label', and 'description'.
    """
    try:
        r = requests.get(
            "https://lookup.dbpedia.org/api/search",
            params={"query": mention, "maxResults": max_results},
            timeout=10
        )
        r.raise_for_status()

        root = ET.fromstring(r.text)
        results = []
        for result in root.findall('Result'):
            uri = result.findtext('URI', '')
            label = result.findtext('Label', '')
            description = result.findtext('Description', '')[:100] if result.findtext('Description') else ''
            results.append({
                'uri': uri,
                'label': label,
                'description': description
            })
        return results

    except Exception as e:
        print(f"Lookup API error for '{mention}': {e}")
        return []

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/siddharth/Desktop/DBpedia/agentic-kgqa')
    from src.entity_linking import RedisEntityLinking

    test = ["Keanu Reeves", "NYC", "Einstein", "Paris"]

    for mention in test:
        print(f"\n{mention}:")
        results = lookup_entity(mention)
        for r in results:
            print(f"  {r['uri']} | {r['label']}")

    el = RedisEntityLinking()
    print("\n\n--- REDIS COMPARISON ---")
    for mention in test:
        print(f"\n{mention}:")
        result = el.lookup(mention, top_k=3)
        if len(result) > 0:
            for uri, row in result.iterrows():
                print(f"  {uri} | score: {row['score']:.3f}")
        else:
            print("  NO RESULT")
