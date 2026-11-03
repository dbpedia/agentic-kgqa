import io
import json
from unittest import mock

from src import evaluate


def _run_against(bindings):
    """Call evaluate._execute_sparql with a fake endpoint that returns these bindings."""
    body = json.dumps({"head": {"vars": ["uri"]}, "results": {"bindings": bindings}}).encode()
    response = mock.MagicMock()
    response.__enter__.return_value = io.BytesIO(body)
    with mock.patch.object(evaluate.urllib.request, "urlopen", return_value=response):
        return evaluate._execute_sparql("SELECT ?uri WHERE { ?uri ?p ?o }")


def test_large_result_sets_are_scored_on_every_row():
    # Same 250 answers, returned in a different order (no ORDER BY in either query).
    rows = [{"uri": {"type": "uri", "value": f"http://dbpedia.org/resource/E{i}"}} for i in range(250)]
    gold = _run_against(rows)
    generated = _run_against(list(reversed(rows)))

    assert evaluate._result_set_match(generated, gold)
    assert evaluate._prf(generated, gold) == (1.0, 1.0, 1.0)
