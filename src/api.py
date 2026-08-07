"""text2sparql API endpoint for the KGQA agent, with streaming UI."""

import json
import logging

import fastapi
from fastapi.responses import HTMLResponse, StreamingResponse

from src.agent import KGQAAgent, MODELS, DEFAULT_MODEL

# Slug -> OpenRouter model ID mapping for path-based endpoints
MODEL_SLUGS = {m["id"].split("/")[-1]: m["id"] for m in MODELS}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = fastapi.FastAPI(title="TEXT2SPARQL API - Agentic KGQA")

DBPEDIA_DATASETS = [
    "https://text2sparql.aksw.org/2025/dbpedia/",
    "https://text2sparql.aksw.org/2026/dbpedia/",
]
CORPORATE_DATASETS = [
    "https://text2sparql.aksw.org/2026/corporate/",
]
KNOWN_DATASETS = DBPEDIA_DATASETS + CORPORATE_DATASETS

agent = None
corporate_agent = None


@app.on_event("startup")
async def startup():
    global agent, corporate_agent
    agent = KGQAAgent()
    logger.info("DBpedia KGQA Agent initialised.")
    try:
        from src.corporate_agent import CorporateKGQAAgent
        corporate_agent = CorporateKGQAAgent()
        logger.info("Corporate KGQA Agent initialised.")
    except Exception as e:
        logger.warning(f"Corporate agent not available (indexes not built?): {e}")


# --- Challenge API endpoint (unchanged) ---


def _get_agent_for_dataset(dataset: str):
    if dataset in CORPORATE_DATASETS:
        if corporate_agent is None:
            raise fastapi.HTTPException(503, "Corporate agent not available — run scripts/build_corporate_index.py first")
        return corporate_agent
    return agent


@app.get("/answer")
async def get_answer(question: str, dataset: str):
    if dataset not in KNOWN_DATASETS:
        raise fastapi.HTTPException(404, "Unknown dataset")

    a = _get_agent_for_dataset(dataset)
    sparql = a.answer(question)

    return {
        "dataset": dataset,
        "question": question,
        "query": sparql,
    }


# --- Model list endpoint ---


@app.get("/models")
async def get_models():
    return {"models": MODELS, "default": DEFAULT_MODEL}


# --- Streaming endpoint (SSE) ---


def _event_stream(question: str, model: str | None = None, dataset: str | None = None):
    """Generator that yields SSE events from the agent pipeline."""
    a = _get_agent_for_dataset(dataset) if dataset in CORPORATE_DATASETS else agent
    for event_type, data in a.answer_stream(question, model=model):
        payload = json.dumps({"type": event_type, "data": data}, default=str)
        yield f"event: {event_type}\ndata: {payload}\n\n"


@app.get("/stream")
async def stream_answer(question: str, model: str | None = None, dataset: str | None = None):
    return StreamingResponse(
        _event_stream(question, model=model, dataset=dataset),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Model-in-URL endpoints ---


def _resolve_model_slug(model_slug: str) -> str:
    if model_slug not in MODEL_SLUGS:
        raise fastapi.HTTPException(404, f"Unknown model '{model_slug}'. Available: {', '.join(sorted(MODEL_SLUGS))}")
    return MODEL_SLUGS[model_slug]


@app.get("/m/{model_slug}/answer")
async def model_get_answer(model_slug: str, question: str, dataset: str):
    model_id = _resolve_model_slug(model_slug)
    if dataset not in KNOWN_DATASETS:
        raise fastapi.HTTPException(404, "Unknown dataset")
    a = _get_agent_for_dataset(dataset)
    sparql = a.answer(question, model=model_id)
    return {"dataset": dataset, "question": question, "query": sparql, "model": model_id}


@app.get("/m/{model_slug}/stream")
async def model_stream_answer(model_slug: str, question: str, dataset: str | None = None):
    model_id = _resolve_model_slug(model_slug)
    return StreamingResponse(
        _event_stream(question, model=model_id, dataset=dataset),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- UI ---


@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTML_PAGE


HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agentic KGQA</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e4e4e7; --muted: #8b8d97; --accent: #6366f1;
    --green: #22c55e; --amber: #f59e0b; --blue: #3b82f6;
    --font: 'Inter', system-ui, -apple-system, sans-serif;
    --mono: 'JetBrains Mono', 'Fira Code', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--font); background: var(--bg); color: var(--text);
    min-height: 100vh; display: flex; flex-direction: column; align-items: center;
    padding: 2rem 1rem;
  }
  h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: .25rem; }
  .subtitle { color: var(--muted); font-size: .85rem; margin-bottom: 1.5rem; }
  .container { width: 100%; max-width: 780px; }
  .input-row {
    display: flex; gap: .5rem; margin-bottom: 1.5rem;
  }
  input[type=text] {
    flex: 1; padding: .65rem .85rem; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--text); font-size: .95rem; font-family: var(--font);
    outline: none; transition: border-color .15s;
  }
  input[type=text]:focus { border-color: var(--accent); }
  button {
    padding: .65rem 1.3rem; border-radius: 8px; border: none;
    background: var(--accent); color: #fff; font-weight: 600;
    font-size: .9rem; cursor: pointer; transition: opacity .15s;
  }
  button:hover { opacity: .85; }
  button:disabled { opacity: .4; cursor: not-allowed; }

  /* Steps timeline */
  .steps { display: flex; flex-direction: column; gap: 0; }
  .step {
    position: relative; padding: .75rem 1rem .75rem 2.5rem;
    border-left: 2px solid var(--border); margin-left: .75rem;
  }
  .step:last-child { border-left-color: transparent; }
  .step::before {
    content: ''; position: absolute; left: -.45rem; top: .85rem;
    width: .7rem; height: .7rem; border-radius: 50%;
    background: var(--border); border: 2px solid var(--bg);
  }
  .step.active::before { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
  .step.done::before { background: var(--green); }
  .step-label {
    font-weight: 600; font-size: .85rem; margin-bottom: .35rem;
    display: flex; align-items: center; gap: .4rem;
  }
  .step-label .spinner {
    width: 14px; height: 14px; border: 2px solid var(--border);
    border-top-color: var(--amber); border-radius: 50%;
    animation: spin .6s linear infinite; display: inline-block;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .step-body {
    background: var(--surface); border-radius: 6px; padding: .6rem .8rem;
    font-family: var(--mono); font-size: .78rem; line-height: 1.5;
    color: var(--muted); overflow-x: auto; white-space: pre-wrap;
    word-break: break-word; max-height: 320px; overflow-y: auto;
  }
  .step-body.sparql { color: var(--blue); font-size: .85rem; }

  /* Examples */
  .examples { margin-bottom: 1rem; }
  .examples summary { color: var(--muted); font-size: .8rem; cursor: pointer; margin-bottom: .5rem; }
  .example-chips { display: flex; flex-wrap: wrap; gap: .4rem; }
  .chip {
    padding: .3rem .7rem; border-radius: 6px; font-size: .78rem;
    background: var(--surface); border: 1px solid var(--border);
    color: var(--muted); cursor: pointer; transition: border-color .15s;
  }
  .chip:hover { border-color: var(--accent); color: var(--text); }

  /* Model selector */
  .model-row { display: flex; align-items: center; gap: .5rem; margin-bottom: 1rem; }
  .model-row label { font-size: .8rem; color: var(--muted); white-space: nowrap; }
  select {
    padding: .45rem .7rem; border-radius: 6px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--text); font-size: .82rem; font-family: var(--font);
    outline: none; cursor: pointer; transition: border-color .15s;
  }
  select:focus { border-color: var(--accent); }
</style>
</head>
<body>
<div class="container">
  <h1>Agentic KGQA</h1>
  <p class="subtitle">Natural language to SPARQL against DBpedia</p>

  <details class="examples" open>
    <summary>Example questions</summary>
    <div class="example-chips">
      <span class="chip" onclick="askExample(this)">What is the birthplace of Keanu Reeves?</span>
      <span class="chip" onclick="askExample(this)">How many unique authors have written science fiction novels?</span>
      <span class="chip" onclick="askExample(this)">Is the Eiffel Tower located in Paris?</span>
      <span class="chip" onclick="askExample(this)">What are the 10 most populated countries?</span>
      <span class="chip" onclick="askExample(this)">Who designed the Python programming language?</span>
    </div>
  </details>

  <div class="model-row">
    <label for="model">Model:</label>
    <select id="model"></select>
    <label for="dataset" style="margin-left:1rem">Dataset:</label>
    <select id="dataset">
      <option value="https://text2sparql.aksw.org/2026/dbpedia/">DBpedia 2026</option>
      <option value="https://text2sparql.aksw.org/2025/dbpedia/">DBpedia 2025</option>
      <option value="https://text2sparql.aksw.org/2026/corporate/">Corporate CK25</option>
    </select>
  </div>

  <div class="input-row">
    <input type="text" id="q" placeholder="Ask a question..." autofocus
           onkeydown="if(event.key==='Enter') run()">
    <button id="btn" onclick="run()">Ask</button>
  </div>

  <div id="steps" class="steps"></div>
</div>

<script>
// Populate model dropdown
fetch('/models').then(r => r.json()).then(d => {
  const sel = document.getElementById('model');
  for (const m of d.models) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.label;
    if (m.id === d.default) opt.selected = true;
    sel.appendChild(opt);
  }
});

function askExample(el) {
  document.getElementById('q').value = el.textContent;
  run();
}

function addStep(id, label, active) {
  const div = document.createElement('div');
  div.className = 'step' + (active ? ' active' : '');
  div.id = 'step-' + id;
  div.innerHTML = `
    <div class="step-label">${active ? '<span class="spinner"></span>' : ''} ${label}</div>
    <div class="step-body" id="body-${id}"></div>
  `;
  document.getElementById('steps').appendChild(div);
  return div;
}

function finishStep(id) {
  const el = document.getElementById('step-' + id);
  if (!el) return;
  el.classList.remove('active');
  el.classList.add('done');
  const spinner = el.querySelector('.spinner');
  if (spinner) spinner.remove();
}

function setBody(id, text, cls) {
  const el = document.getElementById('body-' + id);
  if (!el) return;
  el.textContent = text;
  if (cls) el.classList.add(cls);
}

function formatEntities(data) {
  let out = '';
  for (const [mention, candidates] of Object.entries(data)) {
    out += `${mention}\\n`;
    for (const c of candidates) {
      const badge = c.source === 'redis' ? '(Redis)' : '(heuristic)';
      out += `  ${c.uri}  score=${c.score}  ${badge}\\n`;
    }
  }
  return out || '(none)';
}

function formatOntology(data) {
  let out = '';
  for (const [concept, results] of Object.entries(data)) {
    out += `${concept}\\n`;
    for (const r of results) {
      out += `  ${r.uri}  (${r.type}, score=${r.score})\\n`;
    }
  }
  return out || '(none)';
}

let currentStep = null;

function run() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const btn = document.getElementById('btn');
  btn.disabled = true;
  document.getElementById('steps').innerHTML = '';
  currentStep = null;

  const model = document.getElementById('model').value;
  const dataset = document.getElementById('dataset').value;
  const es = new EventSource('/stream?question=' + encodeURIComponent(q) + '&model=' + encodeURIComponent(model) + '&dataset=' + encodeURIComponent(dataset));

  es.addEventListener('step_start', e => {
    const d = JSON.parse(e.data).data;
    if (currentStep) finishStep(currentStep);
    currentStep = d.step;
    addStep(d.step, d.label, true);
  });

  es.addEventListener('analyse', e => {
    const d = JSON.parse(e.data).data;
    setBody('analyse',
      `Entities: ${(d.entities||[]).join(', ')}\\n` +
      `Answer type: ${d.answer_type}\\n` +
      `Concepts: ${(d.concepts||[]).join(', ')}`
    );
  });

  es.addEventListener('entity_linking', e => {
    const d = JSON.parse(e.data).data;
    setBody('entity_linking', formatEntities(d));
  });

  es.addEventListener('ontology_lookup', e => {
    const d = JSON.parse(e.data).data;
    setBody('ontology_lookup', formatOntology(d));
  });

  es.addEventListener('sparql', e => {
    const d = JSON.parse(e.data).data;
    setBody('sparql_generation', d.query, 'sparql');
  });

  es.addEventListener('execution', e => {
    const d = JSON.parse(e.data).data;
    setBody(currentStep, formatExecution(d.result, d.problem));
  });

  es.addEventListener('revision', e => {
    const d = JSON.parse(e.data).data;
    setBody(currentStep, d.query, 'sparql');
  });

  es.addEventListener('done', e => {
    if (currentStep) finishStep(currentStep);
    es.close();
    btn.disabled = false;
  });

  es.onerror = () => {
    es.close();
    btn.disabled = false;
    if (currentStep) finishStep(currentStep);
  };
}

function formatExecution(data, problem) {
  const disclaimer = '(!) Results from the DBpedia SPARQL endpoint.\\n\\n';
  if (!data) return '';
  if (data.type === 'error') {
    return disclaimer + 'Error: ' + data.message;
  }
  if (data.type === 'ask') {
    return disclaimer + 'Result: ' + (data.result ? 'YES' : 'NO');
  }
  if (data.type === 'select') {
    if (data.rows.length === 0) {
      let out = problem ? `Problem: ${problem}` : '(no results)';
      return disclaimer + out;
    }
    const vars = data.vars;
    let out = disclaimer;
    out += vars.join('  |  ') + '\\n';
    out += vars.map(v => '-'.repeat(v.length + 4)).join('') + '\\n';
    for (const row of data.rows) {
      out += vars.map(v => row[v] || '').join('  |  ') + '\\n';
    }
    if (data.total > 20) out += `\\n... and ${data.total - 20} more rows`;
    return out;
  }
  return disclaimer + JSON.stringify(data, null, 2);
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>
"""
