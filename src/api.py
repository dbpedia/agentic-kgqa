"""text2sparql API endpoint for the KGQA agent, with streaming UI."""

import json
import logging

import fastapi
from fastapi.responses import HTMLResponse, StreamingResponse

from src.agent import KGQAAgent, MODELS, DEFAULT_MODEL

# Slug -> OpenRouter model ID mapping for path-based endpoints
MODEL_SLUGS = {m["id"].split("/")[-1]: m["id"] for m in MODELS}
from src.evaluate import evaluate_stream, evaluate_all_stream, list_results, load_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = fastapi.FastAPI(title="TEXT2SPARQL API - Agentic KGQA")

KNOWN_DATASETS = [
    "https://text2sparql.aksw.org/2025/dbpedia/",
    "https://text2sparql.aksw.org/2026/dbpedia/",
]

agent = None


@app.on_event("startup")
async def startup():
    global agent
    agent = KGQAAgent()
    logger.info("KGQA Agent initialised.")


# --- Challenge API endpoint (unchanged) ---


@app.get("/answer")
async def get_answer(question: str, dataset: str):
    if dataset not in KNOWN_DATASETS:
        raise fastapi.HTTPException(404, "Unknown dataset")

    sparql = agent.answer(question)

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


def _event_stream(question: str, model: str | None = None):
    """Generator that yields SSE events from the agent pipeline."""
    for event_type, data in agent.answer_stream(question, model=model):
        payload = json.dumps({"type": event_type, "data": data}, default=str)
        yield f"event: {event_type}\ndata: {payload}\n\n"


@app.get("/stream")
async def stream_answer(question: str, model: str | None = None):
    return StreamingResponse(
        _event_stream(question, model=model),
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
    sparql = agent.answer(question, model=model_id)
    return {"dataset": dataset, "question": question, "query": sparql, "model": model_id}


@app.get("/m/{model_slug}/stream")
async def model_stream_answer(model_slug: str, question: str):
    model_id = _resolve_model_slug(model_slug)
    return StreamingResponse(
        _event_stream(question, model=model_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Evaluation endpoint (SSE) ---


def _eval_stream(n: int, model: str | None = None, shuffle: bool = True):
    """Generator that yields SSE events from the evaluation pipeline."""
    if model == "all":
        gen = evaluate_all_stream(agent, models=MODELS, n=n, shuffle=shuffle)
    else:
        gen = evaluate_stream(agent, n=n, model=model, shuffle=shuffle)
    for event_type, data in gen:
        payload = json.dumps({"type": event_type, "data": data}, default=str)
        yield f"event: {event_type}\ndata: {payload}\n\n"


@app.get("/evaluate")
async def evaluate(n: int = 10, model: str | None = None, shuffle: bool = True):
    return StreamingResponse(
        _eval_stream(n, model=model, shuffle=shuffle),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Saved evaluation results ---


@app.get("/eval-results")
async def get_eval_results():
    return list_results()


@app.get("/eval-results/{result_id}")
async def get_eval_result(result_id: str):
    result = load_result(result_id)
    if result is None:
        raise fastapi.HTTPException(404, "Result not found")
    return result


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
  input[type=number] {
    width: 70px; padding: .45rem .6rem; border-radius: 6px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--text); font-size: .85rem; font-family: var(--font);
    outline: none; text-align: center;
  }
  input[type=number]:focus { border-color: var(--accent); }

  /* Tabs */
  .tabs { display: flex; gap: 0; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border); }
  .tab {
    padding: .55rem 1.2rem; font-size: .85rem; font-weight: 600; cursor: pointer;
    color: var(--muted); border-bottom: 2px solid transparent; transition: all .15s;
    background: none; border-radius: 0; border-top: none; border-left: none; border-right: none;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* Eval results */
  .eval-summary {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: .5rem;
    margin-bottom: 1rem;
  }
  .eval-stat {
    background: var(--surface); border-radius: 8px; padding: .6rem .8rem;
    text-align: center; border: 1px solid var(--border);
  }
  .eval-stat .val { font-size: 1.4rem; font-weight: 700; }
  .eval-stat .lbl { font-size: .7rem; color: var(--muted); margin-top: .15rem; }
  .eval-table { width: 100%; border-collapse: collapse; font-size: .8rem; }
  .eval-table th {
    text-align: left; padding: .5rem .6rem; border-bottom: 1px solid var(--border);
    color: var(--muted); font-weight: 600; font-size: .75rem; text-transform: uppercase;
  }
  .eval-table td { padding: .5rem .6rem; border-bottom: 1px solid var(--border); vertical-align: top; }
  .eval-table tr:hover td { background: var(--surface); }
  .badge {
    display: inline-block; padding: .15rem .45rem; border-radius: 4px;
    font-size: .7rem; font-weight: 600;
  }
  .badge.pass { background: #16a34a22; color: var(--green); }
  .badge.fail { background: #ef444422; color: #ef4444; }
  .eval-detail { margin-top: .3rem; font-family: var(--mono); font-size: .72rem; color: var(--muted); white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
  .eval-progress { color: var(--muted); font-size: .85rem; margin-bottom: 1rem; display: flex; align-items: center; gap: .5rem; }
  .eval-progress .spinner { width: 14px; height: 14px; border: 2px solid var(--border); border-top-color: var(--amber); border-radius: 50%; animation: spin .6s linear infinite; display: inline-block; }
  .past-runs-list { display: flex; flex-wrap: wrap; gap: .4rem; }
  .past-run {
    padding: .3rem .65rem; border-radius: 6px; font-size: .75rem;
    background: var(--surface); border: 1px solid var(--border);
    color: var(--muted); cursor: pointer; transition: border-color .15s;
    display: inline-flex; align-items: center; gap: .35rem;
  }
  .past-run:hover { border-color: var(--accent); color: var(--text); }
  .past-run .pr-acc { font-weight: 700; color: var(--accent); }
  .saved-badge {
    display: inline-block; padding: .15rem .4rem; border-radius: 4px;
    font-size: .7rem; font-weight: 600; background: #16a34a22; color: var(--green);
    margin-left: .5rem;
  }
</style>
</head>
<body>
<div class="container">
  <h1>Agentic KGQA</h1>
  <p class="subtitle">Natural language to SPARQL against DBpedia 2015-10</p>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('ask')">Ask</button>
    <button class="tab" onclick="switchTab('eval')">Evaluate</button>
    <button class="tab" onclick="switchTab('history')">History</button>
  </div>

  <!-- Ask tab -->
  <div id="tab-ask" class="tab-content active">
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
    </div>

    <div class="input-row">
      <input type="text" id="q" placeholder="Ask a question about DBpedia..." autofocus
             onkeydown="if(event.key==='Enter') run()">
      <button id="btn" onclick="run()">Ask</button>
    </div>

    <div id="steps" class="steps"></div>
  </div>

  <!-- Evaluate tab -->
  <div id="tab-eval" class="tab-content">
    <div id="past-runs" style="margin-bottom:1rem"></div>
    <div class="input-row">
      <label style="color:var(--muted);font-size:.85rem;white-space:nowrap;align-self:center">Sample:</label>
      <input type="number" id="eval-n" value="10" min="1" max="100">
      <label style="color:var(--muted);font-size:.85rem;white-space:nowrap;align-self:center">questions</label>
      <select id="eval-model"></select>
      <button id="eval-btn" onclick="runEval()">Run</button>
      <button id="eval-full-btn" onclick="runEval(true)" style="background:var(--green)">Full benchmark (100)</button>
    </div>
    <div id="eval-progress"></div>
    <!-- Single-model results -->
    <div id="eval-single" style="display:none">
      <div id="eval-summary"></div>
      <table class="eval-table" id="eval-table">
        <thead><tr><th>#</th><th>Question</th><th>Nodes</th><th>Preds</th><th>Inner</th><th>Outer</th><th>Match</th></tr></thead>
        <tbody id="eval-body"></tbody>
      </table>
    </div>
    <!-- All-models comparison -->
    <div id="eval-all" style="display:none">
      <canvas id="eval-chart" width="760" height="340" style="margin-bottom:1.5rem"></canvas>
      <div id="eval-model-cards"></div>
    </div>
  </div>

  <!-- History tab -->
  <div id="tab-history" class="tab-content">
    <p style="color:var(--muted);font-size:.85rem;margin-bottom:1rem">DeepSeek V3.2 performance across prompt versions (full benchmark, n=100).</p>
    <canvas id="history-chart" style="width:100%;max-width:760px;height:380px;margin-bottom:1.5rem"></canvas>
    <div id="history-table-wrap"></div>
  </div>
</div>

<script>
// Populate model dropdowns
fetch('/models').then(r => r.json()).then(d => {
  const sel = document.getElementById('model');
  const evalSel = document.getElementById('eval-model');
  for (const m of d.models) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.label;
    if (m.id === d.default) opt.selected = true;
    sel.appendChild(opt);
    evalSel.appendChild(opt.cloneNode(true));
  }
  // Add "All models" option to eval selector
  const allOpt = document.createElement('option');
  allOpt.value = 'all';
  allOpt.textContent = 'All models (comparison)';
  evalSel.appendChild(allOpt);
});
// Load past runs on startup
refreshPastRuns();

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
  const es = new EventSource('/stream?question=' + encodeURIComponent(q) + '&model=' + encodeURIComponent(model));

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
  const disclaimer = '(!) Results from local DBpedia 2015-10 endpoint.\\n\\n';
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

// --- Tab switching ---
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[onclick="switchTab('${tab}')"]`).classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
  if (tab === 'eval') refreshPastRuns();
  if (tab === 'history') loadHistory();
}

// --- Past runs ---
function refreshPastRuns() {
  fetch('/eval-results').then(r => r.json()).then(runs => {
    const el = document.getElementById('past-runs');
    if (!runs.length) { el.innerHTML = ''; return; }
    let html = '<details open><summary style="font-size:.8rem;color:var(--muted);cursor:pointer;margin-bottom:.5rem">Past runs</summary><div class="past-runs-list">';
    for (const r of runs.slice(0, 20)) {
      const ts = r.timestamp ? new Date(r.timestamp).toLocaleString() : r.id;
      const label = r.mode === 'all'
        ? `All (${r.models_count} models, n=${r.n})`
        : `${(r.model || '').split('/').pop()} (n=${r.n})`;
      const acc = r.accuracy != null ? `<span class="pr-acc">${r.accuracy}%</span>` : '';
      const ver = r.prompt_version ? `<span style="font-size:.65rem;color:var(--green);font-weight:700">${r.prompt_version}</span>` : '';
      html += `<span class="past-run" onclick="loadSavedResult('${r.id}')" title="${ts}">${ver} ${label} ${acc}</span>`;
    }
    html += '</div></details>';
    el.innerHTML = html;
  });
}

function loadSavedResult(id) {
  const prog = document.getElementById('eval-progress');
  prog.innerHTML = '<span class="spinner"></span> Loading saved result...';
  document.getElementById('eval-single').style.display = 'none';
  document.getElementById('eval-all').style.display = 'none';

  fetch('/eval-results/' + encodeURIComponent(id)).then(r => r.json()).then(data => {
    prog.innerHTML = '';
    if (data.mode === 'all') {
      renderSavedAll(data);
    } else {
      renderSavedSingle(data);
    }
  }).catch(() => {
    prog.innerHTML = '<span style="color:#ef4444">Failed to load result.</span>';
  });
}

function renderSavedSingle(data) {
  document.getElementById('eval-single').style.display = 'block';
  document.getElementById('eval-summary').innerHTML = makeSummaryCards(data.summary)
    + `<span class="saved-badge">Saved ${new Date(data.timestamp).toLocaleString()}</span>`;
  const tbody = document.getElementById('eval-body');
  tbody.innerHTML = '';
  for (const q of (data.questions || [])) {
    tbody.appendChild(makeEvalRow(q));
  }
}

function renderSavedAll(data) {
  document.getElementById('eval-all').style.display = 'block';
  const cards = document.getElementById('eval-model-cards');
  cards.innerHTML = `<div style="margin-bottom:.6rem"><span class="saved-badge">Saved ${new Date(data.timestamp).toLocaleString()}</span></div>`;
  const models = data.models || [];
  for (let mi = 0; mi < models.length; mi++) {
    const m = models[mi];
    const color = CHART_COLORS[mi % CHART_COLORS.length];
    cards.innerHTML += `
      <div style="background:var(--surface);border:1px solid var(--border);border-left:3px solid ${color};border-radius:8px;padding:.8rem 1rem;margin-bottom:.6rem">
        <div style="font-weight:600;font-size:.9rem;margin-bottom:.4rem">${escHtml(m.model_label)}</div>
        <div style="display:flex;gap:1.2rem;font-size:.8rem;color:var(--muted)">
          <span>Match: <b style="color:var(--accent)">${m.summary.accuracy}%</b></span>
          <span>Nodes: <b>${m.summary.bgp_nodes_accuracy ?? m.summary.bgp_accuracy ?? 0}%</b></span>
          <span>Preds: <b>${m.summary.bgp_predicates_accuracy ?? m.summary.bgp_accuracy ?? 0}%</b></span>
          <span>Inner: <b>${m.summary.inner_ops_accuracy}%</b></span>
          <span>Outer: <b>${m.summary.outer_ops_accuracy}%</b></span>
        </div>
      </div>`;
  }
  drawComparisonChart(models);
}

// --- Evaluation ---
const CHART_COLORS = ['#6366f1','#22c55e','#f59e0b','#3b82f6','#ef4444','#ec4899'];

function runEval(full) {
  const n = full ? 100 : (parseInt(document.getElementById('eval-n').value) || 10);
  const shuffle = !full;
  const model = document.getElementById('eval-model').value;
  const isAll = model === 'all';
  const btn = document.getElementById('eval-btn');
  const fullBtn = document.getElementById('eval-full-btn');
  btn.disabled = true;
  fullBtn.disabled = true;

  // Reset
  document.getElementById('eval-single').style.display = 'none';
  document.getElementById('eval-all').style.display = 'none';
  document.getElementById('eval-summary').innerHTML = '';
  document.getElementById('eval-body').innerHTML = '';
  document.getElementById('eval-model-cards').innerHTML = '';

  const prog = document.getElementById('eval-progress');
  prog.innerHTML = '<span class="spinner"></span> Starting ' + (full ? 'full benchmark' : 'evaluation') + '...';

  const es = new EventSource('/evaluate?n=' + n + '&model=' + encodeURIComponent(model) + '&shuffle=' + shuffle);

  const done = () => { btn.disabled = false; fullBtn.disabled = false; };
  if (isAll) {
    runEvalAll(es, done, prog);
  } else {
    runEvalSingle(es, done, prog);
  }
}

function runEvalSingle(es, done, prog) {
  document.getElementById('eval-single').style.display = 'block';

  es.addEventListener('eval_start', e => {
    const d = JSON.parse(e.data).data;
    const ver = d.prompt_version ? ` [${d.prompt_version}]` : '';
    prog.innerHTML = `<span class="spinner"></span> Evaluating ${d.total} questions with ${d.model}${ver} (judge: ${d.judge})...`;
  });

  es.addEventListener('eval_progress', e => {
    const d = JSON.parse(e.data).data;
    prog.innerHTML = `<span class="spinner"></span> Question ${d.index}/${d.total}: ${escHtml(d.question).slice(0,60)}...`;
  });

  es.addEventListener('eval_question', e => {
    const d = JSON.parse(e.data).data;
    document.getElementById('eval-body').appendChild(makeEvalRow(d));
  });

  es.addEventListener('eval_done', e => {
    const d = JSON.parse(e.data).data;
    prog.innerHTML = '';
    document.getElementById('eval-summary').innerHTML = makeSummaryCards(d);
  });

  es.addEventListener('eval_saved', e => {
    document.getElementById('eval-summary').innerHTML += `<span class="saved-badge">Saved</span>`;
    refreshPastRuns();
    es.close();
    done();
  });

  es.onerror = () => { es.close(); done(); prog.innerHTML = '<span style="color:#ef4444">Error — check server logs.</span>'; };
}

function runEvalAll(es, done, prog) {
  document.getElementById('eval-all').style.display = 'block';
  let currentModelLabel = '';
  let currentModelIdx = 0;
  let totalModels = 0;

  es.addEventListener('eval_all_start', e => {
    const d = JSON.parse(e.data).data;
    totalModels = d.models.length;
    const ver = d.prompt_version ? ` [${d.prompt_version}]` : '';
    prog.innerHTML = `<span class="spinner"></span> Comparing ${totalModels} models${ver} on ${d.total_questions} questions (judge: ${d.judge})...`;
  });

  es.addEventListener('eval_model_start', e => {
    const d = JSON.parse(e.data).data;
    currentModelLabel = d.model_label;
    currentModelIdx = d.model_index;
    prog.innerHTML = `<span class="spinner"></span> Model ${d.model_index}/${d.total_models}: ${d.model_label}...`;
  });

  es.addEventListener('eval_progress', e => {
    const d = JSON.parse(e.data).data;
    prog.innerHTML = `<span class="spinner"></span> ${currentModelLabel} — question ${d.index}/${d.total}: ${escHtml(d.question).slice(0,50)}...`;
  });

  es.addEventListener('eval_model_done', e => {
    const d = JSON.parse(e.data).data;
    const cards = document.getElementById('eval-model-cards');
    const color = CHART_COLORS[(currentModelIdx - 1) % CHART_COLORS.length];
    cards.innerHTML += `
      <div style="background:var(--surface);border:1px solid var(--border);border-left:3px solid ${color};border-radius:8px;padding:.8rem 1rem;margin-bottom:.6rem">
        <div style="font-weight:600;font-size:.9rem;margin-bottom:.4rem">${escHtml(d.model_label)}</div>
        <div style="display:flex;gap:1.2rem;font-size:.8rem;color:var(--muted)">
          <span>Match: <b style="color:var(--accent)">${d.summary.accuracy}%</b></span>
          <span>BGP: <b>${d.summary.bgp_accuracy}%</b></span>
          <span>Inner: <b>${d.summary.inner_ops_accuracy}%</b></span>
          <span>Outer: <b>${d.summary.outer_ops_accuracy}%</b></span>
        </div>
      </div>`;
  });

  es.addEventListener('eval_all_done', e => {
    const d = JSON.parse(e.data).data;
    prog.innerHTML = '';
    drawComparisonChart(d.models);
  });

  es.addEventListener('eval_saved', e => {
    const cards = document.getElementById('eval-model-cards');
    cards.innerHTML = `<div style="margin-bottom:.6rem"><span class="saved-badge">Saved</span></div>` + cards.innerHTML;
    refreshPastRuns();
    es.close();
    done();
  });

  es.onerror = () => { es.close(); done(); prog.innerHTML = '<span style="color:#ef4444">Error — check server logs.</span>'; };
}

function makeEvalRow(d) {
  const c = d.comparison;
  const badge = (v) => `<span class="badge ${v ? 'pass' : 'fail'}">${v ? 'PASS' : 'FAIL'}</span>`;
  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${d.question_id}</td>
    <td>
      ${escHtml(d.question)}
      <details><summary style="font-size:.7rem;color:var(--muted);cursor:pointer;margin-top:.3rem">Details</summary>
        <div class="eval-detail"><b>Expected:</b>\\n${escHtml(d.expected)}\\n\\n<b>Generated:</b>\\n${escHtml(d.generated)}\\n\\n<b>Explanation:</b> ${escHtml(c.explanation || '')}</div>
      </details>
    </td>
    <td>${badge(c.bgp_nodes ?? c.bgp)}</td>
    <td>${badge(c.bgp_predicates ?? c.bgp)}</td>
    <td>${badge(c.inner_ops)}</td>
    <td>${badge(c.outer_ops)}</td>
    <td>${badge(c.match)}</td>`;
  return row;
}

function makeSummaryCards(d) {
  // Handle both old (bgp_accuracy) and new (bgp_nodes/bgp_predicates) formats
  const nodes = d.bgp_nodes_accuracy != null ? d.bgp_nodes_accuracy : d.bgp_accuracy || 0;
  const preds = d.bgp_predicates_accuracy != null ? d.bgp_predicates_accuracy : d.bgp_accuracy || 0;
  return `<div class="eval-summary">
    <div class="eval-stat"><div class="val" style="color:var(--accent)">${d.accuracy}%</div><div class="lbl">Overall match</div></div>
    <div class="eval-stat"><div class="val">${nodes}%</div><div class="lbl">BGP Nodes</div></div>
    <div class="eval-stat"><div class="val">${preds}%</div><div class="lbl">BGP Predicates</div></div>
    <div class="eval-stat"><div class="val">${d.inner_ops_accuracy}%</div><div class="lbl">Inner ops</div></div>
    <div class="eval-stat"><div class="val">${d.outer_ops_accuracy}%</div><div class="lbl">Outer ops</div></div>
  </div>`;
}

// --- Chart rendering (pure canvas, no dependencies) ---
function drawComparisonChart(models) {
  const canvas = document.getElementById('eval-chart');
  // Use fixed logical size — canvas CSS may report 0 if recently shown
  const W = 760, H = 380;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const axes = ['accuracy', 'bgp_nodes_accuracy', 'bgp_predicates_accuracy', 'inner_ops_accuracy', 'outer_ops_accuracy'];
  const axisLabels = ['Overall', 'BGP Nodes', 'BGP Preds', 'Inner ops', 'Outer ops'];
  const n = models.length;

  // Layout constants
  const leftPad = 45, rightPad = 15;
  const chartTop = 15, chartBot = H - 80; // room for x-labels + legend
  const chartH = chartBot - chartTop;
  const chartW = W - leftPad - rightPad;

  const groupGap = 24; // px between groups
  const barGap = 2;    // px between bars in a group
  const groupCount = axes.length;
  const totalGaps = (groupCount - 1) * groupGap;
  const groupW = (chartW - totalGaps) / groupCount;
  const barW = Math.max(6, Math.min(24, (groupW - (n - 1) * barGap) / n));
  const barsW = n * barW + (n - 1) * barGap; // actual width of bars in group

  // Y-axis grid + labels
  ctx.textBaseline = 'middle';
  for (let pct = 0; pct <= 100; pct += 25) {
    const y = chartBot - (pct / 100) * chartH;
    ctx.strokeStyle = '#2a2d3a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(leftPad, y); ctx.lineTo(W - rightPad, y); ctx.stroke();
    ctx.fillStyle = '#8b8d97'; ctx.font = '11px system-ui'; ctx.textAlign = 'right';
    ctx.fillText(pct + '%', leftPad - 6, y);
  }

  // Draw groups
  for (let gi = 0; gi < groupCount; gi++) {
    const groupX = leftPad + gi * (groupW + groupGap);
    const barsStart = groupX + (groupW - barsW) / 2; // centre bars in group

    // X-axis label
    ctx.fillStyle = '#e4e4e7'; ctx.textAlign = 'center'; ctx.font = 'bold 11px system-ui';
    ctx.textBaseline = 'top';
    ctx.fillText(axisLabels[gi], groupX + groupW / 2, chartBot + 8);

    // Bars
    for (let mi = 0; mi < n; mi++) {
      const s = models[mi].summary;
      let val = s[axes[gi]];
      // Fallback: old results have bgp_accuracy instead of bgp_nodes/bgp_predicates
      if (val == null && axes[gi] === 'bgp_nodes_accuracy') val = s.bgp_accuracy;
      if (val == null && axes[gi] === 'bgp_predicates_accuracy') val = s.bgp_accuracy;
      val = val || 0;
      const x = barsStart + mi * (barW + barGap);
      const h = Math.max(1, (val / 100) * chartH);
      const color = CHART_COLORS[mi % CHART_COLORS.length];

      // Rounded-top bar
      ctx.fillStyle = color;
      const r = Math.min(3, barW / 4);
      const bx = x, bw = barW, by = chartBot - h, bh = h;
      ctx.beginPath();
      ctx.moveTo(bx + r, by);
      ctx.lineTo(bx + bw - r, by);
      ctx.quadraticCurveTo(bx + bw, by, bx + bw, by + r);
      ctx.lineTo(bx + bw, chartBot);
      ctx.lineTo(bx, chartBot);
      ctx.lineTo(bx, by + r);
      ctx.quadraticCurveTo(bx, by, bx + r, by);
      ctx.fill();

      // Value on top of bar
      ctx.fillStyle = '#fff'; ctx.font = 'bold 9px system-ui'; ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(Math.round(val), x + bw / 2, by - 2);
    }
  }

  // Legend (wrapping rows)
  ctx.textBaseline = 'middle';
  const legendTop = chartBot + 30;
  const legendItemH = 16;
  let lx = leftPad, ly = legendTop;
  ctx.font = '11px system-ui';
  for (let mi = 0; mi < n; mi++) {
    const color = CHART_COLORS[mi % CHART_COLORS.length];
    const label = models[mi].model_label;
    const itemW = 14 + ctx.measureText(label).width + 20;
    if (lx + itemW > W - rightPad && lx > leftPad) {
      lx = leftPad; ly += legendItemH;
    }
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly - 5, 10, 10);
    ctx.fillStyle = '#e4e4e7'; ctx.textAlign = 'left';
    ctx.fillText(label, lx + 14, ly);
    lx += itemW;
  }
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// --- History tab ---
const HISTORY_COLORS = {
  accuracy: '#6366f1',
  bgp_nodes_accuracy: '#22c55e',
  bgp_predicates_accuracy: '#f59e0b',
  inner_ops_accuracy: '#3b82f6',
  outer_ops_accuracy: '#ec4899',
};
const HISTORY_LABELS = {
  accuracy: 'Overall',
  bgp_nodes_accuracy: 'BGP Nodes',
  bgp_predicates_accuracy: 'BGP Preds',
  inner_ops_accuracy: 'Inner ops',
  outer_ops_accuracy: 'Outer ops',
};

function loadHistory() {
  const DS_MATCH = 'deepseek';
  fetch('/eval-results').then(r => r.json()).then(runs => {
    // Include both single-model DeepSeek runs AND all-models runs (n>=50) with a version
    const eligible = runs.filter(r => r.n >= 50 && r.prompt_version && (
      (r.mode === 'single' && r.model && r.model.toLowerCase().includes(DS_MATCH)) ||
      r.mode === 'all'
    ));
    if (eligible.length === 0) {
      document.getElementById('history-table-wrap').innerHTML = '<p style="color:var(--muted);font-size:.85rem">No DeepSeek benchmark runs found yet.</p>';
      return;
    }
    // Load full data
    Promise.all(eligible.map(r => fetch('/eval-results/' + encodeURIComponent(r.id)).then(x => x.json())))
      .then(fullRuns => {
        // Extract DeepSeek data from all runs
        const byVersion = {};
        for (const run of fullRuns) {
          const pv = run.prompt_version;
          if (run.mode === 'single' && run.model && run.model.toLowerCase().includes(DS_MATCH)) {
            // Direct single-model DeepSeek run
            if (!byVersion[pv] || run.timestamp > byVersion[pv].timestamp) {
              byVersion[pv] = run;
            }
          } else if (run.mode === 'all' && run.models) {
            // Extract DeepSeek from all-models run
            const dsModel = run.models.find(m => m.model_id && m.model_id.toLowerCase().includes(DS_MATCH));
            if (dsModel) {
              const synthetic = {
                prompt_version: pv,
                model: dsModel.model_id,
                timestamp: run.timestamp,
                summary: dsModel.summary,
              };
              if (!byVersion[pv] || run.timestamp > byVersion[pv].timestamp) {
                byVersion[pv] = synthetic;
              }
            }
          }
        }
        const points = Object.values(byVersion).sort((a, b) =>
          (a.prompt_version || '').localeCompare(b.prompt_version || '')
        );
        drawHistoryChart(points);
        drawHistoryTable(points);
      });
  });
}

function drawHistoryChart(points) {
  const canvas = document.getElementById('history-chart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  if (points.length === 0) return;

  const metrics = Object.keys(HISTORY_COLORS);
  const chartLeft = 50, chartRight = W - 20, chartTop = 20, chartBot = H - 70;
  const chartW = chartRight - chartLeft, chartH = chartBot - chartTop;

  // X axis: one position per point (version + model)
  const xLabels = points.map(p => p.prompt_version || '?');
  const xStep = chartW / Math.max(points.length - 1, 1);

  // Grid
  ctx.strokeStyle = '#2a2d3a'; ctx.lineWidth = 1;
  ctx.font = '11px system-ui'; ctx.fillStyle = '#8b8d97';
  for (let pct = 0; pct <= 100; pct += 25) {
    const y = chartBot - (pct / 100) * chartH;
    ctx.beginPath(); ctx.moveTo(chartLeft, y); ctx.lineTo(chartRight, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(pct + '%', chartLeft - 8, y + 4);
  }

  // X labels
  ctx.textAlign = 'center'; ctx.fillStyle = '#e4e4e7'; ctx.font = '10px system-ui';
  for (let i = 0; i < points.length; i++) {
    const x = chartLeft + i * xStep;
    ctx.save();
    ctx.translate(x, chartBot + 10);
    ctx.rotate(-0.4);
    ctx.textAlign = 'right';
    ctx.fillText(xLabels[i], 0, 0);
    ctx.restore();
  }

  // Lines + dots for each metric
  for (const metric of metrics) {
    const color = HISTORY_COLORS[metric];
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.fillStyle = color;

    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const x = chartLeft + i * xStep;
      // Handle both old (bgp_accuracy) and new (bgp_nodes_accuracy) formats
      const s = points[i].summary || {};
      let val = s[metric];
      if (val == null && metric === 'bgp_nodes_accuracy') val = s.bgp_accuracy;
      if (val == null && metric === 'bgp_predicates_accuracy') val = s.bgp_accuracy;
      if (val == null) val = 0;
      const y = chartBot - (val / 100) * chartH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Dots with value labels
    for (let i = 0; i < points.length; i++) {
      const x = chartLeft + i * xStep;
      const s = points[i].summary || {};
      let val = s[metric];
      if (val == null && metric === 'bgp_nodes_accuracy') val = s.bgp_accuracy;
      if (val == null && metric === 'bgp_predicates_accuracy') val = s.bgp_accuracy;
      if (val == null) val = 0;
      const y = chartBot - (val / 100) * chartH;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 9px system-ui'; ctx.textAlign = 'center';
      ctx.fillText(Math.round(val), x, y - 8);
      ctx.fillStyle = color;
    }
  }

  // Legend
  const legendY = H - 10;
  let lx = chartLeft;
  ctx.font = '11px system-ui';
  for (const metric of metrics) {
    ctx.fillStyle = HISTORY_COLORS[metric];
    ctx.fillRect(lx, legendY - 8, 10, 10);
    ctx.fillStyle = '#e4e4e7'; ctx.textAlign = 'left';
    const label = HISTORY_LABELS[metric];
    ctx.fillText(label, lx + 14, legendY);
    lx += ctx.measureText(label).width + 28;
  }
}

function drawHistoryTable(points) {
  let html = '<table class="eval-table"><thead><tr><th>Version</th><th>Model</th><th>Date</th><th>Overall</th><th>Nodes</th><th>Preds</th><th>Inner</th><th>Outer</th></tr></thead><tbody>';
  for (const p of points) {
    const s = p.summary || {};
    const model = (p.model || '').split('/').pop();
    const date = p.timestamp ? new Date(p.timestamp).toLocaleDateString() : '';
    const nodes = s.bgp_nodes_accuracy ?? s.bgp_accuracy ?? '—';
    const preds = s.bgp_predicates_accuracy ?? s.bgp_accuracy ?? '—';
    html += `<tr>
      <td><b style="color:var(--green)">${p.prompt_version || '?'}</b></td>
      <td>${escHtml(model)}</td>
      <td style="color:var(--muted)">${date}</td>
      <td><b style="color:var(--accent)">${s.accuracy ?? '—'}%</b></td>
      <td>${nodes}%</td>
      <td>${preds}%</td>
      <td>${s.inner_ops_accuracy ?? '—'}%</td>
      <td>${s.outer_ops_accuracy ?? '—'}%</td>
    </tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('history-table-wrap').innerHTML = html;
}
</script>
</body>
</html>
"""
