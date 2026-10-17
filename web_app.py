import os
import sys
from typing import Optional, List
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Ensure repo root is in path
sys.path.insert(0, os.path.abspath("."))
from src.predict import get_predictor

app = FastAPI(
    title="Shakespeare Next-Word Prediction Interactive UI",
    description="Interactive Web UI and API for Word-Level LSTM Model"
)

# Initialize predictor singleton
predictor = get_predictor()

class PredictRequest(BaseModel):
    prompt: str
    top_k: int = 5
    filter_unk: bool = True
    temperature: float = 1.0

class GenerateRequest(BaseModel):
    prompt: str
    num_words: int = 15
    temperature: float = 0.8
    filter_unk: bool = True
    strategy: str = "top_k"  # 'top_k', 'temperature', 'greedy'

class VocabSearchRequest(BaseModel):
    query: str
    limit: int = 20

@app.get("/api/stats")
async def get_stats():
    return {
        "vocab_len": predictor.vocab_len,
        "seq_length": predictor.seq_length,
        "input_size": predictor.input_size,
        "hidden_size": predictor.hidden_size,
        "num_layers": predictor.num_layers,
        "dropout": predictor.dropout,
        "device": str(predictor.device),
        "is_loaded": predictor.is_loaded,
        "accuracy": "12.79%",
        "perplexity": 320.1,
        "val_loss": 5.77,
        "top_words": predictor.top_words[:25]
    }

@app.post("/api/predict")
async def predict_endpoint(req: PredictRequest):
    try:
        res = predictor.predict_next_word(
            prompt=req.prompt,
            top_k=req.top_k,
            filter_unk=req.filter_unk,
            temperature=req.temperature
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_endpoint(req: GenerateRequest):
    try:
        res = predictor.generate_sequence(
            seed_text=req.prompt,
            num_words=req.num_words,
            temperature=req.temperature,
            filter_unk=req.filter_unk,
            strategy=req.strategy
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vocab/search")
async def vocab_search_endpoint(req: VocabSearchRequest):
    matches = predictor.search_vocab(req.query, limit=req.limit)
    return {"query": req.query, "matches": matches, "count": len(matches)}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Shakespeare Next-Word Prediction | Interactive UI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;1,400&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-page: #F8FAFC;
      --bg-card: #FFFFFF;
      --border-color: #E2E8F0;
      --border-focus: #6366F1;
      --text-main: #0F172A;
      --text-muted: #64748B;
      --primary: #4F46E5;
      --primary-hover: #4338CA;
      --primary-light: #EEF2FF;
      --accent: #0EA5E9;
      --success: #10B981;
      --warning: #F59E0B;
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
      --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);
      --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -4px rgba(0,0,0,0.05);
      --radius-sm: 8px;
      --radius-md: 12px;
      --radius-lg: 16px;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-page);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    /* Container */
    .app-container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 24px;
    }

    /* Header */
    header {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 28px 32px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-sm);
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 20px;
    }

    .brand-title {
      font-size: 1.85rem;
      font-weight: 800;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 12px;
      letter-spacing: -0.02em;
    }

    .brand-desc {
      color: var(--text-muted);
      font-size: 0.95rem;
      margin-top: 4px;
    }

    .header-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      padding: 5px 12px;
      background: var(--primary-light);
      color: var(--primary);
      border: 1px solid #E0E7FF;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
    }

    /* Main Grid */
    .main-grid {
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 24px;
      margin-bottom: 24px;
    }

    @media (max-width: 900px) {
      .main-grid {
        grid-template-columns: 1fr;
      }
    }

    /* Cards */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: var(--shadow-sm);
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }

    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Preset Buttons */
    .presets-container {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 14px;
    }

    .preset-btn {
      background: #F1F5F9;
      border: 1px solid var(--border-color);
      border-radius: 9999px;
      padding: 4px 12px;
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .preset-btn:hover {
      background: var(--primary-light);
      color: var(--primary);
      border-color: #C7D2FE;
    }

    /* Input Box */
    .editor-wrapper {
      position: relative;
      margin-bottom: 16px;
    }

    textarea.prompt-input {
      width: 100%;
      height: 120px;
      background: #F8FAFC;
      border: 1.5px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 14px 16px;
      font-family: inherit;
      font-size: 1.05rem;
      color: var(--text-main);
      line-height: 1.6;
      resize: vertical;
      transition: all 0.2s ease;
    }

    textarea.prompt-input:focus {
      outline: none;
      background: #FFFFFF;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }

    /* Controls Bar */
    .controls-row {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
      background: #F8FAFC;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 12px 16px;
      margin-bottom: 16px;
      font-size: 0.88rem;
    }

    .control-item {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .control-item label {
      font-weight: 600;
      color: var(--text-muted);
    }

    input[type="range"] {
      accent-color: var(--primary);
      cursor: pointer;
    }

    /* Buttons */
    .btn {
      background: var(--primary);
      color: #FFFFFF;
      border: none;
      border-radius: var(--radius-sm);
      padding: 9px 16px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }

    .btn:hover {
      background: var(--primary-hover);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
    }

    .btn-secondary {
      background: #F1F5F9;
      color: var(--text-main);
      border: 1px solid var(--border-color);
    }

    .btn-secondary:hover {
      background: #E2E8F0;
      box-shadow: none;
    }

    /* Predictions Column */
    .predictions-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 18px;
    }

    .pred-card {
      background: #FFFFFF;
      border: 1.5px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 12px 16px;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
      overflow: hidden;
    }

    .pred-card:hover {
      border-color: var(--primary);
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
    }

    .pred-progress-bar {
      position: absolute;
      top: 0;
      bottom: 0;
      left: 0;
      background: rgba(79, 70, 229, 0.08);
      z-index: 1;
      pointer-events: none;
      transition: width 0.3s ease;
    }

    .pred-content {
      position: relative;
      z-index: 2;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .pred-word {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .pred-prob {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--primary);
      background: var(--primary-light);
      padding: 4px 10px;
      border-radius: 9999px;
    }

    /* 10-Word Sliding Window */
    .window-container {
      margin-top: 16px;
    }

    .token-chip-group {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }

    .token-chip {
      background: #F1F5F9;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 0.85rem;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;
    }

    .token-chip.known {
      background: #ECFDF5;
      border-color: #A7F3D0;
      color: #065F46;
    }

    .token-chip.oov {
      background: #FEF3C7;
      border-color: #FDE68A;
      color: #92400E;
    }

    .token-chip.padded {
      background: #F8FAFC;
      border-color: #E2E8F0;
      color: #94A3B8;
    }

    .token-id {
      font-size: 0.7rem;
      opacity: 0.75;
    }

    /* Generated Passage Result */
    .generated-passage-card {
      background: #FBFBFB;
      border: 1px solid var(--border-color);
      border-left: 5px solid var(--primary);
      border-radius: var(--radius-md);
      padding: 18px 22px;
      margin-top: 14px;
      font-family: 'Playfair Display', Georgia, serif;
      font-size: 1.15rem;
      line-height: 1.8;
      color: #1E293B;
    }

    .seed-highlight {
      color: #64748B;
    }

    .gen-highlight {
      color: var(--primary);
      font-weight: 600;
    }

    /* Bottom Info Grid */
    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-top: 24px;
    }

    .metric-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 20px;
      box-shadow: var(--shadow-sm);
    }

    .metric-label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
    }

    .metric-value {
      font-size: 1.6rem;
      font-weight: 800;
      color: var(--text-main);
    }

    /* Vocab Search */
    .vocab-search-row {
      display: flex;
      gap: 10px;
      margin-top: 10px;
    }

    .vocab-search-input {
      flex: 1;
      padding: 8px 12px;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      font-size: 0.9rem;
    }

    .vocab-results {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      max-height: 120px;
      overflow-y: auto;
    }
  </style>
</head>
<body>

<div class="app-container">
  <!-- Header -->
  <header>
    <div>
      <h1 class="brand-title">✒️ Shakespeare Next-Word Predictor</h1>
      <p class="brand-desc">Neural Language Model with 2-Layer LSTM & 10-Word Sliding Context Window</p>
    </div>
    <div class="header-badges">
      <span class="badge">📖 10,113 Vocab</span>
      <span class="badge">🧠 2-Layer LSTM</span>
      <span class="badge">🪟 10 Words Context</span>
      <span class="badge">🎯 Perplexity 320.1</span>
      <span class="badge">⚡ Light Theme</span>
    </div>
  </header>

  <!-- Presets -->
  <div class="card" style="margin-bottom: 24px; padding: 16px 20px;">
    <div style="font-size: 0.85rem; font-weight: 700; color: var(--text-muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em;">
      🎭 Quick Shakespeare Presets
    </div>
    <div class="presets-container" id="presets">
      <button class="preset-btn" onclick="setPreset('to be or not to be that is the')">"To be or not to be that is the..."</button>
      <button class="preset-btn" onclick="setPreset('shall i compare thee to a')">"Shall I compare thee to a..."</button>
      <button class="preset-btn" onclick="setPreset('all the world is a')">"All the world is a..."</button>
      <button class="preset-btn" onclick="setPreset('friends romans countrymen lend me your')">"Friends, Romans, countrymen..."</button>
      <button class="preset-btn" onclick="setPreset('now is the winter of our')">"Now is the winter of our..."</button>
      <button class="preset-btn" onclick="setPreset('some are born great some achieve')">"Some are born great..."</button>
    </div>
  </div>

  <!-- Main Grid -->
  <div class="main-grid">
    <!-- Left Column: Input & Generator -->
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">💬 Input Text Sequence</h2>
        <button class="btn btn-secondary" onclick="clearPrompt()" style="padding: 4px 10px; font-size: 0.8rem;">Clear</button>
      </div>

      <div class="editor-wrapper">
        <textarea id="promptInput" class="prompt-input" placeholder="Type words here or pick a preset above..." oninput="handleInput()">to be or not to be that is the</textarea>
      </div>

      <div class="controls-row">
        <div class="control-item">
          <label for="topK">Top-K:</label>
          <input type="range" id="topK" min="3" max="15" value="5" onchange="runPrediction()">
          <span id="topKVal" style="font-weight: 700; min-width: 18px;">5</span>
        </div>

        <div class="control-item">
          <label for="temp">Temp:</label>
          <input type="range" id="temp" min="0.1" max="1.8" step="0.05" value="1.0" onchange="runPrediction()">
          <span id="tempVal" style="font-weight: 700; min-width: 28px;">1.0</span>
        </div>

        <div class="control-item">
          <input type="checkbox" id="filterUnk" checked onchange="runPrediction()">
          <label for="filterUnk" style="cursor: pointer;">Filter &lt;UNK&gt;</label>
        </div>
      </div>

      <!-- Continuous Story Generator -->
      <div style="border-top: 1px solid var(--border-color); padding-top: 16px; margin-top: 16px;">
        <h3 class="card-title" style="font-size: 1rem; margin-bottom: 12px;">✨ Continuous Passage Generator</h3>
        <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
          <label style="font-size: 0.88rem; font-weight: 600; color: var(--text-muted);">Length:</label>
          <select id="genLen" style="padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border-color); font-family: inherit;">
            <option value="10">10 words</option>
            <option value="15" selected>15 words</option>
            <option value="25">25 words</option>
            <option value="40">40 words</option>
          </select>

          <label style="font-size: 0.88rem; font-weight: 600; color: var(--text-muted);">Sampling:</label>
          <select id="genStrategy" style="padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border-color); font-family: inherit;">
            <option value="top_k" selected>Top-K Sampling</option>
            <option value="temperature">Creative (Temperature)</option>
            <option value="greedy">Greedy (Deterministic)</option>
          </select>

          <button class="btn" onclick="generateStory()">✨ Generate Text</button>
        </div>

        <div id="genResult" style="display: none;" class="generated-passage-card"></div>
      </div>
    </div>

    <!-- Right Column: Interactive Predictions & Token Window -->
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">🎯 Next Word Candidates</h2>
        <span style="font-size: 0.8rem; color: var(--text-muted);">Click any word to append</span>
      </div>

      <div id="predictionsContainer" class="predictions-list">
        <!-- Rendered via JS -->
      </div>

      <!-- Sliding Window Debug -->
      <div class="window-container" style="border-top: 1px solid var(--border-color); padding-top: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <h3 class="card-title" style="font-size: 0.95rem;">🪟 10-Word Sliding Context Window</h3>
          <span style="font-size: 0.75rem; color: var(--text-muted);">Model Input</span>
        </div>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">
          The LSTM model reads this exact sequence of 10 tokens:
        </p>
        <div class="token-chip-group" id="tokensContainer">
          <!-- Token chips rendered here -->
        </div>
      </div>
    </div>
  </div>

  <!-- Vocabulary & Model Details Section -->
  <div class="main-grid">
    <!-- Lexicon Lookup -->
    <div class="card">
      <h3 class="card-title">📚 Shakespeare Vocabulary Lookup</h3>
      <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;">
        Search 10,113 words in the trained vocabulary.
      </p>
      <div class="vocab-search-row">
        <input type="text" id="vocabQuery" class="vocab-search-input" placeholder="Search word (e.g., love, king, night)..." oninput="searchVocab()">
      </div>
      <div class="vocab-results" id="vocabResults"></div>
    </div>

    <!-- Training Specs -->
    <div class="card">
      <h3 class="card-title">🧱 Neural Network Architecture</h3>
      <div style="font-size: 0.88rem; color: var(--text-muted); margin-top: 8px; line-height: 1.7;">
        <div>• <strong>Embedding:</strong> 10,113 vocab &rarr; 128 embedding dimensions</div>
        <div>• <strong>LSTM Backbone:</strong> 2 stacked LSTM layers, 256 hidden units, dropout 0.4</div>
        <div>• <strong>Linear Projection:</strong> 256 &rarr; 10,113 class logits</div>
        <div>• <strong>Training:</strong> Adam optimizer, weight decay 1e-4, 25 epochs on Tiny Shakespeare</div>
      </div>
    </div>
  </div>

  <!-- Metric Badges -->
  <div class="info-grid">
    <div class="metric-card">
      <div class="metric-label">Vocabulary Size</div>
      <div class="metric-value">10,113</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Perplexity Metric</div>
      <div class="metric-value">320.1</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Next-Word Accuracy</div>
      <div class="metric-value">12.79%</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Sliding Window Length</div>
      <div class="metric-value">10 Words</div>
    </div>
  </div>
</div>

<script>
  let debounceTimer = null;

  function setPreset(text) {
    document.getElementById('promptInput').value = text;
    runPrediction();
  }

  function clearPrompt() {
    document.getElementById('promptInput').value = '';
    document.getElementById('predictionsContainer').innerHTML = '<div style="color: #94A3B8; padding: 20px; text-align: center;">Enter text to predict next words</div>';
    document.getElementById('tokensContainer').innerHTML = '';
  }

  function handleInput() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runPrediction, 180);
  }

  async function runPrediction() {
    const prompt = document.getElementById('promptInput').value;
    const topK = parseInt(document.getElementById('topK').value);
    const temp = parseFloat(document.getElementById('temp').value);
    const filterUnk = document.getElementById('filterUnk').checked;

    document.getElementById('topKVal').innerText = topK;
    document.getElementById('tempVal').innerText = temp.toFixed(1);

    if (!prompt.trim()) {
      clearPrompt();
      return;
    }

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          prompt: prompt,
          top_k: topK,
          filter_unk: filterUnk,
          temperature: temp
        })
      });

      const data = await resp.json();
      renderPredictions(data.predictions);
      renderTokens(data.window_debug);
    } catch (err) {
      console.error("Prediction error:", err);
    }
  }

  function appendWord(word) {
    const input = document.getElementById('promptInput');
    input.value = (input.value.trim() + ' ' + word).trim();
    runPrediction();
  }

  function renderPredictions(predictions) {
    const container = document.getElementById('predictionsContainer');
    if (!predictions || predictions.length === 0) {
      container.innerHTML = '<div style="color: #94A3B8; padding: 20px; text-align: center;">No predictions returned</div>';
      return;
    }

    let html = '';
    predictions.forEach(p => {
      const widthPct = Math.min(100, Math.max(5, p.percent * 3));
      html += `
        <div class="pred-card" onclick="appendWord('${p.word}')" title="Click to add '${p.word}' to prompt">
          <div class="pred-progress-bar" style="width: ${widthPct}%;"></div>
          <div class="pred-content">
            <span class="pred-word">➕ ${p.word}</span>
            <span class="pred-prob">${p.percent}%</span>
          </div>
        </div>
      `;
    });
    container.innerHTML = html;
  }

  function renderTokens(debugTokens) {
    const container = document.getElementById('tokensContainer');
    if (!debugTokens) {
      container.innerHTML = '';
      return;
    }

    let html = '';
    debugTokens.forEach(t => {
      html += `
        <div class="token-chip ${t.status}" title="${t.status.toUpperCase()} (Index: ${t.index})">
          <strong>${t.word}</strong>
          <span class="token-id">#${t.index}</span>
        </div>
      `;
    });
    container.innerHTML = html;
  }

  async function generateStory() {
    const prompt = document.getElementById('promptInput').value;
    const numWords = parseInt(document.getElementById('genLen').value);
    const strategy = document.getElementById('genStrategy').value;
    const temp = parseFloat(document.getElementById('temp').value);
    const filterUnk = document.getElementById('filterUnk').checked;
    const resultBox = document.getElementById('genResult');

    if (!prompt.trim()) {
      alert("Please enter a prompt first.");
      return;
    }

    resultBox.style.display = 'block';
    resultBox.innerHTML = '<em>Generating Shakespearean sequence...</em>';

    try {
      const resp = await fetch('/api/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          prompt: prompt,
          num_words: numWords,
          temperature: temp,
          filter_unk: filterUnk,
          strategy: strategy
        })
      });

      const data = await resp.json();
      resultBox.innerHTML = `
        <span class="seed-highlight">${data.seed}</span> 
        <span class="gen-highlight">${data.generated_words.join(' ')}</span>
      `;
    } catch (err) {
      resultBox.innerHTML = `<span style="color: red;">Error generating sequence: ${err}</span>`;
    }
  }

  async function searchVocab() {
    const q = document.getElementById('vocabQuery').value.trim();
    const container = document.getElementById('vocabResults');
    if (!q) {
      container.innerHTML = '';
      return;
    }

    try {
      const resp = await fetch('/api/vocab/search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: q, limit: 16})
      });
      const data = await resp.json();
      if (data.matches && data.matches.length > 0) {
        container.innerHTML = data.matches.map(m => `
          <span class="badge" style="cursor: pointer;" onclick="appendWord('${m.word}')" title="Click to insert">
            ${m.word} <small style="opacity: 0.7; margin-left: 4px;">#${m.index}</small>
          </span>
        `).join('');
      } else {
        container.innerHTML = '<span style="font-size: 0.85rem; color: var(--text-muted);">No matches in 10,113 words</span>';
      }
    } catch (e) {
      console.error(e);
    }
  }

  // Initial load
  window.addEventListener('DOMContentLoaded', () => {
    runPrediction();
  });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting Shakespeare Next-Word Prediction UI on http://127.0.0.1:8000 ...")
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)
