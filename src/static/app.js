const API = window.location.origin;
const PROJECT_ID = 1;

document.getElementById('query').addEventListener('keydown', e => {
  if (e.key === 'Enter') ask();
});

document.getElementById('examples').addEventListener('click', e => {
  const chip = e.target.closest('.example-chip');
  if (!chip) return;
  document.getElementById('query').value = chip.dataset.q;
  ask();
});

const HISTORY_KEY = 'breathxRagChatHistory';
const HISTORY_MAX = 15;
let chatHistory = [];
try { chatHistory = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { chatHistory = []; }

function saveHistoryEntry(question, mode, data, isRefusal) {
  chatHistory.unshift({ id: Date.now(), question, mode, data, isRefusal });
  if (chatHistory.length > HISTORY_MAX) chatHistory.length = HISTORY_MAX;
  localStorage.setItem(HISTORY_KEY, JSON.stringify(chatHistory));
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById('historyList');
  const badge = document.getElementById('hamburgerBadge');

  if (!chatHistory.length) {
    list.innerHTML = '<div class="history-empty">No questions asked yet.</div>';
    badge.className = 'hamburger-badge hidden';
    return;
  }
  badge.className = 'hamburger-badge';
  badge.textContent = chatHistory.length;
  list.innerHTML = chatHistory.map(h => `
    <span class="history-chip ${h.isRefusal ? 'refused' : ''}" data-id="${h.id}" title="${esc(h.question)}">
      <span class="hq-dot"></span><span class="hq-text" dir="${isArabic(h.question) ? 'rtl' : 'ltr'}">${esc(h.question)}</span>
    </span>
  `).join('');
}

function openHistorySidebar() {
  document.getElementById('historySidebar').classList.add('open');
  document.getElementById('sidebarOverlay').classList.add('open');
}
function closeHistorySidebar() {
  document.getElementById('historySidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('open');
}

document.getElementById('hamburgerBtn').addEventListener('click', () => {
  const sidebar = document.getElementById('historySidebar');
  sidebar.classList.contains('open') ? closeHistorySidebar() : openHistorySidebar();
});
document.getElementById('historySidebarClose').addEventListener('click', closeHistorySidebar);
document.getElementById('sidebarOverlay').addEventListener('click', closeHistorySidebar);

document.getElementById('historyList').addEventListener('click', e => {
  const chip = e.target.closest('.history-chip');
  if (!chip) return;
  const entry = chatHistory.find(h => String(h.id) === chip.dataset.id);
  if (!entry) return;

  document.getElementById('query').value = entry.question;
  if (entry.mode) document.getElementById('mode').value = entry.mode;

  document.getElementById('status').className = 'status';
  document.getElementById('skeleton').className = 'hidden';
  document.getElementById('empty').className = 'hidden';
  document.getElementById('examples').className = 'hidden';

  if (entry.isRefusal) showRefusal(entry.data);
  else showAnswer(entry.data);

  closeHistorySidebar();
});

document.getElementById('historyClear').addEventListener('click', () => {
  chatHistory = [];
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

renderHistory();

async function ask() {
  const q = document.getElementById('query').value.trim();
  if (!q) return;

  const mode = document.getElementById('mode').value;
  const isRerank = mode === 'rerank';
  const statusEl = document.getElementById('status');
  const skeletonEl = document.getElementById('skeleton');
  const resultsEl = document.getElementById('results');
  const emptyEl = document.getElementById('empty');
  const examplesEl = document.getElementById('examples');
  const answerText = document.getElementById('answerText');
  const btn = document.getElementById('askBtn');

  btn.disabled = true;
  statusEl.className = 'status loading';
  statusEl.innerHTML = '<span class="spinner"></span> Checking safety&hellip;';
  skeletonEl.className = 'skeleton-block';
  resultsEl.className = 'hidden results';
  emptyEl.className = 'hidden';
  examplesEl.className = 'hidden';
  answerText.innerHTML = '';

  const payload = {
    text: q,
    limit: 8,
    retrieval_mode: isRerank ? 'hybrid' : mode,
    rerank: isRerank,
    expand_query: true,
    conversation_history: chatHistory.slice(0, 4).reverse().map(h => ({
      question: h.question,
      answer: h.isRefusal ? null : (h.data?.answer || null),
    })),
  };

  let answerTokens = '';
  let finalData = null;

  try {
    const res = await fetch(`${API}/api/v1/nlp/index/answer/${PROJECT_ID}/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop();

      let eventType = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          handleSSEEvent(eventType, data);
        }
      }
    }

    statusEl.className = 'status';
    skeletonEl.className = 'hidden';
    btn.disabled = false;

    if (finalData) {
      const isRefusal = finalData.risk_assessment?.risk_level === 'refuse_redirect'
                     || finalData.confidence?.generation_allowed === false
                     || !finalData.answer;

      if (isRefusal) showRefusal(finalData);
      else showAnswer(finalData);
      saveHistoryEntry(q, mode, finalData, isRefusal);
    }
  } catch (err) {
    skeletonEl.className = 'hidden';
    statusEl.className = 'status error';
    statusEl.textContent = `Error: ${err.message}`;
    btn.disabled = false;
  }

  function handleSSEEvent(type, data) {
    const PHASE_LABELS = {
      classifying: 'Checking safety',
      retrieving: 'Searching evidence',
      reranking: 'Ranking results',
      checking_evidence: 'Verifying evidence',
      generating: 'Generating answer',
      verifying: 'Verifying citations',
      regenerating: 'Regenerating answer',
    };

    switch (type) {
      case 'phase':
        statusEl.innerHTML = `<span class="spinner"></span> ${PHASE_LABELS[data.phase] || data.phase}&hellip;`;
        if (data.phase === 'generating') {
          skeletonEl.className = 'hidden';
          resultsEl.className = 'results';
          answerText.innerHTML = '<span class="cursor-blink">|</span>';
        }
        break;

      case 'token':
        if (data.full) {
          answerTokens = data.token;
        } else {
          answerTokens += data.token;
        }
        answerText.innerHTML = formatAnswer(answerTokens) + '<span class="cursor-blink">|</span>';
        break;

      case 'done':
        finalData = data;
        statusEl.innerHTML = '<span class="spinner"></span> Verifying citations&hellip;';
        break;
    }
  }
}

function showAnswer(data) {
  const resultsEl = document.getElementById('results');
  const answerText = document.getElementById('answerText');
  const disclaimer = document.getElementById('disclaimer');
  const metricsGrid = document.getElementById('metricsGrid');
  const citationsList = document.getElementById('citationsList');
  const evidenceGrid = document.getElementById('evidenceGrid');
  const riskBadge = document.getElementById('riskBadge');
  const pipelineUsed = document.getElementById('pipelineUsed');
  const unsupportedBox = document.getElementById('unsupportedBox');
  const evidenceCoverage = document.getElementById('evidenceCoverage');

  // Answer — render inline [Doc, p. N] refs as compact chips instead of raw brackets
  answerText.innerHTML = formatAnswer(data.answer || 'No answer generated.');
  const answerIsArabic = isArabic(data.answer);
  answerText.dir = answerIsArabic ? 'rtl' : 'ltr';

  if (data.disclaimer) {
    disclaimer.innerHTML = warningIcon() + `<span>${esc(data.disclaimer)}</span>`;
    disclaimer.className = 'disclaimer';
  } else {
    disclaimer.className = 'disclaimer hidden';
  }

  // Risk badge — surfaces "needs_caution" answers, not just full refusals
  const riskLevel = data.risk_assessment?.risk_level;
  if (riskLevel === 'needs_caution') {
    riskBadge.className = 'risk-badge caution';
    riskBadge.textContent = 'Needs Caution';
  } else if (riskLevel === 'allowed') {
    riskBadge.className = 'risk-badge allowed';
    riskBadge.textContent = 'Allowed';
  } else {
    riskBadge.className = 'risk-badge hidden';
  }

  // Metrics
  const q = data.quality || {};
  const conf = data.confidence || {};
  const faithfulness = q.citation_faithfulness ?? 1;
  const unsupportedRate = q.unsupported_claim_rate ?? 0;
  const citationCount = (q.citations || data.citations || []).length;
  const confLevel = conf.confidence_level;
  const confGood = confLevel === 'high' ? true : (confLevel === 'low' || confLevel === 'insufficient') ? false : null;

  metricsGrid.innerHTML = [
    metric('Faithfulness', faithfulness.toFixed(2), faithfulness >= 0.95),
    metric('Unsupported', unsupportedRate.toFixed(2), unsupportedRate === 0),
    metric('Citations', citationCount, null),
    metric('Chunks', data.evidence_panel?.total_selected ?? 0, null),
    metric('Confidence', capitalize(confLevel) + (conf.top_score != null ? ` &middot; ${conf.top_score.toFixed(2)}` : ''), confGood),
  ].join('');

  // Unsupported claims — only shown when the verifier actually flagged something
  const unsupportedClaims = q.unsupported_claims || [];
  if (unsupportedClaims.length) {
    unsupportedBox.innerHTML = `<div class="uc-title">${unsupportedClaims.length} unsupported claim${unsupportedClaims.length > 1 ? 's' : ''} detected</div>
      <ul dir="${answerIsArabic ? 'rtl' : 'ltr'}">${unsupportedClaims.map(c => `<li>${esc(c.claim)}</li>`).join('')}</ul>`;
    unsupportedBox.className = 'unsupported-box';
  } else {
    unsupportedBox.className = 'unsupported-box hidden';
  }

  // Pipeline actually used to produce this answer
  const pm = data.pipeline_metadata;
  if (pm) {
    pipelineUsed.innerHTML = [
      puChip(`Retrieval: ${capitalize(pm.retrieval_mode)}`, true),
      puChip('Rerank', pm.rerank_enabled),
      puChip('Query expansion', pm.query_expansion_enabled),
      puChip('Claim verification', pm.verify_claims_enabled),
    ].join('');
    pipelineUsed.className = 'pipeline-used';
  } else {
    pipelineUsed.className = 'pipeline-used hidden';
  }

  // Citations — deduplicated by document + page
  const rawCites = data.citations || q.citations || [];
  const seen = new Map();
  rawCites.forEach(c => {
    const key = `${c.document}__${c.page_number}`;
    if (!seen.has(key)) seen.set(key, c);
  });
  const cites = Array.from(seen.values());
  citationsList.innerHTML = cites.length
    ? cites.map(c => `<span class="citation-tag ${c.supported ? 'supported' : 'unsupported'}">${c.supported ? checkIcon() : crossIcon()} ${esc(prettyDocName(c.document))}, p. ${c.page_number}</span>`).join('')
    : '<span class="citation-tag">No citations</span>';

  // Evidence Panel — prefer evidence_panel.chunks (carries is_official_source),
  // fall back to sources[] for older API responses.
  const chunks = data.evidence_panel?.chunks || (data.sources || []).map((s, i) => ({ ...s, rank: i + 1, relevance_score: s.score, text_preview: s.text }));
  evidenceGrid.innerHTML = chunks.length
    ? chunks.map(c => evidenceCard(c)).join('')
    : '<div style="color:var(--muted);font-size:13px">No evidence retrieved.</div>';
  bindEvidenceToggles();

  const coverage = data.evidence_panel?.retrieval_coverage;
  if (coverage && coverage.unique_documents) {
    const range = coverage.page_range || {};
    const pageStr = range.min != null ? ` &middot; pages ${range.min}&ndash;${range.max}` : '';
    evidenceCoverage.textContent = '';
    evidenceCoverage.innerHTML = `${coverage.unique_documents} document${coverage.unique_documents > 1 ? 's' : ''}${pageStr}`;
  } else {
    evidenceCoverage.innerHTML = '';
  }

  document.getElementById('metricsSection').className = 'section';
  document.getElementById('citationsSection').className = 'section';
  document.getElementById('evidenceSection').className = 'section';

  resultsEl.className = 'results';
}

function puChip(label, on) {
  return `<span class="pu-chip ${on ? 'on' : ''}">${on ? checkIcon() : crossIcon()} ${esc(label)}</span>`;
}

function capitalize(s) {
  if (!s) return 'Unknown';
  return s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, ' ');
}

function showRefusal(data) {
  const resultsEl = document.getElementById('results');
  const answerText = document.getElementById('answerText');
  const disclaimer = document.getElementById('disclaimer');
  const riskBadge = document.getElementById('riskBadge');
  const pipelineUsed = document.getElementById('pipelineUsed');
  const evidenceCoverage = document.getElementById('evidenceCoverage');

  const reason = data.risk_assessment?.reason || 'unknown';
  const reasonLabel = {
    possible_emergency: 'Emergency Detected',
    personal_medication_advice: 'Personal Dosing Request',
    animal_or_pet_question: 'Out of Scope',
    clearly_non_clinical: 'Out of Scope',
    outside_asthma_scope: 'Out of Scope',
    empty_query: 'Empty Query',
  }[reason] || 'Refused';

  const msg = data.answer || 'This query cannot be answered.';
  answerText.innerHTML = `<div class="refusal-box">
    <div class="refusal-icon">${alertIcon()}</div>
    <div><h3>${esc(reasonLabel)}</h3><p dir="${isArabic(msg) ? 'rtl' : 'ltr'}">${esc(msg)}</p></div>
  </div>`;
  answerText.dir = 'ltr';
  disclaimer.className = 'disclaimer hidden';
  riskBadge.className = 'risk-badge hidden';
  pipelineUsed.className = 'pipeline-used hidden';
  evidenceCoverage.innerHTML = '';

  const evidenceGrid = document.getElementById('evidenceGrid');
  const chunks = data.evidence_panel?.chunks || (data.sources || []).map((s, i) => ({ ...s, rank: i + 1, relevance_score: s.score, text_preview: s.text }));
  evidenceGrid.innerHTML = chunks.length
    ? chunks.map(c => evidenceCard(c)).join('')
    : '<div style="color:var(--muted);font-size:13px">No evidence retrieved (query blocked before retrieval).</div>';
  bindEvidenceToggles();

  document.getElementById('metricsGrid').innerHTML = '';
  document.getElementById('unsupportedBox').className = 'unsupported-box hidden';
  document.getElementById('citationsList').innerHTML = '';
  document.getElementById('metricsSection').className = 'hidden';
  document.getElementById('citationsSection').className = 'hidden';
  document.getElementById('evidenceSection').className = 'section';

  resultsEl.className = 'results';
}

function metric(label, value, good) {
  const stateCls = good === null ? '' : (good ? 'good' : 'bad');
  return `<div class="metric ${stateCls}"><div class="label">${label}</div><div class="value ${stateCls}">${value}</div></div>`;
}

function evidenceCard(s) {
  const score = s.relevance_score ?? s.score;
  const scoreCls = score >= 0.6 ? 'high' : score >= 0.3 ? 'mid' : 'low';
  const pct = score != null ? Math.max(4, Math.min(100, score * 100)) : 0;
  const text = (s.text_preview ?? s.text ?? '').trim();
  const needsToggle = text.length > 140;
  const docName = esc(prettyDocName(s.document_name || s.file_name || 'Unknown'));
  const docLabel = s.source_url
    ? `<a class="evidence-doc" href="${esc(s.source_url)}" target="_blank" rel="noopener noreferrer">${docName}</a>`
    : `<span class="evidence-doc">${docName}</span>`;
  const officialBadge = s.is_official_source
    ? `<span class="official-badge">${checkIcon()} Official</span>`
    : '';
  return `<div class="evidence-card">
    <div class="evidence-head">
      <div class="evidence-rank">#${s.rank ?? '?'}</div>
      <div class="evidence-body">
        <div class="evidence-doc-row">${docLabel}${officialBadge}</div>
        <div class="evidence-meta">Page ${s.page_number || '?'}${s.section_title ? ' &middot; ' + esc(s.section_title) : ''}</div>
      </div>
      <div class="evidence-score-wrap">
        <div class="evidence-score ${scoreCls}">${score != null ? score.toFixed(3) : '-'}</div>
        <div class="score-bar"><div class="score-bar-fill ${scoreCls}" style="width:${pct}%"></div></div>
      </div>
    </div>
    ${text ? `<div class="evidence-text">${esc(text)}</div>` : ''}
    ${needsToggle ? `<button class="evidence-toggle" type="button">Show more</button>` : ''}
  </div>`;
}

function bindEvidenceToggles() {
  document.querySelectorAll('.evidence-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const textEl = btn.previousElementSibling;
      const expanded = textEl.classList.toggle('expanded');
      btn.textContent = expanded ? 'Show less' : 'Show more';
    });
  });
}

// Renders the model's markdown-lite answer (bold, "- " bullets, "1. " numbered
// items) as proper HTML blocks, and turns inline "[Doc, p. N]" refs into chips.
function formatAnswer(text) {
  const escaped = esc((text || '').trim());

  // The model usually emits real newlines between bullets, but if it collapses
  // everything onto one line, insert breaks before bullet markers ourselves.
  let normalized = escaped.replace(/\r\n/g, '\n');
  if (!normalized.includes('\n')) {
    normalized = normalized.replace(/([.:])\s+-\s+(?=\S)/g, '$1\n- ');
  }

  const lines = normalized.split(/\n+/).map(l => l.trim()).filter(Boolean);

  const blocks = [];
  let list = null;
  lines.forEach(line => {
    const bulletMatch = line.match(/^(?:[-•*]|\d+[.)])\s+(.*)$/);
    if (bulletMatch) {
      if (!list) { list = []; blocks.push({ type: 'list', items: list }); }
      list.push(bulletMatch[1]);
      return;
    }
    list = null;
    const headingMatch = line.match(/^\*\*(.+?)\*\*:?$/);
    if (headingMatch) {
      blocks.push({ type: 'heading', text: headingMatch[1] });
    } else {
      blocks.push({ type: 'para', text: line });
    }
  });

  return blocks.map(b => {
    if (b.type === 'list') {
      return `<ul class="answer-list">${b.items.map(i => `<li>${inlineFormat(i)}</li>`).join('')}</ul>`;
    }
    if (b.type === 'heading') {
      return `<div class="answer-subhead">${inlineFormat(b.text)}</div>`;
    }
    return `<p class="answer-para">${inlineFormat(b.text)}</p>`;
  }).join('');
}

// Applies bold + citation-chip formatting to a single already-HTML-escaped line.
function inlineFormat(line) {
  let html = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\[([^\[\]]+)\]/g, (match, inner) => {
    const parts = inner.split(';').map(p => p.trim()).filter(Boolean);
    const chips = parts.map(part => {
      const m = part.match(/^(.+?),\s*pp?\.?\s*([\d][\d,\s\-–]*)$/i);
      if (!m) return `<span class="inline-cite">${part}</span>`;
      const pages = m[2].replace(/\s+/g, '').replace(/,/g, ', ');
      return `<span class="inline-cite">${esc(prettyDocName(m[1]))} p.${pages}</span>`;
    });
    return chips.join(' ');
  });
  return html;
}

function prettyDocName(name) {
  if (!name) return 'Unknown';
  return name.replace(/\.[a-zA-Z0-9]+$/, '').replace(/_/g, ' ');
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// Content direction is per-answer, not per-page: the same UI answers both
// English and Arabic queries (language fidelity — Arabic in, Arabic out), so
// each rendered block picks its own dir rather than the page defaulting to one.
function isArabic(text) {
  return /[؀-ۿ]/.test(text || '');
}

function checkIcon() { return '<svg viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>'; }
function crossIcon() { return '<svg viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>'; }
function warningIcon() { return '<svg viewBox="0 0 24 24" fill="none"><path d="M12 9v4M12 17h.01M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>'; }
function alertIcon() { return '<svg viewBox="0 0 24 24" fill="none"><path d="M12 8v5M12 16h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.7"/></svg>'; }
document.getElementById('askBtn').addEventListener('click', ask);
