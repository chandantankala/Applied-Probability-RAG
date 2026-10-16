'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const node = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
  const external = (label, href, cls = '') => { const a = node('a', cls, label); a.href = href; a.target = '_blank'; a.rel = 'noopener'; return a; };
  const pdf = (paper, page) => `/api/papers/${encodeURIComponent(paper)}/pdf${page ? `#page=${page}` : ''}`;
  const duration = ms => ms < 1000 ? `${ms.toFixed(1)} ms` : `${(ms / 1000).toFixed(2)} s`;
  let report, papers;

  function audit() {
    const q = report.questions.find(q => q.id === $('audit-question').value);
    const r = report.runs.find(r => r.id === q.id && r.mode === $('audit-method').value);
    const root = $('audit-result');
    root.replaceChildren(node('h3', 'audit-question-text', q.question));
    const summary = node('div', 'audit-summary');
    summary.append(node('span', `hit-badge${r.page_hit_at_5 ? '' : ' miss'}`, r.page_hit_at_5 ? 'Reference page found' : 'Reference page missed'), node('small', '', `${duration(r.timing_ms)} local query · 50 candidates · top 5 passages`));
    const expected = node('div', 'expected-pages');
    expected.append(node('strong', '', `Labelled evidence: ${papers[q.gold_paper].title}`));
    q.gold_pages.forEach(p => expected.append(external(`PDF page ${p} ↗`, pdf(q.gold_paper, p), 'pdf-pill')));
    root.append(summary, expected);
    const list = node('ol', 'returned-sources'); list.setAttribute('aria-label', 'Five returned passages, in rank order');
    r.top_sources.forEach(s => {
      const isGold = s.paper_id === q.gold_paper && q.gold_pages.includes(s.page);
      const row = node('li', isGold ? 'is-gold' : '');
      const link = external(`${papers[s.paper_id].title} ↗`, pdf(s.paper_id, s.page));
      link.append(node('small', '', `PDF page ${s.page}${s.label ? ` · ${s.label}` : ''}`)); row.append(link);
      if (isGold) row.append(node('span', 'reference-tag', 'LABELLED PAGE'));
      list.append(row);
    });
    root.append(list);
  }

  function render(data) {
    report = data; papers = Object.fromEntries(data.papers.map(p => [p.id, p]));
    for (const key of ['papers', 'pages', 'passages']) $(`stat-${key}`).textContent = data.corpus[key];
    $('stat-questions').textContent = data.questions.length;
    $('stat-methods').textContent = data.methods.length;
    $('method-chart').setAttribute('aria-label', data.methods.map(m => `${m.label}: ${m.hits} of ${m.questions} evidence-page hits`).join('; '));
    data.methods.forEach(m => {
      const bar = node('div', `method-bar${m.id === 'hybrid-rerank' ? ' selected' : ''}`);
      const label = node('div', 'bar-label'); label.append(node('span', '', m.label), node('strong', '', `${m.hits}/${m.questions}`));
      const track = node('div', 'bar-track'), fill = node('div', 'bar-fill'); fill.style.width = `${100 * m.hits / m.questions}%`; track.append(fill); bar.append(label, track); $('method-chart').append(bar);
      const row = node('tr'), name = node('td', '', m.label); name.append(node('small', '', m.description));
      row.append(name, node('td', '', `${m.hits}/${m.questions}`), node('td', '', `${m.candidate_hits}/${m.questions}`), node('td', '', duration(m.median_ms))); $('method-table').append(row);
      const option = node('option', '', m.label); option.value = m.id; $('audit-method').append(option);
    });
    $('budget-reduction').textContent = `${data.budget_experiment.median_reduction_percent}%`;
    const maxTime = Math.max(...data.budget_experiment.budgets.map(b => b.median_ms));
    data.budget_experiment.budgets.forEach(b => {
      const selected = b.candidates === 20, bar = node('div', `budget-bar${selected ? ' selected' : ''}`);
      const label = node('div', 'bar-label'), title = node('span', '', `${b.candidates} candidates`);
      if (selected) title.append(node('span', 'selected-tag', 'AGENT DEFAULT'));
      label.append(title, node('strong', '', duration(b.median_ms)));
      const track = node('div', 'bar-track'), fill = node('div', 'bar-fill'); fill.style.width = `${100 * b.median_ms / maxTime}%`; track.append(fill);
      bar.append(label, track, node('small', '', `${b.hits}/${b.questions} evidence-page hits @5 · median local query`)); $('budget-chart').append(bar);
    });
    data.questions.forEach((q, i) => { const o = node('option', '', `${String(i + 1).padStart(2, '0')} · ${q.question}`); o.value = q.id; $('audit-question').append(o); });
    $('audit-method').value = 'hybrid-rerank';
    $('audit-question').addEventListener('change', audit); $('audit-method').addEventListener('change', audit);
    $('inspect-miss').addEventListener('click', () => { $('audit-question').value = 'gaussian-separation'; $('audit-method').value = 'hybrid-rerank'; audit(); });
    audit();
    const labels = {theorem_answer: ['Theorem summary', 'Cited output returned'], contextual_followup: ['Contextual follow-up', 'Same conversation · assumptions and sampling'], out_of_scope: ['Unrelated question', 'Abstention returned']};
    data.live.checks.forEach(c => {
      const details = node('details', 'live-case'), summary = node('summary'), title = node('span', 'live-case-title', labels[c.kind][0]);
      title.append(node('small', '', labels[c.kind][1]));
      summary.append(title, node('span', 'live-case-metrics', `${c.timing_seconds.toFixed(1)} s · ${c.tool_calls} tool calls`)); details.append(summary);
      const body = node('div', 'case-body'); body.append(node('p', 'case-question', c.question));
      body.append(node('p', 'case-meta', `${c.model_calls} model calls · ${c.citations.length} citation links · individual warm-service observation`));
      body.append(node('div', 'recorded-answer', c.answer));
      const links = node('div');
      const seen = new Set(); c.citations.forEach(s => { if (!seen.has(s.url)) { seen.add(s.url); links.append(external(`${s.label} ↗`, s.url, 'pdf-pill')); } });
      body.append(links); details.append(body); $('live-checks').append(details);
    });
    const date = new Date(data.live.checked_at_utc).toLocaleDateString('en-GB', {day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC'});
    $('live-date').textContent = `RENDER FREE · ${date.toUpperCase()}`;
    $('live-limit').textContent = `${data.live.model} · Three development examples; mathematical accuracy and latency percentiles are unmeasured. Support checking uses the same model and can make correlated errors. Original PDF pages remain authoritative.`;
    data.sources.forEach(s => $('source-files').append(external(s.file, s.url)));
    data.papers.forEach(p => { const item = node('div', 'paper-version'); item.append(external(`${p.title} ↗`, pdf(p.id)), node('code', '', `SHA-256 ${p.sha256}`)); $('paper-versions').append(item); });
    $('report-status').hidden = true; $('report').hidden = false;
  }
  fetch('/assets/evaluation.json').then(response => { if (!response.ok) throw new Error('Report unavailable'); return response.json(); }).then(render).catch(() => { $('report-status').textContent = 'The recorded report could not be loaded. Please refresh, or inspect the evaluation files on GitHub.'; });
})();
