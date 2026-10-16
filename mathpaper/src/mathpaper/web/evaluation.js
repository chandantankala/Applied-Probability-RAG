'use strict';
import {binTimings} from './evaluation-metrics.mjs?v=public-charts-1';
(() => {
  const $ = id => document.getElementById(id);
  const node = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
  const external = (label, href, cls = '') => { const a = node('a', cls, label); a.href = href; a.target = '_blank'; a.rel = 'noopener'; return a; };
  const pdf = (paper, page) => `/api/papers/${encodeURIComponent(paper)}/pdf${page ? `#page=${page}` : ''}`;
  const duration = ms => ms < 1000 ? `${ms.toFixed(1)} ms` : `${(ms / 1000).toFixed(2)} s`;
  let report, papers;

  const svgNode = (tag, attrs = {}, text) => { const e = document.createElementNS('http://www.w3.org/2000/svg', tag); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v)); if (text !== undefined) e.textContent = text; return e; };
  function svgChart(width, height, label) { const svg = svgNode('svg', {viewBox: `0 0 ${width} ${height}`, role: 'group', 'aria-label': label, class: 'data-chart'}); svg.append(svgNode('title', {}, label)); return svg; }
  function activate(target, action) { target.addEventListener('click', action); target.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); action(); } }); }

  function methodChart() {
    const svg = svgChart(580, 300, 'Evidence-page hits by retrieval method. Select a bar to inspect that method.');
    const left = 36, top = 25, bottom = 240, width = 524, height = bottom - top;
    for (let n = 0; n <= 10; n += 2) {
      const y = bottom - n / 10 * height;
      svg.append(svgNode('line', {x1: left, x2: left + width, y1: y, y2: y, class: 'chart-gridline'}), svgNode('text', {x: left - 10, y: y + 4, 'text-anchor': 'end', class: 'chart-tick'}, n));
    }
    report.methods.forEach((m, i) => {
      const center = left + (i + .5) * width / report.methods.length, barHeight = m.hits / 10 * height;
      const label = `${m.label}: ${m.hits}/${m.questions} page hits; ${duration(m.median_ms)} median local query. Select method.`;
      const group = svgNode('g', {role: 'button', tabindex: '0', 'aria-label': label, 'aria-pressed': 'false', 'data-chart-mode': m.id, class: 'method-column'});
      group.append(svgNode('title', {}, label), svgNode('rect', {x: center - 34, y: top, width: 68, height: height, rx: 4, class: 'column-background'}), svgNode('rect', {x: center - 34, y: bottom - barHeight, width: 68, height: barHeight, rx: 4, class: 'column-fill'}), svgNode('text', {x: center, y: bottom - barHeight - 10, 'text-anchor': 'middle', class: 'chart-value'}, `${m.hits}/${m.questions}`));
      const names = m.id === 'hybrid-rerank' ? ['Hybrid +', 'reranking'] : [m.label];
      names.forEach((name, line) => group.append(svgNode('text', {x: center, y: bottom + 22 + line * 16, 'text-anchor': 'middle', class: 'chart-category'}, name)));
      activate(group, () => { $('audit-method').value = m.id; audit(); }); svg.append(group);
    });
    $('method-chart').append(svg);
  }

  function histogram() {
    const bins = binTimings(report.budget_experiment.runs);
    const svg = svgChart(900, 292, 'Histogram of local query durations for 20 and 50 candidates, with the same 0.25-second intervals and zero-based axes.');
    const left = 45, top = 32, bottom = 230, width = 830, height = bottom - top, band = width / bins.length;
    svg.append(svgNode('text', {x: left, y: 15, class: 'chart-axis-title'}, 'Number of queries'));
    for (let n = 0; n <= 10; n += 2) {
      const y = bottom - n / 10 * height;
      svg.append(svgNode('line', {x1: left, x2: left + width, y1: y, y2: y, class: 'chart-gridline'}), svgNode('text', {x: left - 12, y: y + 4, 'text-anchor': 'end', class: 'chart-tick'}, n));
    }
    bins.forEach((bin, i) => {
      const x = left + i * band, range = `${(bin.startMs / 1000).toFixed(2)}–${(bin.endMs / 1000).toFixed(2)} s`;
      const label = `${range}: 20 candidates, ${bin.counts[20]} queries; 50 candidates, ${bin.counts[50]} queries.`;
      const group = svgNode('g', {role: 'button', tabindex: '0', 'aria-label': label, class: 'histogram-bin'});
      group.append(svgNode('title', {}, label), svgNode('rect', {x: x + 1, y: top, width: band - 2, height, rx: 3, class: 'bin-highlight'}));
      [20, 50].forEach((budget, j) => {
        const count = bin.counts[budget], barHeight = count / 10 * height, barWidth = (band - 10) / 2;
        const barX = x + 4 + j * (barWidth + 2);
        group.append(svgNode('rect', {x: barX, y: bottom - barHeight, width: barWidth, height: barHeight, rx: 2, class: `histogram-bar budget-${budget}`, 'data-budget': budget, 'data-count': count}));
        if (count) group.append(svgNode('text', {x: barX + barWidth / 2, y: bottom - barHeight - 8, 'text-anchor': 'middle', class: 'histogram-value'}, count));
      });
      const inspect = () => { $('histogram-selection').textContent = label; };
      group.addEventListener('mouseenter', inspect); group.addEventListener('focus', inspect); activate(group, inspect); svg.append(group);
      if (i % 2 === 0) svg.append(svgNode('text', {x, y: bottom + 23, 'text-anchor': 'middle', class: 'chart-tick'}, (bin.startMs / 1000).toFixed(1)));
      const row = node('tr'); row.append(node('td', '', range), node('td', '', bin.counts[20]), node('td', '', bin.counts[50])); $('histogram-counts').append(row);
    });
    svg.append(svgNode('text', {x: left + width, y: bottom + 23, 'text-anchor': 'end', class: 'chart-tick'}, (bins.at(-1).endMs / 1000).toFixed(2)), svgNode('text', {x: left + width / 2, y: 284, 'text-anchor': 'middle', class: 'chart-axis-title'}, 'Local query time (seconds)'));
    $('latency-histogram').append(svg);
  }

  function heatmap() {
    const header = node('tr'), corner = node('th', '', 'Method / question'); corner.scope = 'col'; header.append(corner);
    report.questions.forEach((q, i) => { const th = node('th', '', String(i + 1).padStart(2, '0')); th.scope = 'col'; th.title = q.question; th.setAttribute('aria-label', `Question ${i + 1}: ${q.question}`); header.append(th); }); $('heatmap-head').append(header);
    report.methods.forEach(m => {
      const row = node('tr'), title = node('th', '', m.label); title.scope = 'row'; row.append(title);
      report.questions.forEach((q, i) => {
        const run = report.runs.find(r => r.id === q.id && r.mode === m.id), cell = node('td');
        const button = node('button', `heatmap-cell ${run.page_hit_at_5 ? 'hit' : 'miss'}`, run.page_hit_at_5 ? '✓' : '×'); button.type = 'button'; button.dataset.auditQuestion = q.id; button.dataset.auditMode = m.id;
        const label = `${m.label}, question ${i + 1}: ${q.question} — labelled page ${run.page_hit_at_5 ? 'found' : 'missed'}. Inspect result.`;
        button.setAttribute('aria-label', label); button.title = label;
        button.addEventListener('click', () => { $('audit-question').value = q.id; $('audit-method').value = m.id; audit(); }); cell.append(button); row.append(cell);
      }); $('heatmap-body').append(row);
    });
  }

  function audit() {
    const q = report.questions.find(q => q.id === $('audit-question').value);
    const r = report.runs.find(r => r.id === q.id && r.mode === $('audit-method').value);
    document.querySelectorAll('[data-chart-mode]').forEach(e => { const selected = e.dataset.chartMode === r.mode; e.classList.toggle('selected', selected); e.setAttribute('aria-pressed', selected); });
    document.querySelectorAll('[data-audit-question]').forEach(e => e.setAttribute('aria-pressed', e.dataset.auditQuestion === q.id && e.dataset.auditMode === r.mode));
    const method = report.methods.find(m => m.id === r.mode);
    $('method-selection').textContent = `${method.label} · ${method.hits}/${method.questions} page hits · ${duration(method.median_ms)} median local query`;
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
    data.methods.forEach(m => {
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
    methodChart(); histogram(); heatmap();
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
