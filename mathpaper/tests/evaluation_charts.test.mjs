import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {binTimings} from '../src/mathpaper/web/evaluation-metrics.mjs';

test('histogram uses shared, zero-based intervals with inclusive lower bounds', () => {
  const bins = binTimings([{timing_ms: 0, candidates: 20}, {timing_ms: 249.99, candidates: 50}, {timing_ms: 250, candidates: 20}, {timing_ms: 1000, candidates: 50}]);
  assert.equal(bins.length, 5);
  assert.deepEqual(bins[0], {startMs: 0, endMs: 250, counts: {20: 1, 50: 1}});
  assert.deepEqual(bins[1].counts, {20: 1, 50: 0});
  assert.deepEqual(bins[4].counts, {20: 0, 50: 1});
});

test('recorded histogram preserves all 20 observations in the right intervals', () => {
  const report = JSON.parse(readFileSync(new URL('../src/mathpaper/web/evaluation.json', import.meta.url)));
  const bins = binTimings(report.budget_experiment.runs);
  for (const budget of [20, 50]) assert.equal(bins.reduce((n, b) => n + b.counts[budget], 0), 10);
  assert.equal(bins[4].counts[20], 9);
  assert.equal(bins[5].counts[20], 1);
  assert.equal(bins[10].counts[50], 1);
  assert.equal(bins[11].counts[50], 8);
  assert.equal(bins[12].counts[50], 1);
});

test('invalid durations and bin widths cannot silently disappear from the chart', () => {
  for (const value of [-1, NaN, Infinity]) assert.throws(() => binTimings([{timing_ms: value, candidates: 20}]));
  assert.throws(() => binTimings([{timing_ms: 100, candidates: 50}], 0));
  assert.throws(() => binTimings([]));
});
