// Shared, fixed-width bins keep the two recorded timing distributions comparable.
export function binTimings(runs, widthMs = 250) {
  if (!runs.length || !Number.isFinite(widthMs) || widthMs <= 0) throw new Error('Invalid histogram inputs');
  if (runs.some(r => !Number.isFinite(r.timing_ms) || r.timing_ms < 0 || ![20, 50].includes(r.candidates))) throw new Error('Invalid timing observation');
  const count = Math.floor(Math.max(...runs.map(r => r.timing_ms)) / widthMs) + 1;
  const bins = Array.from({length: count}, (_, i) => ({startMs: i * widthMs, endMs: (i + 1) * widthMs, counts: {20: 0, 50: 0}}));
  runs.forEach(r => bins[Math.floor(r.timing_ms / widthMs)].counts[r.candidates]++);
  return bins;
}
