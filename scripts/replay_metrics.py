"""Recompute published aggregates from frozen cases/traces/labels; no model calls."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import load_cases, summarize_agent, summarize_retrieval

checked = 0
for path in (ROOT/'results/experiments').glob('*/*/*/retrieval_traces.json'):
    split, suite = path.parent.name, path.parent.parent.name
    cases = load_cases(suite, split)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    actual = summarize_retrieval(cases, read(path))
    saved = read(path.with_name('retrieval_metrics.json'))
    # Final packaging adds explanatory provenance to definition; numbers must
    # still be exactly reproducible from the original saved rows.
    actual.pop('definition', None)
    saved.pop('definition', None)
    assert actual == saved, str(path)
    traces, reviews = path.with_name('agent_traces.json'), path.with_name('reviews.ai.json')
    if traces.exists() and reviews.exists() and path.with_name('agent_metrics.ai.json').exists():
        assert summarize_agent(cases, read(traces), read(reviews)) == read(path.with_name('agent_metrics.ai.json')), str(path)
    checked += 1
print('Verified', checked, 'published experiment aggregate sets (semantic labels remain auditable inputs)')
if checked == 0:
    raise SystemExit('No completed experiment results available')
