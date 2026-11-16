"""Freeze the declared pooled-dev selection after all original-source audits.

No model calls, no test data inspection. This completes selection if an offline
judge failed while source auditors independently completed all required labels.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import digest, load_cases, read_json, summarize_agent, write_json

if any((ROOT/'results/experiments').glob('*/*/test/agent_traces.json')):
    raise SystemExit('Refusing dev reselection after any held-out test run')
configs = read_json(ROOT/'configs/experiments.json')
ranking, dev_hashes = [], {}
for config in configs['configs']:
    metrics = []
    for suite in ('basic', 'advanced'):
        folder = ROOT/f"results/experiments/{config['id']}/{suite}/dev"
        cases = load_cases(suite, 'dev')
        traces = read_json(folder/'agent_traces.json')
        reviews = read_json(folder/'reviews.ai.json')
        summary = summarize_agent(cases, traces, reviews)
        write_json(folder/'agent_metrics.ai.json', summary)
        metrics.append(summary)
        dev_hashes[f"{config['id']}/{suite}"] = read_json(folder/'run_identity.json')['implementation_sha256']
    count = sum(sum(c['answerable'] for c in load_cases(suite, 'dev')) for suite in ('basic', 'advanced'))
    quality = sum(m['grounded_answer']['counts']['1']+2*m['grounded_answer']['counts']['2'] for m in metrics)/(2*count)
    total_citations = sum(m['citation_validity']['N'] for m in metrics)
    citation_rate = sum(m['citation_validity']['n'] for m in metrics)/total_citations if total_citations else 0
    seconds = sum(m['mean_seconds']*m['cases'] for m in metrics)/sum(m['cases'] for m in metrics)
    ranking.append({'config': config, 'quality': quality, 'citation_rate': citation_rate, 'mean_seconds': seconds})
ranking.sort(key=lambda item: (-item['quality'], -item['citation_rate'], item['mean_seconds']))
path = ROOT/'results/selection_lock.json'
if path.exists():
    write_json(ROOT/'results/selection_lock.initial.json', read_json(path))
lock = {'selected': ranking[0]['config'], 'ranking': ranking, 'selection_rule': configs['selection_rule'],
        'locked_at_utc': datetime.now(timezone.utc).isoformat(),
        'index_manifest_sha256': digest(ROOT/'data/index_manifest.json'),
        'config_sha256': digest(ROOT/'configs/experiments.json'),
        'implementation_sha256': {f: digest(ROOT/f) for f in ('src/agent.py', 'src/retrieval.py')},
        'dev_implementation_sha256': dev_hashes,
        'benchmark_sha256': {suite: digest(ROOT/f'data/benchmark/{suite}_questions.json') for suite in ('basic', 'advanced')},
        'review_note': 'AI-assisted dev labels plus explicit original-page source audits; no held-out test was run before this lock.',
        'pretest_trace_only_patch': 'Record actual requested tool arguments before execution, including failures. Prompts, retrieval, config and answer policy unchanged;39tests passed. Dev source hashes retained separately.',
        'source_audit_files': [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'results').glob('source_audit_dev*.json'))]}
write_json(path, lock)
print('SELECTED', ranking[0]['config']['id'], 'pooled dev grounded score', ranking[0]['quality'])
