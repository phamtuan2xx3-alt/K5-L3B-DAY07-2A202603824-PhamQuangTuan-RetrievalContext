"""Add final-context/call/usage metrics from real traces, without model calls."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import groups, load_cases, summarize_agent, summarize_retrieval, write_json
from src.pipeline import evidence_complete


def ratio(n, total):
    return {'n': n, 'N': total, 'rate': n/total if total else None}


completed = 0
for path in (ROOT/'results/experiments').glob('*/*/*/agent_metrics.ai.json'):
    folder = path.parent
    suite, split = folder.parent.name, folder.name
    cases = load_cases(suite, split)
    traces = json.loads((folder/'agent_traces.json').read_text(encoding='utf-8'))
    reviews = json.loads((folder/'reviews.ai.json').read_text(encoding='utf-8'))
    if len(traces) != len(cases) or len(reviews) != len(cases):
        continue
    by_id = {r['query_id']: r for r in traces}
    labels = {r['query_id']: r for r in reviews}
    answerable = [c for c in cases if c['answerable']]
    answered = [t for t in traces if t['status'] == 'answered']
    successful = [t for t in traces if t['status'] != 'error']
    context_evidence = sum(t['status'] != 'error' and evidence_complete(groups(c), t.get('context_chunks', []))
                           for c in answerable for t in [by_id[c['id']]])
    calls = [m for t in traces for m in t.get('model_calls', [])]
    usage = Counter()
    for call in calls:
        usage.update({k: v for k, v in (call.get('usage') or {}).items() if type(v) in (int, float)})
    tool_counts = Counter(sum(e.get('event') in ('search', 'read_page') for e in t.get('events', [])) for t in successful)
    per_query_context = [{'query_id': c['id'], 'context_grade': labels[c['id']].get('context_grade', 0),
                          'strict_complete_evidence': by_id[c['id']]['status'] != 'error' and evidence_complete(groups(c), by_id[c['id']].get('context_chunks', [])),
                          'reason': labels[c['id']].get('reason', ''), 'context_ids': by_id[c['id']].get('context_ids', [])}
                         for c in answerable]
    result = {'final_context_evidence_rate': ratio(context_evidence, len(answerable)),
              'definition': 'Actual final context sent to answering model; strict native gold quote+source/page, all multihop groups required; errors scored0. Visual semantic evidence separately source-audited.',
              'answer_citation_coverage': ratio(sum(bool(t.get('citations')) for t in answered), len(answered)),
              'retrieval_tool_call_distribution_successful_queries': dict(sorted(tool_counts.items())),
              'context_grade_distribution_answerable_successful': dict(Counter(labels[c['id']].get('context_grade', 0) for c in answerable if by_id[c['id']]['status'] != 'error')),
              'per_query_context_review': per_query_context, 'model_calls': len(calls),
              'model_http_attempts_recorded': sum(len(m.get('attempts', [])) for m in calls),
              'usage': dict(usage), 'usd_cost': None,
              'cost_limitations': 'No verified USD tariff; usage available only for recorded model responses, failures/setup may not supply token usage.',
              'errors': sum(t['status'] == 'error' for t in traces)}
    write_json(folder/'runtime_metrics.json', result)
    # Deterministic replay of labels. This does not modify answers or gold.
    write_json(path, summarize_agent(cases, traces, reviews))
    retrieval = json.loads((folder/'retrieval_traces.json').read_text(encoding='utf-8'))
    summary = summarize_retrieval(cases, retrieval)
    summary['definition'] += ' This context rate is the original-query retrieval packing diagnostic; actual final agent context rate is in runtime_metrics.json.'
    write_json(folder/'retrieval_metrics.json', summary)
    completed += 1
print('Completed final-context/runtime metrics for',completed,'complete run sets')
