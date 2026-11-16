"""Apply explicit original-page review corrections, preserving initial AI labels.

This changes evaluation labels only, never answers/configs/benchmark gold. Use
before dev selection or for final test source audit; no quality tuning from test.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import load_cases, summarize_agent, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--audit', type=Path, required=True)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--suite', choices=['basic', 'advanced'], required=True)
parser.add_argument('--split', choices=['dev', 'test'], required=True)
args = parser.parse_args()
audit = json.loads(args.audit.read_text(encoding='utf-8'))
reviews_path = args.run/'reviews.ai.json'
reviews = json.loads(reviews_path.read_text(encoding='utf-8')) if reviews_path.exists() else []
by_id = {r['query_id']: r for r in reviews}
for record in audit['reviews']:
    if record['query_id'] not in by_id:
        # A source audit can independently complete labels after a judge
        # transport/JSON failure. Do not fabricate an initial judge response.
        target = {'query_id': record['query_id'], 'initial_ai_label': None,
                  'label_origin': 'Independent original-page source audit; judge label unavailable'}
        reviews.append(target)
        by_id[record['query_id']] = target
    target = by_id[record['query_id']]
    documentwide_absence = record.get('abstention_correct_applicable') is True and record['recommended'].get('abstention_correct') is True
    if not record.get('original_pages_checked') and not documentwide_absence:
        raise ValueError('Source audit must identify original pages checked')
    target.setdefault('initial_ai_label', {k: v for k, v in target.items() if k not in ('judge_call', 'initial_ai_label')})
    for field in ('answer_grade', 'context_grade', 'citation_support', 'abstention_correct', 'pass'):
        if field in record['recommended']:
            target[field] = record['recommended'][field]
    target.update({'reason': record['reason'], 'source_audit_path': args.audit.as_posix(),
                   'source_pages_checked': record['original_pages_checked'],
                   'reviewer': 'Codex original-page source audit; AI-authored, not human certification'})
write_json(reviews_path, reviews)
traces = json.loads((args.run/'agent_traces.json').read_text(encoding='utf-8'))
write_json(args.run/'agent_metrics.ai.json', summarize_agent(load_cases(args.suite, args.split), traces, reviews))
print('Applied',len(audit['reviews']),'source-reviewed labels; original labels retained')
