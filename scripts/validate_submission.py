"""Offline structural audit. Never imports agent/runtime clients or reads credentials."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import unicodedata

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


class Audit:
    def __init__(self, root, allow_incomplete=False):
        self.root = Path(root)
        self.allow_incomplete = allow_incomplete
        self.errors, self.pending, self.checks = [], [], {}

    def require(self, condition, message):
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def missing(self, name):
        (self.pending if self.allow_incomplete else self.errors).append('Missing/incomplete: '+name)

    def read(self, name):
        path = self.root/name
        if not path.is_file():
            self.missing(name)
            return None
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            self.errors.append('Invalid UTF-8 JSON: '+name)
            return None

    def bbox(self, box, page, label):
        good = isinstance(box, list) and len(box) == 4
        good = good and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
        if good:
            x0, y0, x1, y1 = box
            good = 0 <= x0 < x1 <= page.rect.width+.05 and 0 <= y0 < y1 <= page.rect.height+.05
        self.require(good, label+': invalid/out-of-page bbox')

    def evidence(self, item, doc, texts, label):
        if not self.require(isinstance(item, dict), label+': evidence must be object'):
            return
        self.require(item.get('source_id') == 'wdr2024', label+': source_id')
        page = item.get('page')
        if not self.require(type(page) is int and 1 <= page <= len(doc), label+': PDF page'):
            return
        quote = item.get('evidence')
        self.require(isinstance(quote, str) and bool(quote.strip()) and normalized(quote) in texts[page-1], label+': gold not in original native page')
        self.bbox(item.get('bbox'), doc[page-1], label)

    def benchmark(self, suite, doc, texts):
        rows = self.read(f'data/benchmark/{suite}_questions.json')
        if rows is None:
            return []
        if not self.require(isinstance(rows, list), suite+': benchmark must be array'):
            return []
        ids = [r.get('id') for r in rows]
        self.require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids), suite+': duplicate/missing IDs')
        questions = [normalized(r.get('question', '')).casefold() for r in rows]
        self.require(len(questions) == len(set(questions)), suite+': duplicate questions across splits')
        counts = Counter()
        for row in rows:
            label = str(row.get('id'))
            self.require(row.get('split') in ('dev', 'test'), label+': split')
            self.require(type(row.get('answerable')) is bool, label+': answerable must be boolean')
            self.require(bool(row.get('question')), label+': missing question')
            self.require(row.get('group') in ('text', 'table', 'chart', 'diagram', 'image', 'diagram_image', 'multi_hop', 'unanswerable', 'absent'), label+': group')
            counts[(row.get('split'), row.get('answerable'))] += 1
            if row.get('answerable'):
                self.require(bool(row.get('expected_answer')), label+': expected_answer')
                gold = row.get('gold', [])
                self.require(isinstance(gold, list) and (bool(gold) or row.get('group') == 'multi_hop'), label+': missing gold')
                for item in gold:
                    self.evidence(item, doc, texts, label)
                if row.get('group') == 'multi_hop':
                    groups = row.get('evidence_groups')
                    self.require(isinstance(groups, list) and len(groups) >= 2 and all(isinstance(g, list) and g for g in groups), label+': required evidence_groups')
                    self.require(row.get('all_groups_required') is True, label+': multi-hop must require all evidence groups')
                    if isinstance(groups, list):
                        pages = set()
                        for group in groups:
                            for item in group:
                                self.evidence(item, doc, texts, label)
                                pages.add(item.get('page'))
                        self.require(len(pages) >= 2 or len({json.dumps(g, sort_keys=True) for g in groups}) >= 2, label+': indistinct multihop groups')
            else:
                self.require(row.get('expected_behavior') == 'search_and_abstain' and bool(row.get('absence_reason')), label+': abstention behavior/reason')
        minima = {('dev', True): 15, ('dev', False): 2, ('test', True): 5, ('test', False): 2}
        if suite == 'advanced':
            minima.update({('dev', False): 3, ('test', True): 24, ('test', False): 6})
            groups = Counter(r.get('group') for r in rows if r.get('split') == 'test' and r.get('answerable'))
            groups['diagram'] += groups['image']+groups['diagram_image']
            for group, minimum in dict(table=6, diagram=5, chart=5, multi_hop=4, text=4).items():
                self.require(groups[group] >= minimum, f'advanced test {group}: minimum {minimum}')
        for key, minimum in minima.items():
            self.require(counts[key] >= minimum, f'{suite} {key}: minimum {minimum}')
        self.checks[suite+'_counts'] = {str(k): v for k, v in counts.items()}
        return rows

    def indexes(self, chunks):
        manifest = self.read('data/index_manifest.json')
        if manifest is None:
            return
        if manifest.get('status') != 'complete':
            self.missing('final index status')
        path = self.root/manifest.get('chunks_path', 'data/chunks.json')
        self.require(path.is_file() and manifest.get('chunks_sha256') == digest(path), 'Index corpus hash mismatch')
        self.require(manifest.get('chunks') == len(chunks), 'Index corpus count mismatch')
        ids = [c['id'] for c in chunks]
        for space, dim in [('e5', 384), ('qwen', 4096)]:
            info = manifest.get('spaces', {}).get(space)
            filename = f'data/embeddings/{space}_final.npz'
            path = self.root/filename
            if not info or not path.is_file():
                self.missing(filename+' / manifest space')
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    vectors, actual_ids = data['vectors'], data['ids'].tolist()
                    self.require(actual_ids == ids and len(set(actual_ids)) == len(ids), space+': IDs/order/corpus coverage')
                    self.require(vectors.shape == (len(ids), dim) and info.get('dimensions') == dim, space+': dimensions')
                    self.require(np.isfinite(vectors).all(), space+': nonfinite vectors')
                    # BF16-normalized outputs are quantized before export to
                    # float32; observed norm deviation is <=0.00388. Qdrant
                    # cosine re-normalizes stored vectors. Keep a documented
                    # bounded BF16 tolerance rather than rejecting valid data.
                    tolerance = .005 if space == 'qwen' else .0001
                    self.require(np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=tolerance), space+': vectors not L2 normalized')
                    self.require(info.get('vectors_sha256') == hashlib.sha256(vectors.tobytes()).hexdigest(), space+': vector hash')
                    self.require(info.get('points') == len(ids), space+': point count')
                    self.require(bool(info.get('model')) and bool(info.get('revision')) and bool(info.get('namespace')), space+': model provenance')
                    self.require((self.root/info.get('path', '__missing__')).is_dir(), space+': vector store missing')
            except (ValueError, KeyError, OSError):
                self.errors.append(space+': invalid embedding archive')

    def experiments(self, suites):
        lock = self.read('results/selection_lock.json')
        if lock is None:
            return
        for field, filename in [('index_manifest_sha256', 'data/index_manifest.json'), ('config_sha256', 'configs/experiments.json')]:
            path = self.root/filename
            self.require(path.is_file() and lock.get(field) == digest(path), 'Selection lock mismatch: '+field)
        for suite in suites:
            path = self.root/f'data/benchmark/{suite}_questions.json'
            self.require(path.is_file() and lock.get('benchmark_sha256', {}).get(suite) == digest(path), 'Selection benchmark hash mismatch: '+suite)
        selected = lock.get('selected', {}).get('id')
        self.require(bool(selected) and bool(lock.get('locked_at_utc')), 'Missing selection identity/time')
        scores = {}
        for config in {selected, 'baseline_e5'}-{None}:
            for suite, cases in suites.items():
                if suite == 'advanced' and config != selected:
                    continue
                ids = {c['id'] for c in cases if c['split'] == 'test'}
                base = f'results/experiments/{config}/{suite}/test/'
                traces = self.read(base+'agent_traces.json')
                reviews = self.read(base+'reviews.ai.json')
                if traces is None or reviews is None:
                    continue
                for name, rows in [('traces', traces), ('reviews', reviews)]:
                    actual = [r.get('query_id') for r in rows]
                    self.require(len(actual) == len(set(actual)) and set(actual) == ids, base+name+': must cover every published test ID exactly once')
                by_id = {r['query_id']: r for r in reviews}
                cases_by_id = {c['id']: c for c in cases}
                for trace in traces:
                    review = by_id.get(trace['query_id'], {})
                    self.require(type(review.get('pass')) is bool and bool(review.get('reason')), base+trace['query_id']+': explicit review/reason required')
                    if trace.get('error') or trace.get('status') == 'error':
                        self.require(review.get('pass') is False, base+trace['query_id']+': system errors must fail')
                    citations = trace.get('citations', [])
                    context_ids = set(trace.get('context_ids', []))
                    self.require(all(c in context_ids and '['+c+']' in trace.get('answer', '') for c in citations),
                                 base+trace['query_id']+': citation missing from actual final context/answer')
                    support = review.get('citation_support', [])
                    self.require(len(support) == len(citations) and all(type(x) is bool for x in support),
                                 base+trace['query_id']+': citation label cardinality/type mismatch')
                    case = cases_by_id[trace['query_id']]
                    if review.get('pass'):
                        if case['answerable']:
                            self.require(trace.get('status') == 'answered' and review.get('answer_grade') == 2 and bool(citations) and all(support),
                                         base+trace['query_id']+': pass conflicts with complete grounded/citation labels')
                        else:
                            self.require(trace.get('status') == 'abstain' and review.get('abstention_correct') is True,
                                         base+trace['query_id']+': pass conflicts with abstention labels')
                n = sum(r.get('pass') is True for r in reviews if r.get('query_id') in ids)
                scores[config+'/'+suite] = {'n': n, 'N': len(ids), 'threshold_met': bool(ids) and n*10 >= len(ids)*9}
                metrics = self.read(base+'agent_metrics.ai.json')
                if metrics is not None:
                    self.require(metrics.get('cases') == len(ids) and metrics.get('advanced_pass', {}).get('N') == len(ids), base+'metrics: errors/missing results excluded from denominator')
        self.checks['test_scores_all_published_cases'] = scores
        self.checks['bonus'] = {'threshold_met': scores.get(str(selected)+'/advanced', {}).get('threshold_met'), 'eligibility': 'Not certified: original-page final review and all advanced requirements still required', 'structural_basic_requires_90_percent': False}
        self.checks['review_status'] = 'AI-assisted reviews; validator does not certify human approval or semantic correctness'

    def secret_scan(self):
        result = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=self.root, capture_output=True)
        if result.returncode:
            self.errors.append('Cannot enumerate Git shareable candidates')
            return
        candidates = set(result.stdout.decode('utf-8', errors='replace').split('\0'))-{''}
        patterns = [re.compile(rb'sk-agp-[A-Za-z0-9_-]{12,}'), re.compile(rb'(?i)["\x27]?(?:access_token|refresh_token|id_token)["\x27]?\s*[:=]\s*["\x27]([A-Za-z0-9._~-]{20,})["\x27]')]
        hits, scanned = [], 0
        for filename in sorted(candidates):
            parts = Path(filename).parts
            # Never open credential/runtime paths even when accidentally tracked.
            if any(p in ('.runtime', '.git') or p == '.env' or p.startswith('.env.') and p != '.env.example' for p in parts):
                if '.runtime' in parts or '.env' in parts:
                    self.errors.append('Sensitive path is a Git shareable candidate: '+filename)
                continue
            path = self.root/filename
            if not path.is_file() or path.is_symlink():
                continue
            data = path.read_bytes()
            scanned += 1
            if any(pattern.search(data) for pattern in patterns):
                hits.append(filename)
        self.require(not hits, 'Possible secrets in shareable paths: '+', '.join(hits))
        self.checks['secret_scan'] = {'files_scanned': scanned, 'flagged_paths': hits, 'values_redacted': True}

    def run(self):
        pdf = self.root/'data/source/world-development-report-2024.pdf'
        if not pdf.is_file():
            self.missing(str(pdf.relative_to(self.root)))
            return
        with fitz.open(pdf) as doc:
            texts = [normalized(p.get_text()) for p in doc]
            source = self.read('data/source_manifest.json')
            if source is not None:
                self.require(source.get('sha256') == digest(pdf) and source.get('pdf_pages') == len(doc), 'Source manifest hash/page count mismatch')
            chunks = self.read('data/chunks.json')
            if isinstance(chunks, list):
                ids = [c.get('id') for c in chunks]
                self.require(all(ids) and len(set(ids)) == len(ids), 'Chunks: duplicate/missing IDs')
                for c in chunks:
                    label = str(c.get('id'))
                    self.require(all(k in c for k in ('id', 'text', 'source_id', 'source', 'page', 'metadata')) and bool(c.get('text')), label+': chunk schema')
                    self.require(c.get('source_id') == 'wdr2024', label+': chunk source')
                    p = c.get('page')
                    if self.require(type(p) is int and 1 <= p <= len(doc), label+': chunk PDF page'):
                        self.bbox(c.get('metadata', {}).get('bbox'), doc[p-1], label)
                    self.require(not any(k in c for k in ('gold', 'expected_answer', 'answerable', 'absence_reason')), label+': benchmark labels in corpus')
                useful = {i+1 for i, text in enumerate(texts) if text.strip()}
                covered = {c.get('page') for c in chunks}
                self.require(useful <= covered, 'Corpus omits native-text pages: '+str(sorted(useful-covered)))
                self.checks['corpus_coverage'] = {'native_nonblank_pages': len(useful), 'indexed_pages': len(covered), 'chunks': len(chunks)}
                self.indexes(chunks)
            elif chunks is not None:
                self.errors.append('Chunks must be JSON array')
            suites = {s: self.benchmark(s, doc, texts) for s in ('basic', 'advanced')}
            all_ids = [c['id'] for cases in suites.values() for c in cases]
            self.require(len(all_ids) == len(set(all_ids)), 'Cross-suite duplicate IDs')
            self.experiments(suites)
        self.checks.setdefault('bonus', {'eligibility': 'Pending final experiments and original-page review; structural basic submission has no 90-percent requirement'})
        self.checks.setdefault('review_status', 'AI-authored labels; no human approval certified')
        self.secret_scan()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    audit = Audit(args.root, args.allow_incomplete)
    try:
        audit.run()
    except Exception as error:
        # Deliberately omit exception strings, which can contain sensitive input.
        audit.errors.append('Validation aborted safely: '+type(error).__name__)
    report = {'validated_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'error' if audit.errors else 'pending' if audit.pending else 'structurally_valid', 'errors': audit.errors, 'pending': audit.pending, 'checks': audit.checks, 'semantic_correctness_certified': False}
    output = args.root/'results/submission_validation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 1 if audit.errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
