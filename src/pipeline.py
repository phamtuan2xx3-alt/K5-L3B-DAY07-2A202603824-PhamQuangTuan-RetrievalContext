"""Shared page/evidence contracts for the Day 7 pipeline.

Gold is consumed only by offline evaluation, never by the model adapter.
"""
import re
import unicodedata


def chunk_blocks(blocks, max_chars=900, overlap=120):
    if not 0 <= overlap < max_chars:
        raise ValueError('overlap must be smaller than max_chars')
    chunks = []
    for block_index, block in enumerate(blocks):
        text = block['text']
        start = 0
        part = 0
        while start < len(text):
            end = min(len(text), start+max_chars)
            if end < len(text):
                boundary = text.rfind(' ', start+max_chars//2, end)
                if boundary > start:
                    end = boundary
            if text[start:end].strip():
                chunks.append({'id': f"wdr2024:p{block['page']}:b{block_index}:c{part}",
                               'source_id': 'wdr2024', 'source': 'data/source/world-development-report-2024.pdf',
                               'page': block['page'], 'text': text[start:end],
                               'metadata': {'bbox': block['bbox'], 'kind': block.get('kind', 'text'),
                                            'offset_start': start, 'offset_end': end}})
            if end == len(text):
                break
            start = max(start+1, end-overlap)
            part += 1
    return chunks


def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def evidence_complete(groups, chunks):
    def supports(evidence, chunk):
        return (evidence['page'] == chunk['page']
                and ('source_id' not in evidence or evidence['source_id'] == chunk.get('source_id'))
                and normalized(evidence['evidence']) in normalized(chunk['text']))
    return bool(groups) and all(any(supports(evidence, chunk) for evidence in alternatives for chunk in chunks)
                                for alternatives in groups)


def public_case(case):
    return {key: case[key] for key in ('id', 'question', 'filter') if key in case}


def aggregate_advanced(cases, reviews):
    if len({c['id'] for c in cases}) != len(cases) or len({r['query_id'] for r in reviews}) != len(reviews):
        raise ValueError('Duplicate case/review ID')
    by_id = {r['query_id']: r for r in reviews}
    if set(by_id) != {c['id'] for c in cases}:
        raise ValueError('Reviews must cover exactly the published cases')
    groups = {}
    passed = errors = 0
    for case in cases:
        review = by_id[case['id']]
        if type(review.get('pass')) is not bool or review.get('error') and review['pass']:
            raise ValueError('Explicit pass label required; system errors cannot pass')
        group = groups.setdefault(case['group'], {'n': 0, 'N': 0})
        group['n'] += review['pass']
        group['N'] += 1
        passed += review['pass']
        errors += bool(review.get('error'))
    for group in groups.values():
        group['rate'] = group['n']/group['N']
    total = len(cases)
    return {'overall': {'n': passed, 'N': total, 'rate': passed/total if total else None},
            'groups': groups, 'errors': errors,
            'bonus_threshold_met': total >= 30 and passed*10 >= total*9,
            'note': 'Threshold only; all corpus, benchmark composition, evidence and submission requirements must also be met.'}
