"""Finalize example inputs and build two independent, verifiable Qdrant indexes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import fitz
import numpy as np

from .ingest import validate_chunks
from .retrieval import LocalE5, VectorStore

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def finalize_chunks():
    chunks = json.loads((ROOT/'data/processed/chunks_native.json').read_text(encoding='utf-8'))
    descriptions_path = ROOT/'data/processed/visual_descriptions.json'
    if not descriptions_path.exists():
        raise RuntimeError('Visual processing not complete; do not silently finalize a text-only index')
    descriptions = json.loads(descriptions_path.read_text(encoding='utf-8'))
    doc = fitz.open(ROOT/'data/source/world-development-report-2024.pdf')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT.parent/'K5-L3B-DAY06-PHAMQUANGTUAN-2A202603824-DataIngestion/data/models/multilingual-e5-small/tokenizer.json'))
    for item in descriptions:
        if item.get('error'):
            continue
        page_number = item['page']
        text = item['text']
        # Preserve paragraph boundaries; split only to keep the same E5-safe
        # corpus across embedding models. Overlap retains neighboring labels.
        start = part = 0
        while start < len(text):
            end = min(len(text), start+1300)
            if end < len(text):
                cut = text.rfind(' ', start+300, end)
                if cut > start:
                    end = cut
            while len(tokenizer.encode('passage: '+text[start:end]).ids) > 480:
                end = start+int((end-start)*.8)
            bbox = list(doc[page_number-1].rect)
            chunks.append({'id': f'wdr2024:p{page_number}:vision:c{part}', 'text': text[start:end],
                           'source_id': 'wdr2024', 'source': 'https://www.worldbank.org/en/publication/wdr2024',
                           'page': page_number,
                           'metadata': {'version': '2024', 'kind': 'vision_description', 'bbox': bbox,
                                        'printed_page': str(page_number-32), 'generated_by': item['model'],
                                        'model_revision': item['revision'], 'verification': 'AI generated; original PDF is authoritative',
                                        'truncated_description': item.get('truncated', False),
                                        'original_page': f'data/source/world-development-report-2024.pdf#page={page_number}'}})
            if end == len(text):
                break
            start = max(start+1, end-120)
            part += 1
    validate_chunks(chunks)
    write_json(ROOT/'data/chunks.json', chunks)
    coverage = json.loads((ROOT/'data/processed/page_coverage.json').read_text(encoding='utf-8'))
    by_page = {r['page']: r for r in descriptions}
    for page in coverage:
        record = by_page.get(page['page'])
        page['vision_status'] = ('error' if record.get('error') else 'described') if record else 'not_applicable_or_no_caption'
        if record and (record.get('error') or record.get('truncated')):
            page['issues'].append(record.get('error', 'Vision description reached output limit; original page remains available'))
    write_json(ROOT/'data/processed/page_coverage.json', coverage)
    write_json(ROOT/'data/processing_summary.json', {'pages': len(coverage), 'chunks': len(chunks),
               'chunk_kinds': dict(Counter(c['metadata']['kind'] for c in chunks)),
               'vision_pages': len(descriptions), 'vision_errors': sum(bool(r.get('error')) for r in descriptions),
               'vision_output_limit_pages': [r['page'] for r in descriptions if r.get('truncated')],
               'unprocessed_pages': [p for p in coverage if p['status'] == 'blank' or p['vision_status'] == 'error'],
               'limitations': ['Native text + generated visual descriptions are indexed; original crop remains evidence authority.',
                              'AI vision descriptions may omit or misread details; source-page audit required for final labels.']})
    print('FINAL_CHUNKS', len(chunks), flush=True)


def build_indexes():
    chunks_path = ROOT/'data/chunks.json'
    chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
    digest = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    manifest = {'status': 'building', 'chunks_path': 'data/chunks.json', 'chunks_sha256': digest,
                'chunks': len(chunks), 'backend': 'Qdrant local; same backend as Day 6', 'spaces': {}}
    write_json(ROOT/'data/index_manifest.json', manifest)
    native = json.loads((ROOT/'data/processed/chunks_native.json').read_text(encoding='utf-8'))
    native_ids = [c['id'] for c in native]
    e5 = LocalE5()
    from .colab_embeddings import ColabEmbedder
    qwen = ColabEmbedder()
    for space, embedder, dimensions, filename in [('e5', e5, 384, 'e5_native.npz'), ('qwen', qwen, 4096, 'qwen_native.npz')]:
        started = time.perf_counter()
        data = np.load(ROOT/'data/embeddings'/filename)
        native_hash = hashlib.sha256((ROOT/'data/processed/chunks_native.json').read_bytes()).hexdigest()
        native_manifest = json.loads((ROOT/f'data/embeddings/{space}_manifest.json').read_text(encoding='utf-8'))
        expected_native_hash = native_manifest.get('corpus_sha256') if space == 'e5' else native_manifest.get('source_chunks_sha256')
        if expected_native_hash != native_hash:
            raise RuntimeError('Native text changed after embedding; do not attach stale vectors to unchanged IDs')
        expected_model = e5.config if space == 'e5' else json.loads((ROOT/'configs/colab_embedding.json').read_text(encoding='utf-8'))
        if any(native_manifest.get(k) != expected_model[k] for k in ('model', 'revision', 'dimensions')):
            raise RuntimeError('Native-vector model provenance differs from requested embedding space')
        if native_manifest.get('normalization') != 'L2':
            raise RuntimeError('Native-vector normalization differs from index policy')
        if list(data['ids']) != native_ids:
            raise RuntimeError('Vector IDs differ from native corpus')
        if 'done' in data and not data['done'].all():
            raise RuntimeError('Remote batch incomplete')
        vectors = list(data['vectors'])
        remaining = chunks[len(native):]
        for offset in range(0, len(remaining), 32):
            batch = remaining[offset:offset+32]
            vectors.extend(embedder.embed_documents([c['text'] for c in batch]))
            print('INDEX_EMBED', space, len(native)+min(offset+32, len(remaining)), '/', len(chunks), flush=True)
        matrix = np.asarray(vectors, dtype=np.float32)
        if space == 'qwen':
            cfg = json.loads((ROOT/'configs/colab_embedding.json').read_text(encoding='utf-8'))
            namespace = json.dumps({k: cfg[k] for k in ('model', 'revision', 'dimensions', 'normalization', 'precision', 'query_instruction')}, sort_keys=True)
            model, revision = cfg['model'], cfg['revision']
        else:
            namespace = e5.namespace
            model, revision = e5.config['model'], e5.config['revision']
        path = f'data/qdrant/{space}-{digest[:12]}'
        store = VectorStore(ROOT/path, dimensions, namespace)
        try:
            store.upsert(chunks, matrix)
            assert store.count() == len(chunks)
            check = store.search(matrix[0], 1)[0]
            assert check['score'] > .99
        finally:
            store.close()
        np.savez_compressed(ROOT/f'data/embeddings/{space}_final.npz', vectors=matrix, ids=np.array([c['id'] for c in chunks]))
        manifest['spaces'][space] = {'path': path, 'dimensions': dimensions, 'namespace': namespace,
                                    'model': model, 'revision': revision, 'points': len(chunks),
                                    'vectors_sha256': hashlib.sha256(matrix.tobytes()).hexdigest(),
                                    'build_seconds': time.perf_counter()-started}
        write_json(ROOT/'data/index_manifest.json', manifest)
    manifest['status'] = 'complete'
    write_json(ROOT/'data/index_manifest.json', manifest)
    e5.close()
    print('INDEX_COMPLETE', len(chunks), 'chunks x 2 spaces', flush=True)


def restore_indexes():
    """Reconstruct ignored Qdrant directories from submitted final vectors."""
    manifest = json.loads((ROOT/'data/index_manifest.json').read_text(encoding='utf-8'))
    path = ROOT/manifest['chunks_path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['chunks_sha256']:
        raise ValueError('Submitted corpus hash differs from frozen index manifest')
    chunks = json.loads(path.read_text(encoding='utf-8'))
    for space, info in manifest['spaces'].items():
        arrays = np.load(ROOT/f'data/embeddings/{space}_final.npz')
        matrix = arrays['vectors']
        if list(arrays['ids']) != [c['id'] for c in chunks] or hashlib.sha256(matrix.tobytes()).hexdigest() != info['vectors_sha256']:
            raise ValueError('Submitted vector IDs/hash differ from index manifest')
        store = VectorStore(ROOT/info['path'], info['dimensions'], info['namespace'])
        try:
            store.upsert(chunks, matrix)
            assert store.count() == len(chunks)
        finally:
            store.close()
        print('RESTORED', space, len(chunks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['finalize', 'build', 'restore'])
    args = parser.parse_args()
    {'finalize': finalize_chunks, 'build': build_indexes, 'restore': restore_indexes}[args.command]()
