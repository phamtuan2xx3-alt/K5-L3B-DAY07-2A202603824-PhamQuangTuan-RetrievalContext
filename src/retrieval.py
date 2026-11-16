"""Example-schema retrieval over the Day 6 backend (Qdrant local) and BM25."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]


class VectorStore:
    def __init__(self, path, dimensions, namespace):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path/'embedding_space.json'
        expected = {'dimensions': dimensions, 'namespace': namespace, 'metric': 'Cosine'}
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != expected:
            raise ValueError('Embedding space mismatch; use a separate index')
        self.client = QdrantClient(path=str(self.path))
        self.namespace, self.dimensions = namespace, dimensions
        if not self.client.collection_exists('chunks'):
            self.client.create_collection('chunks', vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE))
        cfg = self.client.get_collection('chunks').config.params.vectors
        if cfg.size != dimensions or cfg.distance != models.Distance.COSINE:
            self.close()
            raise ValueError('Existing collection configuration mismatch')
        manifest.write_text(json.dumps(expected, indent=2), encoding='utf-8')

    def upsert(self, chunks, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), self.dimensions) or not np.isfinite(vectors).all():
            raise ValueError('Invalid embedding matrix')
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=.01):
            raise ValueError('Expected L2-normalized embeddings')
        if len({c['id'] for c in chunks}) != len(chunks):
            raise ValueError('Duplicate chunk ID')
        for start in range(0, len(chunks), 128):
            points = [models.PointStruct(id=str(uuid5(NAMESPACE_URL, self.namespace+'|'+c['id'])), vector=v.tolist(),
                                          payload={**c, 'embedding_namespace': self.namespace})
                      for c, v in zip(chunks[start:start+128], vectors[start:start+128])]
            self.client.upsert('chunks', points=points, wait=True)

    def search(self, vector, top_k=3):
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (self.dimensions,) or not np.isfinite(array).all():
            raise ValueError('Query embedding space mismatch')
        points = self.client.query_points('chunks', query=array.tolist(), limit=top_k,
                                          search_params=models.SearchParams(exact=True), with_payload=True).points
        return [{**p.payload, 'score': float(p.score), 'rank': i+1} for i, p in enumerate(points)]

    def count(self):
        return self.client.count('chunks', exact=True).count

    def close(self):
        self.client.close()


def rrf(rankings, top_k=10, constant=60):
    scores, records = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, result in enumerate(ranking, 1):
            identifier = result['id']
            if identifier in seen:
                continue
            seen.add(identifier)
            records[identifier] = result
            scores[identifier] = scores.get(identifier, 0)+1/(constant+rank)
    order = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:top_k]
    return [{**records[identifier], 'score': scores[identifier], 'rank': i+1, 'fusion': 'RRF(k=60)'}
            for i, identifier in enumerate(order)]


def tokens(text):
    # Figure/table identifiers are meaningful tokens, not two unrelated numbers.
    return re.findall(r'[a-z0-9]+\.[0-9]+|\w+', text.casefold(), flags=re.UNICODE)


class LocalE5:
    """Use pinned Day 6 weights directly; no download or Day 6 store mutation."""
    def __init__(self):
        self.day6 = ROOT.parent/'K5-L3B-DAY06-PHAMQUANGTUAN-2A202603824-DataIngestion'
        self.config = json.loads((self.day6/'configs/local_embedding.json').read_text(encoding='utf-8'))
        self.dimensions = 384
        self.namespace = f"{self.config['model']}@{self.config['revision']}|384|L2|passage-query"
        python = self.day6/'starter/.venv/Scripts/python.exe'
        if not python.exists():
            raise RuntimeError('Existing Day 6 Python environment not found; see setup instructions')
        (ROOT/'.runtime').mkdir(exist_ok=True)
        self.log = (ROOT/'.runtime/e5-worker.log').open('a', encoding='utf-8')
        self.worker = subprocess.Popen([str(python), '-u', str(ROOT/'scripts/e5_worker.py')],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                       text=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'USE_TF': '0'})
        self.responses = queue.Queue()
        self.lock = threading.Lock()
        def read_output():
            for line in self.worker.stdout:
                self.responses.put(line)
            self.responses.put(None)
        threading.Thread(target=read_output, daemon=True).start()
        self.runtime_info = self._response()
        if not self.runtime_info.get('ready') or self.runtime_info['dimensions'] != 384:
            self.close()
            raise RuntimeError('Day 6 E5 worker did not initialize correctly')

    def _response(self):
        try:
            line = self.responses.get(timeout=180)
        except queue.Empty:
            raise RuntimeError('Day 6 worker timed out; inspect .runtime/e5-worker.log') from None
        if line is None:
            raise RuntimeError('Day 6 worker exited; inspect .runtime/e5-worker.log')
        response = json.loads(line)
        if response.get('error'):
            raise ValueError(response['error'])
        return response

    def embed_batch(self, texts, task):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        with self.lock:
            self.worker.stdin.write(json.dumps({'task': task, 'texts': texts}, ensure_ascii=True)+'\n')
            self.worker.stdin.flush()
            return np.asarray(self._response()['vectors'], dtype=np.float32)

    def embed_documents(self, texts):
        return self.embed_batch(texts, 'passage: ')

    def embed_query(self, question):
        return self.embed_batch([question], 'query: ')[0].tolist()

    def close(self):
        if self.worker.poll() is None:
            self.worker.stdin.write('{"command":"close"}\n')
            self.worker.stdin.flush()
            try:
                self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait()
        self.log.close()


class Retriever:
    def __init__(self, manifest_path=ROOT/'data/index_manifest.json'):
        from rank_bm25 import BM25Okapi
        self.manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        if self.manifest.get('status') != 'complete':
            raise RuntimeError('Index ingestion is incomplete')
        chunk_file = ROOT/self.manifest['chunks_path']
        if hashlib.sha256(chunk_file.read_bytes()).hexdigest() != self.manifest['chunks_sha256']:
            raise RuntimeError('Corpus differs from frozen index')
        self.chunks = json.loads(chunk_file.read_text(encoding='utf-8'))
        self.by_id = {c['id']: c for c in self.chunks}
        self.by_page = {}
        for chunk in self.chunks:
            self.by_page.setdefault(chunk['page'], []).append(chunk)
        self.lexical = BM25Okapi([tokens(c['text']) for c in self.chunks])
        self.stores = {name: VectorStore(ROOT/info['path'], info['dimensions'], info['namespace'])
                       for name, info in self.manifest['spaces'].items()}
        self._e5 = self._qwen = None
        self.query_cache = {}

    def dense(self, query, space, top_k):
        key = (query, space)
        if key not in self.query_cache:
            if space == 'e5':
                if self._e5 is None:
                    self._e5 = LocalE5()
                vector = self._e5.embed_query(query)
            else:
                if self._qwen is None:
                    from .colab_embeddings import ColabEmbedder
                    self._qwen = ColabEmbedder()
                vector = self._qwen.embed_query(query)
            self.query_cache[key] = vector
        return self.stores[space].search(self.query_cache[key], top_k)

    def search(self, query, method='hybrid_qwen', top_k=6):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Nonempty query required')
        scores = self.lexical.get_scores(tokens(query))
        order = np.argsort(-scores, kind='stable')[:max(top_k, 30)]
        lexical = [{**self.chunks[int(i)], 'score': float(scores[i]), 'rank': n+1} for n, i in enumerate(order)]
        if method == 'bm25':
            return lexical[:top_k]
        if method in ('dense_e5', 'dense_qwen'):
            return self.dense(query, method.removeprefix('dense_'), top_k)
        if method in ('hybrid_e5', 'hybrid_qwen'):
            return rrf([lexical, self.dense(query, method.removeprefix('hybrid_'), 30)], top_k=top_k)
        raise ValueError('Unknown retrieval method')

    def close(self):
        for store in self.stores.values():
            store.close()
        if self._e5 is not None:
            self._e5.close()
