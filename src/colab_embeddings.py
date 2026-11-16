"""Small client adapter with the same embedding interface as Day 6."""
import json
from pathlib import Path
import urllib.error
import urllib.request

from .colab_bridge import validate_vectors

ROOT = Path(__file__).resolve().parents[1]


class ColabEmbedder:
    def __init__(self, base_url='http://127.0.0.1:8765', key_path=None):
        self.base_url = base_url.rstrip('/')
        self.config = json.loads((ROOT/'configs/colab_embedding.json').read_text(encoding='utf-8'))
        self.model_name = self.config['model']
        self.dimensions = self.config['dimensions']
        self.revision = self.config['revision']
        index_path = ROOT/'data/index_manifest.json'
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding='utf-8'))
            stored = index.get('spaces', {}).get('qwen')
            if stored:
                namespace = json.loads(stored['namespace'])
                for field in ('model', 'revision', 'dimensions', 'normalization', 'precision', 'query_instruction'):
                    if namespace.get(field) != self.config[field]:
                        raise RuntimeError(f'Query embedding configuration differs from frozen Qwen index: {field}')
        self.key_path = Path(key_path) if key_path else ROOT/'.runtime/bridge-key'
        self.last_usage = []

    def embed_batch(self, texts, input_type):
        if not texts:
            return []
        key = self.key_path.read_text(encoding='ascii').strip()
        vectors = []
        self.last_usage = []
        for start in range(0, len(texts), self.config['max_batch']):
            batch = texts[start:start+self.config['max_batch']]
            body = json.dumps({'model': self.model_name, 'input': batch, 'input_type': input_type}, ensure_ascii=False).encode()
            request = urllib.request.Request(self.base_url+'/v1/embeddings', data=body,
                                              headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
            try:
                with urllib.request.urlopen(request, timeout=self.config['request_timeout_seconds']+40) as response:
                    result = json.load(response)
            except urllib.error.HTTPError as exc:
                detail = json.loads(exc.read().decode()).get('error', 'Embedding request failed')
                raise RuntimeError(f'Embedding API {exc.code}: {detail}') from None
            if result['model'] != self.model_name or result['revision'] != self.revision:
                raise RuntimeError('Bridge model revision mismatch')
            chunk = [entry['embedding'] for entry in result['data']]
            validate_vectors(chunk, count=len(batch), dimensions=self.dimensions)
            vectors.extend(chunk)
            self.last_usage.append({'usage': result['usage'], 'bridge': result['bridge']})
        return vectors

    def embed_documents(self, texts):
        return self.embed_batch(texts, 'document')

    def embed_query(self, question):
        return self.embed_batch([question], 'query')[0]
