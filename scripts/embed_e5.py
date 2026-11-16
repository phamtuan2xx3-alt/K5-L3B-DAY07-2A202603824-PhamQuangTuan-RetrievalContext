"""Reuse Day 6 weights locally on exactly the same Day 7 chunk corpus."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval import LocalE5

path = ROOT/'data/processed/chunks_native.json'
chunks = json.loads(path.read_text(encoding='utf-8'))
output = ROOT/'data/embeddings'
output.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
embedder = LocalE5()
vectors = []
for offset in range(0, len(chunks), 128):
    vectors.extend(embedder.embed_documents([c['text'] for c in chunks[offset:offset+128]]))
    print('E5_PROGRESS', min(offset+128, len(chunks)), '/', len(chunks), flush=True)
array = np.asarray(vectors, dtype=np.float32)
np.savez_compressed(output/'e5_native.npz', vectors=array, ids=np.array([c['id'] for c in chunks]))
manifest = {'model': embedder.config['model'], 'revision': embedder.config['revision'], 'dimensions': 384,
            'namespace': embedder.namespace, 'normalization': 'L2', 'chunks': len(chunks),
            'corpus_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'seconds': time.perf_counter()-started,
            'truncated_inputs': 0, 'model_downloads': 0, 'day6_model_path': 'data/models/multilingual-e5-small'}
(output/'e5_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
embedder.close()
print(json.dumps(manifest))
