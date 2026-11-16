"""Batch existing example chunks with real Qwen embeddings and resumable NPZ."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

chunk_file = Path('/content/day07_chunks.json')
records = json.loads(chunk_file.read_text())
ids = np.array([record['id'] for record in records])
corpus_hash = hashlib.sha256(chunk_file.read_bytes()).hexdigest()
checkpoint = Path('/content/day07_qwen_vectors.npz')
vectors = np.zeros((len(records), 4096), dtype=np.float32)
done = np.zeros(len(records), dtype=bool)
if checkpoint.exists():
    previous = np.load(checkpoint)
    if str(previous['corpus_hash']) == corpus_hash:
        assert np.array_equal(previous['ids'], ids)
        vectors, done = previous['vectors'], previous['done']
total_tokens = 0
start = time.perf_counter()
for offset in range(0, len(records), 32):
    selected = [i for i in range(offset, min(offset+32, len(records))) if not done[i]]
    if not selected:
        continue
    response = colab_embed([records[i]['text'] for i in selected], 'document')
    assert response['model'] == 'Qwen/Qwen3-Embedding-8B' and response['settings']['dimensions'] == 4096
    vectors[selected] = response['embeddings']
    done[selected] = True
    total_tokens += response['tokens']
    np.savez_compressed(checkpoint, vectors=vectors, ids=ids, done=done, corpus_hash=corpus_hash)
    print('EMBED_PROGRESS', int(done.sum()), '/', len(records), flush=True)
assert done.all() and np.isfinite(vectors).all()
manifest = {'model': MODEL_ID, 'revision': MODEL_REVISION, 'dimensions': 4096, 'normalization': 'L2',
            'precision': 'bfloat16', 'source_chunks_sha256': corpus_hash, 'chunks': len(records),
            'completed': int(done.sum()), 'tokens_this_run': total_tokens, 'seconds_this_run': time.perf_counter()-start,
            'truncated_inputs': 0, 'gpu': torch.cuda.get_device_name(0)}
Path('/content/day07_qwen_manifest.json').write_text(json.dumps(manifest, indent=2))
print('EMBED_COMPLETE:' + json.dumps(manifest), flush=True)
