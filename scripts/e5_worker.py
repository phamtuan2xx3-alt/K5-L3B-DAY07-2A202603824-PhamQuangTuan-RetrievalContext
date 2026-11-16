"""Persistent local E5 worker, executed with the existing Day 6 Python env."""
import json
from pathlib import Path
import sys

import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
day6 = ROOT.parent/'K5-L3B-DAY06-PHAMQUANGTUAN-2A202603824-DataIngestion'
index_path = ROOT/'data/index_manifest.json'
if index_path.exists():
    index = json.loads(index_path.read_text(encoding='utf-8'))
    stored = index.get('spaces', {}).get('e5')
    if stored:
        config = json.loads((day6/'configs/local_embedding.json').read_text(encoding='utf-8'))
        for field in ('model', 'revision', 'dimensions'):
            if stored[field] != config[field]:
                raise RuntimeError('Day 6 query model provenance differs from frozen E5 index: '+field)
        expected_namespace = f"{config['model']}@{config['revision']}|384|L2|passage-query"
        if stored['namespace'] != expected_namespace or config['document_task'] != 'passage: ' or config['query_task'] != 'query: ':
            raise RuntimeError('Day 6 embedding namespace/prefix differs from frozen E5 index')
torch.set_num_threads(min(6, torch.get_num_threads()))
model = SentenceTransformer(str(day6/'data/models/multilingual-e5-small'), local_files_only=True,
                            device='cuda' if torch.cuda.is_available() else 'cpu', trust_remote_code=False)
model.max_seq_length = 512
print(json.dumps({'ready': True, 'dimensions': model.get_sentence_embedding_dimension(),
                  'torch': torch.__version__, 'device': str(model.device)}), flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get('command') == 'close':
            break
        texts = [request['task']+text for text in request['texts']]
        if any(len(model.tokenizer.encode(t, truncation=False)) > 512 for t in texts):
            raise ValueError('E5 input exceeds 512 tokens; split instead of truncating')
        vectors = model.encode(texts, prompt='', batch_size=32, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        result = {'vectors': vectors.tolist()}
    except Exception as error:
        result = {'error': type(error).__name__ + ': ' + str(error)}
    print(json.dumps(result, ensure_ascii=True), flush=True)
