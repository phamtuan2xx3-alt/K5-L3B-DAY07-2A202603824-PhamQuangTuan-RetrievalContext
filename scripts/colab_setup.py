"""Execute with colab exec; keeps an actual embedding model in the live kernel."""
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

print('Installing model dependencies...', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'transformers==4.57.3', 'sentence-transformers==5.1.2',
                'huggingface-hub==0.36.0', 'safetensors>=0.4.3'], check=True)
import torch
from sentence_transformers import SentenceTransformer

assert torch.cuda.is_available(), 'No CUDA GPU was allocated'
device = torch.cuda.get_device_properties(0)
print(json.dumps({'gpu': device.name, 'vram_gb': round(device.total_memory / 2**30, 2),
                  'torch': torch.__version__}), flush=True)

MODEL_ID = 'Qwen/Qwen3-Embedding-8B'
MODEL_REVISION = '1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'
QUERY_INSTRUCTION = 'Given a question in Vietnamese or English, retrieve relevant passages from reports that answer the question.'
print('Loading pinned Qwen3-Embedding-8B weights (download is on Colab only)...', flush=True)
started = time.perf_counter()
embedding_model = SentenceTransformer(
    MODEL_ID, revision=MODEL_REVISION, device='cuda', trust_remote_code=False,
    model_kwargs={'torch_dtype': torch.bfloat16, 'attn_implementation': 'sdpa', 'use_safetensors': True},
    tokenizer_kwargs={'padding_side': 'left'})
embedding_model.max_seq_length = 8192
embedding_model.eval()

def colab_embed(texts, input_type):
    """No truncation; normalized full 4096-dimensional outputs."""
    if input_type not in ('query', 'document'):
        raise ValueError('Unsupported input type')
    if not isinstance(texts, list) or not 1 <= len(texts) <= 32 or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError('Invalid batch')
    inputs = ([f'Instruct: {QUERY_INSTRUCTION}\nQuery: {t}' for t in texts]
              if input_type == 'query' else texts)
    lengths = [len(embedding_model.tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in inputs]
    if max(lengths) > embedding_model.max_seq_length:
        raise ValueError('Input exceeds 8192 tokens; split input instead of truncating')
    before = time.perf_counter()
    with torch.inference_mode():
        vectors = embedding_model.encode(inputs, prompt='', batch_size=4, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False)
    return {'model': MODEL_ID, 'revision': MODEL_REVISION, 'embeddings': vectors.tolist(),
            'settings': {'dimensions': int(vectors.shape[1]), 'normalization': 'L2',
                         'query_instruction': QUERY_INSTRUCTION, 'max_sequence_length': embedding_model.max_seq_length,
                         'precision': str(next(embedding_model.parameters()).dtype).split('.')[-1]},
            'tokens': sum(lengths), 'inference_seconds': time.perf_counter()-before}

smoke = colab_embed(['Hà Nội là thủ đô Việt Nam.', 'Paris is the capital of France.'], 'document')
runtime_info = {'model': MODEL_ID, 'revision': MODEL_REVISION, 'dimensions': len(smoke['embeddings'][0]),
                'gpu': device.name, 'vram_gb': round(device.total_memory / 2**30, 2),
                'precision': str(next(embedding_model.parameters()).dtype),
                'model_load_seconds': time.perf_counter()-started, 'max_sequence_length': 8192,
                'allocated_vram_gb': round(torch.cuda.memory_allocated()/2**30, 2),
                'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'sentence-transformers', 'huggingface-hub')}}
Path('/content/embedding_runtime.json').write_text(json.dumps(runtime_info, indent=2))
print('MODEL_READY:' + json.dumps(runtime_info), flush=True)
