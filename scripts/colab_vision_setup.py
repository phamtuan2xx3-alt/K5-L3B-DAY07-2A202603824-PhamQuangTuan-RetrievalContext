"""Optional document-vision support on the existing A100, without a public API."""
import json
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'PyMuPDF==1.26.6'], check=True)
import fitz
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

VISION_ID = 'Qwen/Qwen3-VL-8B-Instruct'
VISION_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
print('Loading pinned document-vision model on the same A100...', flush=True)
vision_model = Qwen3VLForConditionalGeneration.from_pretrained(
    VISION_ID, revision=VISION_REVISION, dtype=torch.bfloat16,
    attn_implementation='sdpa', device_map={'': 'cuda:0'}, trust_remote_code=False)
vision_processor = AutoProcessor.from_pretrained(VISION_ID, revision=VISION_REVISION, max_pixels=1280*28*28)
vision_model.eval()
source_path = Path('/content/world-development-report-2024.pdf')
if not source_path.exists():
    download = requests.get('https://www.healthsystemsmatter.com/files/GeneralReports/World%20Development%20Report%202024.pdf', timeout=90)
    download.raise_for_status()
    source_path.write_bytes(download.content)
import hashlib
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == '7f545194763ef2ffd1347c14bcf5a270c622b9c3a29e1c8c8b081aaee6c00df5'
vision_pdf = fitz.open(source_path)

def colab_describe_page(page_number, prompt=None):
    page = vision_pdf[page_number-1]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6))
    image = Image.frombytes('RGB', [pixmap.width, pixmap.height], pixmap.samples)
    image.thumbnail((1500, 1500))
    if prompt is None:
        prompt = ('Describe every informative table, chart, map, diagram or image on this report page in English. '
                  'For each, copy its figure/table identifier and caption, explain the actual visual relationships, axes, '
                  'legend, units and readable values. Preserve row/column association for tables. '
                  'Do not infer invisible values. Mark uncertainty explicitly. Ignore ordinary prose and decoration. '
                  'Return compact factual descriptions, not advice. Native page text to help read labels:\n' + page.get_text()[:7000])
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
    inputs = vision_processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors='pt').to('cuda')
    before = time.perf_counter()
    with torch.inference_mode():
        generated = vision_model.generate(**inputs, max_new_tokens=1100, do_sample=False)
    output = vision_processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return {'page': page_number, 'text': output, 'model': VISION_ID, 'revision': VISION_REVISION,
            'seconds': time.perf_counter()-before, 'method': 'AI vision description; verify against original page',
            'input_tokens': int(inputs.input_ids.shape[1]), 'output_tokens': int(generated.shape[1]-inputs.input_ids.shape[1]),
            'truncated': int(generated.shape[1]-inputs.input_ids.shape[1]) >= 1100}

cover_check = colab_describe_page(1, 'Read only the actual title visible on this cover. Do not guess.')
assert 'Middle' in cover_check['text'] and 'Trap' in cover_check['text'], cover_check['text']
Path('/content/vision_preflight.json').write_text(json.dumps(cover_check, indent=2))
print('VISION_READY:' + json.dumps(cover_check), flush=True)
