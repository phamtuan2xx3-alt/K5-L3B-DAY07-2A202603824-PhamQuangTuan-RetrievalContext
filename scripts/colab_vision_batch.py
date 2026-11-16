"""Batched vision extraction, resuming per-page checkpoints from the first run."""
import json
from pathlib import Path
import re
import time

vision_processor.tokenizer.padding_side = 'left'
visual_output = Path('/content/wdr_visual_descriptions.jsonl')
already = {}
if visual_output.exists():
    for line in visual_output.read_text().splitlines():
        record = json.loads(line)
        if record.get('text') and not record.get('error'):
            already[record['page']] = record
visual_pages = [i+1 for i, page in enumerate(vision_pdf)
                if i+1 >= 32 and any(re.match(r'\s*(Figure|Table|Map)\s+[A-Za-z0-9]+\.[0-9]+', b[4])
                                    for b in page.get_text('blocks') if b[6] == 0)]
remaining = [p for p in visual_pages if p not in already]
print('VISION_BATCH_START', len(remaining), 'remaining of', len(visual_pages), flush=True)
for offset in range(0, len(remaining), 4):
    selected = remaining[offset:offset+4]
    conversations = []
    for page_number in selected:
        page = vision_pdf[page_number-1]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        image.thumbnail((1400, 1400))
        prompt = ('Describe every informative table/chart/map/diagram on this page in compact English (at most 300 words). '
                  'Copy figure/table identifiers. Explain actual visual relationships and readable row/column values, '
                  'axes, legend and units. Do not invent unreadable values; state uncertainty. '
                  'Ignore prose and decoration. Native labels for reference:\n'+page.get_text()[:5000])
        conversations.append([{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}])
    started = time.perf_counter()
    try:
        inputs = vision_processor.apply_chat_template(conversations, tokenize=True, padding=True,
                                                       add_generation_prompt=True, return_dict=True,
                                                       return_tensors='pt').to('cuda')
        with torch.inference_mode():
            generated = vision_model.generate(**inputs, max_new_tokens=650, do_sample=False)
        tokens = generated[:, inputs.input_ids.shape[1]:]
        descriptions = vision_processor.batch_decode(tokens, skip_special_tokens=True)
        for number, page_number in enumerate(selected):
            count = int((tokens[number] != vision_processor.tokenizer.pad_token_id).sum())
            record = {'page': page_number, 'text': descriptions[number], 'model': VISION_ID, 'revision': VISION_REVISION,
                      'seconds': (time.perf_counter()-started)/len(selected), 'output_tokens': count,
                      'method': 'AI vision description, batch4; original PDF remains authoritative', 'truncated': count >= 650}
            already[page_number] = record
            with visual_output.open('a') as file:
                file.write(json.dumps(record, ensure_ascii=False)+'\n')
    except Exception as error:
        print('VISION_BATCH_ERROR', selected, type(error).__name__, str(error), flush=True)
        for page_number in selected:
            record = {'page': page_number, 'error': type(error).__name__+': '+str(error)}
            already[page_number] = record
            with visual_output.open('a') as file:
                file.write(json.dumps(record)+'\n')
    Path('/content/wdr_visual_descriptions.json').write_text(json.dumps(list(already.values()), ensure_ascii=False, indent=2))
    print('VISION_PROGRESS', len(already), '/', len(visual_pages), 'batch_pages', selected,
          'batch_seconds', round(time.perf_counter()-started, 2), flush=True)
print('VISION_COMPLETE', len(already), flush=True)
