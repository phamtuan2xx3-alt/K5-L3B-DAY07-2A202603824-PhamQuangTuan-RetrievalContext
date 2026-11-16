"""Describe all useful visual pages, checkpointing independently of benchmarks."""
import json
from pathlib import Path
import re

visual_output = Path('/content/wdr_visual_descriptions.jsonl')
already = {}
if visual_output.exists():
    for line in visual_output.read_text().splitlines():
        record = json.loads(line)
        if 'text' in record and not record.get('error'):
            already[record['page']] = record
visual_pages = [i+1 for i, page in enumerate(vision_pdf)
                if i+1 >= 32 and any(re.match(r'\s*(Figure|Table|Map)\s+[A-Za-z0-9]+\.[0-9]+', block[4])
                                    for block in page.get_text('blocks') if block[6] == 0)]
print('VISUAL_PAGE_COUNT', len(visual_pages), flush=True)
for count, page_number in enumerate(visual_pages, 1):
    if page_number in already:
        continue
    try:
        record = colab_describe_page(page_number)
    except Exception as error:
        record = {'page': page_number, 'error': type(error).__name__ + ': ' + str(error)}
    with visual_output.open('a') as file:
        file.write(json.dumps(record, ensure_ascii=False)+'\n')
    already[page_number] = record
    print('VISION_PROGRESS', count, '/', len(visual_pages), 'page', page_number, 'error', bool(record.get('error')), flush=True)
Path('/content/wdr_visual_descriptions.json').write_text(json.dumps(list(already.values()), ensure_ascii=False, indent=2))
print('VISION_COMPLETE', len(already), flush=True)
