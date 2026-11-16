"""PDF-wide native extraction with stable example-compatible page/region IDs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import fitz

ROOT = Path(__file__).resolve().parents[1]
CAPTION = re.compile(r'\b(Figure|Table|Map)\s+([A-Z][0-9]*|[0-9]+)\.[0-9]+', re.IGNORECASE)
SOURCE_URL = 'https://www.worldbank.org/en/publication/wdr2024'


def clean(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def validate_chunks(chunks):
    ids = set()
    for chunk in chunks:
        if not all(key in chunk for key in ('id', 'text', 'source_id', 'source', 'page', 'metadata')):
            raise ValueError('Missing example schema field')
        if chunk['id'] in ids or not chunk['text'].strip() or type(chunk['page']) is not int or chunk['page'] < 1:
            raise ValueError('Duplicate ID, empty text, or invalid original PDF page')
        if not chunk['source_id'] or not chunk['source'] or len(chunk['metadata'].get('bbox', [])) != 4:
            raise ValueError('Missing source or page region')
        ids.add(chunk['id'])


def extract_pdf(path, source_id='wdr2024', max_chars=1300, overlap=180):
    doc = fitz.open(path)
    chunks, pages, regions = [], [], []
    for number, page in enumerate(doc, 1):
        width, height = page.rect.width, page.rect.height
        blocks = [{'bbox': [round(x, 2) for x in b[:4]], 'text': clean(b[4])}
                  for b in page.get_text('blocks', sort=True) if b[6] == 0 and clean(b[4])]
        page_text = page.get_text(sort=True)
        caption_count = 0
        for block in blocks:
            match = CAPTION.match(block['text'])
            # The contents list repeats figure names. Keep full native text but
            # do not create fake figure crops from the table of contents.
            if not match or source_id == 'wdr2024' and number < 32:
                continue
            label = match.group(0)
            bbox = block['bbox']
            full_width = bbox[2]-bbox[0] > width*.60
            if full_width:
                left, right = max(0, bbox[0]-5), min(width, max(bbox[2]+5, width-35))
            elif bbox[0] < width/2:
                left, right = max(0, bbox[0]-5), width/2+10
            else:
                left, right = width/2-10, min(width, bbox[2]+5)
            bottom = height-28
            for other in blocks:
                ob = other['bbox']
                if ob[1] <= bbox[3] or ob[0] >= right or ob[2] <= left:
                    continue
                if re.match(r'^(Sources?|Notes?):', other['text'], re.I):
                    bottom = min(bottom, ob[3]+4)
                elif CAPTION.match(other['text']):
                    bottom = min(bottom, ob[1]-3)
            region_box = [round(left, 2), bbox[1], round(right, 2), round(max(bbox[3], bottom), 2)]
            region_id = f"{source_id}:p{number}:region{caption_count}"
            region = {'id': region_id, 'page': number, 'label': label, 'caption': block['text'], 'bbox': region_box,
                      'kind': 'table' if match.group(1).lower() == 'table' else 'visual',
                      'native_text': clean(page.get_text(clip=fitz.Rect(region_box), sort=True)),
                      'crop': f'data/crops/p{number:03d}-region{caption_count}.png',
                      'crop_method': 'PyMuPDF 1.6x matrix and original page bbox; local reference, not separately redistributed',
                      'limitations': 'Automatic approximate region; verify original whole page for boundaries and visual claims.'}
            regions.append(region)
            caption_count += 1
        # Combine adjacent native blocks without crossing an original page.
        groups, current = [], []
        for block in blocks:
            if current and sum(len(b['text'])+1 for b in current)+len(block['text']) > max_chars:
                groups.append(current)
                current = []
            current.append(block)
        if current:
            groups.append(current)
        for gi, group in enumerate(groups):
            text = '\n'.join(b['text'] for b in group)
            bbox = union([b['bbox'] for b in group])
            start = part = 0
            while start < len(text):
                end = min(len(text), start+max_chars)
                if end < len(text):
                    cut = text.rfind(' ', start+max_chars//2, end)
                    if cut > start:
                        end = cut
                chunks.append({'id': f'{source_id}:p{number}:g{gi}:c{part}', 'text': text[start:end],
                               'source_id': source_id, 'source': SOURCE_URL, 'page': number,
                               'metadata': {'version': '2024', 'bbox': bbox, 'kind': 'native_text',
                                            'printed_page': str(number-32) if number >= 33 else None,
                                            'page_size': [round(width, 2), round(height, 2)],
                                            'extraction': 'PyMuPDF native text blocks; NFC and whitespace normalization',
                                            'offset_start': start, 'offset_end': end}})
                if end == len(text):
                    break
                start = max(start+1, end-overlap)
                part += 1
        pages.append({'page': number, 'printed_page': str(number-32) if number >= 33 else None,
                      'status': 'native_text' if blocks else 'blank', 'native_characters': len(page_text.strip()),
                      'native_blocks': len(blocks), 'visual_regions': caption_count,
                      'raster_images': len(page.get_images()), 'vector_paths': len(page.get_drawings()),
                      'issues': [] if blocks else ['No native text; inspect blank/decorative/scanned page.']})
    validate_chunks(chunks)
    return chunks, pages, regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, default=ROOT/'data/source/world-development-report-2024.pdf')
    args = parser.parse_args()
    chunks, pages, regions = extract_pdf(args.pdf)
    # Use Day 6's local tokenizer to keep an identical, E5-compatible corpus
    # for both embedding spaces; never download the already available model.
    from tokenizers import Tokenizer
    model_path = ROOT.parent/'K5-L3B-DAY06-PHAMQUANGTUAN-2A202603824-DataIngestion/data/models/multilingual-e5-small'
    tokenizer = Tokenizer.from_file(str(model_path/'tokenizer.json'))
    safe = []
    for chunk in chunks:
        queue = [chunk]
        while queue:
            item = queue.pop(0)
            length = len(tokenizer.encode('passage: '+item['text']).ids)
            if length <= 480:
                item['metadata']['e5_input_tokens'] = length
                safe.append(item)
                continue
            cut = item['text'].rfind(' ', 0, len(item['text'])//2)
            cut = cut if cut > 0 else len(item['text'])//2
            for suffix, text in (('a', item['text'][:cut]), ('b', item['text'][cut:])):
                queue.append({**item, 'id': item['id']+suffix, 'text': text.strip(), 'metadata': {**item['metadata']}})
    chunks = safe
    validate_chunks(chunks)
    target = ROOT/'data/processed'
    target.mkdir(parents=True, exist_ok=True)
    for filename, content in [('chunks_native.json', chunks), ('page_coverage.json', pages), ('visual_regions.json', regions)]:
        (target/filename).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    visual_pages = sorted({r['page'] for r in regions})
    (target/'visual_pages.json').write_text(json.dumps(visual_pages), encoding='utf-8')
    manifest = {'source_id': 'wdr2024', 'title': 'World Development Report 2024: The Middle-Income Trap',
                'source': SOURCE_URL, 'pdf_path': str(args.pdf.relative_to(ROOT)),
                'sha256': hashlib.sha256(args.pdf.read_bytes()).hexdigest(), 'pdf_pages': len(pages),
                'native_chunks': len(chunks), 'visual_regions': len(regions), 'visual_pages': len(visual_pages),
                'page_status_counts': dict(Counter(p['status'] for p in pages)),
                'license': 'CC BY 3.0 IGO; PDF page 6. Third-party components may require separate permission.',
                'page_numbering': 'page is original PDF page, 1-based; printed arabic page = PDF page - 32 after front matter',
                'chunking': {'native_max_characters': 1300, 'long_block_overlap_characters': 180, 'e5_max_input_tokens': 480},
                'limitations': ['Visual descriptions pending; captions/native labels alone do not encode all visual relationships.',
                                'Automatic visual regions are approximate; original page is authoritative.']}
    (ROOT/'data/source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
