"""Local original-page audit gallery. Crops are regenerated, not redistributed."""
import argparse
import html
import json
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--suite', choices=['basic', 'advanced', 'all'], default='all')
args = parser.parse_args()
doc = fitz.open(ROOT/'data/source/world-development-report-2024.pdf')
folder = ROOT/'data/crops'
folder.mkdir(exist_ok=True)
sections = []
for suite in (('basic', 'advanced') if args.suite == 'all' else (args.suite,)):
    cases = json.loads((ROOT/f'data/benchmark/{suite}_questions.json').read_text(encoding='utf-8'))
    for case in cases:
        evidence = [e for group in (case.get('evidence_groups') or [case.get('gold', [])]) for e in group]
        pictures = []
        for i, item in enumerate(evidence):
            page = doc[item['page']-1]
            bbox = fitz.Rect(item['bbox']) & page.rect
            if bbox.width < 100 or bbox.height < 100:
                bbox = fitz.Rect(max(0,bbox.x0-30),max(0,bbox.y0-60),min(page.rect.width,bbox.x1+30),min(page.rect.height,bbox.y1+60))
            name = f"{case['id']}-{i}.png"
            page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),clip=bbox).save(folder/name)
            pictures.append(f'<p>Original PDF page {item["page"]}, bbox {item["bbox"]}</p><img src="../data/crops/{name}" style="max-width:100%;border:1px solid #ccc"><pre>{html.escape(item["evidence"])}</pre>')
        expected = case.get('expected_answer') or case.get('absence_reason') or ''
        sections.append(f'<section><h2>{case["id"]} — {html.escape(case["question"])}</h2><p>Expected: {html.escape(expected)}</p>'+''.join(pictures)+'</section>')
output = ROOT/'report/evidence.local.html'
output.write_text('<!doctype html><meta charset="utf-8"><title>Original source audit</title><style>body{max-width:1000px;margin:40px auto;font:16px system-ui}section{padding:24px 0;border-bottom:2px solid #888}pre{white-space:pre-wrap}</style><h1>Original-page evidence audit</h1><p>Local crops regenerated from the licensed source; AI-authored expected answers require verification.</p>'+''.join(sections),encoding='utf-8')
print('Generated',output,'with',len(sections),'cases')
