"""Query the original Day 6 store locally using its own compatible environment.

Publishes metadata and excerpt hashes only, never copyrighted source prose.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval import LocalE5

query = 'Trường Hogwarts có những hoạt động học tập và sinh hoạt nào?'
embedder = LocalE5()
try:
    vector = embedder.embed_query(query)
finally:
    embedder.close()
code = '''
import hashlib,json,sys
from pathlib import Path
from qdrant_client import QdrantClient
request=json.load(sys.stdin)
client=QdrantClient(path=request['path'])
try:
    collections=[]
    for item in client.get_collections().collections:
        points=client.query_points(item.name,query=request['vector'],limit=3,with_payload=True).points
        collections.append({'collection':item.name,'points':client.count(item.name,exact=True).count,
            'results':[{'chunk_id':p.payload['chunk_id'],'document_id':p.payload['document_id'],
                        'page':p.payload['page'],'score':float(p.score),
                        'excerpt_sha256':hashlib.sha256(p.payload['text'].encode()).hexdigest()} for p in points]})
    print(json.dumps(collections))
finally:
    client.close()
'''
day6 = ROOT.parent/'K5-L3B-DAY06-PHAMQUANGTUAN-2A202603824-DataIngestion'
# Request data travels over stdin, not argv or a shareable source file.
worker_code = ROOT/'.runtime/day6-query-worker.py'
worker_code.write_text(code, encoding='utf-8')
process = subprocess.run([str(day6/'starter/.venv/Scripts/python.exe'), str(worker_code)],
                         input=json.dumps({'path': str(day6/'data/qdrant'), 'vector': vector}),
                         capture_output=True, text=True, encoding='utf-8', timeout=120, check=True)
result = {'query': query, 'embedding_model': 'intfloat/multilingual-e5-small', 'dimensions': 384,
          'connection': 'Real query against original Day 6 Qdrant using the original compatible Python environment. No upsert/delete, no source text sent to API.',
          'collections': json.loads(process.stdout)}
(ROOT/'results/day6_query_connection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print('Day 6 query verified:', len(result['collections']), 'collections, original IDs/pages/scores retained; text redacted')
