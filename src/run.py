"""Small lab CLI; query input is separate from every benchmark gold file."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .agent import run_agent
from .retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description='Day 7 source-grounded report agent')
parser.add_argument('command', choices=['verify', 'search', 'answer', 'restore'])
parser.add_argument('--question')
parser.add_argument('--method', choices=['bm25', 'dense_e5', 'dense_qwen', 'hybrid_e5', 'hybrid_qwen'], default='hybrid_qwen')
parser.add_argument('--top-k', type=int, default=3)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
if args.command == 'verify':
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/validate_submission.py')], cwd=ROOT, check=True)
elif args.command == 'restore':
    from .indexing import restore_indexes
    restore_indexes()
else:
    if not args.question or not args.question.strip():
        parser.error('--question is required')
    retriever = Retriever()
    try:
        if args.command == 'search':
            result = retriever.search(args.question, args.method, args.top_k)
        else:
            lock = ROOT/'results/selection_lock.json'
            config = json.loads(lock.read_text(encoding='utf-8'))['selected'] if lock.exists() else {
                'id': 'interactive', 'method': args.method, 'top_k': 8, 'context_budget': 16000,
                'max_rounds': 3, 'page_expansion': True, 'model': 'gemini-3.8-flash-high'}
            result = run_agent(args.question, retriever, config)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=True, indent=2))
    finally:
        retriever.close()
