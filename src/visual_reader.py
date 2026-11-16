"""Question-directed original-page reading in the existing authenticated Colab kernel."""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import threading

import fitz

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
_VISUAL_LOCK = threading.Lock()


def parse_visual_output(output):
    lines = [line[len('VISUAL_RESULT:'):] for line in output.splitlines() if line.startswith('VISUAL_RESULT:')]
    if len(lines) != 1:
        raise RuntimeError('Colab did not return exactly one visual result')
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise RuntimeError('Colab returned invalid visual JSON') from error
    if not isinstance(result, dict) or result.get('error'):
        raise RuntimeError('Colab visual inference failed: ' + str(result))
    return result


class VisualReader:
    def __init__(self, config=None, *, pdf_path=None, cache_path=None):
        self.config = dict(config) if config is not None else json.loads((ROOT/'configs/colab_embedding.json').read_text(encoding='utf-8'))
        self.pdf_path = Path(pdf_path or ROOT/'data/source/world-development-report-2024.pdf')
        self.pdf_sha256 = hashlib.sha256(self.pdf_path.read_bytes()).hexdigest()
        with fitz.open(self.pdf_path) as document:
            self.page_sizes = [(float(p.rect.width), float(p.rect.height)) for p in document]
        self.cache_path = Path(cache_path or ROOT/'data/visual_cache/visual.sqlite')
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.cache_path)) as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS visual (key TEXT PRIMARY KEY, result TEXT NOT NULL)')
            connection.commit()

    def _validate(self, response, page, question):
        expected = {'page': page, 'question': question, 'model': MODEL, 'revision': REVISION,
                    'pdf_sha256': self.pdf_sha256, 'page_size': list(self.page_sizes[page-1])}
        if not isinstance(response, dict) or any(response.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Visual response page, question, document or model provenance mismatch')
        if not isinstance(response.get('text'), str) or not response['text'].strip():
            raise RuntimeError('Visual response has no actual answer')
        if type(response.get('truncated')) is not bool:
            raise RuntimeError('Visual response lacks truncation provenance')

    def read(self, page: int, question: str) -> dict:
        if type(page) is not int or not 1 <= page <= len(self.page_sizes):
            raise ValueError('page must be a 1-based original PDF page')
        if not isinstance(question, str) or not question.strip() or len(question) > 10000:
            raise ValueError('question must contain 1 to 10000 characters')
        question = question.strip()
        key = hashlib.sha256(json.dumps([self.pdf_sha256, MODEL, REVISION, page, question], ensure_ascii=False).encode()).hexdigest()
        with _VISUAL_LOCK:
            with closing(sqlite3.connect(self.cache_path)) as connection:
                row = connection.execute('SELECT result FROM visual WHERE key=?', (key,)).fetchone()
            response = json.loads(row[0]) if row else self._rpc(page, question)
            self._validate(response, page, question)
            if row is None:
                with closing(sqlite3.connect(self.cache_path)) as connection:
                    connection.execute('INSERT OR REPLACE INTO visual VALUES (?, ?)', (key, json.dumps(response)))
                    connection.commit()
        width, height = self.page_sizes[page-1]
        return {'id': f'wdr2024:p{page}:visual:{key[:20]}', 'text': response['text'], 'source_id': 'wdr2024',
                'source': 'https://www.worldbank.org/en/publication/wdr2024', 'page': page,
                'metadata': {'version': '2024', 'kind': 'visual_answer', 'bbox': [0., 0., width, height],
                             'page_size': [width, height], 'printed_page': None, 'question': question,
                             'model': MODEL, 'revision': REVISION, 'pdf_sha256': self.pdf_sha256,
                             'truncated': response['truncated'],
                             'extraction': 'Question-directed AI vision on original PDF page; verify against original'}}

    def _rpc(self, page, question):
        # JSON literals are encoded as a Python string, never interpolated into shell code.
        payload = json.dumps({'page': page, 'question': question}, ensure_ascii=True)
        code = ("import json\n" + f"_visual_request = json.loads({payload!r})\n"
                "try:\n"
                "    _visual_response = colab_read_visual(_visual_request['page'], _visual_request['question'])\n"
                "except Exception as _visual_error:\n"
                "    _visual_response = {'error': type(_visual_error).__name__ + ': ' + str(_visual_error)}\n"
                "print('VISUAL_RESULT:' + json.dumps(_visual_response), flush=True)\n")
        runtime = ROOT/'.runtime'
        runtime.mkdir(exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix='visual-', suffix='.py', dir=runtime)
        def wsl_path(path):
            path = Path(path).resolve()
            return '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]
        timeout = float(self.config.get('request_timeout_seconds', 180))
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
                file.write(code)
            process = subprocess.run(['wsl.exe', '-d', self.config['distro'], '--', 'timeout', '-k', '5', str(timeout+15),
                                      '/opt/colab-cli/bin/python', wsl_path(ROOT/'scripts/colab_rpc.py'),
                                      '--session', self.config['session'], '--file', wsl_path(filename), '--timeout', str(timeout)],
                                     capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout+30)
            if process.returncode:
                raise RuntimeError('Colab visual transport failed; check existing runtime and visual setup')
            return parse_visual_output(process.stdout)
        finally:
            Path(filename).unlink(missing_ok=True)
