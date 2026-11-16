"""Loopback embedding API backed by an authenticated, persistent Colab kernel.

No public tunnel, remote HTTP server, or OAuth token in requests/files. CLI
authentication stays in Windows Credential Manager. Model weights stay on Colab.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import closing
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import struct
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def validate_request(body, config):
    if not isinstance(body, dict) or body.get('model', config['model']) != config['model']:
        raise ValueError('Unknown model')
    texts = body.get('input')
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not 1 <= len(texts) <= config['max_batch']:
        raise ValueError('input must be a nonempty string or a bounded list of strings')
    if any(not isinstance(t, str) or not t.strip() or len(t) > config['max_input_chars'] for t in texts):
        raise ValueError('Inputs must be nonempty strings within max_input_chars')
    kind = body.get('input_type', 'document')
    if kind not in ('query', 'document'):
        raise ValueError('input_type must be query or document')
    if body.get('encoding_format', 'float') != 'float':
        raise ValueError('Only float encoding is supported')
    if body.get('dimensions', config['dimensions']) != config['dimensions']:
        raise ValueError('This index uses the configured full embedding dimension')
    return texts, kind


def validate_vectors(vectors, *, count, dimensions):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError('Remote batch size mismatch')
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise ValueError('Remote embedding dimension mismatch')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
            raise ValueError('Remote embedding contains nonfinite values')
        if abs(math.sqrt(sum(v*v for v in vector)) - 1) > .01:
            raise ValueError('Remote embedding is not L2 normalized')


def parse_remote_output(output):
    lines = [line[len('EMBED_RESULT:'):] for line in output.splitlines() if line.startswith('EMBED_RESULT:')]
    if len(lines) != 1:
        raise RuntimeError('Colab did not return exactly one complete embedding result')
    result = json.loads(lines[0])
    if result.get('error'):
        raise RuntimeError('Colab inference failed: ' + str(result['error']))
    return result


def validate_remote_config(response, config):
    if response.get('model') != config['model'] or response.get('revision') != config['revision']:
        raise RuntimeError('Remote model/revision differs from the configured index')
    settings = response.get('settings', {})
    for key in ('dimensions', 'normalization', 'query_instruction', 'max_sequence_length', 'precision'):
        if settings.get(key) != config[key]:
            raise RuntimeError(f'Remote embedding setting differs from configuration: {key}')


class EmbeddingCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)')
            connection.commit()

    @staticmethod
    def key(namespace, kind, text):
        return hashlib.sha256(json.dumps([namespace, kind, text], ensure_ascii=False).encode()).hexdigest()

    def get(self, namespace, kind, text):
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute('SELECT vector FROM embeddings WHERE key=?', (self.key(namespace, kind, text),)).fetchone()
        if row is None:
            return None
        return list(struct.unpack('<' + 'f' * (len(row[0]) // 4), row[0]))

    def put(self, namespace, kind, text, vector):
        blob = struct.pack('<' + 'f' * len(vector), *vector)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute('INSERT OR REPLACE INTO embeddings VALUES (?, ?)', (self.key(namespace, kind, text), blob))
            connection.commit()


class RemoteEmbedder:
    def __init__(self, config):
        self.config = config
        self.cache = EmbeddingCache(ROOT/'data/embedding_cache/qwen3_colab.sqlite')
        self.namespace = json.dumps({k: config[k] for k in ('model', 'revision', 'dimensions', 'normalization',
                                                          'query_instruction', 'max_sequence_length', 'precision')}, sort_keys=True)
        self.lock = threading.Lock()
        self.last_access = time.monotonic()
        self.runtime_active = True

    def command(self, *args):
        return ['wsl.exe', '-d', self.config['distro'], '--', self.config['cli'], *args]

    @staticmethod
    def wsl_path(path):
        path = Path(path).resolve()
        return '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]

    def embed(self, texts, kind):
        started = time.perf_counter()
        with self.lock:
            self.last_access = time.monotonic()
            vectors = [self.cache.get(self.namespace, kind, text) for text in texts]
            hits = sum(v is not None for v in vectors)
            missing = list(dict.fromkeys(t for t, v in zip(texts, vectors) if v is None))
            usage = 0
            inference = 0.
            if missing:
                if not self.runtime_active:
                    raise RuntimeError('GPU was released after inactivity. Restart the configured Colab session and model.')
                payload = base64.b64encode(json.dumps({'texts': missing, 'input_type': kind}, ensure_ascii=False).encode()).decode()
                code = ("import base64, json\n"
                        f"_embed_request = json.loads(base64.b64decode({payload!r}))\n"
                        "try:\n"
                        "    _embed_response = colab_embed(_embed_request['texts'], _embed_request['input_type'])\n"
                        "except Exception as _embed_error:\n"
                        "    _embed_response = {'error': type(_embed_error).__name__ + ': ' + str(_embed_error)}\n"
                        "print('EMBED_RESULT:' + json.dumps(_embed_response), flush=True)\n")
                folder = ROOT/'.runtime'
                folder.mkdir(exist_ok=True)
                fd, filename = tempfile.mkstemp(prefix='embed-', suffix='.py', dir=folder)
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as file:
                        file.write(code)
                    process = subprocess.run(['wsl.exe', '-d', self.config['distro'], '--',
                                              'timeout', '-k', '5', str(self.config['request_timeout_seconds']+15),
                                              '/opt/colab-cli/bin/python', self.wsl_path(ROOT/'scripts/colab_rpc.py'),
                                              '--session', self.config['session'], '--file', self.wsl_path(filename),
                                              '--timeout', str(self.config['request_timeout_seconds'])],
                                             capture_output=True, text=True, encoding='utf-8', errors='replace',
                                             timeout=self.config['request_timeout_seconds']+30)
                    if process.returncode:
                        raise RuntimeError('Colab transport failed; check runtime status (no synthetic fallback)')
                    response = parse_remote_output(process.stdout)
                finally:
                    Path(filename).unlink(missing_ok=True)
                validate_remote_config(response, self.config)
                validate_vectors(response['embeddings'], count=len(missing), dimensions=self.config['dimensions'])
                for text, vector in zip(missing, response['embeddings']):
                    self.cache.put(self.namespace, kind, text, vector)
                vectors = [self.cache.get(self.namespace, kind, text) for text in texts]
                usage = response['tokens']
                inference = response['inference_seconds']
            validate_vectors(vectors, count=len(texts), dimensions=self.config['dimensions'])
            self.last_access = time.monotonic()
        return {'object': 'list', 'model': self.config['model'], 'revision': self.config['revision'],
                'data': [{'object': 'embedding', 'index': i, 'embedding': v} for i, v in enumerate(vectors)],
                'usage': {'prompt_tokens': usage, 'total_tokens': usage},
                'bridge': {'input_type': kind, 'cache_hits': hits, 'latency_seconds': time.perf_counter()-started,
                           'inference_seconds': inference, 'runtime_active': self.runtime_active}}

    def idle_monitor(self):
        while True:
            time.sleep(15)
            self.release_if_idle()

    def release_if_idle(self, now=None):
        now = time.monotonic() if now is None else now
        minutes = self.config['idle_shutdown_minutes']
        if not minutes or now-self.last_access < minutes*60 or not self.lock.acquire(blocking=False):
            return
        try:
            if self.runtime_active and now-self.last_access >= minutes*60:
                stopped = subprocess.run(self.command('stop', '-s', self.config['session']), capture_output=True, timeout=90)
                if stopped.returncode == 0:
                    self.runtime_active = False
                    print('Released idle Colab GPU.', flush=True)
        except (OSError, subprocess.TimeoutExpired):
            print('Idle release failed; inspect Colab usage.', flush=True)
        finally:
            self.lock.release()


def run_server(config):
    runtime = ROOT/'.runtime'
    runtime.mkdir(exist_ok=True)
    secret_path = runtime/'bridge-key'
    if not secret_path.exists():
        secret_path.write_text(secrets.token_urlsafe(32), encoding='ascii')
    key = secret_path.read_text(encoding='ascii').strip()
    embedder = RemoteEmbedder(config)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, body):
            blob = json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def authenticated(self):
            return secrets.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+key)

        def do_GET(self):
            if self.path == '/health':
                self.send_json(200, {'status': 'ok', 'model': config['model'], 'revision': config['revision'],
                                     'dimensions': config['dimensions'], 'runtime_active': embedder.runtime_active,
                                     'idle_shutdown_minutes': config['idle_shutdown_minutes']})
            elif self.path == '/v1/models' and self.authenticated():
                self.send_json(200, {'object': 'list', 'data': [{'id': config['model'], 'object': 'model'}]})
            else:
                self.send_json(401 if self.path == '/v1/models' else 404, {'error': 'Unauthorized or unknown route'})

        def do_POST(self):
            if not self.authenticated():
                self.send_json(401, {'error': 'Unauthorized'})
                return
            if self.path == '/v1/activity':
                # Budget guard observes real external batch jobs as activity.
                # This changes no Colab liveness policy and runs no inference.
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 4096:
                    self.send_json(400, {'error': 'Invalid activity body'})
                    return
                self.rfile.read(length)
                embedder.last_access = time.monotonic()
                self.send_json(200, {'status': 'activity_recorded'})
                return
            if self.path != '/v1/embeddings':
                self.send_json(404, {'error': 'Unknown route'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 5_000_000:
                    raise ValueError('Invalid request body size')
                body = json.loads(self.rfile.read(length))
                texts, kind = validate_request(body, config)
                response = embedder.embed(texts, kind)
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {'error': str(exc)})
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                self.send_json(503, {'error': str(exc)})
            else:
                self.send_json(200, response)

        def log_message(self, format, *args):
            # Standard request line only, never request bodies or Authorization.
            print(time.strftime('%Y-%m-%d %H:%M:%S'), format % args, flush=True)

    server = ThreadingHTTPServer((config['host'], config['port']), Handler)
    threading.Thread(target=embedder.idle_monitor, daemon=True).start()
    (runtime/'bridge.pid').write_text(str(os.getpid()))
    print(f"Embedding bridge listening at http://{config['host']}:{config['port']}; model {config['model']}", flush=True)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/colab_embedding.json')
    arguments = parser.parse_args()
    run_server(json.loads(arguments.config.read_text(encoding='utf-8')))
