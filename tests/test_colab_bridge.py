import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.colab_bridge import EmbeddingCache, RemoteEmbedder, validate_request, validate_vectors, parse_remote_output, validate_remote_config


class BridgeContracts(unittest.TestCase):
    def test_rejects_empty_oversize_and_wrong_model(self):
        config = {'model': 'test', 'max_batch': 3, 'max_input_chars': 30, 'dimensions': 2}
        self.assertEqual(validate_request({'model': 'test', 'input': 'hello'}, config), (['hello'], 'document'))
        for request in ({'input': ''}, {'input': ['a'] * 4}, {'input': 'x' * 31},
                        {'input': 'ok', 'model': 'other'}, {'input': 'ok', 'input_type': 'unsupported'}):
            with self.assertRaises(ValueError):
                validate_request(request, config)

    def test_vector_validation_rejects_nan_dimension_and_nonunit(self):
        validate_vectors([[.6, .8]], count=1, dimensions=2)
        for vectors in ([[math.nan, 1]], [[1]], [[2, 0]], []):
            with self.assertRaises(ValueError):
                validate_vectors(vectors, count=1, dimensions=2)

    def test_cache_separates_revision_and_query_document(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = EmbeddingCache(Path(folder)/'cache.sqlite')
            cache.put('model@r1', 'document', 'text', [1., 0.])
            self.assertEqual(cache.get('model@r1', 'document', 'text'), [1., 0.])
            self.assertIsNone(cache.get('model@r2', 'document', 'text'))
            self.assertIsNone(cache.get('model@r1', 'query', 'text'))

    def test_transport_requires_one_structured_response(self):
        response = parse_remote_output('Colab log\nEMBED_RESULT:{"embeddings":[[1,0]],"tokens":2}\n')
        self.assertEqual(response['tokens'], 2)
        for output in ('no result', 'EMBED_RESULT:{}\nEMBED_RESULT:{}'):
            with self.assertRaises(RuntimeError):
                parse_remote_output(output)

    def test_idle_release_does_not_stop_an_active_request_and_releases_once(self):
        with tempfile.TemporaryDirectory() as folder, patch('src.colab_bridge.ROOT', Path(folder)):
            config = dict(model='test', revision='r1', dimensions=2, normalization='L2',
                          query_instruction='test', max_sequence_length=100, precision='float32',
                          distro='Debian', cli='colab', session='test', idle_shutdown_minutes=30)
            remote = RemoteEmbedder(config)
            remote.last_access = 0
            with patch('src.colab_bridge.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run:
                remote.release_if_idle(now=1799)
                run.assert_not_called()
                remote.lock.acquire()
                remote.release_if_idle(now=1801)
                run.assert_not_called()
                remote.lock.release()
                remote.release_if_idle(now=1801)
                self.assertFalse(remote.runtime_active)
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0][-3:], ['stop', '-s', 'test'])
                remote.release_if_idle(now=1900)
                run.assert_called_once()

    def test_remote_instruction_mismatch_never_enters_cache(self):
        config = dict(model='test', revision='r1', dimensions=2, normalization='L2',
                      query_instruction='new instruction', max_sequence_length=100, precision='float32')
        response = {'model': 'test', 'revision': 'r1', 'settings': {k: v for k, v in config.items() if k not in ('model', 'revision')}}
        validate_remote_config(response, config)
        response['settings']['query_instruction'] = 'old instruction'
        with self.assertRaisesRegex(RuntimeError, 'query_instruction'):
            validate_remote_config(response, config)

    def test_query_adapter_rejects_same_dimension_wrong_revision(self):
        import json
        from src.colab_embeddings import ColabEmbedder
        with tempfile.TemporaryDirectory() as folder, patch('src.colab_embeddings.ROOT', Path(folder)):
            root = Path(folder)
            (root/'configs').mkdir()
            (root/'data').mkdir()
            config = dict(model='test', revision='new', dimensions=2, normalization='L2',
                          precision='float32', query_instruction='search')
            (root/'configs/colab_embedding.json').write_text(json.dumps(config), encoding='utf-8')
            namespace = {**config, 'revision': 'old'}
            (root/'data/index_manifest.json').write_text(json.dumps({'spaces': {'qwen': {'namespace': json.dumps(namespace)}}}), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'revision'):
                ColabEmbedder()


if __name__ == '__main__':
    unittest.main()
