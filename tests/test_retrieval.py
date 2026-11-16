import tempfile
from pathlib import Path
import unittest
from src.retrieval import VectorStore, rrf


class RetrievalTests(unittest.TestCase):
    def test_real_qdrant_roundtrip_and_namespace_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            store = VectorStore(Path(folder), dimensions=2, namespace='test-r1')
            chunk = {'id': 'c1', 'text': 'facts', 'source_id': 'doc', 'source': 'source.pdf',
                     'page': 3, 'metadata': {'bbox': [0, 0, 10, 10]}}
            store.upsert([chunk], [[.6, .8]])
            self.assertEqual(store.search([.6, .8], 3)[0]['id'], 'c1')
            self.assertEqual(store.search([.6, .8], 3)[0]['page'], 3)
            store.close()
            with self.assertRaises(ValueError):
                VectorStore(Path(folder), dimensions=2, namespace='other-model')

    def test_rrf_deduplicates_without_summing_raw_scores(self):
        a = [{'id': 'x', 'score': 500}, {'id': 'y', 'score': 1}]
        b = [{'id': 'y', 'score': .99}, {'id': 'z', 'score': .9}]
        result = rrf([a, b], top_k=3)
        self.assertEqual(result[0]['id'], 'y')
        self.assertEqual(len({x['id'] for x in result}), 3)
