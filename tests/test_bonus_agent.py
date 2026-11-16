import json
import unittest

from src.bonus_agent import resolve_page, run_bonus_agent


class BonusAgentTests(unittest.TestCase):
    def test_page_resolution_only_uses_discovered_pages(self):
        self.assertEqual(resolve_page(50, {82}), (82, 'printed-page offset +32'))
        self.assertEqual(resolve_page(82, {82}), (82, 'original PDF page'))
        with self.assertRaises(ValueError):
            resolve_page(99, {82})

    def test_bad_page_request_is_returned_to_model_for_recovery(self):
        chunk = {'id': 'x', 'source_id': 'wdr2024', 'source': 'fixture', 'page': 82,
                 'text': 'Verified fact', 'metadata': {'bbox': [0, 0, 100, 100]}}
        class Retriever:
            by_page = {82: [chunk]}
            def search(self, *args, **kwargs):
                return [chunk]
        class Client:
            def __init__(self): self.n = 0
            def chat(self, messages, **kwargs):
                self.n += 1
                if self.n < 3:
                    name, args = ('search', {'query': 'fact'}) if self.n == 1 else ('read_page', {'page': 200})
                    return {'tool_calls': [{'id': str(self.n), 'function': {'name': name, 'arguments': json.dumps(args)}}]}, {}
                assert 'tool_error' in str(messages)
                return {'content': json.dumps({'status': 'answered', 'answer': 'Fact [x]', 'citations': ['x']})}, {}
        result = run_bonus_agent('fact?', Retriever(), {'method': 'bm25', 'top_k': 3, 'context_budget': 2000, 'max_rounds': 3}, client=Client())
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(sum(e['event'] == 'tool_error' for e in result['events']), 1)

    def test_visual_tool_cannot_read_gold_or_undiscovered_pages(self):
        with self.assertRaises(ValueError):
            resolve_page(True, {1})

