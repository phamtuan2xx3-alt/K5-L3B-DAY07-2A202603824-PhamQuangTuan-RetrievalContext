import unittest
from src.agent import AgentRunError, pack_context, parse_final, run_agent


class LabAgentTests(unittest.TestCase):
    def test_context_keeps_only_complete_chunks_and_sources(self):
        chunks = [{'id': 'a', 'page': 9, 'text': 'facts', 'metadata': {'bbox': [1, 2, 3, 4]}}]
        text, kept = pack_context(chunks+chunks, budget=200)
        self.assertEqual(len(kept), 1)
        self.assertIn('PDF page 9', text)
        self.assertIn('[a]', text)
        self.assertEqual(pack_context(chunks, budget=2), ('', []))

    def test_final_citations_must_exist_and_be_inline(self):
        result = parse_final('{"status":"answered","answer":"Fact [a]","citations":["a"]}', ['a'])
        self.assertEqual(result['status'], 'answered')
        for message in ('{"status":"answered","answer":"Fact [wrong]","citations":["wrong"]}',
                        '{"status":"answered","answer":"Fact","citations":["a"]}'):
            with self.assertRaises(ValueError):
                parse_final(message, ['a'])
        self.assertEqual(parse_final('{"status":"abstain","answer":"Not enough evidence","citations":[]}', [])['status'], 'abstain')

    def test_failed_tool_retains_the_actual_requested_parameters(self):
        class Client:
            def chat(self, *args, **kwargs):
                return {'tool_calls': [{'id': 'call1', 'function': {'name': 'search', 'arguments': '{"query":"public query"}'}}]}, {'model': 'fixture'}
        class BrokenRetriever:
            def search(self, *args, **kwargs):
                raise RuntimeError('fixture transport failure')
        with self.assertRaises(AgentRunError) as raised:
            run_agent('public question', BrokenRetriever(), {'method': 'bm25', 'top_k': 3, 'context_budget': 200, 'max_rounds': 1}, Client())
        request = next(e for e in raised.exception.trace['events'] if e['event'] == 'tool_request')
        self.assertEqual(request['tool'], 'search')
        self.assertEqual(request['arguments'], {'query': 'public query'})
        self.assertEqual(request['native_tool_call_id'], 'call1')
