import unittest
from src.pipeline import chunk_blocks, evidence_complete, aggregate_advanced, public_case


class PipelineContracts(unittest.TestCase):
    def test_chunk_keeps_page_region_and_does_not_cross_pages(self):
        blocks = [{'page': 9, 'bbox': [1, 2, 3, 4], 'text': 'a ' * 600}]
        chunks = chunk_blocks(blocks, max_chars=900, overlap=120)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c['page'] == 9 and c['metadata']['bbox'] == [1, 2, 3, 4] for c in chunks))
        self.assertTrue(all(len(c['text']) <= 900 for c in chunks))
        self.assertEqual(len(set(c['id'] for c in chunks)), len(chunks))

    def test_multihop_requires_every_group_but_allows_alternatives(self):
        groups = [[{'page': 1, 'evidence': 'alpha'}, {'page': 2, 'evidence': 'beta'}], [{'page': 3, 'evidence': 'gamma'}]]
        chunks = [{'page': 2, 'text': 'beta'}, {'page': 3, 'text': 'gamma'}]
        self.assertTrue(evidence_complete(groups, chunks))
        self.assertFalse(evidence_complete(groups, chunks[:1]))

    def test_advanced_errors_stay_in_denominator(self):
        out = aggregate_advanced([{'id': 'a', 'group': 'text'}, {'id': 'b', 'group': 'text'}],
                                 [{'query_id': 'a', 'pass': True}, {'query_id': 'b', 'pass': False, 'error': 'timeout'}])
        self.assertEqual(out['overall'], {'n': 1, 'N': 2, 'rate': .5})
        self.assertEqual(out['errors'], 1)
        self.assertFalse(out['bonus_threshold_met'])
        with self.assertRaises(ValueError):
            aggregate_advanced([{'id': 'a', 'group': 'text'}], [])

    def test_gold_never_sent_to_agent(self):
        case = {'id': 'x', 'question': 'why?', 'gold': ['secret'], 'expected_answer': 'secret',
                'answerable': False, 'group': 'table', 'absence_reason': 'secret'}
        self.assertEqual(public_case(case), {'id': 'x', 'question': 'why?'})


if __name__ == '__main__':
    unittest.main()
