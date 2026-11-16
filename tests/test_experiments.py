"""Offline contract tests. Synthetic fixtures assess code, not benchmark quality."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from src import experiments as exp


class ExperimentContracts(unittest.TestCase):
    def test_missing_and_failed_retrieval_remain_in_denominator(self):
        cases = [{'id': 'a', 'answerable': True}, {'id': 'b', 'answerable': True},
                 {'id': 'c', 'answerable': True}, {'id': 'd', 'answerable': False}]
        result = exp.summarize_retrieval(cases, [{'query_id': 'a', 'hit1': True, 'hit3': True,
                                                'context_hit': True}, {'query_id': 'b', 'error': 'timeout'}])
        self.assertEqual(result['Evidence Hit@3'], {'n': 1, 'N': 3, 'rate': 1/3})
        self.assertEqual(result['errors'], 1)

    def test_every_multihop_group_required(self):
        case = {'evidence_groups': [[{'source_id': 's', 'page': 1, 'evidence': 'alpha'}],
                                    [{'source_id': 's', 'page': 2, 'evidence': 'beta'}]],
                'gold': [{'page': 99, 'evidence': 'misleading alternative'}]}
        chunks = [{'source_id': 's', 'page': 1, 'text': 'alpha'},
                  {'source_id': 's', 'page': 2, 'text': 'beta'}]
        self.assertFalse(exp.evidence_complete(exp.groups(case), chunks[:1]))
        self.assertTrue(exp.evidence_complete(exp.groups(case), chunks))

    def test_agent_boundary_contains_only_question_retriever_config(self):
        case = {'id': 'fixture', 'question': 'public question', 'answerable': True,
                'expected_answer': 'secret', 'gold': [{'evidence': 'secret'}],
                'evidence_groups': [[{'evidence': 'secret'}]], 'absence_reason': 'secret'}
        config = {'id': 'fixture', 'method': 'bm25'}
        retriever = object()
        with tempfile.TemporaryDirectory() as folder, patch.object(exp, 'run_agent', return_value={
                'status': 'abstain', 'answer': 'insufficient', 'citations': [], 'events': [],
                'seconds': 0}) as run:
            exp.agent_run(retriever, [case], config, Path(folder), workers=1)
            run.assert_called_once_with('public question', retriever, config)

    def test_advanced_denominator_includes_error(self):
        cases = [{'id': 'a', 'answerable': True, 'group': 'text'},
                 {'id': 'b', 'answerable': True, 'group': 'text'}]
        traces = [{'query_id': 'a', 'status': 'answered', 'events': [], 'seconds': 1},
                  {'query_id': 'b', 'status': 'error', 'events': [], 'seconds': 1}]
        reviews = [{'query_id': 'a', 'answer_grade': 2, 'citation_support': [True], 'pass': True},
                   {'query_id': 'b', 'pass': False}]
        result = exp.summarize_agent(cases, traces, reviews)
        self.assertEqual(result['advanced_pass'], {'n': 1, 'N': 2, 'rate': .5})
        self.assertEqual(result['groups']['text']['N'], 2)
        self.assertEqual(result['errors'], 1)
        with self.assertRaises(ValueError):
            exp.summarize_agent(cases, traces[:1], reviews)


if __name__ == '__main__':
    unittest.main()
