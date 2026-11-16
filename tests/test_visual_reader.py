import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz

from src.visual_reader import VisualReader, MODEL, REVISION, parse_visual_output


class VisualReaderContracts(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.pdf = Path(self.folder.name)/'page.pdf'
        doc = fitz.open()
        doc.new_page(width=576, height=756)
        doc.save(self.pdf)
        doc.close()
        self.reader = VisualReader({'session': 'test', 'distro': 'Ubuntu', 'request_timeout_seconds': 20},
                                   pdf_path=self.pdf, cache_path=Path(self.folder.name)/'cache.sqlite')
        self.result = {'page': 1, 'question': 'Which line is blue?', 'text': 'The blue line denotes investment.',
                       'model': MODEL, 'revision': REVISION, 'pdf_sha256': hashlib.sha256(self.pdf.read_bytes()).hexdigest(),
                       'page_size': [576., 756.], 'truncated': False}

    def test_actual_answer_schema_and_cache(self):
        with patch.object(self.reader, '_rpc', return_value=self.result) as rpc:
            chunk = self.reader.read(1, self.result['question'])
            self.assertEqual(chunk['text'], self.result['text'])
            self.assertTrue(chunk['id'].startswith('wdr2024:p1:visual:'))
            self.assertEqual(chunk['metadata']['bbox'], [0., 0., 576., 756.])
            self.assertEqual(chunk['metadata']['revision'], REVISION)
            self.assertEqual(self.reader.read(1, self.result['question']), chunk)
            rpc.assert_called_once()
        with patch.object(self.reader, '_rpc', return_value={**self.result, 'question': 'What units?'}) as rpc:
            self.reader.read(1, 'What units?')
            rpc.assert_called_once()

    def test_rejects_wrong_provenance_before_caching(self):
        for changes in ({'page': 2}, {'model': 'wrong'}, {'revision': 'wrong'}, {'pdf_sha256': 'wrong'},
                        {'question': 'wrong'}, {'page_size': [10, 10]}, {'text': ''}):
            with self.subTest(changes=changes), patch.object(self.reader, '_rpc', return_value={**self.result, **changes}):
                with self.assertRaises(RuntimeError):
                    self.reader.read(1, self.result['question'])

    def test_input_and_transport_errors_are_real(self):
        for page, question in ((0, 'x'), (2, 'x'), (True, 'x'), (1, ''), (1, 'x'*10001)):
            with self.assertRaises(ValueError):
                self.reader.read(page, question)
        with patch.object(self.reader, '_rpc', side_effect=RuntimeError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                self.reader.read(1, 'question')
        for output in ('nothing', 'VISUAL_RESULT:{}\nVISUAL_RESULT:{}', 'VISUAL_RESULT:[]',
                       'VISUAL_RESULT:{"error":"out of memory"}', 'VISUAL_RESULT:bad'):
            with self.assertRaises(RuntimeError):
                parse_visual_output(output)

    def test_rpc_uses_bounded_existing_kernel_and_cleans_script(self):
        import json
        with patch('src.visual_reader.subprocess.run', return_value=SimpleNamespace(
                returncode=0, stdout='VISUAL_RESULT:' + json.dumps(self.result))) as run:
            self.assertEqual(self.reader._rpc(1, self.result['question']), self.result)
            arguments, options = run.call_args
            command = arguments[0]
            self.assertIn('/opt/colab-cli/bin/python', command)
            self.assertEqual(command[command.index('--session')+1], 'test')
            self.assertEqual(options['timeout'], 50)
            script_path = command[command.index('--file')+1]
            local_path = Path(script_path[5].upper()+':'+script_path[6:])
            self.assertFalse(local_path.exists())
        with patch('src.visual_reader.subprocess.run', return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(RuntimeError, 'transport failed'):
                self.reader._rpc(1, 'question')


if __name__ == '__main__':
    unittest.main()
