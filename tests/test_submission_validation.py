import importlib.util
from pathlib import Path
import tempfile
import unittest

import fitz

spec = importlib.util.spec_from_file_location('validator', Path(__file__).resolve().parents[1]/'scripts/validate_submission.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class SubmissionValidationTests(unittest.TestCase):
    def test_pending_is_distinct_from_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = validator.Audit(directory, True)
            self.assertIsNone(audit.read('missing.json'))
            self.assertTrue(audit.pending)
            self.assertFalse(audit.errors)
            Path(directory, 'bad.json').write_text('{', encoding='utf-8')
            audit.read('bad.json')
            self.assertTrue(audit.errors)

    def test_evidence_matches_original_and_bounds(self):
        document = fitz.open()
        page = document.new_page(width=200, height=200)
        page.insert_text((20, 40), 'Original evidence 42 percent')
        audit = validator.Audit('.')
        item = {'source_id': 'wdr2024', 'page': 1, 'evidence': 'Original evidence 42 percent', 'bbox': [10, 10, 190, 190]}
        audit.evidence(item, document, [validator.normalized(page.get_text())], 'case')
        self.assertFalse(audit.errors)
        item.update(evidence='Invented evidence', bbox=[0, 0, 201, 200])
        audit.evidence(item, document, [validator.normalized(page.get_text())], 'case')
        self.assertEqual(len(audit.errors), 2)
        document.close()


if __name__ == '__main__':
    unittest.main()
