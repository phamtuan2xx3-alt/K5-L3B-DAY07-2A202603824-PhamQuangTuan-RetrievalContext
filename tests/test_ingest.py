from pathlib import Path
import tempfile
import unittest
import fitz
from src.ingest import extract_pdf, validate_chunks


class IngestionTests(unittest.TestCase):
    def test_original_pdf_page_and_visual_region_survive(self):
        with tempfile.TemporaryDirectory() as folder:
            pdf = Path(folder)/'source.pdf'
            doc = fitz.open()
            doc.new_page()
            page = doc.new_page()
            page.insert_text((50, 80), 'Figure 1.1 Example process')
            page.insert_text((50, 120), 'Investment -> Infusion -> Innovation')
            page.insert_text((50, 180), 'Source: WDR team.')
            doc.save(pdf)
            chunks, pages, regions = extract_pdf(pdf, source_id='fixture')
            self.assertEqual([p['page'] for p in pages], [1, 2])
            self.assertEqual(pages[0]['status'], 'blank')
            self.assertTrue(chunks)
            self.assertTrue(all(c['page'] == 2 for c in chunks))
            self.assertTrue(all(c['metadata']['bbox'] for c in chunks))
            self.assertEqual(regions[0]['label'], 'Figure 1.1')
            self.assertEqual(regions[0]['page'], 2)
            validate_chunks(chunks)

    def test_validator_rejects_duplicate_and_invalid_provenance(self):
        chunk = {'id': 'a', 'source_id': 'x', 'source': 'source.pdf', 'page': 1,
                 'text': 'text', 'metadata': {'bbox': [0, 0, 100, 100]}}
        validate_chunks([chunk])
        with self.assertRaises(ValueError):
            validate_chunks([chunk, chunk])
        with self.assertRaises(ValueError):
            validate_chunks([{**chunk, 'page': 0}])
