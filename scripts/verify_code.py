"""Run real unit/integration checks and retain a small machine-readable record."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = {'validated_at_utc': datetime.now(timezone.utc).isoformat(),
          'command': 'py scripts/verify_code.py', 'tests_run': result.testsRun,
          'failures': len(result.failures), 'errors': len(result.errors),
          'successful': result.wasSuccessful(),
          'scope': 'Code/unit/isolated real Qdrant integration only; not benchmark accuracy',
          'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for folder in ('src', 'tests') for p in sorted((ROOT/folder).glob('*.py'))}}
(ROOT/'results/code_validation.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: v for k, v in record.items() if k != 'source_sha256'}, indent=2))
raise SystemExit(0 if result.wasSuccessful() else 1)
