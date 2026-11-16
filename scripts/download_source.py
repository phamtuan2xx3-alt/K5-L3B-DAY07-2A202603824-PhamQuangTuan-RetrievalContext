"""Get the licensed source from its accessible mirror and verify frozen bytes."""
import hashlib
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT/'data/source/world-development-report-2024.pdf'
expected = '7f545194763ef2ffd1347c14bcf5a270c622b9c3a29e1c8c8b081aaee6c00df5'
if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
    print('Source already present and hash verified; no download')
else:
    response = requests.get('https://www.healthsystemsmatter.com/files/GeneralReports/World%20Development%20Report%202024.pdf', timeout=90)
    response.raise_for_status()
    if not response.content.startswith(b'%PDF') or hashlib.sha256(response.content).hexdigest() != expected:
        raise RuntimeError('Source bytes differ from frozen report version; do not silently substitute')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    print('Source downloaded and hash verified')
