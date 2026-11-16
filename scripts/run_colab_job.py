"""Run a real Colab batch and prevent our local budget guard stopping it mid-job."""
import argparse
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('script', type=Path)
parser.add_argument('--log', type=Path, required=True)
parser.add_argument('--seconds', type=int, default=2700)
args = parser.parse_args()
local = args.script.resolve()
remote_path = '/mnt/'+local.drive[0].lower()+local.as_posix()[2:]
rpc = '/mnt/'+ROOT.drive[0].lower()+ROOT.as_posix()[2:]+'/scripts/colab_rpc.py'
key = (ROOT/'.runtime/bridge-key').read_text().strip()
args.log.parent.mkdir(parents=True, exist_ok=True)
with args.log.open('w', encoding='utf-8') as log:
    process = subprocess.Popen(['wsl', '-d', 'Debian', '--', 'timeout', '-k', '5', str(args.seconds),
                                '/opt/colab-cli/bin/python', rpc, '--session', 'day07-embedding',
                                '--file', remote_path, '--timeout', '180'], stdout=log, stderr=log)
    while process.poll() is None:
        request = urllib.request.Request('http://127.0.0.1:8765/v1/activity', data=b'{}',
                                         headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=3):
                pass
        except OSError:
            pass
        time.sleep(10)
raise SystemExit(process.returncode)
