"""Run in the Colab CLI venv; reconnect to the same kernel with a fresh frontend.

CLI 0.8.0 reuses a stale websocket session_id on this runtime. This adapter uses
the installed CLI's authentication/state/runtime classes without editing state,
and deliberately leaves the frontend session_id unset for each connection.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from colab_cli.common import state
from colab_cli.runtime import ColabRuntime
import jupyter_kernel_client


def enable_wsl_dns_fallback():
    """Use Windows DNS only if WSL DNS fails; TLS still validates the hostname."""
    try:
        from colab_windows_keyring import WINDOWS_PYTHON
    except ImportError:
        return
    original = socket.getaddrinfo

    def resolve(host, port, family=0, type=0, proto=0, flags=0):
        try:
            return original(host, port, family, type, proto, flags)
        except socket.gaierror:
            if not isinstance(host, str) or family not in (0, socket.AF_INET):
                raise
            code = ('import json,socket,sys; '
                    'print(json.dumps(socket.getaddrinfo(sys.argv[1],int(sys.argv[2]),socket.AF_INET,socket.SOCK_STREAM)))')
            result = subprocess.run([WINDOWS_PYTHON, '-c', code, host, str(port)], capture_output=True,
                                    text=True, timeout=15)
            if result.returncode:
                raise
            return [(a, b, c, d, tuple(e)) for a, b, c, d, e in json.loads(result.stdout)]
    socket.getaddrinfo = resolve


def main():
    enable_wsl_dns_fallback()
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True)
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    session = state.get_session(args.session)
    if not session.kernel_id:
        raise RuntimeError('No existing model kernel; execute setup first')
    runtime = ColabRuntime(session.url, session.token, kernel_id=session.kernel_id, session_id=None)
    errors = []

    def output_hook(message):
        if message.get('output_type') == 'stream':
            stream = sys.stderr if message.get('name') == 'stderr' else sys.stdout
            stream.write(message.get('text', ''))
            stream.flush()
        elif message.get('output_type') == 'error':
            errors.append(message.get('ename', 'RemoteError'))

    try:
        # CLI 0.8.0's 10-second websocket handshake can return before the
        # connection is ready on this WSL network. Wait explicitly and close
        # failed sockets before retrying; never send inference on a dead socket.
        client_class = getattr(jupyter_kernel_client, 'KernelClient', None) or jupyter_kernel_client.JupyterKernelClient
        for attempt in range(2):
            try:
                kernel = client_class(server_url=session.url, token=session.token, kernel_id=session.kernel_id,
                                      client_kwargs={'subprotocol': jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
                                                     'timeout': 45,
                                                     'extra_params': {'colab-runtime-proxy-token': session.token}},
                                      headers={'X-Colab-Client-Agent': 'colab-cli', 'X-Colab-Runtime-Proxy-Token': session.token})
                kernel._own_kernel = False
                runtime._kernel_client = kernel
                kernel.start(timeout=45)
                if not kernel._manager.client.connection_ready.is_set():
                    raise RuntimeError('Colab websocket handshake did not become ready')
                runtime._apply_ws_hook()
                break
            except Exception:
                runtime.stop(shutdown_kernel=False)
                runtime._kernel_client = None
                if attempt == 1:
                    raise
                time.sleep(2)
        runtime.execute_code(args.file.read_text(encoding='utf-8'), output_hook=output_hook, timeout=args.timeout)
    finally:
        runtime.stop(shutdown_kernel=False)
    if errors:
        raise RuntimeError('Remote execution failed: ' + ', '.join(errors))


if __name__ == '__main__':
    main()
