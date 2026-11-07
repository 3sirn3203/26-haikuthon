"""Launch the pinned legacy vLLM text server and own its process lifecycle."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from .config import PROJECT_ROOT, server_address


def build_server_command(config, executable='vllm'):
    host, port = server_address(config)
    if host not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Managed serving requires a local server_url; use --external-server for another host')
    model, server = config['model'], config['server']
    return [
        str(executable), 'serve', model['id'],
        '--revision', model['revision'],
        '--served-model-name', model['id'],
        '--host', host, '--port', str(port), '--dtype', model['dtype'],
        '--tensor-parallel-size', str(server['tensor_parallel_size']),
        '--language-model-only',
        '--max-model-len', str(server['max_model_len']),
        '--max-num-seqs', str(server['max_num_seqs']),
        '--max-num-batched-tokens', str(server['max_num_batched_tokens']),
        '--gpu-memory-utilization', str(server['gpu_memory_utilization']),
        '--enable-prefix-caching',
        '--reasoning-parser', 'qwen3',
        '--generation-config', 'vllm',
        '--structured-outputs-config', json.dumps({
            'backend': 'xgrammar', 'disable_any_whitespace': False,
        }),
    ]


def _server_environment():
    environment = os.environ.copy()
    environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment.get('PATH', '')
    environment.setdefault('HF_HOME', str(PROJECT_ROOT / '.cache/huggingface'))
    return environment


def _check_port_available(config):
    host, port = server_address(config)
    family = socket.AF_INET6 if ':' in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(
                f'Cannot start vLLM on {host}:{port}: {exc}. '
                'Use --external-server to connect to an already-running server.'
            ) from exc


def wait_for_server(process, config):
    """Wait for the owned process to advertise the requested model, or raise."""
    deadline = time.monotonic() + config['server']['startup_timeout_seconds']
    endpoint = config['server_url'].rstrip('/') + '/v1/models'
    last_error = 'not ready'
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'vLLM exited before becoming ready (exit code {process.returncode}); see console output')
        try:
            with urlopen(endpoint, timeout=min(5, max(0.1, deadline - time.monotonic()))) as response:
                models = json.load(response)
            if config['model']['id'] not in {model['id'] for model in models['data']}:
                raise RuntimeError('The server answered with a different model than requested')
            if process.poll() is not None:
                raise RuntimeError('vLLM exited while becoming ready')
            print(f'vLLM ready: {config["server_url"]}', flush=True)
            return
        except (URLError, OSError, ValueError, KeyError, TypeError) as exc:
            last_error = str(exc)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(1, remaining))
    raise RuntimeError(f'vLLM startup timed out: {last_error}')


def _stop_server(process, timeout):
    # This group was created with start_new_session=True. No PID files or
    # process-name matching are used, so external servers are never terminated.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


@contextmanager
def managed_server(config, *, external=False, executable='vllm'):
    """Start/wait/stop our server, or leave an explicit external server alone."""
    if external:
        yield None
        return
    command = build_server_command(config, executable)
    _check_port_available(config)
    print('Server command:', json.dumps(command), flush=True)
    try:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=_server_environment(), start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f'vLLM executable not found: {executable}. Prepare the environment using '
            'stage2_prompt_generation/requirements.txt or provide --vllm-executable.'
        ) from exc
    try:
        wait_for_server(process, config)
        yield process
    finally:
        _stop_server(process, config['server']['shutdown_timeout_seconds'])


def serve(config, executable='vllm'):
    """Run a foreground server until it exits or the caller interrupts it."""
    with managed_server(config, executable=executable) as process:
        return process.wait()
