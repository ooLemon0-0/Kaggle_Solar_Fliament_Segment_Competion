"""Process locks for cooperative GPU access; locks release on process exit."""
import contextlib
import functools
import os
import threading
import time
from pathlib import Path

_local = threading.local()
_gate = threading.RLock()
_wait_seconds = 0.0

@contextlib.contextmanager
def file_lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b'0'); stream.flush()
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if not blocking:
                        raise BlockingIOError(f'Lock already held: {path}') from exc
                    time.sleep(.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

def gpu_wait_seconds():
    return _wait_seconds

@contextlib.contextmanager
def gpu_window():
    global _wait_seconds
    if os.environ.get('KAGGLE_COOPERATIVE_GPU') != '1':
        yield
        return
    with _gate:
        if getattr(_local, 'depth', 0):
            _local.depth += 1
            try:
                yield
            finally:
                _local.depth -= 1
            return
        path = Path(__file__).resolve().parents[2] / 'output/benchmark/routes_v3/gpu_window.lock'
        started = time.monotonic()
        with file_lock(path):
            _wait_seconds += time.monotonic() - started
            _local.depth = 1
            try:
                yield
            finally:
                try:
                    import torch
                    if torch.cuda.is_initialized():
                        torch.cuda.synchronize()
                finally:
                    _local.depth = 0

def exclusive_gpu(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with gpu_window():
            return fn(*args, **kwargs)
    return wrapped
