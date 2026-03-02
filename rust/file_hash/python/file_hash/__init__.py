from __future__ import annotations
"""
file_hash — High-performance concurrent chunked SHA-256.

Uses native Rust backend when available, falls back to pure Python.
"""
from pathlib import Path
from typing import Union

__version__ = "0.1.0"

# ── Backend detection ────────────────────────────────────────
_BACKEND = "python"

try:
    from file_hash._native import (
        hash_file as _native_hash_file,
        hash_files as _native_hash_files,
        has_sha_ni,
        backend_info,
    )
    _BACKEND = "native"
except ImportError:
    has_sha_ni = lambda: False
    backend_info = lambda: "file_hash pure-Python fallback backend"


def hash_file(path: Union[str, Path]) -> str:
    """
    Compute chunked parallel SHA-256 of a file.
    Returns hex-encoded hash string.
    """
    if _BACKEND == "native":
        return _native_hash_file(str(path))
    
    from file_hash._fallback import hash_file as _py_hash_file
    return _py_hash_file(str(path))


def hash_files(paths: list[Union[str, Path]]) -> list[tuple[str, str]]:
    """
    Hash multiple files in parallel.
    Returns list of (path, hash) tuples.
    """
    if _BACKEND == "native":
        return _native_hash_files([str(p) for p in paths])
    
    from file_hash._fallback import hash_file as _py_hash_file
    return [(str(p), _py_hash_file(str(p))) for p in paths]


def get_backend() -> str:
    """Return which backend is active: 'native' or 'python'."""
    return _BACKEND


def _cli():
    """Entry point for `file-hasher` console script."""
    import sys
    import time

    if len(sys.argv) < 2:
        print(f"file_hash v{__version__} ({backend_info()})")
        print("Usage: file-hasher <file1> [file2] ...")
        sys.exit(1)

    start = time.perf_counter()
    paths = [Path(p) for p in sys.argv[1:]]

    for p in paths:
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(1)

    results = hash_files(paths)
    elapsed = time.perf_counter() - start

    for filepath, digest in results:
        print(f"{digest}  {filepath}")

    total_size = sum(Path(p).stat().st_size for p, _ in results)
    size_gb = total_size / (1024 ** 3)
    print(
        f"\n{len(results)} file(s), {size_gb:.2f} GB, "
        f"{elapsed:.2f}s ({size_gb / elapsed:.2f} GB/s)",
        file=sys.stderr,
    )