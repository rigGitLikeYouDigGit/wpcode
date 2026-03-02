# python/file_hash/_fallback.py
"""Pure-Python fallback using concurrent.futures."""
from __future__ import annotations
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

CHUNK_SIZE = 512 * 1024 * 1024
READ_BUFFER = 8 * 1024 * 1024
MAX_WORKERS = min(os.cpu_count() or 4, 8)


def _hash_chunk(file_path: str, offset: int, length: int) -> tuple[int, bytes]:
    h = hashlib.sha256()
    remaining = length
    with open(file_path, "rb") as f:
        f.seek(offset)
        while remaining > 0:
            data = f.read(min(READ_BUFFER, remaining))
            if not data:
                break
            h.update(data)
            remaining -= len(data)
    return (offset // CHUNK_SIZE, h.digest())


def hash_file(file_path: str, chunk_size: int = CHUNK_SIZE) -> str:
    file_size = os.path.getsize(file_path)

    if file_size <= chunk_size:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(READ_BUFFER)
                if not data:
                    break
                h.update(data)
        return h.hexdigest()

    chunks = []
    offset = 0
    while offset < file_size:
        length = min(chunk_size, file_size - offset)
        chunks.append((file_path, offset, length))
        offset += length

    digests: dict[int, bytes] = {}
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as pool:
        futures = {
            pool.submit(_hash_chunk, *args): args[1]
            for args in chunks
        }
        for future in as_completed(futures):
            idx, digest = future.result()
            digests[idx] = digest

    final = hashlib.sha256()
    for i in sorted(digests.keys()):
        final.update(digests[i])
    return final.hexdigest()