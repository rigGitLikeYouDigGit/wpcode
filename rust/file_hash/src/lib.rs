// file_hash/src/lib.rs
// Can be compiled as both a CLI binary and a Python extension (via PyO3)

#[cfg(feature = "python")]
pub mod python;

use rayon::prelude::*;
use sha2::{Sha256, Digest};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const CHUNK_SIZE: u64 = 512 * 1024 * 1024;     // 512 MB
const READ_BUFFER: usize = 8 * 1024 * 1024;     // 8 MB

/// Hash a single chunk of a file at the given offset.
fn hash_chunk(path: &Path, offset: u64, length: u64) -> Vec<u8> {
    let mut file = File::open(path).expect("Failed to open file");
    file.seek(SeekFrom::Start(offset)).expect("Seek failed");

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; READ_BUFFER];
    let mut remaining = length as usize;

    while remaining > 0 {
        let to_read = remaining.min(READ_BUFFER);
        let bytes_read = file.read(&mut buffer[..to_read]).expect("Read failed");
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
        remaining -= bytes_read;
    }

    return hasher.finalize().to_vec();
}

/// Compute chunked parallel SHA-256 of a file.
/// Returns the hex-encoded final hash.
pub fn hash_file(path: &Path) -> String {
    let file_size = std::fs::metadata(path)
        .expect("Cannot stat file")
        .len();

    // Small file fast path
    if file_size <= CHUNK_SIZE {
        let digest = hash_chunk(path, 0, file_size);
        return hex::encode(digest);
    }

    // Build chunk descriptors
    let chunks: Vec<(u64, u64)> = (0..)
        .map(|i| {
            let offset = i * CHUNK_SIZE;
            let length = (file_size - offset).min(CHUNK_SIZE);
            (offset, length)
        })
        .take_while(|(offset, _)| *offset < file_size)
        .collect();

    // Parallel hash using rayon
    let digests: Vec<Vec<u8>> = chunks
        .par_iter()
        .map(|(offset, length)| hash_chunk(path, *offset, *length))
        .collect();  // rayon preserves order

    // Combine
    let mut final_hasher = Sha256::new();
    for digest in &digests {
        final_hasher.update(digest);
    }

    hex::encode(final_hasher.finalize())
}

/// Hash multiple files, returning Vec<(path, hash)>
pub fn hash_files(paths: &[&Path]) -> Vec<(String, String)> {
    paths.par_iter()
        .map(|p| {
            let hash = hash_file(p);
            (p.display().to_string(), hash)
        })
        .collect()
}