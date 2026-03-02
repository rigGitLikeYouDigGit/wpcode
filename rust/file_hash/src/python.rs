// src/python.rs
use pyo3::prelude::*;
use std::path::Path;

/// Hash a single file, returning hex-encoded SHA-256.
#[pyfunction]
#[pyo3(signature = (path,))]
fn hash_file(path: &str) -> PyResult<String> {
    Ok(crate::hash_file(Path::new(path)))
}

/// Hash multiple files in parallel.
/// Returns list of (path, hash) tuples.
#[pyfunction]
#[pyo3(signature = (paths,))]
fn hash_files(paths: Vec<String>) -> PyResult<Vec<(String, String)>> {
    let path_refs: Vec<&Path> = paths.iter().map(|p| Path::new(p.as_str())).collect();
    Ok(crate::hash_files(&path_refs))
}

/// Check if SHA-NI hardware acceleration is available.
#[pyfunction]
fn has_sha_ni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("sha")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Return backend info string.
#[pyfunction]
fn backend_info() -> String {
    let sha_ni = if cfg!(target_arch = "x86_64") {
        is_x86_feature_detected!("sha")
    } else {
        false
    };
    
    format!(
        "file_hash native backend | rayon threads: {} | SHA-NI: {}",
        rayon::current_num_threads(),
        if sha_ni { "enabled" } else { "unavailable" }
    )
}

// Module name must match the last segment of tool.maturin.module-name
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_file, m)?)?;
    m.add_function(wrap_pyfunction!(hash_files, m)?)?;
    m.add_function(wrap_pyfunction!(has_sha_ni, m)?)?;
    m.add_function(wrap_pyfunction!(backend_info, m)?)?;
    Ok(())
}