// file_hash/src/main.rs — CLI interface
use std::path::Path;
use std::time::Instant;


fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    
    if args.is_empty() {
        eprintln!("Usage: file_hash <file1> [file2] ...");
        std::process::exit(1);
    }

    // Output format: one line per file, "hash  path"
    // (matches sha256sum format for familiarity)
    let start = Instant::now();
    
    for arg in &args {
        let path = Path::new(arg);
        let hash = file_hash::hash_file(path);
        println!("{}  {}", hash, arg);
    }

    let elapsed = start.elapsed();
    eprintln!("Completed in {:.2}s", elapsed.as_secs_f64());
}