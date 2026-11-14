//! The time of one CRC by input length, with the calls back to back.
//!
//! `indep_<len>` hashes the buffer afresh each time, so the calls don't depend on each other and
//! the CPU can overlap them. `chain_<len>` feeds one `Hasher`, so each CRC starts from the result
//! of the one before. Prints Criterion-style lines for `compare.py`.

use std::hint::black_box;
use std::time::{Duration, Instant};

/// The median time of one call of `f`, from batches of about two milliseconds.
fn time(mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    let mut calls = 0u32;
    while start.elapsed() < Duration::from_millis(2) {
        for _ in 0..16 {
            f();
        }
        calls += 16;
    }
    let mut batches: Vec<f64> = (0..15)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..calls {
                f();
            }
            start.elapsed().as_nanos() as f64 / f64::from(calls)
        })
        .collect();
    batches.sort_by(|a, b| a.partial_cmp(b).unwrap());
    batches[batches.len() / 2]
}

fn report(what: &str, len: usize, ns: f64) {
    println!("{what}/{len}");
    println!("                        time:   [{ns:.2} ns {ns:.2} ns {ns:.2} ns]");
}

fn main() {
    let data: Vec<u8> = (0..4096u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    let mut lens: Vec<usize> = (64..=2304).step_by(64).collect();
    lens.extend([1000, 1023, 1025, 2047, 3072, 4096]);
    lens.sort_unstable();
    for len in lens {
        let buf = &data[..len];
        report(
            "indep",
            len,
            time(|| {
                black_box(crc32fast::hash(black_box(buf)));
            }),
        );
        let mut hasher = crc32fast::Hasher::new();
        report("chain", len, time(|| hasher.update(black_box(buf))));
        black_box(hasher.finalize());
    }
}
