//! Does a CRC that uses wider vectors slow down the code around it?
//!
//! A benchmark of the CRC alone can't tell: it runs the kernel in a loop, with nothing else to
//! slow down. This alternates a CRC of `len` bytes with a fixed amount of scalar work and times
//! the two apart. The scalar work is a chain of dependent multiplies, so it takes a fixed number
//! of cycles, and its time follows the core's frequency.
//!
//! Each pattern runs in blocks that alternate with blocks of the same scalar work and no CRC, so
//! that the two are measured within milliseconds of each other and a drifting machine cancels out.
//! For every pattern `<work>/<len>`, where `<work>` is the number of multiplies between CRCs (`1k`
//! takes about a microsecond), it prints Criterion-style lines for `compare.py`:
//!
//! - `rel_`: the scalar work's time next to the CRCs, as a percentage of the same work's time
//!   without them (printed as ns). 100 means the CRC doesn't slow down what follows it, and 103
//!   means it does by 3%.
//! - `crc_`: the time of one CRC.
//! - `scalar_`: the time of the scalar work after one CRC.
//!
//! Compare builds that differ only in vector width. If wider vectors lower the frequency, `rel_`
//! is higher in the build that uses them, and `crc_` plus `scalar_` says whether the faster CRC
//! makes up for it.

use std::hint::black_box;
use std::time::{Duration, Instant};

const LENS: [usize; 2] = [1024, 65536];
const WORK: [(&str, u64); 4] = [
    ("1k", 1_000),
    ("30k", 30_000),
    ("1m", 1_000_000),
    ("4m", 4_000_000),
];

/// Measured time per pattern, split evenly between blocks with and without the CRC.
const BUDGET: Duration = Duration::from_millis(600);
/// Blocks are at least this long, and at least `MIN_BLOCK_ITERS` iterations.
const BLOCK: Duration = Duration::from_millis(25);
const MIN_BLOCK_ITERS: usize = 8;
/// This much of each block is not measured (and at least one iteration), so that the frequency
/// has settled into the block's pattern: Intel CPUs keep it lowered for a millisecond or two
/// after the last instruction that asked for it.
const SETTLE: Duration = Duration::from_millis(5);

/// Scalar work whose duration is a fixed number of cycles: each step waits for the one before,
/// and the shift keeps the compiler from reassociating the chain.
#[inline(never)]
fn spin(mut x: u64, n: u64) -> u64 {
    for _ in 0..n {
        x = (x ^ (x >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    x
}

/// One iteration: a CRC of `buf` (none if `crc` is false), then the scalar work. Returns the
/// nanoseconds each took.
#[inline(always)]
fn iteration(buf: &[u8], crc: bool, work: u64, x: &mut u64) -> (u64, u64) {
    let t0 = Instant::now();
    if crc {
        *x ^= u64::from(crc32fast::hash(black_box(buf)));
    }
    let t1 = Instant::now();
    *x = spin(black_box(*x), black_box(work));
    let t2 = Instant::now();
    ((t1 - t0).as_nanos() as u64, (t2 - t1).as_nanos() as u64)
}

fn median(v: &mut [u64]) -> f64 {
    v.sort_unstable();
    v[v.len() / 2] as f64
}

fn report(what: &str, work: &str, len: usize, ns: f64) {
    println!("{what}_{work}/{len}");
    println!("                        time:   [{ns:.1} ns {ns:.1} ns {ns:.1} ns]");
}

fn main() {
    let data: Vec<u8> = (0..65536u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    let mut x = 1u64;
    for &len in &LENS {
        for &(name, work) in &WORK {
            let buf = &data[..len];

            // Size the blocks from the time of a few iterations.
            let start = Instant::now();
            for _ in 0..3 {
                iteration(buf, true, work, &mut x);
            }
            let per_iter = (start.elapsed() / 3).max(Duration::from_nanos(1));
            let iters_in = |d: Duration| (d.as_nanos() / per_iter.as_nanos()) as usize + 1;
            let block_iters = iters_in(BLOCK).max(MIN_BLOCK_ITERS);
            let settle_iters = iters_in(SETTLE).min(block_iters / 2);

            let (mut crc_ns, mut after_crc, mut alone) = (Vec::new(), Vec::new(), Vec::new());
            let start = Instant::now();
            let mut blocks = 0;
            while blocks < 8 || start.elapsed() < BUDGET {
                let with_crc = blocks % 2 == 1;
                for i in 0..block_iters {
                    let (c, s) = iteration(buf, with_crc, work, &mut x);
                    if i < settle_iters {
                        continue;
                    }
                    if with_crc {
                        crc_ns.push(c);
                        after_crc.push(s);
                    } else {
                        alone.push(s);
                    }
                }
                blocks += 1;
            }

            let after_crc = median(&mut after_crc);
            report("rel", name, len, 100.0 * after_crc / median(&mut alone));
            report("crc", name, len, median(&mut crc_ns));
            report("scalar", name, len, after_crc);
        }
    }
    black_box(x);
}
