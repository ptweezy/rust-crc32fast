//! Specialized checksum code for the x86 CPU architecture, using carry-less multiplication to
//! "fold" the input, as described in:
//!
//! Gopal, V., Ozturk, E., Guilford, J., Wolrich, G., Feghali, W., Dixon, M., & Karakoyunlu, D. (2009).
//! _Fast CRC computation for generic polynomials using PCLMULQDQ instruction_. Intel.
//! (Mirror link: <https://fossies.org/linux/zlib-ng/doc/crc-pclmulqdq.pdf>, accessed 2024-05-20)
//!
//! The kernel's structure and constants follow libdeflate's `lib/x86/crc32_pclmul_template.h` by
//! Eric Biggers (MIT licensed): eight vector accumulators in the main loop, a tree of folds (eight
//! vectors to four, two, one, then down to 128 bits) that consumes the leftover input on the way,
//! a final partial block folded with one overlapping load, and a three-multiply reduction from 128
//! bits to the 32-bit CRC.
//!
//! `fold_crc` is written once against the `Vector` trait and compiled for three vector widths:
//! 128-bit `PCLMULQDQ`, and 256-bit (`AVX2`) and 512-bit (`AVX-512`) `VPCLMULQDQ`, which fold two
//! and four 128-bit lanes per instruction. The wider two rely on intrinsics stabilized in Rust 1.89
//! and are only compiled when `build.rs` sets the `stable_vpclmulqdq` cfg, leaving the crate MSRV
//! unchanged otherwise. The best variant supported by the running CPU is chosen at runtime.
//!
//! Each `unsafe fn` here is unsafe only because it needs the CPU features in its
//! `#[target_feature]` list, or those of the function it's inlined into. None of them has a
//! memory-safety precondition: every load is bounds-checked.

#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

use arch::__m128i;
#[cfg(stable_vpclmulqdq)]
use arch::{__m256i, __m512i};

/// Which SIMD implementation to use, chosen once at construction from the CPU's features.
#[derive(Clone, Copy)]
enum Kind {
    /// 128-bit `PCLMULQDQ`.
    Sse,
    /// 256-bit `VPCLMULQDQ`.
    #[cfg(stable_vpclmulqdq)]
    Avx2,
    /// 256- and 512-bit `VPCLMULQDQ` with `AVX-512`.
    #[cfg(stable_vpclmulqdq)]
    Avx512,
    /// As `Avx512`, but with 512-bit vectors only for long inputs, for CPUs that downclock too
    /// eagerly when they're used.
    #[cfg(stable_vpclmulqdq)]
    Avx512Ymm,
    /// As `Avx512`, and long inputs cancel half their chunks with XORs instead of folding them
    /// (`cancel_and_fold_512`), for CPUs where that's faster.
    #[cfg(stable_vpclmulqdq)]
    Avx512Cancel,
}

#[cfg(stable_vpclmulqdq)]
impl Kind {
    /// The `AVX-512` kind for this CPU. It depends on the CPU model, and CPUID is slow, especially
    /// in a virtual machine, so it's computed once.
    #[inline]
    fn avx512() -> Self {
        use core::sync::atomic::{AtomicU8, Ordering};

        // 0 = not computed yet, then 1 + the index in this list.
        const KINDS: [Kind; 3] = [Kind::Avx512, Kind::Avx512Ymm, Kind::Avx512Cancel];
        static CACHE: AtomicU8 = AtomicU8::new(0);
        match CACHE.load(Ordering::Relaxed) {
            i @ 1..=3 => KINDS[i as usize - 1],
            _ => {
                let i = avx512_kind_from_cpuid();
                CACHE.store(i as u8 + 1, Ordering::Relaxed);
                KINDS[i]
            }
        }
    }
}

/// Returns the index in `Kind::avx512`'s list of the `AVX-512` kind for this CPU:
///
/// - `Avx512Ymm` on the Intel CPUs whose frequency drops too eagerly when 512-bit registers are
///   used only occasionally: Ice Lake and Tiger Lake, the list libdeflate uses.
/// - `Avx512Cancel` on AMD Zen 5 in 64-bit mode, which runs one `VPCLMULQDQ` per two cycles but
///   three or four 512-bit XORs per cycle. Intel CPUs run 512-bit XORs and `VPCLMULQDQ` on the
///   same two ports, so trading multiplies for XORs would slow them down.
/// - `Avx512` everywhere else.
#[cfg(stable_vpclmulqdq)]
#[cold]
#[allow(unused_unsafe)] // `__cpuid` is a safe function on newer toolchains
fn avx512_kind_from_cpuid() -> usize {
    const GENUINE_INTEL: (u32, u32, u32) = (0x756e_6547, 0x4965_6e69, 0x6c65_746e);
    const AUTHENTIC_AMD: (u32, u32, u32) = (0x6874_7541, 0x6974_6e65, 0x444d_4163);

    // SAFETY: Only called for CPUs with AVX-512, which all have the CPUID instruction.
    let vendor = unsafe { arch::__cpuid(0) };
    // SAFETY: As above.
    let eax = unsafe { arch::__cpuid(1) }.eax;
    let base_family = (eax >> 8) & 0xf;
    let family = if base_family == 0xf {
        base_family + ((eax >> 20) & 0xff)
    } else {
        base_family
    };
    let model = ((eax >> 4) & 0xf) | ((eax >> 12) & 0xf0);

    // Registers EBX, EDX, ECX spell out the vendor in that order.
    match (vendor.ebx, vendor.edx, vendor.ecx) {
        // Ice Lake server (106, 108) and client (126), Tiger Lake (140, 141).
        GENUINE_INTEL if family == 6 && matches!(model, 106 | 108 | 126 | 140 | 141) => 1,
        // The cancelling loop keeps 22 vectors in registers, which 32-bit x86 doesn't have.
        AUTHENTIC_AMD if family == 0x1a && cfg!(target_arch = "x86_64") => 2,
        _ => 0,
    }
}

#[derive(Clone)]
pub struct State {
    kind: Kind,
}

impl State {
    #[cfg(not(feature = "std"))]
    #[inline]
    fn detect() -> Option<Kind> {
        if cfg!(target_feature = "pclmulqdq")
            && cfg!(target_feature = "sse2")
            && cfg!(target_feature = "sse4.1")
            && cfg!(target_feature = "ssse3")
        {
            #[cfg(stable_vpclmulqdq)]
            {
                if cfg!(target_feature = "avx512f")
                    && cfg!(target_feature = "avx512vl")
                    && cfg!(target_feature = "vpclmulqdq")
                {
                    return Some(Kind::avx512());
                }
                if cfg!(target_feature = "avx2") && cfg!(target_feature = "vpclmulqdq") {
                    return Some(Kind::Avx2);
                }
            }

            return Some(Kind::Sse);
        }

        None
    }

    #[cfg(feature = "std")]
    #[inline]
    fn detect() -> Option<Kind> {
        if is_x86_feature_detected!("pclmulqdq")
            && is_x86_feature_detected!("sse2")
            && is_x86_feature_detected!("sse4.1")
            && is_x86_feature_detected!("ssse3")
        {
            #[cfg(stable_vpclmulqdq)]
            {
                if is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512vl")
                    && is_x86_feature_detected!("vpclmulqdq")
                {
                    return Some(Kind::avx512());
                }
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("vpclmulqdq") {
                    return Some(Kind::Avx2);
                }
            }

            return Some(Kind::Sse);
        }

        None
    }

    #[inline]
    pub fn new() -> Option<Self> {
        // SAFETY: `detect` only returns a `Kind` whose instructions the CPU supports.
        Self::detect().map(|kind| Self { kind })
    }

    #[inline]
    pub fn update(&self, state: u32, buf: &[u8]) -> u32 {
        // SAFETY: The kernels are only unsafe because the CPU might lack their target features.
        // `detect` checked every feature listed on the kernel for `self.kind`, and `kind` can't be
        // changed after `new`.
        unsafe {
            match self.kind {
                Kind::Sse => calculate(state, buf),
                #[cfg(stable_vpclmulqdq)]
                Kind::Avx2 => calculate_avx2(state, buf),
                #[cfg(stable_vpclmulqdq)]
                Kind::Avx512 => calculate_avx512::<MIN_ZMM_BYTES, false>(state, buf),
                #[cfg(stable_vpclmulqdq)]
                Kind::Avx512Ymm => {
                    calculate_avx512::<MIN_ZMM_BYTES_DOWNCLOCKING, false>(state, buf)
                }
                #[cfg(stable_vpclmulqdq)]
                Kind::Avx512Cancel => calculate_avx512::<MIN_ZMM_BYTES, true>(state, buf),
            }
        }
    }
}

/// Multipliers for folding a 128-bit lane forward across `D` bits, as `(low, high)`: the lane's low
/// and high 64-bit halves are multiplied by `x^(D+31) mod G(x)` and `x^(D-33) mod G(x)`
/// respectively, bit-reflected. These are the values libdeflate's `scripts/gen-crc32-consts.py`
/// generates.
type Mults = (i64, i64);

const FOLD_128: Mults = (0xae689191, 0xccaa009e);
const FOLD_256: Mults = (0xf1da05aa, 0x81256527);
const FOLD_512: Mults = (0x8f352d95, 0x1d9513d7);
const FOLD_1024: Mults = (0x33fff533, 0x910eeec1);
#[cfg(stable_vpclmulqdq)]
const FOLD_2048: Mults = (0xce3371cb, 0xe95c1271);
#[cfg(stable_vpclmulqdq)]
const FOLD_4096: Mults = (0x1072db28, 0x0c30f51d);

/// Barrett reduction constants: `floor(x^95 / G(x))` and `G(x)`, bit-reflected.
const BARRETT: Mults = (0xb4e5b025f7011641_u64 as i64, 0x1db710641);

/// Inputs this long and up use 512-bit vectors in the `AVX-512` kernel; shorter ones use 256-bit
/// vectors with the same features.
#[cfg(stable_vpclmulqdq)]
const MIN_ZMM_BYTES: usize = 64;

/// As `MIN_ZMM_BYTES`, on CPUs that downclock eagerly when 512-bit vectors are used. This is where
/// 512-bit vectors started before they were used for shorter inputs.
#[cfg(stable_vpclmulqdq)]
const MIN_ZMM_BYTES_DOWNCLOCKING: usize = 2048;

/// SSE kernel: 128-bit `PCLMULQDQ`.
#[target_feature(
    enable = "pclmulqdq",
    enable = "sse2",
    enable = "sse4.1",
    enable = "ssse3"
)]
unsafe fn calculate(crc: u32, data: &[u8]) -> u32 {
    // Below 16 bytes there isn't even a single full block to load, so use the scalar fallback.
    if data.len() < 16 {
        return crate::baseline::update_short(crc, data);
    }
    fold_crc::<__m128i, false>(crc, data)
}

/// `AVX2` kernel: 256-bit `VPCLMULQDQ`, folding two 128-bit lanes per instruction.
#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
#[target_feature(
    enable = "pclmulqdq",
    enable = "sse2",
    enable = "sse4.1",
    enable = "ssse3",
    enable = "avx",
    enable = "avx2",
    enable = "vpclmulqdq"
)]
unsafe fn calculate_avx2(crc: u32, data: &[u8]) -> u32 {
    if data.len() < 16 {
        return crate::baseline::update_short(crc, data);
    }
    fold_crc::<__m256i, false>(crc, data)
}

/// `AVX-512` kernel: 512-bit `VPCLMULQDQ` (four 128-bit lanes per instruction) for inputs of
/// `MIN_ZMM` bytes or more, and 256-bit for shorter ones. Either way, `AVX-512VL` lets the
/// compiler merge each fold's two XORs into one `vpternlogq` and use 32 vector registers.
#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
#[target_feature(
    enable = "pclmulqdq",
    enable = "sse2",
    enable = "sse4.1",
    enable = "ssse3",
    enable = "avx",
    enable = "avx2",
    enable = "vpclmulqdq",
    enable = "avx512f",
    enable = "avx512vl"
)]
unsafe fn calculate_avx512<const MIN_ZMM: usize, const CANCEL: bool>(crc: u32, data: &[u8]) -> u32 {
    let n = data.len();
    if n < 16 {
        return crate::baseline::update_short(crc, data);
    }
    if n < MIN_ZMM {
        fold_crc::<__m256i, false>(crc, data)
    } else {
        fold_crc::<__m512i, CANCEL>(crc, data)
    }
}

/// A vector of one or more 128-bit lanes for `fold_crc`. Every method is `#[inline(always)]` so
/// that it compiles with the target features of the kernel that `fold_crc` is inlined into.
trait Vector: Copy {
    /// Width in bytes.
    const LEN: usize;
    /// Multipliers for folding across one, two, four, and eight vectors.
    const M1: Mults;
    const M2: Mults;
    const M4: Mults;
    const M8: Mults;

    /// Loads the first `LEN` bytes of `a`.
    unsafe fn load(a: &[u8]) -> Self;
    /// Zero-extends a 128-bit vector.
    unsafe fn from_128(x: __m128i) -> Self;
    unsafe fn xor(self, other: Self) -> Self;
    /// Repeats `m` in every 128-bit lane.
    unsafe fn splat(m: Mults) -> Self;
    /// Folds each lane of `self` forward by the distance `m` was computed for, and adds the
    /// result to `dst`.
    unsafe fn fold(self, dst: Self, m: Self) -> Self;
    /// Folds the lanes together into one, consuming any whole vectors of a narrower width that
    /// remain in `rest`.
    unsafe fn reduce(self, rest: &mut &[u8]) -> __m128i;

    /// Starts the main loop by cancelling chunks with XORs; see `cancel_and_fold_512`. Only
    /// called when `fold_crc`'s `CANCEL` is set, which only 512-bit vectors support.
    unsafe fn cancel_and_fold(_init: __m128i, _rest: &mut &[u8]) -> [Self; 8] {
        unreachable!()
    }
}

impl Vector for __m128i {
    const LEN: usize = 16;
    const M1: Mults = FOLD_128;
    const M2: Mults = FOLD_256;
    const M4: Mults = FOLD_512;
    const M8: Mults = FOLD_1024;

    #[inline(always)]
    unsafe fn load(a: &[u8]) -> Self {
        arch::_mm_loadu_si128(a[..16].as_ptr() as *const __m128i)
    }

    #[inline(always)]
    unsafe fn from_128(x: __m128i) -> Self {
        x
    }

    #[inline(always)]
    unsafe fn xor(self, other: Self) -> Self {
        arch::_mm_xor_si128(self, other)
    }

    #[inline(always)]
    unsafe fn splat(m: Mults) -> Self {
        arch::_mm_set_epi64x(m.1, m.0)
    }

    #[inline(always)]
    unsafe fn fold(self, dst: Self, m: Self) -> Self {
        let lo = arch::_mm_clmulepi64_si128(self, m, 0x00);
        let hi = arch::_mm_clmulepi64_si128(self, m, 0x11);
        arch::_mm_xor_si128(arch::_mm_xor_si128(dst, lo), hi)
    }

    #[inline(always)]
    unsafe fn reduce(self, _rest: &mut &[u8]) -> __m128i {
        self
    }
}

#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
impl Vector for __m256i {
    const LEN: usize = 32;
    const M1: Mults = FOLD_256;
    const M2: Mults = FOLD_512;
    const M4: Mults = FOLD_1024;
    const M8: Mults = FOLD_2048;

    #[inline(always)]
    unsafe fn load(a: &[u8]) -> Self {
        arch::_mm256_loadu_si256(a[..32].as_ptr() as *const __m256i)
    }

    #[inline(always)]
    unsafe fn from_128(x: __m128i) -> Self {
        arch::_mm256_zextsi128_si256(x)
    }

    #[inline(always)]
    unsafe fn xor(self, other: Self) -> Self {
        arch::_mm256_xor_si256(self, other)
    }

    #[inline(always)]
    unsafe fn splat(m: Mults) -> Self {
        arch::_mm256_set_epi64x(m.1, m.0, m.1, m.0)
    }

    #[inline(always)]
    unsafe fn fold(self, dst: Self, m: Self) -> Self {
        let lo = arch::_mm256_clmulepi64_epi128(self, m, 0x00);
        let hi = arch::_mm256_clmulepi64_epi128(self, m, 0x11);
        arch::_mm256_xor_si256(arch::_mm256_xor_si256(dst, lo), hi)
    }

    #[inline(always)]
    unsafe fn reduce(self, _rest: &mut &[u8]) -> __m128i {
        let lo = arch::_mm256_castsi256_si128(self);
        let hi = arch::_mm256_extracti128_si256(self, 1);
        lo.fold(hi, __m128i::splat(FOLD_128))
    }
}

#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
impl Vector for __m512i {
    const LEN: usize = 64;
    const M1: Mults = FOLD_512;
    const M2: Mults = FOLD_1024;
    const M4: Mults = FOLD_2048;
    const M8: Mults = FOLD_4096;

    #[inline(always)]
    unsafe fn load(a: &[u8]) -> Self {
        arch::_mm512_loadu_si512(a[..64].as_ptr() as *const _)
    }

    #[inline(always)]
    unsafe fn from_128(x: __m128i) -> Self {
        arch::_mm512_zextsi128_si512(x)
    }

    #[inline(always)]
    unsafe fn xor(self, other: Self) -> Self {
        arch::_mm512_xor_si512(self, other)
    }

    #[inline(always)]
    unsafe fn splat(m: Mults) -> Self {
        arch::_mm512_set_epi64(m.1, m.0, m.1, m.0, m.1, m.0, m.1, m.0)
    }

    #[inline(always)]
    unsafe fn fold(self, dst: Self, m: Self) -> Self {
        let lo = arch::_mm512_clmulepi64_epi128(self, m, 0x00);
        let hi = arch::_mm512_clmulepi64_epi128(self, m, 0x11);
        arch::_mm512_xor_si512(arch::_mm512_xor_si512(dst, lo), hi)
    }

    #[inline(always)]
    unsafe fn reduce(self, rest: &mut &[u8]) -> __m128i {
        let m = __m256i::splat(FOLD_256);
        let lo = arch::_mm512_castsi512_si256(self);
        let hi = arch::_mm512_extracti64x4_epi64(self, 1);
        let mut y = lo.fold(hi, m);
        if rest.len() >= 32 {
            y = y.fold(__m256i::load(rest), m);
            *rest = &rest[32..];
        }
        y.reduce(rest)
    }

    #[inline(always)]
    unsafe fn cancel_and_fold(init: __m128i, rest: &mut &[u8]) -> [Self; 8] {
        cancel_and_fold_512(init, rest)
    }
}

/// Invokes `$m!(i, args...)` for `i` in `0..16`, with `i` as a literal.
#[cfg(stable_vpclmulqdq)]
macro_rules! each_of_16 {
    ($m:ident $(, $arg:tt)*) => {
        $m!(0 $(, $arg)*);
        $m!(1 $(, $arg)*);
        $m!(2 $(, $arg)*);
        $m!(3 $(, $arg)*);
        $m!(4 $(, $arg)*);
        $m!(5 $(, $arg)*);
        $m!(6 $(, $arg)*);
        $m!(7 $(, $arg)*);
        $m!(8 $(, $arg)*);
        $m!(9 $(, $arg)*);
        $m!(10 $(, $arg)*);
        $m!(11 $(, $arg)*);
        $m!(12 $(, $arg)*);
        $m!(13 $(, $arg)*);
        $m!(14 $(, $arg)*);
        $m!(15 $(, $arg)*);
    };
}

/// Inputs this long and up start with `cancel_and_fold_512`. It needs 4 KiB: one 2 KiB block to
/// cancel chunks in, and the 2 KiB after it that those chunks are XORed into. (`fold_crc` refers to
/// it on every toolchain, so it isn't gated on `stable_vpclmulqdq` like the function.)
const MIN_CANCEL_BYTES: usize = 4096;

/// Starts the 512-bit main loop on CPUs whose vector units can run many XORs alongside the
/// carry-less multiplies, which limit the fold's throughput (AMD Zen 5 runs one `VPCLMULQDQ` per
/// two cycles, whatever its width, but three or four 512-bit XORs per cycle).
///
/// `G(x)` divides `G(x^512)`, which is `G(x)^512` since coefficients are bits. So XORing a 64-byte
/// chunk into the chunks `32 - e` places after it, for each exponent `e < 32` of `G(x)`, and then
/// clearing it, adds a multiple of `G(x)` to the message and leaves its CRC unchanged. That
/// "cancels" the chunk with 14 XORs and no multiplies. (This is the idea behind the Chorba
/// algorithm, <https://arxiv.org/abs/2412.16398>.) This cancels every even-numbered chunk and folds
/// the odd-numbered ones into four accumulators as usual, halving the multiplies per byte.
///
/// Rather than pushing each cancelled chunk into the 14 later chunks, each chunk pulls in the
/// cancelled chunks that land on it, from the last 16 of them, which stay in registers. Of the 14
/// offsets, the 9 even ones land on chunks that are cancelled in turn, and the 5 odd ones on
/// folded chunks. Each loop iteration covers 2 KiB (32 chunks), so the 16 saved chunks rotate
/// through fixed registers. Afterwards, the cancelled chunks are pulled into the next 2 KiB, which
/// is folded into all eight accumulators, and the main loop carries on without cancelling.
///
/// Returns the eight accumulators as of the end of the consumed input, with the even ones (whose
/// chunks were cancelled) folded in from the final 2 KiB.
#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
#[inline(always)]
unsafe fn cancel_and_fold_512(init: __m128i, rest: &mut &[u8]) -> [__m512i; 8] {
    debug_assert!(rest.len() >= MIN_CANCEL_BYTES);
    let zero = arch::_mm512_setzero_si512();
    let m = __m512i::splat(FOLD_4096);
    // `h[k]` is the cancelled value of chunk `start - 32 + 2k`, where `start` is the first chunk of
    // the current 2 KiB block. Each slot is overwritten as its replacement is computed, so while
    // pair `P` is processed, slots below `P` already hold chunks of the current block.
    let mut h = [zero; 16];
    // `acc[j]` accumulates the odd chunks `8i + 2j + 1`.
    let mut acc = [zero; 4];

    // Processes pair `P` of `block`: the cancelled chunk `2P` and the folded chunk `2P + 1`. Each
    // pulls in the cancelled chunks at the offsets of its parity: the even offsets
    // 6, 10, 16, 20, 22, 24, 28, 30, 32 are in slots `P + 16 - o/2`, and the odd offsets
    // 9, 21, 25, 27, 31 in slots `P + (33 - o)/2`, both mod 16. In the first block, slots `P` and
    // up would be chunks before the input, which are zero, so those pulls are skipped, the CRC
    // state is added to chunk 0, and the accumulators start from their first chunk.
    macro_rules! pair {
        ($p:literal, $block:ident, $first:literal) => {{
            const P: usize = $p;
            const FIRST: bool = $first;
            macro_rules! pull {
                ($x:ident, $k:literal) => {
                    if !FIRST || (P + $k) % 16 < P {
                        $x = $x.xor(h[(P + $k) % 16]);
                    }
                };
            }
            let mut even = __m512i::load(&$block[128 * P..]);
            if FIRST && P == 0 {
                even = even.xor(__m512i::from_128(init));
            }
            pull!(even, 13);
            pull!(even, 11);
            pull!(even, 8);
            pull!(even, 6);
            pull!(even, 5);
            pull!(even, 4);
            pull!(even, 2);
            pull!(even, 1);
            pull!(even, 0);
            h[P] = even;
            let mut odd = __m512i::load(&$block[128 * P + 64..]);
            pull!(odd, 12);
            pull!(odd, 6);
            pull!(odd, 4);
            pull!(odd, 3);
            pull!(odd, 1);
            acc[P % 4] = if FIRST && P < 4 {
                odd
            } else {
                acc[P % 4].fold(odd, m)
            };
        }};
    }

    let block = &rest[..2048];
    each_of_16!(pair, block, true);
    *rest = &rest[2048..];
    while rest.len() >= 4096 {
        let block = &rest[..2048];
        each_of_16!(pair, block, false);
        *rest = &rest[2048..];
    }

    // The next 32 chunks still have cancelled chunks to pull in: those less than 32 chunks before
    // them, in slots `Q + k` below 16. Fold them into all eight accumulators; the even ones are
    // empty, so their first chunk is just moved in.
    let block = &rest[..2048];
    let mut v = [zero, acc[0], zero, acc[1], zero, acc[2], zero, acc[3]];
    macro_rules! drain {
        ($q:literal, $block:ident) => {{
            const Q: usize = $q;
            macro_rules! pull {
                ($x:ident, $k:literal) => {
                    if Q + $k < 16 {
                        $x = $x.xor(h[(Q + $k) % 16]);
                    }
                };
            }
            let mut even = __m512i::load(&$block[128 * Q..]);
            pull!(even, 13);
            pull!(even, 11);
            pull!(even, 8);
            pull!(even, 6);
            pull!(even, 5);
            pull!(even, 4);
            pull!(even, 2);
            pull!(even, 1);
            pull!(even, 0);
            v[(2 * Q) % 8] = if Q < 4 {
                even
            } else {
                v[(2 * Q) % 8].fold(even, m)
            };
            let mut odd = __m512i::load(&$block[128 * Q + 64..]);
            pull!(odd, 12);
            pull!(odd, 6);
            pull!(odd, 4);
            pull!(odd, 3);
            pull!(odd, 1);
            v[(2 * Q + 1) % 8] = v[(2 * Q + 1) % 8].fold(odd, m);
        }};
    }
    each_of_16!(drain, block);
    *rest = &rest[2048..];
    v
}

/// Computes the CRC of `data`, which must be at least 16 bytes long, folding it with `V`-wide
/// vectors.
///
/// Inputs of eight vectors or more run through a main loop over eight accumulators. Every
/// shorter input, and the leftover of a longer one, enters the tree of folds at the level its
/// length calls for: each level halves the number of accumulators, then folds in one more group of
/// vectors if that many bytes remain. So the leftover is consumed a group at a time instead of a
/// 16-byte block at a time, and the final partial block takes one more multiply.
#[inline(always)]
unsafe fn fold_crc<V: Vector, const CANCEL: bool>(crc: u32, data: &[u8]) -> u32 {
    debug_assert!(data.len() >= 16);
    let len = V::LEN;
    let init = arch::_mm_cvtsi32_si128(!crc as i32);
    let mut rest = data;

    let mut x = if data.len() >= len {
        let v = if data.len() >= 2 * len {
            let (mut v0, v1) = if data.len() >= 4 * len {
                let [mut v0, mut v1, v2, v3] = if data.len() >= 8 * len {
                    let [mut v0, mut v1, mut v2, mut v3, mut v4, mut v5, mut v6, mut v7] =
                        if CANCEL && data.len() >= MIN_CANCEL_BYTES {
                            V::cancel_and_fold(init, &mut rest)
                        } else {
                            let v = [
                                V::load(rest).xor(V::from_128(init)),
                                V::load(&rest[len..]),
                                V::load(&rest[2 * len..]),
                                V::load(&rest[3 * len..]),
                                V::load(&rest[4 * len..]),
                                V::load(&rest[5 * len..]),
                                V::load(&rest[6 * len..]),
                                V::load(&rest[7 * len..]),
                            ];
                            rest = &rest[8 * len..];
                            v
                        };

                    let m8 = V::splat(V::M8);
                    while rest.len() >= 8 * len {
                        v0 = v0.fold(V::load(rest), m8);
                        v1 = v1.fold(V::load(&rest[len..]), m8);
                        v2 = v2.fold(V::load(&rest[2 * len..]), m8);
                        v3 = v3.fold(V::load(&rest[3 * len..]), m8);
                        v4 = v4.fold(V::load(&rest[4 * len..]), m8);
                        v5 = v5.fold(V::load(&rest[5 * len..]), m8);
                        v6 = v6.fold(V::load(&rest[6 * len..]), m8);
                        v7 = v7.fold(V::load(&rest[7 * len..]), m8);
                        rest = &rest[8 * len..];
                    }

                    // Eight accumulators to four.
                    let m4 = V::splat(V::M4);
                    v0 = v0.fold(v4, m4);
                    v1 = v1.fold(v5, m4);
                    v2 = v2.fold(v6, m4);
                    v3 = v3.fold(v7, m4);
                    if rest.len() >= 4 * len {
                        v0 = v0.fold(V::load(rest), m4);
                        v1 = v1.fold(V::load(&rest[len..]), m4);
                        v2 = v2.fold(V::load(&rest[2 * len..]), m4);
                        v3 = v3.fold(V::load(&rest[3 * len..]), m4);
                        rest = &rest[4 * len..];
                    }
                    [v0, v1, v2, v3]
                } else {
                    let v = [
                        V::load(rest).xor(V::from_128(init)),
                        V::load(&rest[len..]),
                        V::load(&rest[2 * len..]),
                        V::load(&rest[3 * len..]),
                    ];
                    rest = &rest[4 * len..];
                    v
                };

                // Four to two.
                let m2 = V::splat(V::M2);
                v0 = v0.fold(v2, m2);
                v1 = v1.fold(v3, m2);
                if rest.len() >= 2 * len {
                    v0 = v0.fold(V::load(rest), m2);
                    v1 = v1.fold(V::load(&rest[len..]), m2);
                    rest = &rest[2 * len..];
                }
                (v0, v1)
            } else {
                let v0 = V::load(rest).xor(V::from_128(init));
                let v1 = V::load(&rest[len..]);
                rest = &rest[2 * len..];
                (v0, v1)
            };

            // Two to one.
            let m1 = V::splat(V::M1);
            v0 = v0.fold(v1, m1);
            if rest.len() >= len {
                v0 = v0.fold(V::load(rest), m1);
                rest = &rest[len..];
            }
            v0
        } else {
            let v0 = V::load(rest).xor(V::from_128(init));
            rest = &rest[len..];
            v0
        };

        // One vector to 128 bits.
        v.reduce(&mut rest)
    } else {
        // Shorter than one vector (only when `V` is wider than 128 bits): start at 128 bits.
        let x = __m128i::load(rest).xor(init);
        rest = &rest[16..];
        x
    };

    // At most two whole 16-byte blocks remain.
    let m = __m128i::splat(FOLD_128);
    while rest.len() >= 16 {
        x = x.fold(__m128i::load(rest), m);
        rest = &rest[16..];
    }

    // Fold a final partial block of `n` (1..=15) bytes. Together, `x` and the `n` bytes are
    // `16 + n` bytes of message: split them into the first `n` bytes, which are folded forward by
    // 128 bits, and the last 16. The last 16 are the high `16 - n` bytes of `x` followed by the
    // final `n` bytes of the input, which one load ending at the end of `data` reads. That load
    // overlaps bytes already folded in, but `data` is at least 16 bytes long, so it stays in
    // bounds.
    let n = rest.len();
    if n > 0 {
        let shl = shift_left_mask(n);
        let shr = arch::_mm_xor_si128(shl, arch::_mm_set1_epi8(-128));
        let first = arch::_mm_shuffle_epi8(x, shl);
        let tail = __m128i::load(&data[data.len() - 16..]);
        let last = arch::_mm_blendv_epi8(arch::_mm_shuffle_epi8(x, shr), tail, shr);
        x = first.fold(last, m);
    }

    reduce_to_crc(x)
}

/// Returns the `pshufb` mask that shifts a vector `16 - n` bytes toward its high end, for `n` in
/// `1..=16`: byte `i` of the mask is `i + n - 16`, and the negative ones zero their byte.
///
/// XORed with `0x80` in every byte, the same mask instead shifts `n` bytes toward the low end, and
/// has the high bit set in exactly the `n` high bytes that shift empties.
#[inline(always)]
unsafe fn shift_left_mask(n: usize) -> __m128i {
    let seq = arch::_mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    arch::_mm_add_epi8(seq, arch::_mm_set1_epi8(n as i8 - 16))
}

/// Reduces a 128-bit message polynomial to its CRC: multiply by `x^32` and reduce modulo `G(x)`,
/// first folding the low 64 bits into the high 64, then with a Barrett reduction.
///
/// This matches libdeflate, and `crc-pclmul-template.S` in the Linux kernel with `n=32` and
/// `LSB_CRC=1`; see there for the math.
#[inline(always)]
unsafe fn reduce_to_crc(x: __m128i) -> u32 {
    let x = arch::_mm_xor_si128(
        arch::_mm_clmulepi64_si128(x, __m128i::splat(FOLD_128), 0x10),
        arch::_mm_srli_si128(x, 8),
    );
    let barrett = __m128i::splat(BARRETT);
    let t = arch::_mm_clmulepi64_si128(x, barrett, 0x00);
    let t = arch::_mm_clmulepi64_si128(t, barrett, 0x10);
    !(arch::_mm_extract_epi32(arch::_mm_xor_si128(x, t), 2) as u32)
}

#[cfg(test)]
mod test {
    use super::{Kind, State};

    /// Every kernel the running CPU supports, so the narrower ones get tested on wider CPUs too.
    fn supported() -> impl Iterator<Item = State> {
        #[cfg(stable_vpclmulqdq)]
        let avx512 = matches!(
            State::new().expect("not supported").kind,
            Kind::Avx512 | Kind::Avx512Ymm | Kind::Avx512Cancel
        );
        let kinds = [
            Some(Kind::Sse),
            #[cfg(stable_vpclmulqdq)]
            Some(Kind::Avx2).filter(|_| {
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("vpclmulqdq")
            }),
            #[cfg(stable_vpclmulqdq)]
            Some(Kind::Avx512).filter(|_| avx512),
            #[cfg(stable_vpclmulqdq)]
            Some(Kind::Avx512Ymm).filter(|_| avx512),
            #[cfg(stable_vpclmulqdq)]
            Some(Kind::Avx512Cancel).filter(|_| avx512),
        ];
        kinds.into_iter().flatten().map(|kind| State { kind })
    }

    quickcheck::quickcheck! {
        fn check_against_baseline(init: u32, chunks: Vec<(Vec<u8>, usize)>) -> bool {
            supported().all(|spec| {
                let mut baseline = init;
                let mut specialized = init;
                for (chunk, offset) in &chunks {
                    // simulate random alignments by offsetting the slice by up to 15 bytes
                    let chunk = &chunk[(offset & 0xF).min(chunk.len())..];
                    baseline = crate::baseline::update_fast_16(baseline, chunk);
                    specialized = spec.update(specialized, chunk);
                }
                specialized == baseline
            })
        }
    }

    // Exercises every kernel across the sizes and alignments where the strategy switches.
    #[test]
    fn check_large_inputs_against_baseline() {
        let mut data = vec![0u8; 8200];
        let mut s: u32 = 0x1234_5678;
        for b in data.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (s >> 24) as u8;
        }

        let mut lens: Vec<usize> = (0..=160).collect();
        for &edge in &[
            192usize, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
        ] {
            lens.extend(edge - 17..=edge + 17);
        }
        for spec in supported() {
            for &len in &lens {
                for &offset in &[0usize, 1, 3, 7, 8, 15] {
                    if offset + len > data.len() {
                        continue;
                    }
                    let slice = &data[offset..offset + len];
                    for &init in &[0u32, 0xDEAD_BEEF] {
                        assert_eq!(
                            spec.update(init, slice),
                            crate::baseline::update_fast_16(init, slice),
                            "mismatch for len={len} offset={offset} init={init:#x}",
                        );
                    }
                }
            }
        }
    }
}
