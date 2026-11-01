"""Compare interleaved Criterion runs named <round>.<variant>.txt.

Usage: compare.py <results dir> <base variant> <other variant>...

For each benchmark, prints the base median over rounds, and for each other variant its median,
the change against base, and the range of per-round changes (round r against base round r).
"""
import collections
import pathlib
import re
import statistics
import sys

UNIT = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6}
results, base, others = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
times = collections.defaultdict(dict)  # (group, size) -> {(variant, round): ns}

for f in sorted(results.glob("*.txt")):
    rnd, variant = f.stem.split(".")
    name = None
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^(\w+)/(\d+)", line)
        if m:
            name = (m.group(1), int(m.group(2)))
        t = re.search(r"time:\s+\[\S+ \S+ (\S+) (\S+) \S+ \S+\]", line)
        if t and name:
            times[name][(variant, int(rnd))] = float(t.group(1)) * UNIT[t.group(2)]
            name = None


def fmt(ns):
    return f"{ns / 1000:.2f} µs" if ns >= 1000 else f"{ns:.1f} ns"


header = f"{'benchmark':20} {base:>10}"
for v in others:
    header += f" | {v:>10} {'change':>7} {'range':>15}"
print(header)
for key in sorted(times, key=lambda k: (k[1], k[0])):
    d = times[key]
    rounds = sorted({r for _, r in d})
    b = [d[(base, r)] for r in rounds if (base, r) in d]
    if not b:
        continue
    mb = statistics.median(b)
    row = f"{key[0] + '/' + str(key[1]):20} {fmt(mb):>10}"
    for v in others:
        o = [d[(v, r)] for r in rounds if (v, r) in d]
        if not o:
            row += f" | {'':>10} {'':>7} {'':>15}"
            continue
        mo = statistics.median(o)
        per = [100 * (d[(v, r)] - d[(base, r)]) / d[(base, r)]
               for r in rounds if (v, r) in d and (base, r) in d]
        rng = f"{min(per):+.0f}%..{max(per):+.0f}%"
        row += f" | {fmt(mo):>10} {100 * (mo - mb) / mb:+6.1f}% {rng:>15}"
    print(row)
