"""Summarize interleaved Criterion runs named <round>.<base|pr>.txt.

For each benchmark, prints the median over rounds for base and pr, the change between the
medians, and the range of the per-round changes (round r of pr against round r of base).
"""
import collections
import pathlib
import re
import statistics
import sys

UNIT = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6}
times = collections.defaultdict(dict)  # (group, size) -> {(variant, round): ns}

for f in sorted(pathlib.Path(sys.argv[1]).glob("*.txt")):
    rnd, variant = f.stem.split(".")
    name = None
    for line in f.read_text(errors="replace").splitlines():
        m = re.match(r"^(\w+)/(\d+)", line)
        if m:
            name = (m.group(1), int(m.group(2)))
        t = re.search(r"time:\s+\[\S+ \S+ (\S+) (\S+) \S+ \S+\]", line)
        if t and name:
            times[name][(variant, int(rnd))] = float(t.group(1)) * UNIT[t.group(2)]
            name = None

print(f"{'benchmark':22} {'master':>11} {'PR #74':>11} {'change':>8}   per-round range")
for key in sorted(times):
    d = times[key]
    rounds = sorted({r for _, r in d})
    base = [d[("base", r)] for r in rounds if ("base", r) in d]
    pr = [d[("pr", r)] for r in rounds if ("pr", r) in d]
    if not base or not pr:
        continue
    mb, mp = statistics.median(base), statistics.median(pr)
    per = [100 * (d[("pr", r)] - d[("base", r)]) / d[("base", r)]
           for r in rounds if ("base", r) in d and ("pr", r) in d]
    fmt = lambda ns: f"{ns / 1000:.2f} µs" if ns >= 1000 else f"{ns:.2f} ns"
    print(f"{key[0] + '/' + str(key[1]):22} {fmt(mb):>11} {fmt(mp):>11} {100 * (mp - mb) / mb:+7.1f}%"
          f"   {min(per):+.1f}% .. {max(per):+.1f}%")
