"""Progression chart for the blog post: TFLOPS at 4096^3 and 8192^3 for each kernel version.

Run from the repo root:  python plots/progression.py [out.png]

v0, v1 and v3 come from their CSVs in results/. v2 is the GROUP_M = 8 row of the GROUP_M sweep
(bench/run_v2_group.py, LOG.md), which is not saved to a CSV. The v4 and hipBLASLt bars use the
headline medians from Day 6 (LOG.md): both measured back to back in one job, five runs each,
so the two are directly comparable. Run-to-run variation between jobs is about 3%.
"""
import csv
import sys

import matplotlib.pyplot as plt


def tflops(path, M, N, K):
    with open(path) as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            if (int(r["M"]), int(r["N"]), int(r["K"])) == (M, N, K):
                return float(r["tflops"])


# Numbers that live in LOG.md rather than in a CSV
FIXED = {
    "v2": {4096: 91.8, 8192: 89.5},             # GROUP_M sweep, GROUP_M = 8 (one job)
    "v4": {4096: 99.6, 8192: 107.4},            # Day 6 headline medians (one job, five runs)
    "hipBLASLt": {4096: 101.5, 8192: 106.5},    # same job as v4
}

versions = [  # (x-axis label, CSV in results/ or a FIXED key)
    ("v0\n32×32 tiles", "triton_v0"),
    ("v1\n128×128 tiles", "triton_v1"),
    ("v2\nGROUP_M = 8", "v2"),
    ("v3\nautotuned", "triton_v3"),
    ("v4\n+ 16×16 MFMA", "v4"),
    ("hipBLASLt\n(torch.matmul)", "hipBLASLt"),
]

fig, ax = plt.subplots(figsize=(8.5, 4.5))
for i, n in enumerate((4096, 8192)):
    vals = [FIXED[src][n] if src in FIXED else tflops(f"results/{src}.csv", n, n, n)
            for _, src in versions]
    xs = [x + i * 0.4 for x in range(len(versions))]
    bars = ax.bar(xs, vals, width=0.4, label=f"{n}³")
    ax.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=2)
ax.set_xticks([x + 0.2 for x in range(len(versions))], [label for label, _ in versions])
ax.set_ylabel("TFLOPS (FP16, one MI250X GCD)")
ax.set_ylim(0, 120)
ax.set_title("From first kernel to hipBLASLt")
ax.legend(loc="upper left")
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "plots/progression.png"
fig.savefig(out, dpi=150)
print("saved", out)
