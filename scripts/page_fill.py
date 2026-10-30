#!/usr/bin/env python3
"""How full each page is: empty mm at the bottom of each column, measured from the PDF's text and images."""
import re, subprocess, sys, tempfile
import pypdf
pdf = sys.argv[1]
n = len(pypdf.PdfReader(pdf).pages)
MM = 25.4 / 72
for p in range(1, n + 1):
    with tempfile.NamedTemporaryFile(suffix=".html") as t:
        subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), "-bbox", pdf, t.name], check=True)
        s = open(t.name).read()
    w, h = [float(x) for x in re.search(r'width="([\d.]+)" height="([\d.]+)"', s).groups()]
    x = [tuple(float(v) for v in m) for m in re.findall(r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)"', s)]
    left = [v for v in x if v[2] <= w / 2 + 2]
    right = [v for v in x if v[0] >= w / 2 - 2]
    bottom = lambda c: (h - max(v[3] for v in c)) * MM - 19 if c else float("nan")
    print(f"p{p}: empty above the 19 mm margin: left {bottom(left):.1f} mm, right {bottom(right):.1f} mm; top {min(v[1] for v in x) * MM:.1f} mm")
