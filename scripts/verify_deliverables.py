#!/usr/bin/env python3
"""Automatic checks of the manuscript and its evidence chain. Exit 1 on any failure."""
import json, re, subprocess, sys
from pathlib import Path
import pypdf
R = Path(__file__).resolve().parents[1]; P = R / "paper"
fail = []
def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fail.append(msg)
pdf = pypdf.PdfReader(P / "paper_final.pdf")
check(4 <= len(pdf.pages) <= 6, f"page count {len(pdf.pages)} within 4-6")
for extra in ("paper_ieee_candidate",):
    check(4 <= len(pypdf.PdfReader(P / f"{extra}.pdf").pages) <= 6, f"{extra} within 4-6 pages")
b = pdf.pages[0].mediabox; check(abs(float(b.width) - 595.3) < 1 and abs(float(b.height) - 841.9) < 1, "A4 page size")
log = (P / "paper_final.log").read_text(errors="ignore")
check("LaTeX Warning: Citation" not in log and "LaTeX Warning: Reference" not in log, "no unresolved citation or reference")
check("Overfull \\hbox" not in log, "no overfull boxes")
uris = []
for a in pdf.pages[0].get("/Annots") or []:
    o = a.get_object()
    if "/A" in o and "/URI" in o["/A"]:
        uris.append(str(o["/A"]["/URI"]))
for oid in ("0009-0001-1645-8724", "0009-0006-6972-2484", "0009-0002-9203-8710"):
    check(any(oid in u for u in uris), f"ORCID link {oid}")
for mail in ("javier.garay.y@uni.pe", "francisco.rodriguez.h@uni.pe", "jose.segundo.m@uni.pe"):
    check(any(mail in u and u.startswith("mailto:") for u in uris), f"blue mailto link {mail}")
check("urlcolor=blue" in (P / "profile.tex").read_text(), "links set to blue")
check(any("github.com/JoseSegundoM/proprioceptive_risk_mapping" in str(o.get_object()["/A"].get("/URI", "")) for pg in pdf.pages for o in (pg.get("/Annots") or []) if "/A" in o.get_object()), "repository link in the PDF")
for page, t in enumerate(pdf.pages):
    words = (t.extract_text() or "").strip().splitlines()
    if words:
        check(not re.fullmatch(r"\d+", words[-1].strip()), f"no page number on page {page + 1}")
order = re.sub(r"\s+", " ", pdf.pages[0].extract_text())
check(order.find("Segundo Manayay") < order.find("Garay Yovera") < order.find("Huiman") and " and Jose" not in order[:400] and "Yovera, and" not in order[:400], "author order Jose, Javier, Francisco, commas only")
fill = subprocess.run([sys.executable, str(R / "scripts" / "page_fill.py"), str(P / "paper_final.pdf")], capture_output=True, text=True).stdout
gaps = [float(g) for g in re.findall(r"(?:left|right) (-?[\d.]+) mm", fill)]
check(max(gaps) < 14.0, f"pages full: largest gap above the bottom margin {max(gaps):.1f} mm")
tops = [float(t) for t in re.findall(r"top ([\d.]+) mm", fill)]
check(min(tops) >= 36.0, f"EPIC top margin respected: first ink at {min(tops):.1f} mm")
last = pdf.pages[-1]
check(not list(last.images), "no images on the last page")
bbl = (P / "paper_final.bbl").read_text()
items = re.findall(r"\\bibitem\{(\w+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", bbl, re.S)
check(all("\\doilink" in b or "arxiv.org" in b for _, b in items), f"all {len(items)} references show a DOI or arXiv link")
all_uris = [str(o.get_object()["/A"].get("/URI", "")) for pg in pdf.pages for o in (pg.get("/Annots") or []) if "/A" in o.get_object()]
check(sum("doi.org" in u for u in all_uris) >= len(items) - 2, "DOI links are live annotations")
check(any("10.18687/LACCEI2026.1.1.2664" in u for u in all_uris) and any("scopus.com/pages/publications/105048843694" in u for u in all_uris), "LACCEI paper links to its DOI and to Scopus")
for name in ("paper_final", "paper_ieee_candidate", "paper_real_photos"):
    if (P / f"{name}.pdf").exists():
        r = subprocess.run([sys.executable, str(R / "scripts" / "fix_breaks.py"), "--check", str(P / f"{name}.pdf")], capture_output=True, text=True)
        check(r.returncode == 0, f"{name}: every column ends at a full stop or a float")
        check(len(pypdf.PdfReader(P / f"{name}.pdf").pages) == 6, f"{name}: 6 pages")
tex = (P / "main.tex").read_text()
ab = tex[tex.index(r"\begin{abstract}") + 16:tex.index(r"\end{abstract}")]
check(len(ab.split()) <= 200, f"abstract {len(ab.split())} words (<=200, limit from the project summary, not confirmed on the site)")
v = dict(re.findall(r"\\newcommand\{\\v(\w+)\}\{(.*)\}", (P / "tables" / "values.tex").read_text()))
for quote, key in (("0.055", "dTwoOne"), ("0.015", "dThreeTwo"), ("0.935", "fHeldLeg"), ("0.999", "fHeldWhe")):
    check(quote in ab and v[key].lstrip("+") == quote, f"abstract {quote} equals {key}={v[key]}")
check("0.45 " in ab and v["speedLegMin"].startswith("0.45"), f"abstract 0.45 equals speedLegMin={v['speedLegMin']}")
used = set(re.findall(r"\\v([A-Za-z]+)\{\}", tex))
check(used <= set(v), f"all {len(used)} text macros come from results")
for readme in ("README.md", "README.es.md"):
    text_ = (R / readme).read_text()
    missing = [k for k in ("psMz", "psMo", "psMt", "psMh", "psOrc", "expMz", "expMo", "expMt", "expMh", "dOneZero", "dOneZeroLo", "dOneZeroHi", "dTwoOne", "dTwoOneLo", "dTwoOneHi", "dThreeTwo", "dThreeTwoLo", "dThreeTwoHi", "repOneZero", "repTwoOne", "repThreeTwo", "intDiffEqual", "intLegFalls", "intWheFalls", "fHeldLeg", "fHeldWhe", "fCourseLeg", "fCourseWhe", "speedLegMin") if v[k] not in text_]
    check(not missing, f"{readme}: quoted numbers match the results {missing}")
for f in re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", tex):
    check((P / f).exists(), f"figure {f}")
ver = json.load(open(P / "references_verified.json"))
bib = (P / "references.bib").read_text()
cited = set(sum((c.split(",") for c in re.findall(r"\\cite\{([^}]+)\}", tex)), []))
for k in sorted(cited):
    check(f"{{{k}," in bib, f"cited {k} in bib")
check("calibrated digital twins" in tex, "digital twin appears only as a denial")
check(not re.search(r"(switch|transition)\w* between (legs|wheels|modes)", tex, re.I), "no in-mission switching claim")
for name, n in (("E1_perception_sweep", 30000), ("E2_measured_perception", 30000), ("E3_sensitivity", 38400), ("E4_ablation", 10000), ("E5_lambda", 21600), ("E6_course_perception", 20000), ("E7_independent_repeat", 15000), ("E8_spatial_baseline", 12000)):
    import gzip
    c = sum(1 for _ in gzip.open(R / "results" / "mc" / f"{name}.csv.gz", "rt")) - 1
    check(c == n, f"{name} has {c} mission rows")
print("\nFAILED:" if fail else "\nALL CHECKS PASSED", fail if fail else "")
sys.exit(1 if fail else 0)
