#!/usr/bin/env python3
"""A column must end at the end of a sentence. Finds the first column that does not, moves its break to the
last full stop by inserting \\sentbreak in the source, recompiles, repeats. --check only reports."""
import re, subprocess, sys, tempfile, unicodedata
from pathlib import Path
import pypdf
R = Path(__file__).resolve().parents[1]; P = R / "paper"; MM = 25.4 / 72
PDF = str(P / "paper_final.pdf")
ABBR = {"fig.", "figs.", "eq.", "vol.", "no.", "pp.", "al.", "e.g.", "i.e."}


def compile_():
    env = dict(__import__("os").environ, PATH="/usr/bin:" + __import__("os").environ["PATH"])
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-jobname=paper_final", r"-usepretex=\def\PDFPROFILE{interactive}", "main.tex"], cwd=P, env=env, check=True, capture_output=True)


def columns(pdf):
    n = len(pypdf.PdfReader(pdf).pages)
    out = []
    for p in range(1, n + 1):
        with tempfile.NamedTemporaryFile(suffix=".html") as t:
            subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), "-bbox", pdf, t.name], check=True)
            s = open(t.name, encoding="utf8").read()
        w = float(re.search(r'page width="([\d.]+)"', s).group(1))
        words = [(float(a), float(b), float(c), float(d), unicodedata.normalize("NFKC", t_)) for a, b, c, d, t_ in re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', s)]
        for side, sel in (("L", [v for v in words if v[2] <= w / 2 + 3]), ("R", [v for v in words if v[0] >= w / 2 - 3])):
            sel.sort(key=lambda v: (round(v[1] / 3), v[0]))
            out.append((p, side, sel))
    return out


HEADINGS = {"hazard.", "belief.", "rules.", "appearance.", "planning.", "simulation.", "classifier.", "carlo.", "statistics."}


def ends_sentence(word):
    if re.fullmatch(r"[A-Z]\.|[IVX]+\.", word):      # the label of a heading, not a sentence
        return False
    return word.endswith((".", ".)", ".”")) and word.lower() not in ABBR and word.lower() not in HEADINGS


BODY = None


def body_height(words):
    """Largest common word height in the document is the body type; tables and captions are smaller."""
    global BODY
    if BODY is None:
        import collections
        hs = collections.Counter(round(v[3] - v[1], 1) for _, _, sel in columns(PDF) for v in sel)
        total = sum(hs.values())
        BODY = max(h for h, c in hs.items() if c > 0.1 * total)
    return BODY


def closes(sel):
    """Body text ends at a full stop; a table or caption (small type) at the foot of a column also closes it."""
    bottom = max(v[3] for v in sel)
    line = sorted([v for v in sel if v[3] > bottom - 5], key=lambda v: v[0])
    if max(v[3] - v[1] for v in line) < 0.93 * body_height(sel):
        return True
    return ends_sentence(line[-1][4]) or (len(line) > 1 and line[-2][4] == ".")


def status(cols):
    return [(p, side, sel) for p, side, sel in cols if sel and not closes(sel)]


def insert_break(sel, tex):
    words = [v[4] for v in sel]
    for i in range(len(words) - 2, max(len(words) - 90, 3), -1):
        if not (ends_sentence(words[i]) and re.match(r"[A-Z]", words[i + 1])) or sel[i][3] - sel[i][1] < 0.93 * body_height(sel):
            continue
        following = words[i + 1:i + 5]
        if any(x.endswith("-") for x in following[:3]):
            following = following[:1]
        for n in (3, 2):           # numbers in the text are macros in the source, so fewer words may have to do
            pat = r"(\\textbf\{|\\emph\{)?" + r"\}?\s+".join(re.escape(x).replace(r"\~", "~") for x in following[:n])
            ms = list(re.finditer(pat, tex))
            if len(ms) == 1:
                head, tail = tex[:ms[0].start()], tex[ms[0].start():]
                # Breaks further on were computed for a layout that no longer exists.
                return head + "\\sentbreak " + tail.replace("\\sentbreak ", ""), (len(words) - i - 1, " ".join(following))
    return None, None


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--check"]
    pdf = args[0] if args else str(P / "paper_final.pdf")
    PDF = pdf
    if "--check" in sys.argv:
        cols = columns(pdf)
        for p, side, sel in cols:
            print(f"p{p}{side}: ...{' '.join(v[4] for v in sel[-4:])}  {'OK' if closes(sel) else 'MID-SENTENCE'}")
        sys.exit(1 if status(cols) else 0)
    tex_path = P / "main.tex"
    for it in range(40):
        compile_()
        BODY = None
        cols = columns(pdf)
        bad = status(cols)
        print(f"pass {it}: {len(pypdf.PdfReader(pdf).pages)} pages, {len(bad)} columns end mid-sentence")
        if not bad:
            break
        p, side, sel = bad[0]
        new, info = insert_break(sel, tex_path.read_text())
        if new is None:
            print(f"  cannot place a break for p{p}{side}: ...{' '.join(v[4] for v in sel[-6:])}")
            break
        print(f"  p{p}{side}: break before '{info[1]}' ({info[0]} words moved)")
        tex_path.write_text(new)
