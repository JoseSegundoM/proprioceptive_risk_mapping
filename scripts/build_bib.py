#!/usr/bin/env python3
"""references.bib from registry metadata (Crossref by DOI, arXiv). Only the self-citation comes from a project document."""
import json, re, urllib.request
from pathlib import Path
P = Path(__file__).resolve().parents[1] / "paper"
v = json.load(open(P / "references_verified.json"))
DOI = dict(kolvenbach2019="10.1109/lra.2019.2896732", elnoor2024="10.1109/lra.2024.3418270")
SKIP = {"guo2017", "zhao2023", "tremblay2021"}
for k, d in DOI.items():
    m = json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.crossref.org/works/{d}", headers={"User-Agent": "reference-check/1.0"}), timeout=30))["message"]
    v[k] = dict(status="crossref", title=m["title"][0], authors=[f"{a.get('given','')} {a.get('family','')}".strip() for a in m["author"]], venue=(m.get("container-title") or [""])[0],
                volume=m.get("volume"), issue=m.get("issue"), page=m.get("page"), year=m["issued"]["date-parts"][0][0], doi=m["DOI"], type=m.get("type"))
json.dump(v, open(P / "references_verified.json", "w"), indent=1, ensure_ascii=False)
out = []
for k, e in v.items():
    if k in SKIP or e["status"] in ("not_found", "lookup_error"):
        continue
    au = " and ".join(a.replace("*", "") for a in e["authors"]).replace("&", r"\&")
    title = "{{" + e["title"].replace("&", r"\&").replace("’", "'") + "}}"
    if e["status"] == "arxiv":
        out.append(f"@misc{{{k},\n author={{{au}}},\n title={title},\n howpublished={{arXiv:{e['arxiv']}}},\n year={{{e['year']}}}\n}}")
        continue
    conf = "proceedings" in e.get("type", "") or "Conference" in e["venue"] or "Science and Systems" in e["venue"]
    f = [f"author={{{au}}}", f"title={title}", f"year={{{e['year']}}}"]
    f.append(("booktitle" if conf else "journal") + f"={{{e['venue'].replace('&', chr(92) + '&')}}}")
    for a, b in (("volume", "volume"), ("issue", "number")):
        if e.get(a) and not conf:
            f.append(f"{b}={{{e[a]}}}")
    if e.get("page"):
        f.append(f"pages={{{e['page'].replace('-', '--')}}}")
    f.append(f"doi={{{e['doi']}}}")
    out.append(f"@{'inproceedings' if conf else 'article'}{{{k},\n " + ",\n ".join(f) + "\n}")
out.append("""@inproceedings{segundo2026laccei,
 author={J. L. {Segundo Manayay} and J. Y. {Garay Yovera} and R. G. {Ch\\'avez Ch\\'avez} and F. {Rodr\\'iguez Huiman} and F. A. {Castro Suazo} and R. N. {Antayhua Alvarez} and J. A. {Espinoza Coronel}},
 title={{A multimodal 3D perception-based autonomous robotic architecture for underground mining: Predictive planning and dynamic risk-aware control}},
 booktitle={Proc. LACCEI International Multi-Conference for Engineering, Education and Technology},
 year={2026}
}""")
(P / "references.bib").write_text("\n\n".join(out) + "\n")
print(len(out), "entries")
