#!/usr/bin/env python3
"""Look up each candidate reference in Crossref or arXiv and record what the registry returns. No field is typed by hand."""
import json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path
import xml.etree.ElementTree as ET
OUT = Path(__file__).resolve().parents[1] / "paper" / "references_verified.json"
Q = [
("bjelonic2019","Keep Rollin—Whole-Body Motion Control and Planning for Wheeled Quadrupedal Robots","Bjelonic"),
("lee2024","Learning robust autonomous navigation and locomotion for wheeled-legged robots","Lee"),
("mattamala2025","Wild Visual Navigation: Fast Traversability Learning via Pre-Trained Models and Online Self-Supervision","Mattamala"),
("frey2023","Fast Traversability Estimation for Wild Visual Navigation","Frey"),
("wellhausen2019","Where Should I Walk? Predicting Terrain Properties From Images Via Self-Supervised Learning","Wellhausen"),
("bednarek2019","What am I touching? Learning to classify terrain via haptic sensing","Bednarek"),
("kolvenbach2019","Haptic Inspection of Planetary Soils With Legged Robots","Kolvenbach"),
("fahmi2020","STANCE: Locomotion Adaptation Over Soft Terrain","Fahmi"),
("cai2024","EVORA: Deep Evidential Traversability Learning for Risk-Aware Off-Road Autonomy","Cai"),
("fan2021","STEP: Stochastic Traversability Evaluation and Planning for Risk-Aware Off-road Navigation","Fan"),
("cai2022","Risk-Aware Off-Road Navigation via a Learned Speed Distribution Map","Cai"),
("ewen2022","These Maps are Made for Walking: Real-Time Terrain Property Estimation for Mobile Robots","Ewen"),
("elnoor2024","ProNav: Proprioceptive Traversability Estimation for Legged Robot Navigation in Outdoor Environments","Elnoor"),
("hart1968","A Formal Basis for the Heuristic Determination of Minimum Cost Paths","Hart"),
("todorov2012","MuJoCo: A physics engine for model-based control","Todorov"),
("miki2022","Learning robust perceptive locomotion for quadrupedal robots in the wild","Miki"),
("tranzatto2022","CERBERUS in the DARPA Subterranean Challenge","Tranzatto"),
("bissiri2016","A general framework for updating belief distributions","Bissiri"),
("grunwald2017","Inconsistency of Bayesian Inference for Misspecified Linear Models, and a Proposal for Repairing It","Grünwald"),
("guo2017","On Calibration of Modern Neural Networks","Guo"),
("breiman2001","Random Forests","Breiman"),
("tremblay2021","Multimodal dynamics modeling for off-road autonomous vehicles","Tremblay"),
("zhao2023","Fast LIO2 Fast Direct LiDAR-Inertial Odometry","Xu"),
]
def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
out = {}
for key, title, first in Q:
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": title, "query.author": first, "rows": 5})
    try:
        items = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "reference-check/1.0"}), timeout=30))["message"]["items"]
    except Exception as e:
        out[key] = dict(status="lookup_error", error=str(e), query=title); continue
    best = None
    for it in items:
        t = (it.get("title") or [""])[0]
        if norm(title)[:30] == norm(t)[:30] or norm(t)[:30] in norm(title):
            best = it; break
    if not best:
        out[key] = dict(status="not_found", query=title, top=[(i.get("title") or [""])[0] for i in items[:3]]); continue
    a = best.get("author", [])
    y = None
    for f in ("published-print", "published-online", "published", "issued"):
        if best.get(f, {}).get("date-parts", [[None]])[0][0]:
            y = best[f]["date-parts"][0][0]; break
    out[key] = dict(status="crossref", title=best["title"][0], authors=[f"{x.get('given','')} {x.get('family','')}".strip() for x in a], first_family=a[0].get("family") if a else None,
                    venue=(best.get("container-title") or [""])[0], volume=best.get("volume"), issue=best.get("issue"), page=best.get("page"), year=y, doi=best.get("DOI"), type=best.get("type"))
    time.sleep(0.2)
for key, aid in (("gao2026","2603.04470"),("khattak2025","2505.06483")):
    x = urllib.request.urlopen(f"http://export.arxiv.org/api/query?id_list={aid}", timeout=30).read()
    root = ET.fromstring(x); ns = {"a": "http://www.w3.org/2005/Atom"}
    e = root.find("a:entry", ns)
    out[key] = dict(status="arxiv", arxiv=aid, title=" ".join(e.find("a:title", ns).text.split()), authors=[n.find("a:name", ns).text for n in e.findall("a:author", ns)], year=int(e.find("a:published", ns).text[:4]))
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
for k, v in out.items():
    print(k, "|", v["status"], "|", (v.get("title") or str(v.get("top")))[:80], "|", v.get("venue", "")[:40], v.get("volume"), v.get("page"), v.get("year"), v.get("doi"))
