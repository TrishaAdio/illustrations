"""
Search Wikimedia Commons for reference photographs, filtered by licence and size.

Written as a tool rather than a throwaway because casting a plate is a recurring
job: each new character or scene needs a sitter, and the constraints are always
the same - a permissive licence, enough resolution to draw from, and a real
photograph rather than a scanned book page.

Commons full-text search happily returns djvu and pdf scans whose *contents*
mention the query, so results are filtered to image extensions. Requests are
spaced out because the API returns 429 readily.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "kalam-reference-search/0.1 (illustration reference casting)"}
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

OPEN_HINTS = ("public domain", "pd-", "cc0", "cc by", "cc-by", "attribution")
CLOSED_HINTS = ("non-free", "fair use", "nc", "nd")


def api(pause: float = 1.2, **params) -> dict:
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                out = json.load(r)
            time.sleep(pause)
            return out
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2.5 * (attempt + 1))
            _ = e
    return {}


def search(term: str, limit: int = 30) -> list[str]:
    d = api(action="query", list="search", srsearch=term, srnamespace=6, srlimit=limit)
    return [r["title"] for r in d.get("query", {}).get("search", [])]


def category(cat: str, limit: int = 60) -> list[str]:
    d = api(action="query", list="categorymembers", cmtitle=f"Category:{cat}",
            cmtype="file", cmlimit=limit)
    return [m["title"] for m in d.get("query", {}).get("categorymembers", [])]


def details(titles: list[str]) -> list[dict]:
    titles = [t for t in titles if t.lower().endswith(IMG_EXT)]
    out: list[dict] = []
    for i in range(0, len(titles), 10):
        d = api(action="query", titles="|".join(titles[i:i + 10]), prop="imageinfo",
                iiprop="url|size|extmetadata")
        for p in d.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [None])[0]
            if not ii:
                continue
            em = ii.get("extmetadata", {})
            lic = em.get("LicenseShortName", {}).get("value", "?")
            out.append({
                "title": p["title"],
                "w": ii.get("width", 0),
                "h": ii.get("height", 0),
                "lic": lic,
                "url": ii.get("url", ""),
                "desc": em.get("ImageDescription", {}).get("value", "")[:110],
            })
    return out


def permissive(lic: str) -> bool:
    low = lic.lower()
    if any(h in low for h in CLOSED_HINTS) and "cc by" not in low:
        return False
    return any(h in low for h in OPEN_HINTS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-edge", type=int, default=1000)
    ap.add_argument("--portrait-only", action="store_true",
                    help="keep only taller-than-wide images")
    a = ap.parse_args()

    queries = [
        ("search", "elderly Indian man spectacles portrait photograph"),
        ("search", "old Bengali man portrait dhoti photograph"),
        ("search", "Indian old man moustache spectacles face"),
        ("category", "Old men of India"),
        ("category", "Men of India with eyeglasses"),
        ("category", "Portrait photographs of old men"),
        ("category", "Elderly people of India"),
    ]

    seen: set[str] = set()
    keep: list[dict] = []
    for kind, term in queries:
        try:
            titles = search(term) if kind == "search" else category(term)
        except Exception as e:
            print(f"!! {term}: {e}")
            continue
        try:
            rows = details(titles)
        except Exception as e:
            print(f"!! details for {term}: {e}")
            continue
        hits = 0
        for r in rows:
            if r["title"] in seen:
                continue
            if max(r["w"], r["h"]) < a.min_edge:
                continue
            if a.portrait_only and r["h"] <= r["w"]:
                continue
            if not permissive(r["lic"]):
                continue
            seen.add(r["title"])
            keep.append(r)
            hits += 1
        print(f"{term[:46]:<48} {hits} kept")

    print("\n" + "=" * 96)
    for r in sorted(keep, key=lambda x: -max(x["w"], x["h"])):
        print(f'{r["w"]:>5}x{r["h"]:<5} {r["lic"][:22]:<24} {r["title"][8:70]}')
        print(f'      {r["url"]}')
    print(f"\n{len(keep)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
