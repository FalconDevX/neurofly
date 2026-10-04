"""Podmienia slajd 1 w decku z build_mvp.js na ręcznie poprawiony slajd tytułowy z NeuroFly.pptx.

    python merge_title.py NeuroFly_mvp_raw.pptx NeuroFly_mvp.pptx [--src NeuroFly.pptx]

Kopiuje XML slajdu, jego obrazy i notatki; układ wskazuje na TITLE w nowym decku (ten sam układ z build.js).
"""
import argparse
import posixpath
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
REL_LAYOUT = "relationships/slideLayout"
REL_NOTES = "relationships/notesSlide"


def first_slide(z):
    pres = z.read("ppt/presentation.xml").decode("utf8")
    rid = re.search(r'<p:sldIdLst>\s*<p:sldId [^>]*r:id="([^"]+)"', pres).group(1)
    rels = z.read("ppt/_rels/presentation.xml.rels").decode("utf8")
    target = re.search(rf'<Relationship [^>]*Id="{rid}"[^>]*/>', rels).group(0)
    return "ppt/" + re.search(r'Target="([^"]+)"', target).group(1)


def rels_of(part):
    d, f = posixpath.split(part)
    return f"{d}/_rels/{f}.rels"


def parse_rels(xml):
    out = []
    for m in re.finditer(r"<Relationship ([^>]*)/>", xml):
        a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        out.append(a)
    return out


def resolve(base_part, target):
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_part), target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", type=Path, default=HERE / "NeuroFly.pptx")
    a = ap.parse_args()

    src, raw = zipfile.ZipFile(a.src), zipfile.ZipFile(a.raw)
    s_slide, r_slide = first_slide(src), first_slide(raw)
    s_rels, r_rels = parse_rels(src.read(rels_of(s_slide)).decode()), parse_rels(raw.read(rels_of(r_slide)).decode())
    r_layout = next(r["Target"] for r in r_rels if r["Type"].endswith(REL_LAYOUT))
    r_notes = next((r["Target"] for r in r_rels if r["Type"].endswith(REL_NOTES)), None)

    files = {n: raw.read(n) for n in raw.namelist()}
    ctypes = files["[Content_Types].xml"].decode("utf8")
    new_rels = []
    for r in s_rels:
        t = r["Target"]
        if r["Type"].endswith(REL_LAYOUT):
            t = r_layout
        elif r["Type"].endswith(REL_NOTES):
            if r_notes is None:
                continue
            notes_xml = src.read(resolve(s_slide, r["Target"]))
            if b"r:id=" not in notes_xml and b"r:embed=" not in notes_xml:
                files[resolve(r_slide, r_notes)] = notes_xml
            t = r_notes
        elif r.get("TargetMode") != "External":
            part = resolve(s_slide, t)
            name = "ppt/media/title_" + posixpath.basename(part)
            files[name] = src.read(part)
            ext = posixpath.splitext(name)[1][1:].lower()
            if f'Extension="{ext}"' not in ctypes:
                mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}[ext]
                ctypes = ctypes.replace("<Default ", f'<Default Extension="{ext}" ContentType="{mime}"/><Default ', 1)
            t = posixpath.relpath(name, posixpath.dirname(r_slide))
        extra = ' TargetMode="External"' if r.get("TargetMode") == "External" else ""
        new_rels.append(f'<Relationship Id="{r["Id"]}" Type="{r["Type"]}" Target="{t}"{extra}/>')

    files[r_slide] = src.read(s_slide)
    files[rels_of(r_slide)] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(new_rels) + "</Relationships>"
    ).encode("utf8")
    files["[Content_Types].xml"] = ctypes.encode("utf8")

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in ["[Content_Types].xml"] + [n for n in files if n != "[Content_Types].xml"]:
            z.writestr(n, files[n])
    print("zapisano", a.out)


if __name__ == "__main__":
    main()
