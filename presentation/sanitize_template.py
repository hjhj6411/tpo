#!/usr/bin/env python3
"""Strip inherited baggage out of assets/template.pptx.

The template is the author's own deck with the slides removed, so it arrives
carrying three things that belong to *that* file and not to this one:

1. `<p:embeddedFontLst>` — 2.9 MB of 맑은 고딕 embedded as `.fntdata`, SUBSET to
   the glyphs the original 76 slides used. New slides carry Korean text outside
   that subset, so PowerPoint has to reconcile an embedded font that does not
   cover the document. With a CJK face that stalls the app on open. 맑은 고딕
   ships with Windows/Office, so embedding buys nothing here.
2. `docProps/app.xml` — declares `<Slides>76</Slides>`, `<Notes>48</Notes>` and a
   `TitlesOfParts` vector of 79 entries. python-pptx copies this part verbatim
   and never rewrites it, so a 14-slide deck shipped metadata describing 76.
3. `docProps/thumbnail.jpeg` + `core.xml` — the previous deck's preview image and
   its authorship (creator, revision 122, March creation date).

Run once; the result is committed. Idempotent.

    python sanitize_template.py [path/to/template.pptx]
"""
import os
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "assets", "template.pptx")

RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
FONT_REL = "/relationships/font"
THUMB_REL = "/metadata/thumbnail"   # NOT "/relationships/thumbnail"

APP_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/'
    'extended-properties" xmlns:vt="http://schemas.openxmlformats.org/'
    'officeDocument/2006/docPropsVTypes">'
    "<Application>Microsoft Office PowerPoint</Application>"
    "<PresentationFormat>A4 용지(210x297mm)</PresentationFormat>"
    "<ScaleCrop>false</ScaleCrop><LinksUpToDate>false</LinksUpToDate>"
    "<SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged>"
    "<AppVersion>16.0000</AppVersion>"
    "</Properties>"
)

CORE_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/'
    'metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:dcterms="http://purl.org/dc/terms/" '
    'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    "<dc:title>공학연구인턴십 성과발표</dc:title>"
    "<dc:creator>조현준</dc:creator>"
    "<cp:lastModifiedBy>조현준</cp:lastModifiedBy>"
    "<cp:revision>1</cp:revision>"
    "</cp:coreProperties>"
)


def drop_rels(xml: str, *substrings: str) -> str:
    """Remove <Relationship> elements whose text contains any substring.

    Note `<Relationships>` shares a prefix with `<Relationship>`; match the
    self-closing child form explicitly so the container tag is never touched.
    """
    def repl(m):
        return "" if any(sub in m.group(0) for sub in substrings) else m.group(0)

    return re.sub(r"<Relationship\s[^>]*?/>", repl, xml)


def main() -> None:
    src = zipfile.ZipFile(TPL)
    items = {n: src.read(n) for n in src.namelist()}
    src.close()

    dropped = []

    # 1. embedded fonts: the list, the parts, the rels, the content-type Default
    pres = items["ppt/presentation.xml"].decode("utf-8")
    if "<p:embeddedFontLst>" in pres:
        pres = re.sub(r"<p:embeddedFontLst>.*?</p:embeddedFontLst>", "", pres, flags=re.S)
        items["ppt/presentation.xml"] = pres.encode("utf-8")
        dropped.append("p:embeddedFontLst")

    for name in [n for n in items if n.startswith("ppt/fonts/")]:
        dropped.append(name)
        del items[name]

    rels = items["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    items["ppt/_rels/presentation.xml.rels"] = drop_rels(rels, FONT_REL).encode("utf-8")

    # 2. the previous deck's thumbnail
    for name in [n for n in items if n.startswith("docProps/thumbnail")]:
        dropped.append(name)
        del items[name]
    root_rels = items["_rels/.rels"].decode("utf-8")
    items["_rels/.rels"] = drop_rels(root_rels, THUMB_REL).encode("utf-8")

    # 3. stale document properties (python-pptx copies these verbatim)
    items["docProps/app.xml"] = APP_XML.encode("utf-8")
    items["docProps/core.xml"] = CORE_XML.encode("utf-8")

    # 4. content types: no fntdata Default, no thumbnail/font Overrides left over
    ct = items["[Content_Types].xml"].decode("utf-8")
    ct = re.sub(r'<Default Extension="fntdata"[^>]*/>', "", ct)
    ct = re.sub(r'<Override PartName="/ppt/fonts/[^"]*"[^>]*/>', "", ct)
    ct = re.sub(r'<Override PartName="/docProps/thumbnail[^"]*"[^>]*/>', "", ct)
    # jpeg Default is only there for the thumbnail; keep it only if a jpeg remains
    if not any(n.lower().endswith((".jpeg", ".jpg")) for n in items):
        ct = re.sub(r'<Default Extension="jpeg"[^>]*/>', "", ct)
    items["[Content_Types].xml"] = ct.encode("utf-8")

    tmp = TPL + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        # [Content_Types].xml must be the first entry in an OPC package
        out.writestr("[Content_Types].xml", items.pop("[Content_Types].xml"))
        for name, blob in items.items():
            out.writestr(name, blob)
    shutil.move(tmp, TPL)

    verify(TPL)
    print(f"sanitized {TPL}")
    for d in dropped:
        print("  dropped:", d)
    print(f"  size now: {os.path.getsize(TPL)/1e6:.2f} MB")


def verify(path: str) -> None:
    """A dangling relationship is exactly what makes PowerPoint offer to repair."""
    import posixpath
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    problems = []
    for n in names:
        if not n.endswith(".rels"):
            continue
        base = posixpath.dirname(posixpath.dirname(n))
        for m in re.finditer(r"<Relationship\s[^>]*?/>", z.read(n).decode("utf-8")):
            tag = m.group(0)
            if 'TargetMode="External"' in tag:
                continue
            tgt = re.search(r'Target="([^"]+)"', tag).group(1)
            resolved = posixpath.normpath(posixpath.join(base, tgt)).lstrip("/")
            if resolved not in names:
                problems.append(f"{n} -> {tgt} (missing)")
    ct = z.read("[Content_Types].xml").decode("utf-8")
    exts = set(re.findall(r'<Default Extension="([^"]+)"', ct))
    overrides = set(re.findall(r'<Override PartName="([^"]+)"', ct))
    for n in names:
        if n == "[Content_Types].xml" or n.endswith("/"):
            continue
        if "/" + n in overrides or n.rsplit(".", 1)[-1].lower() in exts:
            continue
        problems.append(f"{n} has no content type")
    for o in overrides:
        if o.lstrip("/") not in names:
            problems.append(f"content-type Override for missing part {o}")
    z.close()
    if problems:
        raise SystemExit("package is inconsistent:\n  " + "\n  ".join(problems))
    print("  verified: every relationship and content type resolves")


if __name__ == "__main__":
    main()
