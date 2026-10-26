import re
import zipfile

z = zipfile.ZipFile("i3dgs复现汇报_final.pptx")

pres = z.read("ppt/presentation.xml").decode("utf-8")
ids = re.findall(r'r:id="(rId\d+)"', pres)
print("sldIdLst r:ids:", ids)

rels = z.read("ppt/_rels/presentation.xml.rels").decode("utf-8")
m = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
print("\norder:")
for i, rid in enumerate(ids, 1):
    tgt = m.get(rid, "?")
    print(f"  page {i:2d} -> {tgt}")

print("\nall slide parts:")
parts = sorted(
    [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
    key=lambda s: int(re.search(r"(\d+)", s.split("/")[-1]).group(1)),
)
used = {m.get(r, "") for r in ids}
for p in parts:
    tag = "USED" if p in used else "orphan"
    print(" ", p, tag)
