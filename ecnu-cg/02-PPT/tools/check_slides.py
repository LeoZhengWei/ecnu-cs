import re
import zipfile

z = zipfile.ZipFile("i3dgs复现汇报_final.pptx")
slides = sorted(
    [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
    key=lambda s: int(re.search(r"(\d+)", s.split("/")[-1]).group(1)),
)
print("slide count:", len(slides))
for n in slides:
    xml = z.read(n).decode("utf-8", "ignore")
    txt = "".join(re.findall(r"<a:t>([^<]*)</a:t>", xml))
    flag = ""
    if "跑通了什么" in txt:
        flag = "  <<< NEW 11"
    elif "卡在哪" in txt:
        flag = "  <<< OLD 11"
    print(n, "|", txt[:46], flag)
