import re
import zipfile

z = zipfile.ZipFile("图形和视觉前沿技术报告-i3dgs复现.docx")
xml = z.read("word/document.xml").decode("utf-8")
txt = re.sub(r"<[^>]+>", "", xml)

keys = ["25.95", "0.786", "0.206", "62.7", "362 044", "十二处", "RTX 4090", "sm_89", "图 1", "表 5", "1.888", "156"]
for k in keys:
    print(("OK   " if k in txt else "MISS "), k)

imgs = [n for n in z.namelist() if n.startswith("word/media/") and n.endswith((".jpeg", ".png", ".jpg"))]
print("images embedded:", len(imgs))
print("text length:", len(txt))
