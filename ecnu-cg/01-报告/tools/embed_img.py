"""把 report.html 里的三张渲染图压缩并内嵌为 base64，生成 report_docx.html 供转换。
原因：html-to-doc 对相对路径 <img> 不落图，改用 data URI 可确保嵌入。"""

import base64
import io
import os
import re

from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "report.html")
DST = os.path.join(BASE, "report_docx.html")
TARGET_W = 620
QUALITY = 82

html = open(SRC, encoding="utf-8").read()


def to_data_uri(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w > TARGET_W:
        im = im.resize((TARGET_W, int(h * TARGET_W / w)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=QUALITY, optimize=True)
    raw = buf.getvalue()
    print("  %-24s %dx%d -> %dx%d  %d KB" % (os.path.basename(path), w, h, im.size[0], im.size[1], len(raw) // 1024))
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def repl(m):
    src = m.group(1)
    p = os.path.join(BASE, src.replace("/", os.sep))
    if not os.path.exists(p):
        print("  MISSING", p)
        return m.group(0)
    uri = to_data_uri(p)
    return m.group(0).replace(src, uri)


print("embedding images:")
html2 = re.sub(r'src="(img/[^"]+)"', repl, html)

open(DST, "w", encoding="utf-8").write(html2)
print("written:", DST, len(html2), "chars")
