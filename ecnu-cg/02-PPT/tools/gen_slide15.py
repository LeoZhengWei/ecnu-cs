# -*- coding: utf-8 -*-
"""生成第 15 页（附录：关键步骤的核心代码）。
代码块以 VS Code Dark+ 风格 PNG 截图嵌入（render_code_png.py 生成），
标题条与文件路径仍用 Text，保证可编辑、可搜索。
"""
import io, os

def esc(s):
    return (s.replace('&', '&#38;')
             .replace('<', '&#60;')
             .replace('>', '&#62;')
             .replace('{', '&#123;')
             .replace('}', '&#125;'))

HEAD_C = '#8FBBFF'
PATH_C = '#7F8C9F'
CARD = '#1E1E1E'
DPR = 2
IMG_W = 568

BLOCKS = [
    # (步骤标签, 文件:行, 截图)
    ('① ② ③  检索 → 粗筛 → 建边', 'scene/scene_model_loop_closure.py:179', '01'),
    ('④  协视图贪心扩张（式 1）', 'poses/ba_problem.py:648', '02'),
    ('⑤  局部 BA：LM 迭代 + Schur 消元', 'poses/mini_ba.py:285', '03'),
    ('⑥  回环检测与补边', 'poses/ba_problem.py:1054', '04'),
    ('⑦-a  分块深度对齐', 'scene/mono_depth.py:315', '05'),
    ('⑦-b  渐进层级与合并', 'scene/hierarchy_cut.py:105', '06'),
]

def card(block):
    head, path, tag = block
    png = 'code_png/%s.png' % tag
    from PIL import Image
    w, h = Image.open(png).size
    assert w == IMG_W * DPR, (png, w)
    disp_h = round(h / DPR, 1)
    out = []
    out.append(
        "        <Box style={{ backgroundColor: '%s', borderRadius: 5, "
        "padding: '5px 11px 5px 11px', flexDirection: 'column' }}>" % CARD)
    out.append(
        "            <Text style={{ fontSize: 13, fontWeight: 'bold', color: '%s', "
        "marginBottom: 0 }}>%s</Text>" % (HEAD_C, esc(head)))
    out.append(
        "            <Text style={{ fontFamily: 'Consolas', fontSize: 10, color: '%s', "
        "marginBottom: 2 }}>%s</Text>" % (PATH_C, esc(path)))
    out.append(
        "            <Image src=\"%s\" style={{ width: %d, height: %s }} />" % (png, IMG_W, disp_h))
    out.append("        </Box>")
    return '\n'.join(out)

LEFT = BLOCKS[:3]
RIGHT = BLOCKS[3:]

parts = []
parts.append("<Slide style={{ width: '1280px', height: '720px', backgroundColor: '#FFFFFF', padding: '0px', flexDirection: 'column' }}>")
parts.append("    <Box style={{ width: '1280px', height: '8px', backgroundColor: '#1E4FA8' }} />")
parts.append("    <Box style={{ padding: '14px 40px 0px 40px', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end' }}>")
parts.append("        <Text style={{ fontSize: 32, fontWeight: 'bold', color: '#1E4FA8' }}>附：七个步骤在源码里长什么样</Text>")
parts.append("        <Text style={{ fontSize: 15, color: '#8B97A8' }}>15 / 15</Text>")
parts.append("    </Box>")
parts.append("    <Box style={{ padding: '5px 40px 0px 40px' }}>")
parts.append("        <Text style={{ fontSize: 17, color: '#4A5568' }}>真实源码截图，行号对应第 7 页七步链：<b>①②③ 定跟谁比、④ 定跟谁一起算、⑤ 才真解位姿、⑥⑦ 修漂移与压显存</b>。</Text>")
parts.append("    </Box>")
parts.append("    <Box style={{ padding: '8px 40px 0px 40px', flexDirection: 'row', gap: 14 }}>")
parts.append("        <Box style={{ flex: 1, flexDirection: 'column', gap: 10 }}>")
for b in LEFT:
    parts.append(card(b))
parts.append("        </Box>")
parts.append("        <Box style={{ flex: 1, flexDirection: 'column', gap: 10 }}>")
for b in RIGHT:
    parts.append(card(b))
parts.append("        </Box>")
parts.append("    </Box>")
parts.append("    <Box style={{ padding: '7px 40px 0px 40px' }}>")
parts.append("        <Text style={{ fontSize: 13.5, color: '#8B97A8', lineHeight: 1.35 }}>行号指向片段起点；为投影宽度做了换行与变量缩写（w = edge_weight），逻辑未改。第 7 页已按源码默认值修正：回环触发为<b>加权图距 &gt; 10</b>；层级合并触发为<b>可并数 &gt; max(10%, 20 万)</b>，且是 <b>kNN 加权平均</b>而非 KL。</Text>")
parts.append("    </Box>")
parts.append("</Slide>")

txt = '\n'.join(parts) + '\n'
io.open('slides_v2/15.slide', 'w', encoding='utf-8', newline='\n').write(txt)
print('WROTE slides_v2/15.slide  lines=%d  chars=%d' % (txt.count('\n'), len(txt)))
