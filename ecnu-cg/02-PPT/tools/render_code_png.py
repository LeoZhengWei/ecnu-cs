# -*- coding: utf-8 -*-
"""把第 15 页的六段代码渲染成 VS Code Dark+ 风格 PNG（2x），供 PPT 以图片形式嵌入。"""
import io, os, re, sys
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import (Keyword, Name, String, Number, Comment,
                            Operator, Punctuation, Token)

DPR = 2
W = 568 * DPR          # 显示宽度 568
LH = 37                # 行高（@2x）
PAD_T = 12
GUTTER = 68            # 行号列宽（@2x）
CODE_X = GUTTER + 8

BG = (30, 30, 30)          # #1E1E1E
LN = (110, 118, 129)       # #6E7681
C = {
    'kw':    (86, 156, 214),    # #569CD6
    'ctrl':  (197, 134, 192),   # #C586C0
    'func':  (220, 220, 170),   # #DCDCAA
    'str':   (206, 145, 120),   # #CE9178
    'num':   (181, 206, 168),   # #B5CEA8
    'com':   (106, 153, 85),    # #6A9955
    'op':    (212, 212, 212),   # #D4D4D4
    'name':  (156, 220, 254),   # #9CDCFE
    'self':  (86, 156, 214),
    'built': (78, 201, 176),    # #4EC9B0
    'punct': (212, 212, 212),
    'plain': (212, 212, 212),
}
CTRL_WORDS = {'if', 'else', 'elif', 'for', 'while', 'in', 'return', 'and',
              'or', 'not', 'is', 'with', 'as', 'import', 'from', 'break',
              'continue', 'pass', 'lambda', 'yield', 'try', 'except'}

FONT_CODE = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 26)
FONT_BOLD = ImageFont.truetype('C:/Windows/Fonts/consolab.ttf', 26)
FONT_CJK = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
FONT_LN = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 22)

CJK = re.compile(r'[\u2e80-\u9fff\u3000-\u303f\uff00-\uffef\u2018\u2019\u201c\u201d]')

BLOCKS = [
    ('01', 179, [
        'cossim = self.vpr_features @ vpr_feature.T  # 全局描述子',
        '_, cand = torch.topk(cossim_best, n_candidates)  # top-20',
        'n_m = self.matcher.batched_evaluate_match(...)  # XFeat',
        '_, best = torch.topk(n_m, n)  # 留 top-5',
        'self.matcher(..., remove_outliers=True, use_lightglue=True)  # 建边',
    ]),
    ('02', 648, [
        'while len(selected_ids) < get_n_keyframes:  # 扩到 20 帧',
        '    for sid in selected_ids:',
        '        for nb in self.pose_graph.neighbors(sid):',
        '            if nb not in selected_ids:  # 边权 = 1/|M|',
        '                neighbor_counts[nb] += 1.0 / w(sid, nb)',
        '    selected_ids.add(neighbor_counts.most_common(1)[0][0])',
    ]),
    ('03', 285, [
        'for it in range(self.iters):',
        '    jtj_cam, ..., resid = self._assemble_jtj(...)',
        '    jtj_cam.diagonal().mul_(1 + lm)  # LM 阻尼',
        '    jtj_xyz_inv = self.inv_3x3(jtj_xyz)  # C^-1',
        '    S = jtj_cam - jtj_xyz_inv @ jtj_cam_xyz.T @ jtj_cam_xyz',
        '    L = torch.linalg.cholesky_ex(S)[0]  # 舒尔补 S',
        '    dcam = torch.linalg.solve_triangular(...)  # ΔT',
        '    dxyz = torch.bmm(b.view(-1, 1, 3), jtj_xyz_inv)',
    ]),
    ('04', 1054, [
        'if dists.max() > self.min_loop_size:  # 图距 > 10',
        '    clusters_t = split_into_clusters(dists)  # 分两簇',
        'is_loop = (min(n_robust_pairs) > n // 2 and',
        '           min(max_matches_robustness) > 0.5 * max(...))',
        'if len(match.kpts) >= self.min_matches_edge:  # τm=500',
        '    self.pose_graph.add_match(i, j, 1.0 / len(match.kpts))',
    ]),
    ('05', 315, [
        'tri_idepth = 1 / (pts3d[has_pt3d] @ R.T + t)[..., -1]',
        'mono_al, g_s, g_o = align_samples(tri_idepth, sampled)',
        'valid = err < m_err + grid_outlier_mult * err_dev',
        'return align_depth_grid(..., 4, g_s, g_o)  # 4x4 分块',
    ]),
    ('06', 105, [
        'mask = screen_size < self.hierarchy_screen_size_threshold  # 0.5 px',
        'if mask.sum() > max(len(self.xyz) * 0.1, 200_000):',
        '    n_merged = self.create_hierarchy_level(cam_centre, mask)',
        '# 合并权重 = opacity * screen_size^2（非 KL 投影）',
        'weights = opa.sigmoid() * screen_size[selected] ** 2',
    ]),
]


def colorize(line):
    """返回 [(text, rgb, is_cjk_aware)] 段列表。"""
    out = []
    for ttype, val in lex(line, PythonLexer()):
        if not val:
            continue
        val = val.rstrip('\n')
        if not val:
            continue
        if ttype in Comment:
            col = C['com']
        elif ttype in String:
            col = C['str']
        elif ttype in Number:
            col = C['num']
        elif ttype in Keyword:
            col = C['ctrl'] if val in CTRL_WORDS else C['kw']
        elif ttype in Operator.Word:
            col = C['ctrl'] if val in CTRL_WORDS else C['op']
        elif ttype in Punctuation:
            col = C['punct']
        elif ttype in Name.Builtin.Pseudo:
            col = C['self']
        elif ttype in Name.Builtin:
            col = C['built']
        elif ttype in Name.Function:
            col = C['func']
        elif ttype in Name.Attribute or ttype in Name:
            col = C['name']
        elif ttype in Operator:
            col = C['op']
        else:
            col = C['plain']
        out.append((val, col))
    return out


def render(tag, start_ln, lines, path):
    h = PAD_T * 2 + LH * len(lines)
    img = Image.new('RGB', (W, h), BG)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        base = PAD_T + i * LH + 27
        # 行号
        d.text((GUTTER - 4, base), str(start_ln + i), font=FONT_LN,
               fill=LN, anchor='rs')
        x = CODE_X
        for seg, col in colorize(line):
            # 按中英文切开，各用各的字体
            parts = re.split(r'([\u2e80-\u9fff\u3000-\u303f\uff00-\uffef\u2018\u2019\u201c\u201d]+)', seg)
            for p in parts:
                if not p:
                    continue
                f = FONT_CJK if CJK.search(p) else FONT_CODE
                d.text((x, base), p, font=f, fill=col, anchor='ls')
                x += f.getlength(p)
        if x > W - 6:
            print('  !! 超宽 %s 行%d  x=%.0f/%d' % (tag, start_ln + i, x, W))
    img.save(path, optimize=True)
    return img.size


os.makedirs('code_png', exist_ok=True)
for tag, ln, lines in BLOCKS:
    p = 'code_png/%s.png' % tag
    w, h = render(tag, ln, lines, p)
    print('%-3s -> %s  %dx%d  (显示 %dx%.1f)  %d B' %
          (tag, p, w, h, w / DPR, h / DPR, os.path.getsize(p)))
