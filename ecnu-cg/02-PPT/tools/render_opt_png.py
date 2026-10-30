# -*- coding: utf-8 -*-
"""把「核心优化」四段代码渲染成 VS Code Dark+ 风格 PNG（2x），供第 13 页以图片形式嵌入。"""
import os, re
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import CppLexer, PythonLexer
from pygments.token import (Keyword, Name, String, Number, Comment,
                            Operator, Punctuation)

DPR = 2
W = 579 * DPR          # 显示宽度 579（<= 卡片内宽）
LH = 36                # 行高（@2x）
PAD_T = 11
GUTTER = 66            # 行号列宽（@2x）
CODE_X = GUTTER + 8

BG = (30, 30, 30)
LN = (110, 118, 129)
C = {
    'kw':    (86, 156, 214),
    'ctrl':  (197, 134, 192),
    'prep':  (197, 134, 192),
    'type':  (78, 201, 176),
    'func':  (220, 220, 170),
    'str':   (206, 145, 120),
    'num':   (181, 206, 168),
    'com':   (106, 153, 85),
    'op':    (212, 212, 212),
    'name':  (156, 220, 254),
    'built': (78, 201, 176),
    'punct': (212, 212, 212),
    'plain': (212, 212, 212),
}
CTRL_WORDS = {'if', 'else', 'elif', 'for', 'while', 'in', 'return', 'and',
              'or', 'not', 'is', 'with', 'as', 'import', 'from', 'break',
              'continue', 'pass', 'lambda', 'yield', 'try', 'except'}

FONT_CODE = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 26)
FONT_CJK = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
FONT_LN = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 22)

CJK = re.compile(r'[\u2e80-\u9fff\u3000-\u303f\uff00-\uffef\u2018\u2019\u201c\u201d]')
SPLIT = re.compile(r'([\u2e80-\u9fff\u3000-\u303f\uff00-\uffef\u2018\u2019\u201c\u201d]+)')

LEXERS = {'cpp': CppLexer(), 'py': PythonLexer()}


def colorize(line, lang):
    out = []
    for ttype, val in lex(line, LEXERS[lang]):
        if not val:
            continue
        val = val.rstrip('\n')
        if not val:
            continue
        if ttype in Comment.Preproc:
            col = C['prep']
        elif ttype in Comment:
            col = C['com']
        elif ttype in String:
            col = C['str']
        elif ttype in Number:
            col = C['num']
        elif ttype in Keyword.Type:
            col = C['type']
        elif ttype in Keyword:
            col = C['ctrl'] if val in CTRL_WORDS else C['kw']
        elif ttype in Keyword.Constant:
            col = C['kw']
        elif ttype in Operator.Word:
            col = C['ctrl'] if val in CTRL_WORDS else C['op']
        elif ttype in Punctuation:
            col = C['punct']
        elif ttype in Name.Builtin:
            col = C['built']
        elif ttype in Name.Function:
            col = C['func']
        elif ttype in Name:
            col = C['name']
        elif ttype in Operator:
            col = C['op']
        else:
            col = C['plain']
        out.append((val, col))
    return out


BLOCKS = [
    ('01', 'cpp', 133, [
        '// 2. SE3 Jacobian (Left Perturbation convention).',
        '// d(p_cam) / d(xi) = [ I | -[p_cam]_cross ]',
        'float du_dtx = du_drx;               // 平移块：就是投影雅可比',
        'float du_dwx = du_dry * (-p_cam_z) + du_drz * (p_cam_y);',
        'float du_dwy = du_drx * (p_cam_z) + du_drz * (-p_cam_x);',
        'float du_dwz = du_drx * (-p_cam_y) + du_dry * (p_cam_x);',
        'int jac_base = residual_base * 2 * n_cams * 6;   // 步长 6，不是 9',
        'int cam_param_offset = camera_idx * 6;',
        'jacobian_se3_ptr[jac_base + cam_param_offset + 3] = du_dwx;',
    ]),
    ('02', 'cpp', 186, [
        '// Taylor expansion for small angles to avoid division by zero',
        'if (theta < 1e-5f) {',
        '    A = 1.0f - theta_sq * (1.0f / 6.0f);      // A = sin(th)/th',
        '    B = 0.5f - theta_sq * (1.0f / 24.0f);     // B = (1-cos th)/th^2',
        "} else { A = sinf(theta) * inv_theta; /* ... */ }",
        '// Construct R_upd = I + A*K + B*K^2   九个元素直算，不建临时矩阵',
        'float R_upd_00 = 1.0f - B*(wy2 + wz2);',
        'float R_upd_01 = -A*w2 + B*wxy;',
        'float t_upd_0 = u0 + B*c1_0 + C*c2_0;       // t = V*u 同理',
    ]),
    ('03', 'py', 296, [
        'jtj_cam.diagonal().mul_(1 + lm)              # LM 阻尼',
        'jtj_cam.diagonal().clamp_min_(3e7 * self.ep)',
        '# Solve with Schur Complement',
        'jtj_xyz_inv = self.inv_3x3(jtj_xyz)          # 每点一个 3x3，逐点求逆',
        'BD = (torch.bmm(jtj_xyz_inv, jtj_cam_xyz.view(-1, 3, self.n_cam_params))',
        '      .view(-1, self.n_cam_params).T)',
        'BmECm1Et = jtj_cam - BD @ jtj_cam_xyz        # S = B - E C^-1 E^T',
        'L_cam = torch.linalg.cholesky_ex(BmECm1Et)[0]',
        'v = jacxr_cam - BD @ jacxr_xyz               # 右端做同样的消元',
        'dcam = torch.linalg.solve_triangular(L_cam, v.unsqueeze(-1), upper=False)',
    ]),
    ('04', 'py', 506, [
        '# cholesky_ex does not support HIP stream capture -> AMD 上降级',
        'if make_cuda_graph and not torch.version.hip:',
        '    self.optimizer = torch.cuda.make_graphed_callables(',
        '        self.optimizer, (Rts_init, f_init, xyz_init, centre, uv,',
        '                         cam_indices, fixed_masks_init), 2)',
        "# Graph Safe Updates (No 'if' on tensors)：循环里不许有数据相关分支",
        'poses = torch.where(success_mask, poses_tmp, poses)',
        'f     = torch.where(success_mask, f_tmp, f)',
        'xyz   = torch.where(success_mask, xyz_tmp, xyz)',
        'lm    = torch.where(success_mask, lm / self.k, lm * self.k)',
    ]),
]


def render(tag, lang, start_ln, lines, path):
    h = PAD_T * 2 + LH * len(lines)
    img = Image.new('RGB', (W, h), BG)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        base = PAD_T + i * LH + 26
        d.text((GUTTER - 4, base), str(start_ln + i), font=FONT_LN,
               fill=LN, anchor='rs')
        x = CODE_X
        for seg, col in colorize(line, lang):
            for p in SPLIT.split(seg):
                if not p:
                    continue
                f = FONT_CJK if CJK.search(p) else FONT_CODE
                d.text((x, base), p, font=f, fill=col, anchor='ls')
                x += f.getlength(p)
        if x > W - 6:
            print('  !! 超宽 %s 行%d  x=%.0f/%d' % (tag, start_ln + i, x, W))
    img.save(path, optimize=True)
    return img.size


os.makedirs('opt_png', exist_ok=True)
for tag, lang, ln, lines in BLOCKS:
    p = 'opt_png/%s.png' % tag
    w, h = render(tag, lang, ln, lines, p)
    print('%-3s -> %s  %dx%d  (显示 %dx%.1f)  %d B' %
          (tag, p, w, h, w / DPR, h / DPR, os.path.getsize(p)))
