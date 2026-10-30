# -*- coding: utf-8 -*-
"""生成 P14「重建过程：同一机位，12 个瞬间」。

时刻来自 i3dgs 训练循环逐帧打点（纯净运行，无写盘开销）：
  84 帧主体 26.1 s，收尾重试搁置帧至 54.5 s；
  单帧延迟中位 87 ms、p90 756 ms、最大 1.72 s。
右侧竖卡集中列出重建耗时，左侧 12 格同机位快照。
"""
import io

# (缩略图序号, 输入帧号, 累计时刻秒, 关键帧数)
DATA = [
    (1, 36, 4.4, 8),
    (2, 40, 5.3, 9),
    (3, 44, 6.7, 11),
    (4, 48, 7.6, 12),
    (5, 52, 9.0, 14),
    (6, 60, 12.0, 18),
    (7, 64, 14.5, 22),
    (8, 68, 17.2, 26),
    (9, 72, 21.2, 32),
    (10, 76, 22.7, 34),
    (11, 80, 23.6, 35),
    (12, 84, 26.1, 81),
]

CELL = """                <Box style={{{{ flexDirection: 'column', alignItems: 'center' }}}}>
                    <Image src="prog_fix/tiles/t{idx:02d}.jpg" style={{{{ width: 156, height: 208{border} }}}} />
                    <Box style={{{{ flexDirection: 'row', alignItems: 'baseline', gap: 4, marginTop: 4 }}}}>
                        <Text style={{{{ fontSize: 11, color: '#8B97A8' }}}}>#{frame}</Text>
                        <Text style={{{{ fontSize: 15, fontWeight: 'bold', color: '{tcolor}' }}}}>{t} s</Text>
                        <Text style={{{{ fontSize: 11, color: '#8B97A8' }}}}>{kf} KF</Text>
                    </Box>
                </Box>
"""

head = """<Slide style={{ width: '1280px', height: '720px', backgroundColor: '#FFFFFF', padding: '0px', flexDirection: 'column' }}>
    <Box style={{ width: '1280px', height: '8px', backgroundColor: '#1E4FA8' }} />
    <Box style={{ padding: '14px 40px 0px 40px', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <Text style={{ fontSize: 33, fontWeight: 'bold', color: '#1E4FA8' }}>重建过程：同一机位，12 个瞬间</Text>
        <Text style={{ fontSize: 15, color: '#8B97A8' }}>14 / 16</Text>
    </Box>
    <Box style={{ padding: '6px 40px 0px 40px' }}>
        <Text style={{ fontSize: 19, color: '#4A5568', lineHeight: 1.35 }}>85 张照片全程 <b>54.5 秒</b>。12 格<b>相机位姿与内参完全固定</b>，唯一变量是“模型建到哪儿”；格下蓝字是该帧<b>处理完的时刻</b>。</Text>
    </Box>

    <Box style={{ padding: '9px 40px 0px 40px', flexDirection: 'row', gap: 16 }}>
        <Box style={{ width: 976, flexDirection: 'column', gap: 8 }}>
"""

rows = []
for r in (DATA[:6], DATA[6:]):
    cells = "".join(
        CELL.format(idx=i, frame=f, t=("%.1f" % t), kf=k,
                    border=", border: '2px solid #1E4FA8'" if i == 12 else "",
                    tcolor="#1E4FA8" if i == 12 else "#0F2E66")
        for (i, f, t, k) in r
    )
    rows.append("""            <Box style={{ flexDirection: 'row', gap: 8, justifyContent: 'space-between' }}>
%s            </Box>
""" % cells)

card = """        </Box>

        <Box style={{ width: 208, backgroundColor: '#F7F9FC', borderTop: '4px solid #1E4FA8', borderRadius: 4, padding: '12px 14px', flexDirection: 'column' }}>
            <Text style={{ fontSize: 15, fontWeight: 'bold', color: '#1E4FA8' }}>重建花了多少时间</Text>
            <Text style={{ fontSize: 12, color: '#8B97A8', marginTop: 2 }}>单帧延迟（= 下一张可拍前）</Text>
            <Box style={{ flexDirection: 'column', gap: 3, marginTop: 8 }}>
                <Box style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <Text style={{ fontSize: 13.5, color: '#4A5568' }}>中位</Text>
                    <Text style={{ fontSize: 21, fontWeight: 'bold', color: '#1E4FA8' }}>87 ms</Text>
                </Box>
                <Box style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <Text style={{ fontSize: 13.5, color: '#4A5568' }}>p90</Text>
                    <Text style={{ fontSize: 21, fontWeight: 'bold', color: '#1A2230' }}>756 ms</Text>
                </Box>
                <Box style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <Text style={{ fontSize: 13.5, color: '#4A5568' }}>最慢一帧</Text>
                    <Text style={{ fontSize: 21, fontWeight: 'bold', color: '#1A2230' }}>1.72 s</Text>
                </Box>
            </Box>
            <Box style={{ height: 1, backgroundColor: '#D6DCE5', marginTop: 11, marginBottom: 9 }} />
            <Text style={{ fontSize: 12, color: '#8B97A8' }}>整段流程（85 张）</Text>
            <Box style={{ flexDirection: 'column', gap: 4, marginTop: 7 }}>
                <Box style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <Text style={{ fontSize: 13, color: '#4A5568' }}>84 帧主体</Text>
                    <Text style={{ fontSize: 17, fontWeight: 'bold', color: '#1E4FA8' }}>26.1 s</Text>
                </Box>
                <Box style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <Text style={{ fontSize: 13, color: '#4A5568' }}>收尾重试</Text>
                    <Text style={{ fontSize: 17, fontWeight: 'bold', color: '#1A2230' }}>28.4 s</Text>
                </Box>
                <Box style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <Text style={{ fontSize: 13, color: '#4A5568' }}>全程</Text>
                    <Text style={{ fontSize: 17, fontWeight: 'bold', color: '#1A2230' }}>54.5 s</Text>
                </Box>
            </Box>
            <Box style={{ height: 1, backgroundColor: '#D6DCE5', marginTop: 10, marginBottom: 8 }} />
            <Text style={{ fontSize: 12, color: '#8B97A8' }}>与论文对照（论文 · 本次）</Text>
            <Box style={{ flexDirection: 'column', gap: 3, marginTop: 6 }}>
                <Box style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <Text style={{ fontSize: 13, color: '#4A5568' }}>平均</Text>
                    <Text style={{ fontSize: 14.5, color: '#1A2230' }}>324 ms · <b style={{ color: '#1E4FA8' }}>311 ms</b></Text>
                </Box>
                <Box style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <Text style={{ fontSize: 13, color: '#4A5568' }}>峰值</Text>
                    <Text style={{ fontSize: 14.5, color: '#1A2230' }}>386 ms · <b style={{ color: '#1E4FA8' }}>1.72 s</b></Text>
                </Box>
                <Box style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <Text style={{ fontSize: 13, color: '#4A5568' }}>全流程</Text>
                    <Text style={{ fontSize: 14.5, color: '#1A2230' }}>8 min 50 s → <b style={{ color: '#1E4FA8' }}>54.5 s</b></Text>
                </Box>
            </Box>
            <Text style={{ fontSize: 11, color: '#8B97A8', lineHeight: 1.3, marginTop: 9 }}>像素量为论文典型场景约 4 倍，故峰值更高；离线一栏为 COLMAP＋3DGS 同场景。</Text>
        </Box>
    </Box>

    <Box style={{ marginTop: 'auto', padding: '8px 40px 12px 40px', flexDirection: 'column', gap: 5 }}>
        <Box style={{ backgroundColor: '#F0F5FC', borderLeft: '5px solid #1E4FA8', padding: '8px 14px' }}>
            <Text style={{ fontSize: 17, color: '#1A2230', lineHeight: 1.38 }}>第 36 张在 <b>4.4 s</b> 就建到 8 个关键帧——画面还是噪点，但<b>哪块没拍到已经看得出来</b>；第 60 张（<b>12.0 s</b>）跨过 18 个关键帧陡然清晰；第 84 张（<b>26.1 s</b>）81 个关键帧、噪点收净。之后 28 秒只<b>重试搁置帧</b>，不影响边拍边看。</Text>
        </Box>
        <Text style={{ fontSize: 12, color: '#8B97A8', lineHeight: 1.4 }}>逐帧打点实测（纯净运行，无写盘开销）：84 帧主体 26.1 s；单帧中位 87 ms、p90 756 ms、最大 1.72 s。图像取自 --save_every 4 快照轮，两轮关键帧序列一致（8 → 81）。</Text>
    </Box>
</Slide>
"""

out = head + "\n".join(rows) + card
path = "slides_v2/14.slide"
io.open(path, "w", encoding="utf-8", newline="\n").write(out)
print("wrote", path, len(out.encode()), "bytes,", out.count("\n"), "lines")
