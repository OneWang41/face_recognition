# 综艺嘉宾出镜秒数统计

基于 InsightFace（`buffalo_l`：SCRFD 检测 + ArcFace 特征）的人脸比对方案。
**无需训练、无需标注**，只用现成的嘉宾照片建底库即可。

## 1. 安装

```bash
pip install -r requirements.txt
# 有 NVIDIA GPU：把 requirements 里的 onnxruntime 换成 onnxruntime-gpu，运行时加 --gpu
```

首次运行会自动下载 `buffalo_l` 模型（约 300MB，需要联网一次），之后可离线运行。

## 2. 准备目录

```
gallery/                videos/
  张三/                   ep01.mp4
    1.jpg                 ep02.mp4
    2.jpg                 ...
  李四/                   ep10.mp4
    1.jpg
  ...（共 33 人）
```

- 每位嘉宾 1~5 张清晰正脸照即可，尽量覆盖不同妆造/光照。
- 参考图最好单人；若一张图里多张脸，脚本会取最大那张并告警。
- `output/gallery_report.txt` 会列出每人成功提取了几张参考脸，先检查它再跑全片。

## 3. 运行

```bash
python screen_time.py --gallery ./gallery --videos ./videos --out ./output --gpu
```

输出：
- `output/ep01.csv … ep10.csv`：每期每位嘉宾的 `sampled_frames` 与 `seconds`
- `output/summary.csv`：嘉宾 × 各期 的秒数矩阵 + 总秒数/分钟
- `output/crops/`：每位嘉宾的若干命中截图（用于人工抽检，`--save-crops 0` 关闭）

## 4. 秒数怎么来的

按固定间隔抽帧，每个抽样帧代表 `dt = step / 视频FPS` 秒。某嘉宾出现的抽样帧数
× `dt` 即为出镜秒数。开启跟踪后，短暂侧脸/遮挡造成的漏检会被同一条轨迹续接，
减少低估。

## 5. 调参（决定精确率，务必做）

先剪 1~2 段已知答案的短片标定，再跑全片：

| 参数 | 作用 | 建议 |
|------|------|------|
| `--sim-threshold` | 判定为某嘉宾的最低余弦相似度 | 默认 0.40；偏松→张冠李戴，偏紧→漏判。33 人较多，建议在 0.40~0.48 间试 |
| `--margin-threshold` | top1 须比 top2 身份高出的差值，降低相似脸混淆 | 默认 0.04，可升到 0.06~0.08 |
| `--sample-fps` | 每秒分析几帧 | 默认 6 |
| `--min-det-score` / `--min-face-px` | 过滤低质/过小人脸 | 噪声多时调高 |
| `--max-age` | 轨迹容忍的最长漏检步数（也用于 gap 续接） | 默认 12（≈2 秒@6fps） |
| `--no-track`（`--use-track false`） | 关闭跟踪，纯逐帧比对 | 想要最保守、不续接时用 |
| `--bridge-gaps false` | 不把续接的空隙计入出镜 | 想要绝对保守时用 |

调参方法：用截图目录 `crops/` 和某段已知答案的 csv 反复比对，找到误判（把 A 认成 B）
和漏判都可接受的阈值组合。

## 6. 注意

- 统计的是「脸可见」时长。纯背身/后脑勺、严重侧脸人脸识别无能为力，跟踪只能续接
  短暂缺失；若需要把「人在画面但背对镜头」也算出镜，需另加行人重识别（ReID）。
- `UNKNOWN`（未达阈值的脸）默认不计入任何嘉宾；如某期某人秒数异常偏低，多半是
  阈值太紧或参考图不够，先看 `crops/` 抽检。
