#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variety-show guest screen-time counter (with annotated video + interval table).

Pipeline:
  1. Build a face gallery from reference photos (one sub-folder per guest).
  2. For each episode video, sample frames at a fixed cadence (default 6 fps),
     detect + align faces, extract 512-d ArcFace embeddings.
  3. Match each face against the gallery (cosine similarity + top1/top2 margin).
  4. A lightweight IoU+embedding tracker stabilises identity and bridges short
     occlusion / side-face gaps.
  5. Accumulate per-guest screen time; write per-episode + summary CSVs.
  6. Draw boxes + Chinese names on each sampled frame and mux them into an
     annotated review video; write a per-guest appearance-interval table
     (start/end timestamps of every continuous on-screen segment) for spot-check.

No training / labelling is required: recognition is pure embedding comparison
against the reference gallery.

Usage:
  python screen_time.py --gallery ./gallery --videos ./videos --out ./output --gpu
  # quick test on the first 2 minutes of each episode first:
  python screen_time.py --gallery ./gallery --videos ./videos --out ./output --max-seconds 120
"""

import argparse
import csv
import os
import sys
import glob
from collections import defaultdict

import numpy as np
import cv2

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional
    def tqdm(it, **kw):
        return it

# InsightFace is imported lazily inside build_app() so that --help works
# without the heavy dependency installed.


# --------------------------------------------------------------------------- #
# Configuration defaults (override via CLI)
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    sample_fps=6.0,        # frames analysed per second of video
    det_size=640,          # detector input size (square)
    min_det_score=0.55,    # discard low-confidence detections
    min_face_px=40,        # discard faces smaller than this (shorter side)
    sim_threshold=0.40,    # cosine sim to accept a gallery match (TUNE THIS)
    margin_threshold=0.04, # top1 must beat top2 identity by this much
    # tracker
    assoc_threshold=0.35,  # min combined score to link a det to a track
    w_iou=0.4,             # weight of IoU in association cost
    w_emb=0.6,             # weight of embedding sim in association cost
    max_age=12,            # sampled steps a track survives without a match
    bridge_gaps=False,     # count bridged gaps (<= max_age) as on-screen
    use_track=True,        # set False for plain per-frame matching
    save_crops=8,          # save up to N example crops per guest (0 = off)
    # annotated review video
    annotate=True,         # write an annotated review video per episode
    annotate_scale=0.5,    # output video scale (0.5 keeps file size manageable)
    show_unknown=True,     # also draw boxes for UNKNOWN faces (grey)
    max_seconds=0.0,       # process only first N seconds per episode (0 = all)
)

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".m4v", ".wmv")

# label colours, RGB (PIL space)
COLOR_OK = (40, 200, 80)
COLOR_UNK = (170, 170, 170)


# --------------------------------------------------------------------------- #
# Unicode-safe image I/O (cv2.imread/imwrite fail on non-ASCII paths on Windows)
# --------------------------------------------------------------------------- #
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path, img):
    try:
        ext = os.path.splitext(path)[1] or ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# CJK-capable text drawing (cv2.putText cannot render Chinese)
# --------------------------------------------------------------------------- #
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",  # macOS
]


def load_font(font_path, size):
    from PIL import ImageFont
    paths = ([font_path] if font_path else []) + _FONT_CANDIDATES
    for p in paths:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    print("[annotate] WARNING: no CJK font found; Chinese names will not render. "
          "Pass --font C:\\Windows\\Fonts\\msyh.ttc")
    from PIL import ImageFont as _IF
    return _IF.load_default()


def annotate_frame(frame_bgr, items, font):
    """items: list of (bbox(x1,y1,x2,y2), label_str, color_rgb). Returns BGR frame."""
    if not items:
        return frame_bgr
    from PIL import Image, ImageDraw
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), label, color in items:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = 8 * len(label), 16
        ty = y1 - th - 4
        if ty < 0:
            ty = y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_app(det_size, use_gpu):
    from insightface.app import FaceAnalysis
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if use_gpu else ["CPUExecutionProvider"])
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))
    return app


def detect_faces(app, frame_bgr, min_det_score, min_face_px):
    out = []
    for f in app.get(frame_bgr):
        if float(getattr(f, "det_score", 1.0)) < min_det_score:
            continue
        x1, y1, x2, y2 = f.bbox.astype(int)
        if min(x2 - x1, y2 - y1) < min_face_px:
            continue
        emb = f.normed_embedding.astype(np.float32)
        out.append(((int(x1), int(y1), int(x2), int(y2)), emb))
    return out


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #
def build_gallery(app, gallery_dir, cfg, out_dir):
    names, embs, report = [], [], []
    people = sorted(d for d in os.listdir(gallery_dir)
                    if os.path.isdir(os.path.join(gallery_dir, d)))
    if not people:
        sys.exit(f"[gallery] no guest sub-folders found in {gallery_dir}")

    for name in people:
        pdir = os.path.join(gallery_dir, name)
        imgs = [p for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
                for p in glob.glob(os.path.join(pdir, ext))]
        n_face = 0
        for ip in sorted(imgs):
            img = imread_unicode(ip)
            if img is None:
                report.append(f"  ! unreadable image: {ip}")
                continue
            faces = detect_faces(app, img, cfg["min_det_score"], cfg["min_face_px"])
            if len(faces) == 0:
                report.append(f"  ! no face detected: {ip}")
                continue
            if len(faces) > 1:
                faces.sort(key=lambda fb: (fb[0][2] - fb[0][0]) * (fb[0][3] - fb[0][1]),
                           reverse=True)
                report.append(f"  ~ {len(faces)} faces in {ip}; kept largest")
            names.append(name)
            embs.append(faces[0][1])
            n_face += 1
        if n_face == 0:
            report.append(f"  !! guest '{name}' has ZERO usable reference faces")
        else:
            report.append(f"  ok  {name}: {n_face} reference face(s)")

    if not embs:
        sys.exit("[gallery] no usable reference faces at all; aborting.")

    mat = np.vstack(embs).astype(np.float32)
    row_names = np.array(names)
    id_names = sorted(set(names))
    id_rows = [np.where(row_names == nm)[0] for nm in id_names]

    rep_path = os.path.join(out_dir, "gallery_report.txt")
    with open(rep_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print(f"[gallery] {len(id_names)} guests, {mat.shape[0]} reference faces "
          f"-> report at {rep_path}")
    return mat, row_names, id_names, id_rows


def match_identity(emb, mat, id_names, id_rows, sim_th, margin_th):
    sims = mat @ emb
    id_scores = np.array([sims[rows].max() for rows in id_rows], dtype=np.float32)
    order = np.argsort(id_scores)[::-1]
    top1 = float(id_scores[order[0]])
    top2 = float(id_scores[order[1]]) if len(order) > 1 else -1.0
    if top1 >= sim_th and (top1 - top2) >= margin_th:
        return id_names[order[0]], top1
    return "UNKNOWN", top1


# --------------------------------------------------------------------------- #
# Lightweight tracker
# --------------------------------------------------------------------------- #
def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


class Track:
    __slots__ = ("id", "bbox", "mean_emb", "last_step", "votes", "steps")

    def __init__(self, tid, bbox, emb, step):
        self.id = tid
        self.bbox = bbox
        self.mean_emb = emb.copy()
        self.last_step = step
        self.votes = defaultdict(float)
        self.steps = set()
        self.steps.add(step)

    def update(self, bbox, emb, step):
        self.bbox = bbox
        m = 0.7 * self.mean_emb + 0.3 * emb
        n = np.linalg.norm(m)
        self.mean_emb = m / n if n > 0 else emb
        self.last_step = step
        self.steps.add(step)

    def vote(self, name, sim):
        if name != "UNKNOWN":
            self.votes[name] += sim

    def identity(self):
        if not self.votes:
            return "UNKNOWN"
        return max(self.votes.items(), key=lambda kv: kv[1])[0]


def associate(tracks, dets, cfg):
    pairs = []
    for ti, tr in enumerate(tracks):
        for di, (bbox, emb) in enumerate(dets):
            score = cfg["w_iou"] * iou(tr.bbox, bbox) + cfg["w_emb"] * float(tr.mean_emb @ emb)
            if score >= cfg["assoc_threshold"]:
                pairs.append((score, ti, di))
    pairs.sort(reverse=True)
    used_t, used_d, matches = set(), set(), []
    for score, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((ti, di))
    unmatched = [di for di in range(len(dets)) if di not in used_d]
    return matches, unmatched


# --------------------------------------------------------------------------- #
# Time / interval helpers
# --------------------------------------------------------------------------- #
def fmt_ts(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def steps_to_intervals(steps, dt):
    """Group sorted sampled-step indices into continuous runs -> time intervals.
    Returns list of (start_sec, end_sec, duration_sec)."""
    if not steps:
        return []
    ss = sorted(steps)
    runs, a, prev = [], ss[0], ss[0]
    for x in ss[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((a, prev))
            a = prev = x
    runs.append((a, prev))
    out = []
    for s0, s1 in runs:
        start = s0 * dt
        end = (s1 + 1) * dt  # run covers steps [s0 .. s1] inclusive
        out.append((start, end, end - start))
    return out


def _collect_draw(draw_items, bbox, name, sim, cfg):
    if name == "UNKNOWN":
        if not cfg["show_unknown"]:
            return
        draw_items.append((bbox, f"? {sim:.2f}", COLOR_UNK))
    else:
        draw_items.append((bbox, f"{name} {sim:.2f}", COLOR_OK))


def _maybe_save_crop(frame, bbox, name, crop_dir, crop_count, cfg):
    if cfg["save_crops"] <= 0:
        return
    if crop_count[name] >= cfg["save_crops"]:
        return
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = os.path.join(crop_dir, name)
    os.makedirs(sub, exist_ok=True)
    imwrite_unicode(os.path.join(sub, f"{crop_count[name]:02d}.jpg"), frame[y1:y2, x1:x2])
    crop_count[name] += 1


# --------------------------------------------------------------------------- #
# Per-episode processing
# --------------------------------------------------------------------------- #
def process_video(app, video_path, gallery, cfg, out_dir, ep_tag, font):
    mat, _row_names, id_names, id_rows = gallery
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[video] cannot open {video_path}; skipped")
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if video_fps <= 1e-3:
        video_fps = 25.0
        print(f"[video] FPS unreadable, assuming {video_fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    step = max(1, int(round(video_fps / cfg["sample_fps"])))
    dt = step / video_fps  # seconds represented by one sampled frame
    max_frames = int(cfg["max_seconds"] * video_fps) if cfg["max_seconds"] > 0 else 0

    # annotated review video writer
    writer = None
    anno_path = None
    out_w = max(2, int(round(src_w * cfg["annotate_scale"])) // 2 * 2)
    out_h = max(2, int(round(src_h * cfg["annotate_scale"])) // 2 * 2)
    if cfg["annotate"]:
        anno_dir = os.path.join(out_dir, "annotated")
        os.makedirs(anno_dir, exist_ok=True)
        anno_path = os.path.join(anno_dir, f"{ep_tag}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(anno_path, fourcc, cfg["sample_fps"], (out_w, out_h))
        if not writer.isOpened():
            print(f"[annotate] WARNING: cannot open VideoWriter for {anno_path}; "
                  f"annotation disabled for this episode")
            writer = None

    crop_dir = os.path.join(out_dir, "crops", ep_tag)
    crop_count = defaultdict(int)
    if cfg["save_crops"] > 0:
        os.makedirs(crop_dir, exist_ok=True)

    active, finished, next_tid = [], [], 0
    direct_steps = defaultdict(set)

    frame_idx, sampled = 0, 0
    pbar_total = (min(total_frames, max_frames) if max_frames else total_frames) or None
    pbar = tqdm(total=pbar_total, desc=f"{ep_tag}", unit="f")
    while True:
        ok = cap.grab()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break
        if frame_idx % step == 0:
            ok2, frame = cap.retrieve()
            if not ok2:
                break
            dets = detect_faces(app, frame, cfg["min_det_score"], cfg["min_face_px"])
            draw_items = []

            if cfg["use_track"]:
                matches, unmatched = associate(active, dets, cfg)
                for ti, di in matches:
                    bbox, emb = dets[di]
                    tr = active[ti]
                    tr.update(bbox, emb, sampled)
                    name, sim = match_identity(emb, mat, id_names, id_rows,
                                               cfg["sim_threshold"], cfg["margin_threshold"])
                    tr.vote(name, sim)
                    _collect_draw(draw_items, bbox, name, sim, cfg)
                    _maybe_save_crop(frame, bbox, name, crop_dir, crop_count, cfg)
                for di in unmatched:
                    bbox, emb = dets[di]
                    tr = Track(next_tid, bbox, emb, sampled)
                    next_tid += 1
                    name, sim = match_identity(emb, mat, id_names, id_rows,
                                               cfg["sim_threshold"], cfg["margin_threshold"])
                    tr.vote(name, sim)
                    active.append(tr)
                    _collect_draw(draw_items, bbox, name, sim, cfg)
                    _maybe_save_crop(frame, bbox, name, crop_dir, crop_count, cfg)
                still = []
                for tr in active:
                    if sampled - tr.last_step > cfg["max_age"]:
                        finished.append(tr)
                    else:
                        still.append(tr)
                active = still
            else:
                for bbox, emb in dets:
                    name, sim = match_identity(emb, mat, id_names, id_rows,
                                               cfg["sim_threshold"], cfg["margin_threshold"])
                    if name != "UNKNOWN":
                        direct_steps[name].add(sampled)
                    _collect_draw(draw_items, bbox, name, sim, cfg)
                    _maybe_save_crop(frame, bbox, name, crop_dir, crop_count, cfg)

            # ---- write annotated frame ----
            if writer is not None:
                out_frame = annotate_frame(frame, draw_items, font)
                ts = fmt_ts(sampled * dt)  # ASCII -> safe with cv2.putText
                cv2.putText(out_frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(out_frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 1, cv2.LINE_AA)
                if (out_frame.shape[1], out_frame.shape[0]) != (out_w, out_h):
                    out_frame = cv2.resize(out_frame, (out_w, out_h))
                writer.write(out_frame)

            sampled += 1
        frame_idx += 1
        if pbar is not None and hasattr(pbar, "update"):
            pbar.update(1)
    if hasattr(pbar, "close"):
        pbar.close()
    if writer is not None:
        writer.release()
        print(f"[annotate] {ep_tag}: review video -> {anno_path}")
    cap.release()

    # ---- accumulate per-guest step sets ----
    guest_steps = defaultdict(set)
    if cfg["use_track"]:
        finished.extend(active)
        for tr in finished:
            name = tr.identity()
            if name == "UNKNOWN":
                continue
            steps = sorted(tr.steps)
            if cfg["bridge_gaps"] and len(steps) > 1:
                filled = set(steps)
                for a, b in zip(steps[:-1], steps[1:]):
                    if b - a <= cfg["max_age"]:
                        filled.update(range(a + 1, b))
                steps = filled
            guest_steps[name].update(steps)
    else:
        guest_steps = direct_steps

    # ---- per-episode totals CSV ----
    rows = []
    for name in id_names:
        n = len(guest_steps.get(name, ()))
        rows.append((name, n, round(n * dt, 1)))
    rows.sort(key=lambda r: r[2], reverse=True)
    ep_csv = os.path.join(out_dir, f"{ep_tag}.csv")
    with open(ep_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["guest", "sampled_frames", "seconds"])
        w.writerows(rows)

    # ---- per-episode interval table (for spot-check) ----
    iv_csv = os.path.join(out_dir, f"{ep_tag}_intervals.csv")
    iv_rows = []
    for name in id_names:
        for k, (s0, s1, dur) in enumerate(steps_to_intervals(guest_steps.get(name, set()), dt), 1):
            iv_rows.append((name, k, fmt_ts(s0), fmt_ts(s1), round(dur, 1),
                            round(s0, 2), round(s1, 2)))
    iv_rows.sort(key=lambda r: r[5])  # chronological
    with open(iv_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["guest", "segment", "start_hms", "end_hms",
                    "duration_s", "start_sec", "end_sec"])
        w.writerows(iv_rows)

    print(f"[video] {ep_tag}: dt={dt:.3f}s/frame, sampled {sampled} frames "
          f"-> {ep_csv} ; intervals -> {iv_csv}")
    return {name: round(len(guest_steps.get(name, ())) * dt, 1) for name in id_names}, iv_rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Variety-show guest screen-time counter")
    ap.add_argument("--gallery", required=True, help="dir with one sub-folder per guest")
    ap.add_argument("--videos", required=True, help="dir containing episode videos")
    ap.add_argument("--out", default="./output", help="output dir")
    ap.add_argument("--gpu", action="store_true", help="use CUDA if available")
    ap.add_argument("--font", default="", help="path to a CJK .ttf/.ttc for name labels")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", dest=k,
                            type=lambda s: s.lower() in ("1", "true", "yes", "y"),
                            default=v)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=type(v), default=v)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}

    os.makedirs(args.out, exist_ok=True)
    print("[init] loading model (first run downloads buffalo_l)...")
    app = build_app(cfg["det_size"], args.gpu)
    gallery = build_gallery(app, args.gallery, cfg, args.out)
    id_names = gallery[2]

    font = load_font(args.font, 22) if cfg["annotate"] else None

    videos = sorted(p for p in glob.glob(os.path.join(args.videos, "*"))
                    if os.path.splitext(p)[1].lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"[videos] no video files found in {args.videos}")
    print(f"[videos] found {len(videos)} episode(s)")

    summary = {}
    all_intervals = []
    for vp in videos:
        ep_tag = os.path.splitext(os.path.basename(vp))[0]
        res = process_video(app, vp, gallery, cfg, args.out, ep_tag, font)
        if res is not None:
            sec_map, iv_rows = res
            summary[ep_tag] = sec_map
            for r in iv_rows:
                all_intervals.append([ep_tag] + list(r))

    # ---- master summary matrix ----
    ep_tags = list(summary.keys())
    sum_csv = os.path.join(args.out, "summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["guest"] + ep_tags + ["total_seconds", "total_minutes"])
        for name in id_names:
            secs = [summary[t].get(name, 0.0) for t in ep_tags]
            tot = round(sum(secs), 1)
            w.writerow([name] + secs + [tot, round(tot / 60, 2)])

    # ---- combined interval table across all episodes ----
    comb_csv = os.path.join(args.out, "intervals_all.csv")
    with open(comb_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "guest", "segment", "start_hms", "end_hms",
                    "duration_s", "start_sec", "end_sec"])
        w.writerows(all_intervals)

    print(f"[done] summary -> {sum_csv}")
    print(f"[done] combined intervals -> {comb_csv}")


if __name__ == "__main__":
    main()
