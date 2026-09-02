#!/usr/bin/env python3
"""Interactively build tapo_osd_glyph_templates.json from user recordings.

No external font file is embedded: labels come from the OSD string confirmed
by the user for each sampled frame.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


OUTPUT_NAME = "tapo_osd_glyph_templates.json"
TARGET_PER_DIGIT = 20
ROUND_SIZE = 10
MAX_ROUNDS_WITHOUT_PROGRESS = 3
THRESHOLDS = (225, 235, 240, 245, 250)


def extract_glyphs(gray):
    candidates = []
    for threshold in THRESHOLDS:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        comps = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 7 <= x and x + w <= 925 and 35 <= h <= 60 and 5 <= w <= 35 and area >= 120:
                comps.append((int(x), int(y), int(w), int(h), int(area)))
        if len(comps) == 14:
            candidates.append((float(np.std([c[3] for c in comps])), threshold, mask, sorted(comps)))
    if not candidates:
        return None
    _, threshold, mask, comps = min(candidates, key=lambda x: x[0])
    output = []
    for x, y, w, h, _ in comps:
        piece = mask[max(0, y - 2):min(70, y + h + 2), max(0, x - 2):min(950, x + w + 2)]
        ys, xs = np.where(piece > 0)
        if len(xs) == 0:
            return None
        glyph = piece[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        scale = min(56 / glyph.shape[0], 36 / glyph.shape[1])
        nw, nh = max(1, round(glyph.shape[1] * scale)), max(1, round(glyph.shape[0] * scale))
        glyph = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((64, 40), dtype=np.uint8)
        canvas[(64 - nh) // 2:(64 - nh) // 2 + nh, (40 - nw) // 2:(40 - nw) // 2 + nw] = glyph > 0
        output.append(canvas)
    return output, threshold


def choose_video_frame(video, seconds):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = cap.read()
    actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cap.release()
    return (frame, actual) if ok else (None, actual)


def write_json(path, labels, templates, target=TARGET_PER_DIGIT):
    counts = {d: len(templates[d]) for d in "0123456789"}
    data = {
        "format": "tapo_osd_glyph_templates_v1",
        "label_source": "user-confirmed OSD strings from personal recordings",
        "canvas": [64, 40],
        "target_samples_per_digit": target,
        "labels": labels,
        "digits": {
            d: {
                "sample_count": counts[d],
                "templates": [item["matrix"] for item in templates[d]],
                "template_sources": [item["source"] for item in templates[d]],
            }
            for d in "0123456789"
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        help="録画データフォルダ（省略時は対話入力）")
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_NAME))
    parser.add_argument("--round-size", type=int, default=ROUND_SIZE)
    parser.add_argument("--samples-per-digit", type=int, default=TARGET_PER_DIGIT,
                        help="exact number of templates to keep for every digit (default: 20)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore an existing output JSON and build a new one")
    args = parser.parse_args()
    if args.samples_per_digit < 1:
        raise SystemExit("--samples-per-digit must be at least 1")
    target = args.samples_per_digit

    if args.output.exists() and not args.fresh:
        answer = input(f"{args.output} は既に存在します。再作成・積み増ししますか？ [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("中止しました。既存JSONは変更していません。")
            return
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            labels = existing.get("labels", {})
            templates = defaultdict(list)
            for d in "0123456789":
                bucket = existing.get("digits", {}).get(d, {})
                matrices = bucket.get("templates", [])
                sources = bucket.get("template_sources", [])
                items = [
                    {"source": sources[i] if i < len(sources) else "existing",
                     "method": "existing", "matrix": matrix}
                    for i, matrix in enumerate(matrices)
                ]
                if len(items) > target:
                    indices = [round(i * (len(items) - 1) / (target - 1))
                               for i in range(target)] if target > 1 else [0]
                    items = [items[i] for i in indices]
                templates[d].extend(items)
        except Exception as exc:
            raise SystemExit(f"既存JSONを読めません: {exc}")
    else:
        labels, templates = {}, defaultdict(list)

    folder = (args.input_dir.expanduser() if args.input_dir else
              Path(input("録画データフォルダのパス: ").strip()).expanduser())
    videos = sorted(folder.rglob("*.mp4"))
    if not videos:
        raise SystemExit("MP4が見つかりません。録画フォルダを確認してください。")
    sample_dir = args.output.parent / "tapo_training_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random()
    no_progress = 0

    while True:
        counts = Counter({d: len(templates[d]) for d in "0123456789"})
        missing = [d for d in "0123456789" if counts[d] < target]
        if not missing:
            counts = {d: len(templates[d]) for d in "0123456789"}
            if any(count != target for count in counts.values()):
                raise SystemExit(f"数字別サンプル数が不正です: {counts}")
            write_json(args.output, labels, templates, target)
            print(f"完成: {args.output}")
            print("数字別サンプル数:", dict(counts))
            return

        print(f"不足数字: {', '.join(missing)}。ランダムに {args.round_size} 枚抽出します。")
        progress = 0
        for index in range(args.round_size):
            video = rng.choice(videos)
            cap = cv2.VideoCapture(str(video))
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1.0)
            cap.release()
            frame, seconds = choose_video_frame(video, rng.uniform(0, max(0.1, duration - 0.1)))
            if frame is None:
                continue
            strip = frame[:70, :950]
            sample_path = sample_dir / f"sample_{len(labels):05d}_{video.stem}_{seconds:.1f}s.png"
            cv2.imwrite(str(sample_path), strip)
            print(f"\n[{index + 1}/{args.round_size}] {sample_path}")
            answer = input("OSD正解文字列 (YYYYMMDDHHMMSS / Enter=skip / q=abort): ").strip()
            if answer.lower() == "q":
                write_json(args.output, labels, templates, target)
                raise SystemExit("ユーザー指定でabortしました。途中結果は保存済みです。")
            if len(answer) != 14 or not answer.isdigit():
                continue
            result = extract_glyphs(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY))
            if result is None:
                print("14桁のOSDを切り出せません。スキップします。")
                continue
            glyphs, threshold = result
            key = sample_path.name
            labels[key] = {"osd": answer, "video": str(video), "seconds": seconds, "threshold": threshold}
            for digit, glyph in zip(answer, glyphs):
                # Keep the final font balanced.  A labeled frame contains
                # all 14 digits, so stop adding a digit once its quota is met
                # instead of letting frequently occurring digits dominate.
                if len(templates[digit]) >= target:
                    continue
                templates[digit].append({"source": key, "method": f"binary_{threshold}", "matrix": glyph.tolist()})
            progress += 1
            write_json(args.output, labels, templates, target)

        if progress == 0:
            no_progress += 1
            if no_progress >= MAX_ROUNDS_WITHOUT_PROGRESS:
                write_json(args.output, labels, templates, target)
                raise SystemExit("学習データを積み増せません。OSD表示・録画データを確認してください。")
        else:
            no_progress = 0


if __name__ == "__main__":
    main()
