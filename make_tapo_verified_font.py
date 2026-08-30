#!/usr/bin/env python3
"""Build a verified Tapo OSD digit font and evaluate it on held-out samples.

The labels in tapo_verified_labels.json are transcribed from the pixels in the
OSD, not inferred from the screenshot filename.  Existing tapo_font.json is
never read or overwritten.
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
SAMPLES = BASE / "samples_other"
LABELS_PATH = BASE / "tapo_verified_labels.json"
FONT_PATH = BASE / "tapo_font_verified.json"
REPORT_PATH = BASE / "tapo_font_verified_report.json"

# One sample from each recording group is held out.  These must not enter FONT.
TEST_FILES = {
    "20250818_214826_tp00000_1267.8s.png",
    "20250929_190150_tp00087_1432.0s.png",
    "20260217_143652_tp00043_224.0s.png",
    "20260717_021030_tp00381_1188.5s.png",
    "20260717_021030_tp00381_296.6s.png",
}

CANVAS_H, CANVAS_W = 64, 40
THRESHOLDS = (225, 235, 240, 245, 250)


def normalize(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    glyph = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = glyph.shape
    if h < 8 or w < 2:
        return None
    scale = min(56 / h, 36 / w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    glyph = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    ox, oy = (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = glyph > 0
    return canvas


def components_for_threshold(gray, threshold):
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    items = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # Tapo digits in the 950x70 OSD strip have these approximate bounds.
        if 7 <= x and x + w <= 925 and 35 <= h <= 60 and 5 <= w <= 35 and area >= 120:
            items.append((int(x), int(y), int(w), int(h), int(area)))
    return mask, sorted(items)


def extract_glyphs_from_gray(gray, source_name="image"):

    candidates = []
    for threshold in THRESHOLDS:
        mask, comps = components_for_threshold(gray, threshold)
        # A valid OSD row has 14 digit components; hyphens/colons are rejected
        # by the height filter above.
        if len(comps) != 14:
            continue
        heights = np.array([c[3] for c in comps])
        score = float(np.std(heights))
        candidates.append((score, threshold, mask, comps))
    if not candidates:
        raise RuntimeError(f"14 digit components not found in {source_name}")

    _, threshold, mask, comps = min(candidates, key=lambda row: row[0])
    glyphs = []
    for x, y, w, h, _area in comps:
        pad = 2
        piece = mask[max(0, y - pad):min(mask.shape[0], y + h + pad),
                     max(0, x - pad):min(mask.shape[1], x + w + pad)]
        glyph = normalize(piece)
        if glyph is None:
            raise RuntimeError(f"empty digit component in {source_name}")
        glyphs.append(glyph)
    return glyphs, threshold


def extract_glyphs(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"cannot read {image_path}")
    return extract_glyphs_from_gray(gray, image_path.name)


def shifted_hamming(a, b):
    best = int(np.count_nonzero(a != b))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            shifted = np.zeros_like(a)
            y0, y1 = max(0, dy), min(CANVAS_H, CANVAS_H + dy)
            x0, x1 = max(0, dx), min(CANVAS_W, CANVAS_W + dx)
            shifted[y0:y1, x0:x1] = a[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            best = min(best, int(np.count_nonzero(shifted != b)))
    return best


def recognize(glyph, templates):
    scores = []
    for digit, values in templates.items():
        distances = sorted(shifted_hamming(glyph, t) for t in values)
        scores.append((distances[0], digit))
    scores.sort()
    best, digit = scores[0]
    margin = scores[1][0] - best if len(scores) > 1 else None
    return digit, best, margin


def main():
    labels = json.loads(LABELS_PATH.read_text())
    train = sorted(set(labels) - TEST_FILES)
    test = sorted(TEST_FILES)
    if len(train) != 15 or len(test) != 5 or not set(test) <= set(labels):
        raise RuntimeError("invalid train/test split")

    templates = defaultdict(list)
    train_methods = {}
    for name in train:
        glyphs, threshold = extract_glyphs(SAMPLES / name)
        if len(glyphs) != len(labels[name]):
            raise RuntimeError(f"digit count mismatch in {name}")
        train_methods[name] = threshold
        for digit, glyph in zip(labels[name], glyphs):
            templates[digit].append(glyph)
    missing = [d for d in "0123456789" if not templates[d]]
    if missing:
        raise RuntimeError(f"training set is missing digits: {missing}")

    font = {
        "format": "tapo_verified_bitmap_font_v1",
        "canvas": [CANVAS_H, CANVAS_W],
        "label_source": "manual transcription of visible OSD pixels",
        "training_files": train,
        "excluded_test_files": test,
        "digits": {d: {"count": len(templates[d]), "templates": [x.tolist() for x in templates[d]]}
                   for d in "0123456789"},
    }
    FONT_PATH.write_text(json.dumps(font, ensure_ascii=False), encoding="utf-8")

    results = []
    correct_chars = total_chars = exact_matches = 0
    for name in test:
        glyphs, threshold = extract_glyphs(SAMPLES / name)
        predicted, detail = [], []
        for glyph in glyphs:
            digit, distance, margin = recognize(glyph, templates)
            predicted.append(digit)
            detail.append({"digit": digit, "distance": distance, "margin": margin})
        predicted = "".join(predicted)
        expected = labels[name]
        correct = sum(a == b for a, b in zip(predicted, expected))
        correct_chars += correct
        total_chars += len(expected)
        exact_matches += predicted == expected
        results.append({"file": name, "expected": expected, "predicted": predicted,
                        "correct_characters": correct, "total_characters": len(expected),
                        "character_accuracy": correct / len(expected),
                        "threshold": threshold, "per_digit": detail})

    report = {
        "method": "held-out evaluation: 15 labelled images train, 5 labelled images test",
        "labels": str(LABELS_PATH), "font": str(FONT_PATH),
        "training_files": train, "test_files": test, "training_thresholds": train_methods,
        "character_accuracy": correct_chars / total_chars,
        "correct_characters": correct_chars, "total_characters": total_chars,
        "exact_timestamp_accuracy": exact_matches / len(test),
        "exact_timestamp_matches": exact_matches, "test_image_count": len(test),
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"font: {FONT_PATH}")
    print(f"report: {REPORT_PATH}")
    for row in results:
        print(f"{row['file']}: {row['predicted']}  expected={row['expected']}  "
              f"{row['correct_characters']}/{row['total_characters']}")
    print(f"character accuracy: {report['character_accuracy']:.2%} "
          f"({correct_chars}/{total_chars})")
    print(f"exact timestamp accuracy: {report['exact_timestamp_accuracy']:.2%} "
          f"({exact_matches}/{len(test)})")


if __name__ == "__main__":
    main()


