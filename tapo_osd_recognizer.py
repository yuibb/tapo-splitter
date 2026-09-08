"""OSD glyph extraction and digit matching used by the splitter."""

import cv2
import numpy as np

from tapo_profile import geometry_for_profile


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


def components_for_threshold(gray, threshold, profile=None):
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    geometry = geometry_for_profile(profile)
    slots = geometry["slot_ranges"]
    left_bound = min(left for left, _right in slots)
    right_bound = max(right for _left, right in slots)
    items = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if left_bound <= x and x + w <= right_bound and 35 <= h <= 60 \
                and 5 <= w <= 35 and area >= 120:
            items.append((int(x), int(y), int(w), int(h), int(area)))
    return mask, sorted(items)


def extract_glyphs_from_gray(gray, source_name="image", profile=None):
    candidates = []
    for threshold in THRESHOLDS:
        mask, comps = components_for_threshold(gray, threshold, profile)
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
