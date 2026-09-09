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


_TEMPLATE_CACHE = {}


def _shifted_variants(glyph):
    """Build the nine ±1-pixel variants once for one glyph."""
    variants = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.zeros_like(glyph)
            y0, y1 = max(0, dy), min(CANVAS_H, CANVAS_H + dy)
            x0, x1 = max(0, dx), min(CANVAS_W, CANVAS_W + dx)
            shifted[y0:y1, x0:x1] = glyph[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            variants.append(shifted)
    return np.stack(variants)


def _prepared_templates(templates):
    """Cache stacked templates while keeping legacy dict input compatible."""
    key = id(templates)
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None and cached[0] is templates:
        return cached[1:]

    digits = tuple(sorted(templates))
    stacks = tuple(np.stack(templates[digit]) for digit in digits)
    counts = tuple(len(stack) for stack in stacks)
    equal_counts = len(set(counts)) == 1
    bank = np.stack(stacks) if equal_counts else None
    prepared = (templates, digits, stacks, bank)
    _TEMPLATE_CACHE[key] = prepared
    return prepared[1:]


def recognize(glyph, templates):
    digits, stacks, bank = _prepared_templates(templates)
    shifted = _shifted_variants(glyph)

    if bank is not None:
        # Shape: shifts × digits × templates × pixels.
        distances = np.count_nonzero(
            shifted[:, None, None, :, :] != bank[None, :, :, :, :],
            axis=(3, 4),
        )
        best_by_digit = distances.min(axis=(0, 2))
        scores = [(int(best), digit) for best, digit in zip(best_by_digit, digits)]
    else:
        # Keep compatibility with legacy JSON files with uneven template counts.
        scores = []
        for digit, stack in zip(digits, stacks):
            distances = np.count_nonzero(
                shifted[:, None, :, :] != stack[None, :, :, :],
                axis=(2, 3),
            )
            scores.append((int(distances.min()), digit))
    scores.sort()
    best, digit = scores[0]
    margin = scores[1][0] - best if len(scores) > 1 else None
    return digit, best, margin
