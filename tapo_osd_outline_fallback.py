"""Fallback OSD extraction using the dark glyph outline to bound each digit."""

import cv2
import numpy as np

from tapo_osd_recognizer import normalize

DIGIT_SLOTS = ((7, 43), (50, 88), (99, 137), (148, 185),
               (247, 282), (290, 330), (391, 426), (435, 475),
               (530, 565), (579, 617), (675, 710), (724, 765),
               (819, 854), (867, 905))
OUTLINE_THRESHOLDS = (60, 80, 100)
WHITE_THRESHOLDS = (225, 235, 240, 245, 250)


def _outline_components(gray, threshold):
    mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if 25 <= h <= 60 and 5 <= w <= 35 and area >= 20:
            result.append((x, y, w, h, area, labels == i))
    return result


def _recover_slot(gray, left, right):
    roi = gray[:, left:right]
    outline_candidates = []
    for threshold in OUTLINE_THRESHOLDS:
        for x, y, w, h, area, component in _outline_components(roi, threshold):
            score = abs(h - 49) + abs(w - 27) - min(area, 250) / 100
            outline_candidates.append((score, x, y, w, h))
    outline = min(outline_candidates)[1:] if outline_candidates else None

    best = None
    for white_threshold in WHITE_THRESHOLDS:
        white = cv2.threshold(roi, white_threshold, 255, cv2.THRESH_BINARY)[1]
        n, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
        for i in range(1, n):
            x, y, w, h, area = map(int, stats[i])
            if not (25 <= h <= 62 and 3 <= w <= right - left - 2 and area >= 40):
                continue
            if outline is not None:
                ox, oy, ow, oh = outline
                overlap = max(0, min(x + w, ox + ow) - max(x, ox))
                if overlap < min(w, ow) * 0.35:
                    continue
            piece = white[max(0, y - 2):min(roi.shape[0], y + h + 2),
                          max(0, x - 2):min(roi.shape[1], x + w + 2)]
            glyph = normalize(piece)
            if glyph is None:
                continue
            candidate = (abs(h - 49) + abs(w - 25), glyph)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return best[1] if best else None


def extract_glyphs_from_outline(gray):
    glyphs = []
    for left, right in DIGIT_SLOTS:
        glyph = _recover_slot(gray, left, right)
        if glyph is None:
            raise RuntimeError("outline fallback could not recover all digit slots")
        glyphs.append(glyph)
    return glyphs
