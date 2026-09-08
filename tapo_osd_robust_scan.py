"""Robust OSD extraction stages used only after Fill Scan fails."""

import cv2
import numpy as np

from tapo_osd_recognizer import normalize

DIGIT_SLOTS = ((7, 43), (50, 88), (99, 137), (148, 185),
               (247, 282), (290, 330), (391, 426), (435, 475),
               (530, 565), (579, 617), (675, 710), (724, 765),
               (819, 854), (867, 905))
OUTLINE_THRESHOLDS = (60, 80, 100)
WHITE_THRESHOLDS = (225, 235, 240, 245, 250)
LUMA_SEARCH_OUTLINE_THRESHOLDS = (16, 24, 32, 40, 48, 56, 64, 72,
                                   80, 96, 112, 128, 144, 160)


def _bbox_candidates(mask, width):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if 20 <= h <= 62 and 3 <= w <= width - 2 and area >= 20:
            result.append((abs(h - 49) + abs(w - 25), x, y, w, h))
    return sorted(result)


def _local_bbox(roi):
    blur = cv2.GaussianBlur(roi, (0, 0), 5)
    contrast = cv2.addWeighted(roi, 1.0, blur, -1.0, 128.0)
    candidates = []
    for threshold in (135, 145, 155, 165, 175, 185):
        mask = cv2.threshold(contrast, threshold, 255, cv2.THRESH_BINARY)[1]
        candidates.extend(_bbox_candidates(mask, roi.shape[1]))
    return candidates[0][1:] if candidates else None


def extract_glyphs_from_local_contrast(gray, digit_slots=None):
    """Find digit boxes from local contrast, then extract glyphs from gray."""
    glyphs = []
    for left, right in (digit_slots or DIGIT_SLOTS):
        roi = gray[:, left:right]
        bbox = _local_bbox(roi)
        if bbox is None:
            raise RuntimeError("local contrast could not locate all digit slots")
        x, y, w, h = bbox
        source = roi[max(0, y - 2):min(roi.shape[0], y + h + 2),
                     max(0, x - 2):min(roi.shape[1], x + w + 2)]
        best = None
        for threshold in WHITE_THRESHOLDS:
            mask = cv2.threshold(source, threshold, 255, cv2.THRESH_BINARY)[1]
            glyph = normalize(mask)
            if glyph is not None:
                pixels = int(np.count_nonzero(glyph))
                if best is None or pixels > best[0]:
                    best = (pixels, glyph)
        if best is None:
            raise RuntimeError("local contrast could not extract all glyphs")
        glyphs.append(best[1])
    return glyphs


def _outline_components(gray, threshold):
    mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if 25 <= h <= 60 and 5 <= w <= 35 and area >= 20:
            result.append((x, y, w, h, area, labels == i))
    return result


def _recover_slot(source_gray, detection_gray, left, right):
    source_roi = source_gray[:, left:right]
    detection_roi = detection_gray[:, left:right]
    outline_candidates = []
    for threshold in OUTLINE_THRESHOLDS:
        for x, y, w, h, area, component in _outline_components(detection_roi, threshold):
            score = abs(h - 49) + abs(w - 27) - min(area, 250) / 100
            outline_candidates.append((score, x, y, w, h))
    outline = min(outline_candidates)[1:] if outline_candidates else None

    best = None
    for white_threshold in WHITE_THRESHOLDS:
        white = cv2.threshold(source_roi, white_threshold, 255, cv2.THRESH_BINARY)[1]
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
            piece = white[max(0, y - 2):min(source_roi.shape[0], y + h + 2),
                          max(0, x - 2):min(source_roi.shape[1], x + w + 2)]
            glyph = normalize(piece)
            if glyph is None:
                continue
            candidate = (abs(h - 49) + abs(w - 25), glyph)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return best[1] if best else None


def extract_glyphs_from_outline(gray, digit_slots=None):
    glyphs = []
    for left, right in (digit_slots or DIGIT_SLOTS):
        glyph = _recover_slot(gray, gray, left, right)
        if glyph is None:
            raise RuntimeError("outline fallback could not recover all digit slots")
        glyphs.append(glyph)
    return glyphs


def extract_glyphs_from_contrast_outline(gray, digit_slots=None):
    """Detect dark outlines in a local-contrast image, crop glyphs from gray."""
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    contrast = cv2.addWeighted(gray, 1.0, blur, -1.0, 128.0)
    glyphs = []
    for left, right in (digit_slots or DIGIT_SLOTS):
        glyph = _recover_slot(gray, contrast, left, right)
        if glyph is None:
            raise RuntimeError("contrast outline could not recover all digit slots")
        glyphs.append(glyph)
    return glyphs


def _recover_slot_luma(source_gray, left, right):
    """Recover one glyph using a wider raw-Y dark-outline search.

    This is deliberately a rescue path, not a second digit recognizer.  The
    dark raw-Y component only supplies a geometry hint; the glyph itself is
    re-extracted from the original Y plane and passed to the same Hamming
    matcher as every other path.
    """
    source_roi = source_gray[:, left:right]
    outline_candidates = []
    for threshold in LUMA_SEARCH_OUTLINE_THRESHOLDS:
        dark = cv2.threshold(source_roi, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        for i in range(1, n):
            x, y, w, h, area = map(int, stats[i])
            if 25 <= h <= 62 and 5 <= w <= 35 and area >= 16:
                # Geometry is the judge here: prefer a Tapo-sized dark
                # envelope, but do not let area choose a digit.
                score = abs(h - 49) + abs(w - 27) - min(area, 300) / 120
                outline_candidates.append((score, x, y, w, h))
    if not outline_candidates:
        return None
    ox, oy, ow, oh = min(outline_candidates)[1:]

    best = None
    for white_threshold in WHITE_THRESHOLDS:
        white = cv2.threshold(source_roi, white_threshold, 255,
                              cv2.THRESH_BINARY)[1]
        n, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
        for i in range(1, n):
            x, y, w, h, area = map(int, stats[i])
            if not (25 <= h <= 62 and 3 <= w <= right - left - 2
                    and area >= 40):
                continue
            overlap_x = max(0, min(x + w, ox + ow) - max(x, ox))
            overlap_y = max(0, min(y + h, oy + oh) - max(y, oy))
            if overlap_x < min(w, ow) * 0.35 or overlap_y < min(h, oh) * 0.35:
                continue
            piece = white[max(0, y - 2):min(source_roi.shape[0], y + h + 2),
                          max(0, x - 2):min(source_roi.shape[1], x + w + 2)]
            glyph = normalize(piece)
            if glyph is None:
                continue
            candidate = (abs(h - 49) + abs(w - 25), glyph)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return best[1] if best else None


def extract_glyphs_from_luma_search(gray, digit_slots=None):
    """Use raw-Y outline geometry as a final Fill rescue path."""
    glyphs = []
    for left, right in (digit_slots or DIGIT_SLOTS):
        glyph = _recover_slot_luma(gray, left, right)
        if glyph is None:
            raise RuntimeError("luma search could not recover all digit slots")
        glyphs.append(glyph)
    return glyphs
