"""Shared OSD template loading, recognition, and evidence helpers."""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tapo_osd_robust_scan import (
    extract_glyphs_from_contrast_outline,
    extract_glyphs_from_luma_search,
    extract_glyphs_from_outline,
)
from tapo_osd_recognizer import extract_glyphs_from_gray, recognize

OSD_WIDTH, OSD_HEIGHT = 950, 70
MIN_MARGIN = 80
ROBUST_MIN_MARGIN = 80
STALL_SECONDS = 5.0


def load_templates(font_path):
    raw = json.loads(Path(font_path).read_text(encoding="utf-8"))
    return {d: [np.array(m, dtype=np.uint8) for m in raw["digits"][d]["templates"]]
            for d in "0123456789"}


def _recognize_glyphs(glyphs, templates, minimum_margin):
    digits, margins = [], []
    for glyph in glyphs:
        digit, _distance, margin = recognize(glyph, templates)
        digits.append(digit)
        margins.append(margin)
    value = "".join(digits)
    if min(margins) < minimum_margin:
        raise RuntimeError(f"low recognition margin: {min(margins)}")
    timestamp = datetime.strptime(value, "%Y%m%d%H%M%S")
    return timestamp, value, min(margins)


def recognize_osd(frame, templates):
    roi = frame[:OSD_HEIGHT, :OSD_WIDTH]
    # FastScan supplies the already-cropped grayscale rawvideo directly.
    # Keep accepting BGR frames for refinement and evidence paths.
    gray = roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    try:
        glyphs, threshold = extract_glyphs_from_gray(gray, "video frame")
        timestamp, value, margin = _recognize_glyphs(glyphs, templates, MIN_MARGIN)
        return timestamp, value, threshold, margin
    except Exception as primary_error:
        for method, extractor in (("raw_outline", extract_glyphs_from_outline),
                                  ("contrast_outline", extract_glyphs_from_contrast_outline),
                                  ("luma_search", extract_glyphs_from_luma_search)):
            try:
                glyphs = extractor(gray)
                timestamp, value, margin = _recognize_glyphs(
                    glyphs, templates, ROBUST_MIN_MARGIN)
                return timestamp, value, method, margin
            except Exception:
                continue
        raise primary_error


def save_pair(output, index, before, after, delta):
    prefix = Path(output) / f"jump_{index:02d}"
    cv2.imwrite(str(prefix.with_name(prefix.name + "_before.png")), before["frame"])
    cv2.imwrite(str(prefix.with_name(prefix.name + "_after.png")), after["frame"])
    return {
        "before_frame": prefix.with_name(prefix.name + "_before.png").name,
        "after_frame": prefix.with_name(prefix.name + "_after.png").name,
        "before_video_seconds": before["video_seconds"],
        "after_video_seconds": after["video_seconds"],
        "before_osd": before["formatted"],
        "after_osd": after["formatted"],
        **delta,
    }


def detect_osd_stalls(records):
    stalls, start, previous = [], None, None
    for record in records + ([None] if records else []):
        if record is not None and previous is not None and record["osd_digits"] == previous["osd_digits"]:
            start = previous if start is None else start
        else:
            if start is not None and previous is not None:
                duration = previous["video_seconds"] - start["video_seconds"]
                if duration >= STALL_SECONDS:
                    stalls.append({"video_start_seconds": start["video_seconds"],
                                   "video_end_seconds": previous["video_seconds"],
                                   "duration_seconds": duration, "osd": start["formatted"]})
            start = None
        previous = record
    return stalls
