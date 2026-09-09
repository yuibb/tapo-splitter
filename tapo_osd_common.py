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
from tapo_profile import geometry_for_profile

OSD_WIDTH, OSD_HEIGHT = 950, 70
MIN_MARGIN = 80
ROBUST_MIN_MARGIN = 80
STALL_SECONDS = 5.0

# A frame state is deliberately separate from a digit candidate.  A weak or
# unreadable frame is not a fatal processing error; it is evidence for the
# bounded Full Lane investigation.
VALID = "VALID"
SUSPECT = "SUSPECT"
UNKNOWN = "UNKNOWN"
ERROR = "ERROR"


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


def _osd_gray(frame, profile=None):
    """Extract the configured OSD crop; geometry failures are fatal errors."""
    geometry = geometry_for_profile(profile)
    roi_spec = geometry["roi"]
    if frame.shape[0] == roi_spec["height"] and frame.shape[1] == roi_spec["width"]:
        roi = frame
    else:
        roi = frame[roi_spec["y"]:roi_spec["y"] + roi_spec["height"],
                    roi_spec["x"]:roi_spec["x"] + roi_spec["width"]]
    if roi.shape[:2] != (roi_spec["height"], roi_spec["width"]):
        raise RuntimeError("ProfileのOSD ROIが動画フレーム範囲外です")
    # FastScan supplies the already-cropped grayscale rawvideo directly.
    # Keep accepting BGR frames for refinement and evidence paths.
    return roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def _candidate_from_glyphs(glyphs, templates, method):
    digits, margins = [], []
    for glyph in glyphs:
        digit, _distance, margin = recognize(glyph, templates)
        digits.append(digit)
        margins.append(margin)
    value = "".join(digits)
    try:
        timestamp = datetime.strptime(value, "%Y%m%d%H%M%S")
    except ValueError:
        return {"status": SUSPECT, "reason": "INVALID_DATETIME_CANDIDATE",
                "value": value, "margin": min(margins), "method": method}
    margin = min(margins)
    return {"status": VALID if margin >= MIN_MARGIN else SUSPECT,
            "reason": "STRONG_MARGIN" if margin >= MIN_MARGIN else "LOW_MARGIN",
            "timestamp": timestamp, "value": value, "margin": margin,
            "method": method}


def recognize_osd_state(frame, templates, profile=None):
    """Classify an OSD observation without converting weak OCR into an error.

    VALID is safe for Fast Lane.  SUSPECT and UNKNOWN are deliberately kept
    for temporal investigation; ERROR is reserved for a broken profile/ROI or
    another processing precondition, not ordinary OCR difficulty.
    """
    try:
        geometry = geometry_for_profile(profile)
        gray = _osd_gray(frame, profile)
    except Exception as exc:
        return {"status": ERROR, "reason": "PROFILE_OR_ROI_ERROR",
                "details": str(exc)}

    attempts = []
    try:
        glyphs, threshold = extract_glyphs_from_gray(gray, "video frame", profile)
        return _candidate_from_glyphs(glyphs, templates, threshold)
    except Exception as exc:
        attempts.append(f"fill:{type(exc).__name__}")
    slots = geometry["slot_ranges"]
    for method, extractor in (("raw_outline", extract_glyphs_from_outline),
                              ("contrast_outline", extract_glyphs_from_contrast_outline),
                              ("luma_search", extract_glyphs_from_luma_search)):
        try:
            return _candidate_from_glyphs(extractor(gray, slots), templates, method)
        except Exception as exc:
            attempts.append(f"{method}:{type(exc).__name__}")
    return {"status": UNKNOWN, "reason": "UNREADABLE_ALL_STAGES",
            "details": attempts}


def recognize_osd(frame, templates, profile=None, minimum_margin=None):
    """Backward-compatible strict recognizer for legacy Fast Lane callers."""
    state = recognize_osd_state(frame, templates, profile)
    floor = MIN_MARGIN if minimum_margin is None else minimum_margin
    if "timestamp" in state and state["margin"] >= floor:
        return state["timestamp"], state["value"], state["method"], state["margin"]
    raise RuntimeError(
        f"OSD {state['status']}: {state['reason']}"
        + (f" (margin={state['margin']})" if "margin" in state else "")
    )


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
