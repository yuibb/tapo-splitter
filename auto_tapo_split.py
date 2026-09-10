#!/usr/bin/env python3
"""Tapo production splitter using v1.3.4 Fast/Full Lane detection.

Example:
  .tapo-venv/bin/python auto_tapo_split.py \
      --input-dir rec_sample --output-dir split成果_mov_auto

The source MP4 files are never modified. Video and audio are copied without
re-encoding into MOV containers.
"""

import argparse
import json
import shutil
import subprocess
import sys
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

from ffmpeg_frame_pipeline import (
    osd_range_frames, range_frames, record_from_frame, state_from_frame,
)
from tapo_osd_common import ERROR, SUSPECT, UNKNOWN, VALID, load_templates
from tapo_profile import ProfileError, resolve_profile

BASE = Path(__file__).resolve().parent
DETECTOR = BASE / "detect_tapo_time_jumps.py"
FONT = BASE / "tapo_osd_glyph_templates.json"
SPLITTER_VERSION = "1.3.4"
FULL_LANE_PADDING_SECONDS = 3.0
FULL_LANE_STEP_SECONDS = 1.0
FULL_LANE_DRIFT_TOLERANCE = 3.0
FULL_LANE_MAX_BRIDGE_GAP_SECONDS = 120.0
NAME_LOCK = Lock()


def run(command):
    subprocess.run(command, check=True)


def probe_duration(video):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def round_half_minutes(seconds):
    return round((seconds / 60.0) * 2) / 2


def clock(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def read_samples(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        video_seconds, osd, *_ = line.split("\t")
        rows.append((float(video_seconds), datetime.strptime(osd, "%Y-%m-%d %H:%M:%S")))
    return rows


def nearest_osd(samples, seconds):
    return min(samples, key=lambda item: abs(item[0] - seconds))[1]


def make_error_intervals(report, samples, duration):
    """Group non-VALID observations and bound them by neighboring VALID ones."""
    observations = report.get("non_valid_observations",
                              report.get("recognition_failures", []))
    failed = sorted(float(item["video_seconds"])
                    for item in observations
                    if item.get("status") in {SUSPECT, UNKNOWN, ERROR})
    if not failed or not samples:
        return []
    groups, current = [], []
    for seconds in failed:
        if current and seconds - current[-1] > 7.5:
            groups.append(current)
            current = []
        current.append(seconds)
    if current:
        groups.append(current)
    success_times = sorted(float(item[0]) for item in samples)
    intervals = []
    for group in groups:
        before = [value for value in success_times if value < group[0]]
        after = [value for value in success_times if value > group[-1]]
        start = (before[-1] + group[0]) / 2 if before else 0.0
        end = (group[-1] + after[0]) / 2 if after else duration
        if end > start:
            intervals.append({"start": max(0.0, start),
                              "end": min(duration, end),
                              "reason": "OSD要調査区間"})
    merged = []
    for interval in sorted(intervals, key=lambda item: item["start"]):
        if merged and interval["start"] <= merged[-1]["end"] + 1.0:
            merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
        else:
            merged.append(interval)
    return merged


def rescan_error_intervals(video, report, samples, duration, templates, step=1.0,
                           profile=None):
    """Rescan each coarse error interval and keep only still-unknown spans."""
    broad = make_error_intervals(report, samples, duration)
    recovered = []
    remaining = []
    recovered_jumps = []
    for interval in broad:
        points = []
        for seconds, frame in range_frames(
                video, interval["start"], interval["end"], step):
            try:
                points.append(record_from_frame(seconds, frame, templates, profile))
            except Exception:
                points.append({"video_seconds": seconds, "error": True})
        previous = None
        for point in points:
            if point.get("error"):
                previous = None
                continue
            recovered.append(point)
            if previous is not None:
                vd = point["video_seconds"] - previous["video_seconds"]
                od = (point["timestamp"] - previous["timestamp"]).total_seconds()
                if 0 < vd <= step * 1.5 and abs(od - vd) > 3.0:
                    recovered_jumps.append({
                        "before_video_seconds": previous["video_seconds"],
                        "after_video_seconds": point["video_seconds"],
                        "after_osd": point["formatted"],
                    })
            previous = point
        failed_points = [point["video_seconds"] for point in points if point.get("error")]
        if not failed_points:
            continue
        groups, current = [], []
        for seconds in failed_points:
            if current and seconds - current[-1] > step * 1.5:
                groups.append(current)
                current = []
            current.append(seconds)
        if current:
            groups.append(current)
        good_times = sorted(point["video_seconds"] for point in points if not point.get("error"))
        for group in groups:
            before = [value for value in good_times if value < group[0]]
            after = [value for value in good_times if value > group[-1]]
            start = (before[-1] + group[0]) / 2 if before else interval["start"]
            end = (group[-1] + after[0]) / 2 if after else interval["end"]
            if end > start:
                remaining.append({"start": max(interval["start"], start),
                                  "end": min(interval["end"], end),
                                  "reason": "OSD認識失敗区間"})
    merged = []
    for interval in sorted(remaining, key=lambda item: item["start"]):
        if merged and interval["start"] <= merged[-1]["end"] + 0.5:
            merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
        else:
            merged.append(interval)
    return recovered, merged, recovered_jumps


def _transition_cost(left, right):
    video_delta = right["video_seconds"] - left["video_seconds"]
    if video_delta <= 0:
        return 100.0
    osd_delta = (right["timestamp"] - left["timestamp"]).total_seconds()
    drift = abs(osd_delta - video_delta)
    if drift <= 3.0:
        return drift * 0.15
    return 3.0 + min(drift / 60.0, 20.0) * 0.1


def _decode_multiframe_candidates(frames, center, templates, profile=None):
    """Build candidates from several aligned frames and their composites."""
    observations = []
    for seconds, frame in frames:
        try:
            observations.append(record_from_frame(seconds, frame, templates, profile))
        except Exception:
            pass
    def collect(groups):
        candidates = []
        for group in groups.values():
            best = max(group, key=lambda item: item["margin"])
            support = len(group)
            if support < 2 and best["margin"] < 80:
                continue
            best = dict(best)
            best["video_seconds"] = center
            best["_support"] = support
            best["scan_method"] = "multiframe_consensus"
            candidates.append(best)
        return candidates

    groups = {}
    for item in observations:
        groups.setdefault(item["osd_digits"], []).append(item)
    # Two agreeing raw frames are enough; avoid the expensive composite path.
    candidates = collect(groups)
    if any(item.get("_support", 0) >= 2 for item in candidates):
        return candidates
    if frames:
        gray_stack = np.stack([
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for _, frame in frames
        ])
        composites = (
            ("median", np.median(gray_stack, axis=0).astype(np.uint8)),
            ("brightest", np.max(gray_stack, axis=0)),
            ("darkest", np.min(gray_stack, axis=0)),
        )
        for method, gray in composites:
            try:
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                item = record_from_frame(center, frame, templates, profile)
                item["scan_method"] = f"multiframe_{method}"
                observations.append(item)
            except Exception:
                pass
    groups = {}
    for item in observations:
        groups.setdefault(item["osd_digits"], []).append(item)
    return collect(groups)


def _choose_temporal_path(candidate_sets, before, after):
    """Choose a likely candidate sequence using a continuity cost."""
    paths = []
    for candidates in candidate_sets:
        states = []
        for candidate in candidates:
            emission = max(0.0, 80.0 - candidate["margin"]) / 20.0
            emission -= min(candidate.get("_support", 1), 4) * 0.25
            best_cost = emission
            best_prev = None
            if before is not None:
                best_cost += _transition_cost(before, candidate)
            for previous_index, previous in enumerate(states):
                cost = (previous["cost"] +
                        _transition_cost(previous["item"], candidate) + emission)
                if cost < best_cost:
                    best_cost, best_prev = cost, previous_index
            states.append({"item": candidate, "cost": best_cost, "prev": best_prev})
        paths.append(states)
    usable = [index for index, states in enumerate(paths) if states]
    if not usable:
        return []
    last_index = usable[-1]
    final_states = paths[last_index]
    if after is not None:
        chosen = min(final_states,
                     key=lambda state: state["cost"] + _transition_cost(state["item"], after))
    else:
        chosen = min(final_states, key=lambda state: state["cost"])
    selected = [None] * len(paths)
    for index in range(last_index, -1, -1):
        if not paths[index]:
            continue
        selected[index] = chosen["item"]
        if chosen["prev"] is None:
            break
        chosen = paths[index][chosen["prev"]]
    return selected


def _group_close(values, max_gap):
    groups, current = [], []
    for value in sorted(values):
        if current and value - current[-1] > max_gap:
            groups.append(current)
            current = []
        current.append(value)
    if current:
        groups.append(current)
    return groups


def _merge_intervals(intervals):
    merged = []
    for interval in sorted(intervals, key=lambda item: item["start"]):
        if merged and interval["start"] <= merged[-1]["end"] + 0.5:
            merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
        else:
            merged.append(dict(interval))
    return merged


def _rescan_multiframe_interval(video, interval, samples, templates, profile=None):
    scan_frames = list(osd_range_frames(
        video, interval["start"], interval["end"], 0.5, profile))
    buckets = {}
    for seconds, frame in scan_frames:
        bucket = round(seconds - interval["start"])
        buckets.setdefault(bucket, []).append((seconds, frame))
    if not buckets:
        return [], [interval], []
    first_bucket, last_bucket = min(buckets), max(buckets)
    candidate_sets, centers = [], []
    for bucket in range(first_bucket, last_bucket + 1):
        center = min(interval["end"], interval["start"] + bucket)
        centers.append(center)
        candidate_sets.append(_decode_multiframe_candidates(
            buckets.get(bucket, []), center, templates, profile))
    before = max((item for item in samples if item[0] < interval["start"]),
                 key=lambda item: item[0], default=None)
    after = min((item for item in samples if item[0] > interval["end"]),
                key=lambda item: item[0], default=None)
    before_record = None if before is None else {
        "video_seconds": before[0], "timestamp": before[1]}
    after_record = None if after is None else {
        "video_seconds": after[0], "timestamp": after[1]}
    selected = _choose_temporal_path(candidate_sets, before_record, after_record)
    recovered, failed_times, recovered_jumps = [], [], []
    previous = None
    for center, item in zip(centers, selected):
        if item is None:
            failed_times.append(center)
            previous = None
            continue
        item = dict(item)
        item["video_seconds"] = center
        recovered.append(item)
        if previous is not None and center - previous["video_seconds"] <= 1.5:
            if abs((item["timestamp"] - previous["timestamp"]).total_seconds()
                   - (center - previous["video_seconds"])) > 3.0:
                recovered_jumps.append({
                    "before_video_seconds": previous["video_seconds"],
                    "after_video_seconds": center,
                    "after_osd": item["formatted"],
                })
        previous = item
    remaining = []
    good_times = [item["video_seconds"] for item in recovered]
    for group in _group_close(failed_times, 1.5):
        before_good = [value for value in good_times if value < group[0]]
        after_good = [value for value in good_times if value > group[-1]]
        start = (before_good[-1] + group[0]) / 2 if before_good else interval["start"]
        end = (group[-1] + after_good[0]) / 2 if after_good else interval["end"]
        if end > start:
            remaining.append({"start": start, "end": end,
                              "reason": "OSD認識失敗区間"})
    return recovered, remaining, recovered_jumps


def rescan_error_intervals_multiframe(video, report, samples, duration, templates,
                                      profile=None):
    """Recover error spans using multi-frame consensus and decoding."""
    broad = make_error_intervals(report, samples, duration)
    recovered, remaining, recovered_jumps = [], [], []
    for interval in broad:
        part_recovered, part_remaining, part_jumps = _rescan_multiframe_interval(
            video, interval, samples, templates, profile)
        recovered.extend(part_recovered)
        remaining.extend(part_remaining)
        recovered_jumps.extend(part_jumps)
    return recovered, _merge_intervals(remaining), recovered_jumps


def _full_lane_boundary_samples(interval, samples):
    """Return the nearest known-good samples around an unresolved interval."""
    before = max((item for item in samples if item[0] < interval["start"]),
                 key=lambda item: item[0], default=None)
    after = min((item for item in samples if item[0] > interval["end"]),
                key=lambda item: item[0], default=None)
    return before, after


def _remove_transient_full_lane_ocr(items, before=None, after=None):
    """Remove an isolated wild date read when its neighbors connect normally."""
    context = list(items)
    original_ids = {id(item) for item in items}
    for boundary, side in ((before, "before"), (after, "after")):
        if boundary is None:
            continue
        sample = {
            "video_seconds": boundary[0],
            "timestamp": boundary[1],
            "osd_digits": boundary[1].strftime("%Y%m%d%H%M%S"),
            "formatted": boundary[1].strftime("%Y-%m-%d %H:%M:%S"),
            "margin": None,
            "_boundary": side,
        }
        context.append(sample)
    context.sort(key=lambda item: item["video_seconds"])
    transient_ids = set()
    for left_index, left in enumerate(context):
        for right_index in range(left_index + 2,
                                 min(len(context), left_index + 5)):
            right = context[right_index]
            video_span = right["video_seconds"] - left["video_seconds"]
            osd_span = (right["timestamp"] - left["timestamp"]).total_seconds()
            if (video_span <= 0
                    or abs(osd_span - video_span) > FULL_LANE_DRIFT_TOLERANCE):
                continue
            middle = context[left_index + 1:right_index]
            if any(abs((item["timestamp"] - left["timestamp"]).total_seconds()
                       - (item["video_seconds"] - left["video_seconds"])) > 60.0
                   for item in middle):
                transient_ids.update(id(item) for item in middle)
    return [item for item in items
            if id(item) not in transient_ids and id(item) in original_ids]


def _full_lane_transitions(records, before, after):
    """Find jumps/stalls in a single Full Lane pipe window.

    The recognizer remains the same recognizer.  Full Lane only widens the
    evidence window and lowers the rescue margin to 70; it never changes a
    digit based on geometry or width alone.
    """
    records = _remove_transient_full_lane_ocr(records, before, after)
    context = []
    if before is not None:
        context.append({"video_seconds": before[0], "timestamp": before[1],
                        "osd_digits": before[1].strftime("%Y%m%d%H%M%S"),
                        "formatted": before[1].strftime("%Y-%m-%d %H:%M:%S"),
                        "margin": None})
    context.extend(records)
    if after is not None:
        context.append({"video_seconds": after[0], "timestamp": after[1],
                        "osd_digits": after[1].strftime("%Y%m%d%H%M%S"),
                        "formatted": after[1].strftime("%Y-%m-%d %H:%M:%S"),
                        "margin": None})
    context.sort(key=lambda item: item["video_seconds"])
    deduped = []
    for item in context:
        if deduped and abs(item["video_seconds"] -
                           deduped[-1]["video_seconds"]) < 0.25:
            current = deduped[-1]
            if (item.get("margin") or -1) > (current.get("margin") or -1):
                deduped[-1] = item
            continue
        deduped.append(item)

    jumps, stalls = [], []
    stall_start = None
    stall_reported = False
    for left, right in zip(deduped, deduped[1:]):
        video_delta = right["video_seconds"] - left["video_seconds"]
        if video_delta <= 0:
            stall_start = None
            stall_reported = False
            continue
        osd_delta = (right["timestamp"] - left["timestamp"]).total_seconds()
        drift = osd_delta - video_delta
        if video_delta > FULL_LANE_STEP_SECONDS * 4.0:
            stall_start = None
            stall_reported = False
            # A long unreadable span can hide the exact jump frame.  Keep the
            # boundary mismatch as a bridge candidate; make_segments will
            # still keep it inside the unresolved Err interval.
            if (video_delta <= FULL_LANE_MAX_BRIDGE_GAP_SECONDS
                    and left["osd_digits"] != right["osd_digits"]
                    and abs(drift) > FULL_LANE_DRIFT_TOLERANCE):
                jumps.append({
                    "before_video_seconds": left["video_seconds"],
                    "after_video_seconds": right["video_seconds"],
                    "before_osd": left["formatted"],
                    "after_osd": right["formatted"],
                    "video_elapsed_seconds": video_delta,
                    "osd_elapsed_seconds": osd_delta,
                    "jump_seconds": drift,
                    "scan_method": "full_lane_bridge",
                })
            continue
        if left["osd_digits"] == right["osd_digits"]:
            if video_delta <= FULL_LANE_STEP_SECONDS * 2.5:
                if stall_start is None:
                    stall_start = left
                    stall_reported = False
                if (not stall_reported and
                        right["video_seconds"] - stall_start["video_seconds"] >= 5.0):
                    stalls.append({
                        "video_start_seconds": stall_start["video_seconds"],
                        "video_end_seconds": right["video_seconds"],
                        "duration_seconds": (
                            right["video_seconds"] - stall_start["video_seconds"]),
                        "osd": stall_start["formatted"],
                    })
                    stall_reported = True
            else:
                stall_start = None
                stall_reported = False
            continue
        stall_start = None
        stall_reported = False
        if abs(drift) > FULL_LANE_DRIFT_TOLERANCE:
            jumps.append({
                "before_video_seconds": left["video_seconds"],
                "after_video_seconds": right["video_seconds"],
                "before_osd": left["formatted"],
                "after_osd": right["formatted"],
                "video_elapsed_seconds": video_delta,
                "osd_elapsed_seconds": osd_delta,
                "jump_seconds": drift,
                "scan_method": "full_lane",
            })
    return jumps, stalls


def _full_lane_remaining(interval, failed_times, recovered):
    """Keep only the portions that still have no readable Full Lane sample."""
    failed_times = sorted(
        value for value in failed_times
        if interval["start"] <= value <= interval["end"]
    )
    if not failed_times:
        return []
    groups = _group_close(failed_times, FULL_LANE_STEP_SECONDS * 1.5)
    good_times = sorted(item["video_seconds"] for item in recovered
                        if interval["start"] <= item["video_seconds"] <= interval["end"])
    remaining = []
    for group in groups:
        before = [value for value in good_times if value < group[0]]
        after = [value for value in good_times if value > group[-1]]
        start = (before[-1] + group[0]) / 2 if before else interval["start"]
        end = (group[-1] + after[0]) / 2 if after else interval["end"]
        if end > start:
            remaining.append({"start": max(interval["start"], start),
                              "end": min(interval["end"], end),
                              "reason": "Full LaneでもOSD認識不能"})
    return remaining


def rescan_error_intervals_full_lane(video, report, samples, duration, templates,
                                     profile=None):
    """Investigate only non-VALID spans without treating them as fatal errors.

    Full Lane first asks whether the surrounding VALID OSD observations follow
    normal PTS time.  A continuous gap is harmless even when individual
    frames are SUSPECT or UNKNOWN.  Only a temporal anomaly remains an Err
    span unless Full Lane obtains sufficiently strong direct evidence.
    """
    broad = make_error_intervals(report, samples, duration)
    recovered, remaining, rescued_jumps, rescued_stalls = [], [], [], []
    stats = {
        "intervals": len(broad),
        "pipe_windows": 0,
        "pipe_frames": 0,
        "valid_frames": 0,
        "suspect_frames": 0,
        "unknown_frames": 0,
        "error_frames": 0,
        "continuous_gaps": 0,
        "temporal_anomalies": 0,
        "rescued_jumps": 0,
        "unresolved_intervals": 0,
    }
    for interval in broad:
        scan_start = max(0.0, interval["start"] - FULL_LANE_PADDING_SECONDS)
        scan_end = min(duration, interval["end"] + FULL_LANE_PADDING_SECONDS)
        before, after = _full_lane_boundary_samples(interval, samples)
        valid_observations = []
        stats["pipe_windows"] += 1
        try:
            frames = list(osd_range_frames(
                video, scan_start, scan_end, FULL_LANE_STEP_SECONDS, profile))
        except Exception:
            frames = []
        stats["pipe_frames"] += len(frames)
        for seconds, frame in frames:
            try:
                state = state_from_frame(seconds, frame, templates, profile)
            except Exception:
                stats["error_frames"] += 1
                continue
            if state["status"] == VALID:
                valid_observations.append(state)
            elif state["status"] == SUSPECT:
                stats["suspect_frames"] += 1
            elif state["status"] == UNKNOWN:
                stats["unknown_frames"] += 1
            else:
                stats["error_frames"] += 1
        valid_observations = _remove_transient_full_lane_ocr(
            valid_observations, before, after)
        recovered.extend(
            item for item in valid_observations
            if interval["start"] <= item["video_seconds"] <= interval["end"]
        )
        stats["valid_frames"] += len(valid_observations)

        boundary_drift = None
        if before is not None and after is not None:
            boundary_drift = ((after[1] - before[1]).total_seconds()
                              - (after[0] - before[0]))
        if boundary_drift is not None and abs(boundary_drift) <= FULL_LANE_DRIFT_TOLERANCE:
            # UNKNOWN/SUSPECT frames are a hole in an otherwise continuous
            # timeline, not a video error and not a splitting boundary.
            stats["continuous_gaps"] += 1
            continue

        stats["temporal_anomalies"] += 1
        jumps, stalls = _full_lane_transitions(valid_observations, before, after)
        rescued_jumps.extend(jumps)
        rescued_stalls.extend(stalls)
        # An anomalous span remains explicitly marked until a later Rescue
        # Gate can prove every weak frame.  A direct bridge is recorded above,
        # but never promoted solely because its margin happened to be low.
        remaining.append({"start": interval["start"], "end": interval["end"],
                          "reason": "時系列異常を伴うOSD要調査区間"})
        stats["unresolved_intervals"] += 1

    unique_jumps = []
    for jump in sorted(rescued_jumps, key=lambda item: item["before_video_seconds"]):
        if any(abs(jump["before_video_seconds"] - existing["before_video_seconds"]) < 3.0
               for existing in unique_jumps):
            continue
        unique_jumps.append(jump)
    stats["rescued_jumps"] = len(unique_jumps)
    return (recovered, _merge_intervals(remaining), unique_jumps,
            rescued_stalls, stats)


def make_segments(report, samples, duration, error_intervals=None):
    error_intervals = error_intervals or []
    jump_pairs = []
    for item in report.get("jumps", []):
        cut = (float(item["before_video_seconds"]) +
               float(item["after_video_seconds"])) / 2
        if any(interval["start"] <= cut <= interval["end"]
               for interval in error_intervals):
            continue
        jump_pairs.append((cut, item))
    jump_pairs.sort(key=lambda pair: pair[0])
    cuts = [pair[0] for pair in jump_pairs]
    stalls = []
    for item in report.get("osd_stalls", []):
        start = float(item["video_start_seconds"]) + 5.0
        end = min(float(item["video_end_seconds"]), duration)
        if end > start:
            stalls.append({"start": start, "end": end, "osd": item["osd"]})
    edges = sorted({0.0, duration, *cuts,
                    *(item["start"] for item in stalls),
                    *(item["end"] for item in stalls),
                    *(item["start"] for item in error_intervals),
                    *(item["end"] for item in error_intervals)})
    jump_starts = [datetime.strptime(item["after_osd"], "%Y-%m-%d %H:%M:%S")
                   for _, item in jump_pairs]
    segments = []
    for index, (start, end) in enumerate(zip(edges, edges[1:]), 1):
        if end <= start:
            continue
        matching_stall = next(
            (item for item in stalls if start >= item["start"] - 0.001
             and end <= item["end"] + 0.001), None
        )
        matching_error = next(
            (item for item in error_intervals
             if start >= item["start"] - 0.001
             and end <= item["end"] + 0.001), None
        )
        jump_index = next(
            (i for i, cut in enumerate(cuts) if abs(cut - start) < 0.001), None
        )
        if matching_stall:
            osd_start = datetime.strptime(matching_stall["osd"], "%Y-%m-%d %H:%M:%S")
        elif matching_error:
            osd_start = nearest_osd(samples, start)
        elif jump_index is not None:
            osd_start = jump_starts[jump_index]
        else:
            osd_start = nearest_osd(samples, start)
        segments.append({
            "index": index,
            "start_video_seconds": start,
            "end_video_seconds": end,
            "duration_seconds": end - start,
            "osd_start": osd_start,
            "is_error": matching_stall is not None or matching_error is not None,
        })
    return segments


def unique_path(path):
    with NAME_LOCK:
        if not path.exists():
            return path
        for suffix in range(2, 1000):
            candidate = path.with_name(f"{path.stem}_part{suffix:02d}{path.suffix}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"too many filename collisions: {path}")


def process_video(video, output_root, python_executable, profiles_file, legacy_font=None):
    try:
        profile = resolve_profile(video, profiles_file) if profiles_file else None
    except ProfileError:
        if profiles_file is not None or legacy_font is None:
            raise
        profile = {"id": "legacy-font", "font_path": str(legacy_font),
                   "video_resolution": {"width": None, "height": None}}
    font = Path(profile["font_path"])
    print(f"processing: {video.name} [Profile: {profile['id']}]", flush=True)
    work = output_root / f".work_{video.stem}"
    work.mkdir(parents=True, exist_ok=True)
    report_dir = work / "jump_report"
    run([python_executable, str(DETECTOR), str(video), "--font", str(font),
         "--profile-id", profile["id"],
         "--profile-file", profile["registry_path"],
         "--output", str(report_dir), "--coarse-step", "15",
         "--refine-step", "5"])

    report = json.loads((report_dir / "time_jump_report.json").read_text(encoding="utf-8"))
    report["profile_id"] = profile["id"]
    samples = read_samples(report_dir / "toc_samples.tsv")
    source_duration = probe_duration(video)
    if len(samples) < 2:
        index_path = output_root / f"{video.stem}.md"
        index_path.write_text(
            f"# {video.name}\n\n"
            f"- 使用スプリッター: `{SPLITTER_VERSION}`\n"
            f"- 使用Profile: `{profile['id']}`\n"
            f"- 判定: **分割スキップ**\n"
            f"- 理由: 正常に認識できたOSDサンプルが2点未満（{len(samples)}点）\n"
            f"- 元ファイル時間: `{source_duration:.3f} 秒`\n"
            f"- 元動画は未変更・未処理のまま残しています。\n",
            encoding="utf-8",
        )
        shutil.rmtree(work)
        print(f"skipped: {video.name} (fewer than 2 valid OSD samples)", flush=True)
        return
    templates = load_templates(font)
    (recovered, error_intervals, recovered_jumps, recovered_stalls,
     full_lane_stats) = rescan_error_intervals_full_lane(
        video, report, samples, source_duration, templates, profile)
    samples.extend((item["video_seconds"], item["timestamp"]) for item in recovered)
    existing_jump_times = {
        round(float(item["before_video_seconds"]), 3)
        for item in report.get("jumps", [])
    }
    for jump in recovered_jumps:
        key = round(float(jump["before_video_seconds"]), 3)
        if key not in existing_jump_times:
            report["jumps"].append(jump)
            existing_jump_times.add(key)
    existing_stall_times = {
        round(float(item["video_start_seconds"]), 3)
        for item in report.get("osd_stalls", [])
    }
    for stall in recovered_stalls:
        key = round(float(stall["video_start_seconds"]), 3)
        if key not in existing_stall_times:
            report["osd_stalls"].append(stall)
            existing_stall_times.add(key)
    segments = make_segments(report, samples, source_duration, error_intervals)
    source_manifest = []
    stalls = report.get("osd_stalls", [])

    for segment in segments:
        stamp = segment["osd_start"].strftime("%Y%m%d_%H%M%S")
        length = round_half_minutes(segment["duration_seconds"])
        error_suffix = "_Err" if segment["is_error"] else ""
        base_name = f"{stamp}_{length:.1f}{error_suffix}"
        destination = unique_path(output_root / f"{base_name}.mov")
        command = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{segment['start_video_seconds']:.6f}",
            "-t", f"{segment['duration_seconds']:.6f}",
            "-i", str(video), "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "copy", "-c:a", "copy", str(destination),
        ]
        run(command)
        segment["output_duration_seconds"] = probe_duration(destination)
        segment["output_file"] = destination.name
        segment["length_minutes_rounded"] = length
        source_manifest.append(segment)

    output_total = sum(item["output_duration_seconds"] for item in source_manifest)
    delta = output_total - source_duration
    tolerance = 1.0
    if delta < -tolerance:
        verdict = "異常: 分割後の合計が元ファイルより短い"
    elif delta > tolerance:
        verdict = "警告: 分割後の合計が元ファイルより長い"
    else:
        verdict = "正常範囲"

    index_path = output_root / f"{video.stem}.md"
    lines = [
        f"# {video.name}", "", "無劣化ストリームコピー（H.264/音声 → MOV）による分割一覧。", "",
        f"- 使用スプリッター: `{SPLITTER_VERSION}`（Fast Lane＋Full Lane救済）",
        f"- 使用Profile: `{profile['id']}`",
        f"- 元ファイル時間: `{source_duration:.3f} 秒`",
        f"- 分割後合計時間: `{output_total:.3f} 秒`",
        f"- 差分（分割後−元）: `{delta:+.3f} 秒`",
        f"- シーケンス判定: **{verdict}**（許容値 ±{tolerance:.1f}秒）", "",
        f"- Fast Lane状態: VALID `{report.get('state_counts', {}).get('VALID', len(samples))}` / "
        f"SUSPECT `{report.get('state_counts', {}).get('SUSPECT', 0)}` / "
        f"UNKNOWN `{report.get('state_counts', {}).get('UNKNOWN', 0)}` / "
        f"ERROR `{report.get('state_counts', {}).get('ERROR', 0)}`",
        f"- 未解決の時系列異常区間: `{len(error_intervals)}区間`",
        f"- Full Lane: `{full_lane_stats['pipe_windows']}窓 / "
        f"{full_lane_stats['pipe_frames']}フレーム`、連続ギャップ `"
        f"{full_lane_stats['continuous_gaps']}件`、時系列異常 `"
        f"{full_lane_stats['temporal_anomalies']}件`、Full Lane内ジャンプ候補 `"
        f"{full_lane_stats['rescued_jumps']}件`", "",
        "| No. | 出力ファイル | OSDジャンプ後の開始時刻 | 動画区間(秒) | 実時間(分) |", 
        "|---:|---|---|---:|---:|",
    ]
    if stalls:
        lines += ["", "## 録画停止疑い（OSD同一秒が5秒以上継続）", "",
                  "| OSD | 動画ファイル上の位置 | Err切り出し位置 | 継続時間 |", "|---|---|---|---:|"]
        for stall in stalls:
            lines.append(
                f"| {stall['osd']} | "
                f"{clock(stall['video_start_seconds'])}–{clock(stall['video_end_seconds'])} "
                f"（{stall['video_start_seconds']:.3f}–{stall['video_end_seconds']:.3f}秒） | "
                f"{clock(stall['video_start_seconds'] + 5.0)}–{clock(stall['video_end_seconds'])} | "
                f"{stall['duration_seconds']:.1f}秒 |"
            )
    else:
        lines += ["", "## 録画停止疑い", "", "検出なし（OSD同一秒が5秒以上継続する区間なし）。"]
    for item in source_manifest:
        lines.append(
            f"| {item['index']} | `{item['output_file']}` | "
            f"{item['osd_start']:%Y-%m-%d %H:%M:%S} | "
            f"{item['start_video_seconds']:.3f}–{item['end_video_seconds']:.3f} | "
            f"{item['output_duration_seconds'] / 60:.2f}（表示 {item['length_minutes_rounded']:.1f}） |"
        )
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(work)
    if delta < -tolerance:
        raise RuntimeError(
            f"duration check failed for {video.name}: "
            f"source={source_duration:.3f}s output={output_total:.3f}s delta={delta:+.3f}s"
        )
    print(f"completed: {video.name} -> {len(source_manifest)} clips, {index_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=FONT)
    parser.add_argument("--profiles", type=Path,
                        help="解像度ごとのProfile＋EliteFont定義JSON")
    parser.add_argument("--workers", type=int, default=3,
                        help="number of videos to process in parallel (default: 3)")
    parser.add_argument("--reprocess-existing", action="store_true",
                        help="explicitly reprocess videos with an existing Markdown index")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(args.input_dir.glob("*.mp4"))
    if not args.reprocess_existing:
        videos = [
            video for video in videos
            if not (args.output_dir / f"{video.stem}.md").exists()
            or video.stat().st_mtime > (args.output_dir / f"{video.stem}.md").stat().st_mtime
        ]
    if not videos:
        raise SystemExit(f"no MP4 files found in {args.input_dir}")
    python_executable = sys.executable
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_video, video, args.output_dir,
                        python_executable, args.profiles, args.font): video
            for video in videos
        }
        for future in as_completed(futures):
            video = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed.append(video)
                print(f"failed: {video.name} ({type(exc).__name__})", flush=True)
    if failed:
        names = ", ".join(video.name for video in failed)
        raise SystemExit(f"{len(failed)} file(s) failed; remaining files were processed: {names}")


if __name__ == "__main__":
    main()
