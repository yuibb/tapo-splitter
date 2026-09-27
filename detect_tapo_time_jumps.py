#!/usr/bin/env python3
"""Tapo OSD time-jump detector with temporal OCR correction and range-pipe audit."""

import argparse
from collections import Counter
import json
from pathlib import Path

from tapo_osd_common import detect_osd_stalls, load_templates, save_pair
from tapo_profile import ProfileError, load_profile_by_id
from ffmpeg_frame_pipeline import (
    TOLERANCE, coarse_osd_frames_seek, range_frames, probe_duration,
    record_from_frame, sampling_targets, state_from_frame,
)
from tapo_osd_common import VALID

STEP = 15.0


def unique_events(events, key):
    out = []
    for event in sorted(events, key=lambda item: item[key]):
        if out and abs(event[key] - out[-1][key]) < 3:
            continue
        out.append(event)
    return out


def plan_ranges(ranges):
    """Merge overlapping/contained ranges within one pass only."""
    ordered = sorted(
        ({"start": float(item["start"]), "end": float(item["end"]),
          "source_index": index} for index, item in enumerate(ranges)
         if float(item["end"]) > float(item["start"])),
        key=lambda item: (item["start"], item["end"]),
    )
    planned = []
    for item in ordered:
        if not planned or item["start"] > planned[-1]["end"]:
            planned.append({"start": item["start"], "end": item["end"],
                            "source_indices": [item["source_index"]]})
            continue
        planned[-1]["end"] = max(planned[-1]["end"], item["end"])
        planned[-1]["source_indices"].append(item["source_index"])
    return planned


def _anomalies_from_items(items):
    anomalies = []
    for left, right in zip(items, items[1:]):
        video_delta = right["video_seconds"] - left["video_seconds"]
        if video_delta <= 0:
            continue
        osd_delta = (right["timestamp"] - left["timestamp"]).total_seconds()
        if abs(osd_delta - video_delta) > TOLERANCE:
            anomalies.append((left, right, video_delta, osd_delta))
    return anomalies


def audit_planned_ranges(video, templates, ranges, step, total, non_valid, toc,
                         corrections, profile=None):
    """Audit a pass after merging only overlapping logical windows.

    The physical range is decoded once.  Its observations are then projected
    back onto each logical source window before transient OCR cleanup, so the
    existing temporal semantics remain local to the original window.
    """
    planned = plan_ranges(ranges)
    anomalies = []
    seen = set()
    for physical in planned:
        items, _ = audit_range(
            video, templates, physical["start"], physical["end"], step, total,
            non_valid, toc, corrections, profile, apply_transient=False,
        )
        for source_index in physical["source_indices"]:
            source = ranges[source_index]
            local = [item for item in items
                     if source["start"] - 1e-6 <= item["video_seconds"]
                     <= source["end"] + 1e-6]
            local = remove_transient_ocr(local, corrections)
            for anomaly in _anomalies_from_items(local):
                key = (round(anomaly[0]["video_seconds"], 6),
                       round(anomaly[1]["video_seconds"], 6),
                       round(anomaly[2], 6), round(anomaly[3], 6))
                if key not in seen:
                    seen.add(key)
                    anomalies.append(anomaly)
    return anomalies, {
        "step": float(step),
        "logical_ranges": [{"start": item["start"], "end": item["end"]}
                            for item in ranges],
        "physical_ranges": planned,
        "logical_range_count": len(ranges),
        "physical_range_count": len(planned),
        "merge_count": max(0, len(ranges) - len(planned)),
    }


def _ranges_from_anomalies(anomalies, source_ranges, padding, total):
    """Create the next pass windows while retaining the source bounds."""
    ranges = []
    for left, right, _video_delta, _osd_delta in anomalies:
        containing = [item for item in source_ranges
                      if item["start"] - 1e-6 <= left["video_seconds"]
                      and right["video_seconds"] <= item["end"] + 1e-6]
        bound = min(containing, key=lambda item: item["end"] - item["start"])
        ranges.append({
            "start": max(0.0, bound["start"], left["video_seconds"] - padding),
            "end": min(total, bound["end"], right["video_seconds"] + padding),
        })
    return ranges


def remove_transient_ocr(items, corrections):
    """Drop short wildly-wrong OSD runs when wider neighbors join normally."""
    if len(items) < 3:
        return items
    kept = []
    for index, item in enumerate(items):
        corrected = False
        # Check across up to two neighboring samples on either side. This
        # catches a short 2-sample OCR run such as 2025 -> 2021 -> 2021 -> 2025.
        for radius in (1, 2):
            before_index = index - 1
            after_index = index + radius
            if before_index < 0 or after_index >= len(items):
                continue
            before, after = items[before_index], items[after_index]
            video_span = after["video_seconds"] - before["video_seconds"]
            osd_span = (after["timestamp"] - before["timestamp"]).total_seconds()
            video_from_before = item["video_seconds"] - before["video_seconds"]
            osd_from_before = (item["timestamp"] - before["timestamp"]).total_seconds()
            if (video_span > 0 and abs(osd_span - video_span) <= TOLERANCE
                    and abs(osd_from_before - video_from_before) > 60.0):
                corrections.append({
                    "video_seconds": item["video_seconds"],
                    "read_osd": item["formatted"],
                    "reason": f"前後{radius}サンプルで正常接続する短い大幅時刻誤読",
                })
                corrected = True
                break
        if corrected:
            continue
        kept.append(item)
    return kept


def _reportable_state(item):
    return {key: value for key, value in item.items()
            if key not in {"timestamp", "frame"}}


def audit_range(video, templates, lo, hi, step, total, non_valid, toc, corrections,
                profile=None, apply_transient=True):
    items = []
    try:
        for seconds, frame in range_frames(video, lo, hi, step):
            try:
                state = state_from_frame(seconds, frame, templates, profile)
                if state["status"] == VALID:
                    item = record_from_frame(seconds, frame, templates, profile)
                    items.append(item)
                    toc.append({k: v for k, v in item.items()
                                if k not in {"frame", "timestamp"}})
                else:
                    non_valid.append(_reportable_state(state))
            except Exception as exc:
                non_valid.append({"video_seconds": seconds, "status": "ERROR",
                                  "reason": "PIPE_OR_PROCESSING_ERROR", "details": str(exc)})
    except Exception as exc:
        non_valid.append({"video_seconds": lo, "status": "ERROR",
                          "reason": "PIPE_OR_PROCESSING_ERROR", "details": str(exc)})
    if apply_transient:
        items = remove_transient_ocr(items, corrections)
    return items, _anomalies_from_items(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--profile-id", default="unknown")
    parser.add_argument("--profile-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coarse-step", type=float, default=STEP)
    parser.add_argument("--refine-step", type=float, default=5.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile = None
    if args.profile_file and args.profile_id != "unknown":
        try:
            profile = load_profile_by_id(args.profile_file, args.profile_id)
        except ProfileError as exc:
            raise SystemExit(str(exc)) from exc
    templates = load_templates(args.font)
    total = probe_duration(args.video)
    coarse = []
    non_valid = []
    ocr_corrections = []
    coarse_seek_failures = []
    for seconds, frame in coarse_osd_frames_seek(args.video, step=args.coarse_step,
                                                 profile=profile,
                                                 failures=coarse_seek_failures):
        try:
            state = state_from_frame(seconds, frame, templates, profile)
            if state["status"] == VALID:
                coarse.append(record_from_frame(seconds, frame, templates, profile))
            else:
                non_valid.append(_reportable_state(state))
        except Exception as exc:
            non_valid.append({"video_seconds": seconds, "status": "ERROR",
                              "reason": "PIPE_OR_PROCESSING_ERROR", "details": str(exc)})
    non_valid.extend(coarse_seek_failures)
    coarse = remove_transient_ocr(coarse, ocr_corrections)
    if not coarse:
        raise SystemExit("no recognizable coarse samples")

    toc = [{k: v for k, v in item.items() if k not in {"frame", "timestamp"}}
           for item in coarse]
    jump_events = []
    stall_events = []
    refined_ranges = []
    coarse_windows = []
    for before, after in zip(coarse, coarse[1:]):
        video_delta = after["video_seconds"] - before["video_seconds"]
        osd_delta = (after["timestamp"] - before["timestamp"]).total_seconds()
        if video_delta <= 0 or abs(osd_delta - video_delta) <= TOLERANCE:
            continue
        lo = max(0.0, before["video_seconds"] - 5.0)
        hi = min(total, after["video_seconds"] + 5.0)
        refined_ranges.append([lo, hi])
        coarse_windows.append({"start": lo, "end": hi})

    # Pass 2/3/4: collect all logical windows for the pass, merge only
    # overlapping/contained windows, then reuse the existing Short Range Pipe.
    planner_passes = []
    pass2, planner = audit_planned_ranges(
        args.video, templates, coarse_windows, args.refine_step, total,
        non_valid, toc, ocr_corrections, profile,
    ) if coarse_windows else ([], {"step": args.refine_step,
                                   "logical_ranges": [], "physical_ranges": [],
                                   "logical_range_count": 0,
                                   "physical_range_count": 0, "merge_count": 0})
    planner["pass"] = f"refine_{args.refine_step:g}"
    planner_passes.append(planner)

    pass3_ranges = _ranges_from_anomalies(pass2, coarse_windows, 2.0, total)
    pass3, planner = audit_planned_ranges(
        args.video, templates, pass3_ranges, 2.0, total,
        non_valid, toc, ocr_corrections, profile,
    ) if pass3_ranges else ([], {"step": 2.0, "logical_ranges": [],
                                 "physical_ranges": [], "logical_range_count": 0,
                                 "physical_range_count": 0, "merge_count": 0})
    planner["pass"] = "refine_2"
    planner_passes.append(planner)

    pass4_ranges = _ranges_from_anomalies(pass3, pass3_ranges, 1.0, total)
    final, planner = audit_planned_ranges(
        args.video, templates, pass4_ranges, 1.0, total,
        non_valid, toc, ocr_corrections, profile,
    ) if pass4_ranges else ([], {"step": 1.0, "logical_ranges": [],
                                 "physical_ranges": [], "logical_range_count": 0,
                                 "physical_range_count": 0, "merge_count": 0})
    planner["pass"] = "refine_1"
    planner_passes.append(planner)

    pending_jump = None
    for left, right, vd, od in final:
        drift = od - vd
        if left["osd_digits"] == right["osd_digits"] and vd >= 5:
            stall_events.append({"video_start_seconds": left["video_seconds"],
                                 "video_end_seconds": right["video_seconds"],
                                 "duration_seconds": vd, "osd": left["formatted"]})
            continue
        if pending_jump is not None:
            if abs(drift + pending_jump["drift"]) <= TOLERANCE:
                pending_jump = None
                continue
            jump_events.append(save_pair(args.output, len(jump_events) + 1,
                pending_jump["left"], pending_jump["right"],
                {"video_elapsed_seconds": pending_jump["vd"],
                 "osd_elapsed_seconds": pending_jump["od"],
                 "jump_seconds": pending_jump["drift"]}))
            pending_jump = None
        if left["osd_digits"] != right["osd_digits"]:
            pending_jump = {"left": left, "right": right,
                            "vd": vd, "od": od, "drift": drift}
    if pending_jump is not None:
        jump_events.append(save_pair(args.output, len(jump_events) + 1,
            pending_jump["left"], pending_jump["right"],
            {"video_elapsed_seconds": pending_jump["vd"],
             "osd_elapsed_seconds": pending_jump["od"],
             "jump_seconds": pending_jump["drift"]}))

    jump_events = unique_events(jump_events, "before_video_seconds")
    stall_events = unique_events(stall_events, "video_start_seconds")
    toc.sort(key=lambda item: item["video_seconds"])
    # Accumulate stall duration from the consolidated timeline rather than
    # relying on whichever fixed refinement pass happened to contain a pair.
    stall_events.extend(detect_osd_stalls(toc))
    samples = args.output / "toc_samples.tsv"
    with samples.open("w", encoding="utf-8") as handle:
        handle.write("video_seconds\tosd\tthreshold\tmargin\n")
        for item in toc:
            handle.write(f"{item['video_seconds']:.3f}\t{item['formatted']}\t"
                         f"{item['threshold']}\t{item['margin']}\n")
    state_counts = {status: sum(item.get("status") == status for item in non_valid)
                    for status in ("SUSPECT", "UNKNOWN", "ERROR")}
    report = {"version": "1.4.1", "engine": "public",
              "profile_id": args.profile_id,
              "video": str(args.video), "font": str(args.font),
              "fastscan_transport": "per-sample-input-seek-rawvideo-gray-osd-crop",
              "coarse_transport": {
                  "method": "per-sample-input-seek",
                  "step": args.coarse_step,
                  "target_count": len(sampling_targets(total, args.coarse_step)),
                  "seek_count": len(sampling_targets(total, args.coarse_step)),
                  "failed_targets": len(coarse_seek_failures),
              },
              "passes": [f"{args.coarse_step:g}-second OSD per-sample Seek",
                         f"{args.refine_step:g}-second candidate refinement",
                         "2-second refinement", "1-second final audit"],
              "valid_samples": len(toc),
              "non_valid_observations": non_valid,
              "recognition_failures": [item for item in non_valid
                                       if item.get("status") == "ERROR"],
              "state_counts": {"VALID": len(toc), **state_counts},
              "stage_counts": {
                  "valid": dict(Counter(item.get("stage", "unknown") for item in toc)),
                  "suspect": dict(Counter(item.get("stage", "unreadable")
                                            for item in non_valid
                                            if item.get("status") == "SUSPECT")),
              },
              "ocr_transient_corrections": ocr_corrections,
              "refined_ranges": refined_ranges,
              "batch_planner": {
                  "merge_policy": "overlap-and-containment-only",
                  "cross_pass_merge": False,
                  "near_gap_merge": False,
                  "passes": planner_passes,
                  "merge_count": sum(item["merge_count"] for item in planner_passes),
              },
              "jumps": jump_events,
              "osd_stalls": stall_events, "toc_samples": samples.name}
    path = args.output / "time_jump_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OSD samples: {len(toc)}; jumps: {len(jump_events)}; stalls: {len(stall_events)}")
    print(f"report: {path}")


if __name__ == "__main__":
    main()
