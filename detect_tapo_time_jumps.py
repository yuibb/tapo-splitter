#!/usr/bin/env python3
"""Tapo OSD time-jump detector with temporal OCR correction and range-pipe audit."""

import argparse
import json
from pathlib import Path

from tapo_osd_common import load_templates, save_pair
from ffmpeg_frame_pipeline import (
    TOLERANCE, coarse_osd_frames, range_frames, probe_duration, record_from_frame,
)

STEP = 15.0


def unique_events(events, key):
    out = []
    for event in sorted(events, key=lambda item: item[key]):
        if out and abs(event[key] - out[-1][key]) < 3:
            continue
        out.append(event)
    return out


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


def audit_range(video, templates, lo, hi, step, total, failures, toc, corrections):
    items = []
    try:
        for seconds, frame in range_frames(video, lo, hi, step):
            try:
                item = record_from_frame(seconds, frame, templates)
                items.append(item)
                toc.append({k: v for k, v in item.items() if k not in {"frame", "timestamp"}})
            except Exception as exc:
                failures.append({"video_seconds": seconds, "error": str(exc)})
    except Exception as exc:
        failures.append({"video_seconds": lo, "error": str(exc)})
    items = remove_transient_ocr(items, corrections)
    anomalies = []
    for left, right in zip(items, items[1:]):
        vd = right["video_seconds"] - left["video_seconds"]
        if vd <= 0:
            continue
        od = (right["timestamp"] - left["timestamp"]).total_seconds()
        if abs(od - vd) > TOLERANCE:
            anomalies.append((left, right, vd, od))
    return items, anomalies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coarse-step", type=float, default=STEP)
    parser.add_argument("--refine-step", type=float, default=5.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    templates = load_templates(args.font)
    total = probe_duration(args.video)
    coarse = []
    failures = []
    ocr_corrections = []
    for seconds, frame in coarse_osd_frames(args.video, step=args.coarse_step):
        try:
            coarse.append(record_from_frame(seconds, frame, templates))
        except Exception as exc:
            failures.append({"video_seconds": seconds, "error": str(exc)})
    coarse = remove_transient_ocr(coarse, ocr_corrections)
    if not coarse:
        raise SystemExit("no recognizable coarse samples")

    toc = [{k: v for k, v in item.items() if k not in {"frame", "timestamp"}}
           for item in coarse]
    jump_events = []
    stall_events = []
    refined_ranges = []
    for before, after in zip(coarse, coarse[1:]):
        video_delta = after["video_seconds"] - before["video_seconds"]
        osd_delta = (after["timestamp"] - before["timestamp"]).total_seconds()
        if video_delta <= 0 or abs(osd_delta - video_delta) <= TOLERANCE:
            continue
        lo = max(0.0, before["video_seconds"] - 5.0)
        hi = min(total, after["video_seconds"] + 5.0)
        refined_ranges.append([lo, hi])
        # Pass 2/3/4: only shrink and rescan ranges that remain anomalous.
        _, pass2 = audit_range(args.video, templates, lo, hi, args.refine_step, total, failures, toc, ocr_corrections)
        pass3_ranges = [(max(lo, a[0]["video_seconds"] - 2),
                         min(hi, a[1]["video_seconds"] + 2)) for a in pass2]
        pass3 = []
        for a, b in pass3_ranges:
            _, found = audit_range(args.video, templates, a, b, 2.0, total, failures, toc, ocr_corrections)
            pass3.extend(found)
        pass4_ranges = [(max(lo, a[0]["video_seconds"] - 1),
                         min(hi, a[1]["video_seconds"] + 1)) for a in pass3]
        final = []
        for a, b in pass4_ranges:
            _, found = audit_range(args.video, templates, a, b, 1.0, total, failures, toc, ocr_corrections)
            final.extend(found)
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
    samples = args.output / "toc_samples.tsv"
    with samples.open("w", encoding="utf-8") as handle:
        handle.write("video_seconds\tosd\tthreshold\tmargin\n")
        for item in toc:
            handle.write(f"{item['video_seconds']:.3f}\t{item['formatted']}\t"
                         f"{item['threshold']}\t{item['margin']}\n")
    report = {"version": "1.2.0", "engine": "public",
              "video": str(args.video), "font": str(args.font),
              "fastscan_transport": "rawvideo-gray-osd-crop",
              "passes": [f"{args.coarse_step:g}-second OSD rawvideo FastScan",
                         f"{args.refine_step:g}-second candidate refinement",
                         "2-second refinement", "1-second final audit"],
              "valid_samples": len(toc), "recognition_failures": failures,
              "ocr_transient_corrections": ocr_corrections,
              "refined_ranges": refined_ranges, "jumps": jump_events,
              "osd_stalls": stall_events, "toc_samples": samples.name}
    path = args.output / "time_jump_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OSD samples: {len(toc)}; jumps: {len(jump_events)}; stalls: {len(stall_events)}")
    print(f"report: {path}")


if __name__ == "__main__":
    main()
