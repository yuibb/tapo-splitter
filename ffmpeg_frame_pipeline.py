#!/usr/bin/env python3
"""Tapo frame pipeline used by the public rev2.5d-based detector."""

import argparse
import json
import re
import subprocess
import os
from pathlib import Path

import cv2
import numpy as np

from tapo_osd_common import load_templates, recognize_osd, save_pair, detect_osd_stalls
BASE = Path(__file__).resolve().parent
FONT_PATH = BASE / "tapo_osd_glyph_templates.json"

STEP = 15.0
TOLERANCE = 3.0
STALL_SECONDS = 5.0


def ffmpeg_command(*args):
    command = ["ffmpeg"]
    if os.environ.get("Tapo_VIDEOTOOLBOX") == "1":
        command += ["-hwaccel", "videotoolbox"]
    return command + list(args)


def probe_duration(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
                         check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def jpeg_frames(data):
    start = 0
    while True:
        start = data.find(b"\xff\xd8", start)
        if start < 0:
            return
        end = data.find(b"\xff\xd9", start + 2)
        if end < 0:
            return
        yield data[start:end + 2]
        start = end + 2


def coarse_frames(video, step=STEP):
    command = ffmpeg_command(
        "-v", "info", "-copyts", "-i", str(video), "-an", "-sn",
        "-vf", f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{step})',showinfo",
        "-fps_mode", "vfr",
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    )
    result = subprocess.run(command, check=True, capture_output=True)
    pts = [float(x) for x in re.findall(rb"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)]
    frames = list(jpeg_frames(result.stdout))
    if len(pts) != len(frames):
        raise RuntimeError(f"coarse pipe count mismatch: pts={len(pts)} frames={len(frames)}")
    for seconds, data in zip(pts, frames):
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            yield seconds, image


def range_frames(video, start, end, step):
    """Read a bounded time range through one ffmpeg pipe at a fixed interval."""
    command = ffmpeg_command(
        "-v", "info", "-copyts", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", str(video), "-an", "-sn",
        "-vf", f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{step})',showinfo",
        "-fps_mode", "vfr", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    )
    result = subprocess.run(command, check=True, capture_output=True)
    pts = [float(x) for x in re.findall(rb"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)]
    frames = list(jpeg_frames(result.stdout))
    if len(pts) != len(frames):
        raise RuntimeError(f"range pipe count mismatch: pts={len(pts)} frames={len(frames)}")
    for seconds, data in zip(pts, frames):
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            yield seconds, image


def record_from_frame(seconds, frame, templates):
    timestamp, value, threshold, margin = recognize_osd(frame, templates)
    return {"video_seconds": seconds, "osd_digits": value,
            "formatted": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": threshold, "margin": margin,
            "frame": frame, "timestamp": timestamp}


def seek_record(video, seconds, templates):
    command = ffmpeg_command("-v", "info", "-copyts", "-ss", f"{seconds:.3f}", "-i", str(video),
               "-vf", "showinfo", "-frames:v", "1", "-f", "image2pipe",
               "-vcodec", "mjpeg", "pipe:1")
    result = subprocess.run(command, check=True, capture_output=True)
    match = re.search(rb"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)
    image = next(jpeg_frames(result.stdout), None)
    if not match or image is None:
        raise RuntimeError(f"could not obtain PTS/frame at {seconds:.3f}s")
    frame = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    return record_from_frame(float(match.group(1)), frame, templates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--font", type=Path, default=FONT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    templates = load_templates(args.font)
    total = probe_duration(args.video)
    coarse = []
    failures = []
    for seconds, frame in coarse_frames(args.video):
        try:
            coarse.append(record_from_frame(seconds, frame, templates))
        except Exception as exc:
            failures.append({"video_seconds": seconds, "error": str(exc)})
    if not coarse:
        raise SystemExit("no recognizable coarse samples")

    records = [{k: v for k, v in r.items() if k not in {"frame", "timestamp"}} for r in coarse]
    jumps = []
    stalls = []
    expanded = []
    for before, after in zip(coarse, coarse[1:]):
        expanded.append(before)
        gap = after["video_seconds"] - before["video_seconds"]
        # A dropped coarse sample must not be interpreted as a 10-second pair.
        # Follow the actual PTS and fill large gaps before comparing OSD deltas.
        probe_time = before["video_seconds"] + STEP
        while probe_time < after["video_seconds"] - 1.0:
            try:
                expanded.append(seek_record(args.video, probe_time, templates))
            except Exception as exc:
                failures.append({"video_seconds": probe_time, "error": str(exc)})
            probe_time += STEP
    expanded.append(coarse[-1])

    for before, after in zip(expanded, expanded[1:]):
        video_delta = after["video_seconds"] - before["video_seconds"]
        if video_delta <= 0:
            continue
        osd_delta = (after["timestamp"] - before["timestamp"]).total_seconds()
        drift = osd_delta - video_delta
        if abs(drift) <= TOLERANCE:
            continue
        if after["osd_digits"] == before["osd_digits"]:
            resume = after["video_seconds"]
            while resume < total - 0.1:
                resume += 2.0
                try:
                    probe = seek_record(args.video, resume, templates)
                except Exception as exc:
                    failures.append({"video_seconds": resume, "error": str(exc)})
                    break
                if probe["osd_digits"] != before["osd_digits"]:
                    resume = probe["video_seconds"]
                    break
            stalls.append({"video_start_seconds": before["video_seconds"],
                           "video_end_seconds": min(resume, total),
                           "duration_seconds": min(resume, total) - before["video_seconds"],
                           "osd": before["formatted"]})
        else:
            delta = {"video_elapsed_seconds": video_delta,
                     "osd_elapsed_seconds": osd_delta, "jump_seconds": drift}
            jumps.append(save_pair(args.output, len(jumps) + 1, before, after, delta))

    samples_path = args.output / "osd_samples.tsv"
    with samples_path.open("w", encoding="utf-8") as handle:
        handle.write("video_seconds\tosd\tthreshold\tmargin\n")
        for item in records:
            handle.write(f"{item['video_seconds']:.3f}\t{item['formatted']}\t"
                         f"{item['threshold']}\t{item['margin']}\n")
    report = {"video": str(args.video), "font": str(args.font),
              "sample_interval_seconds": STEP, "jump_tolerance_seconds": TOLERANCE,
              "valid_samples": len(records), "recognition_failures": failures,
              "jumps": jumps, "osd_stall_threshold_seconds": STALL_SECONDS,
              "osd_stalls": stalls, "samples_tsv": samples_path.name}
    path = args.output / "time_jump_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pipe samples: {len(records)}; jumps: {len(jumps)}; stalls: {len(stalls)}")
    print(f"report: {path}")


if __name__ == "__main__":
    main()
