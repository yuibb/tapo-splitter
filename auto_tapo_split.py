#!/usr/bin/env python3
"""Tapo production splitter using Ver 1.0 (15→5→2→1 range-pipe) detection.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock


BASE = Path(__file__).resolve().parent
DETECTOR = BASE / "detect_tapo_time_jumps.py"
FONT = BASE / "tapo_osd_glyph_templates.json"
SPLITTER_VERSION = "1.0"
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


def make_segments(report, samples, duration):
    cuts = sorted(
        (float(item["before_video_seconds"]) + float(item["after_video_seconds"])) / 2
        for item in report.get("jumps", [])
    )
    stalls = []
    for item in report.get("osd_stalls", []):
        start = float(item["video_start_seconds"]) + 5.0
        end = min(float(item["video_end_seconds"]), duration)
        if end > start:
            stalls.append({"start": start, "end": end, "osd": item["osd"]})
    edges = sorted({0.0, duration, *cuts,
                    *(item["start"] for item in stalls),
                    *(item["end"] for item in stalls)})
    jump_starts = [
        datetime.strptime(item["after_osd"], "%Y-%m-%d %H:%M:%S")
        for item in report.get("jumps", [])
    ]
    segments = []
    for index, (start, end) in enumerate(zip(edges, edges[1:]), 1):
        matching_stall = next(
            (item for item in stalls if start >= item["start"] - 0.001
             and end <= item["end"] + 0.001), None
        )
        jump_index = next(
            (i for i, cut in enumerate(cuts) if abs(cut - start) < 0.001), None
        )
        if matching_stall:
            osd_start = datetime.strptime(matching_stall["osd"], "%Y-%m-%d %H:%M:%S")
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
            "is_error": matching_stall is not None,
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


def process_video(video, output_root, python_executable, font):
    print(f"processing: {video.name}", flush=True)
    work = output_root / f".work_{video.stem}"
    work.mkdir(parents=True, exist_ok=True)
    report_dir = work / "jump_report"
    run([python_executable, str(DETECTOR), str(video), "--font", str(font),
         "--output", str(report_dir), "--coarse-step", "15",
         "--refine-step", "5"])

    report = json.loads((report_dir / "time_jump_report.json").read_text(encoding="utf-8"))
    samples = read_samples(report_dir / "toc_samples.tsv")
    source_duration = probe_duration(video)
    segments = make_segments(report, samples, source_duration)
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
        f"- 使用スプリッター: `{SPLITTER_VERSION}`（15→5→2→1秒のrange-pipe方式）",
        f"- 元ファイル時間: `{source_duration:.3f} 秒`",
        f"- 分割後合計時間: `{output_total:.3f} 秒`",
        f"- 差分（分割後−元）: `{delta:+.3f} 秒`",
        f"- シーケンス判定: **{verdict}**（許容値 ±{tolerance:.1f}秒）", "",
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
                        python_executable, args.font): video
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
