#!/usr/bin/env python3
"""Build a user-labelled, diversity-selected Elite20 font JSON.

The builder deliberately keeps collection and selection separate:

* collect at least 35 quality candidates for every digit;
* limit same-video/nearby-time duplication;
* select 20 candidates per digit using appearance, geometry, profile,
  template-shape, and source diversity;
* store Threshold Scan and raw-Luma Search geometry baselines as metadata.

The bundled runtime only needs ``digits[*].templates``.  All other fields are
evidence for geometry checks and for rebuilding the font later.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tapo_osd_recognizer import recognize
from tapo_osd_seed import load_seed, recognize_seed
from tapo_profile import ProfileError, geometry_for_profile, load_profile_by_id


BASE = Path(__file__).resolve().parent
OUTPUT_NAME = "tapo_osd_glyph_templates.json"
DEFAULT_SEED = BASE / "tapo_osd_seed.json"
CANDIDATE_MINIMUM = 35
SELECTED_PER_DIGIT = 20
MAX_CANDIDATES_PER_VIDEO_PER_DIGIT = 6
MIN_SAME_VIDEO_TIME_GAP = 10.0
ROUND_SIZE = 10
MAX_ATTEMPTS = 2000
THRESHOLDS = (225, 235, 240, 245, 250)
LUMA_THRESHOLDS = (16, 24, 32, 40, 48, 56, 64, 72,
                   80, 96, 112, 128, 144, 160)
SLOTS = ((7, 43), (50, 88), (99, 137), (148, 185),
         (247, 282), (290, 330), (391, 426), (435, 475),
         (530, 565), (579, 617), (675, 710), (724, 765),
         (819, 854), (867, 905))


def robust_summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "median": None, "mad": None, "p10": None, "p90": None}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {"count": int(len(values)), "median": median, "mad": mad,
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90))}


def _width_profile(mask):
    """Return occupied width at top/middle/bottom bands of a binary glyph."""
    ys, _ = np.where(mask > 0)
    if len(ys) == 0:
        return [0, 0, 0]
    lo, hi = int(ys.min()), int(ys.max())
    span = max(1, hi - lo + 1)
    result = []
    for start_ratio, end_ratio in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)):
        y0 = lo + int(span * start_ratio)
        y1 = lo + max(1, int(span * end_ratio))
        band = mask[y0:min(mask.shape[0], y1), :]
        xs = np.where(band > 0)[1]
        result.append(int(xs.max() - xs.min() + 1) if len(xs) else 0)
    return result


def _appearance_features(gray, components):
    occupied = np.zeros_like(gray, dtype=np.uint8)
    for x, y, w, h, _area in components:
        cv2.rectangle(occupied, (max(0, x - 2), max(0, y - 2)),
                      (min(gray.shape[1] - 1, x + w + 1),
                       min(gray.shape[0] - 1, y + h + 1)), 255, -1)
    background = gray[occupied == 0]
    if len(background) < 20:
        background = gray.reshape(-1)
    edges = cv2.Canny(gray, 40, 120)
    return [
        float(np.median(background)),
        float(np.mean(background < 80)),
        float(np.mean(background > 200)),
        float(np.percentile(background, 90) - np.percentile(background, 10)),
        float(np.mean(edges[occupied == 0] > 0)) if np.any(occupied == 0) else 0.0,
    ]


def extract_glyphs(gray, profile=None):
    """Extract 14 threshold glyphs plus raw geometry metadata."""
    geometry = geometry_for_profile(profile)
    slots = geometry["slot_ranges"]
    left_bound = min(left for left, _right in slots)
    right_bound = max(right for _left, right in slots)
    candidates = []
    for threshold in THRESHOLDS:
        mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        comps = []
        for i in range(1, n):
            x, y, w, h, area = map(int, stats[i])
            if left_bound <= x and x + w <= right_bound and 35 <= h <= 60 \
                    and 5 <= w <= 35 and area >= 120:
                comps.append((x, y, w, h, area))
        if len(comps) == 14:
            candidates.append((float(np.std([c[3] for c in comps])),
                               threshold, mask, sorted(comps)))
    if not candidates:
        return None
    _, threshold, mask, components = min(candidates, key=lambda item: item[0])
    glyphs, geometry = [], []
    for x, y, w, h, area in components:
        piece = mask[max(0, y - 2):min(gray.shape[0], y + h + 2),
                     max(0, x - 2):min(gray.shape[1], x + w + 2)]
        ys, xs = np.where(piece > 0)
        if len(xs) == 0:
            return None
        glyph = piece[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        scale = min(56 / glyph.shape[0], 36 / glyph.shape[1])
        nw, nh = max(1, round(glyph.shape[1] * scale)), max(1, round(glyph.shape[0] * scale))
        glyph = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((64, 40), dtype=np.uint8)
        canvas[(64 - nh) // 2:(64 - nh) // 2 + nh,
               (40 - nw) // 2:(40 - nw) // 2 + nw] = glyph > 0
        glyphs.append(canvas)
        profile = _width_profile(piece)
        geometry.append({
            "x": x, "y": y, "width": w, "height": h, "area": area,
            "top_width": profile[0], "middle_width": profile[1],
            "bottom_width": profile[2],
        })
    return {"glyphs": glyphs, "threshold": int(threshold),
            "components": geometry, "appearance": _appearance_features(gray, components),
            "height_std": float(np.std([item["height"] for item in geometry]))}


def _luma_bbox(gray, left, right):
    roi = gray[:, left:right]
    candidates = []
    for threshold in LUMA_THRESHOLDS:
        dark = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        for i in range(1, n):
            x, y, w, h, area = map(int, stats[i])
            if 25 <= h <= 62 and 5 <= w <= 35 and area >= 16:
                score = abs(h - 49) + abs(w - 27) - min(area, 300) / 120
                candidates.append((score, x, y, w, h, area))
    return min(candidates)[1:] if candidates else None


def luma_geometry(gray, profile=None):
    result = []
    for left, right in geometry_for_profile(profile)["slot_ranges"]:
        bbox = _luma_bbox(gray, left, right)
        if bbox is None:
            return None
        x, y, w, h, area = bbox
        result.append({"x": left + x, "y": y, "width": w, "height": h, "area": area})
    return result


def choose_video_frame(video, seconds):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = cap.read()
    actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cap.release()
    return (frame, actual) if ok else (None, actual)


def _feature_vector(candidate):
    return np.asarray(candidate.get("feature", [0] * 12), dtype=float)


def _normalizer(pool):
    matrix = np.vstack([_feature_vector(item) for item in pool])
    lo = np.percentile(matrix, 5, axis=0)
    hi = np.percentile(matrix, 95, axis=0)
    return lo, np.maximum(hi - lo, 1.0)


def _template_distance(left, right):
    a = np.asarray(left["matrix"], dtype=np.uint8)
    b = np.asarray(right["matrix"], dtype=np.uint8)
    return float(np.count_nonzero(a != b)) / float(a.size)


def select_elite(candidates, selected_count=SELECTED_PER_DIGIT):
    """Select quality-seeded, max-min diverse candidates."""
    if len(candidates) <= selected_count:
        return list(candidates)
    lo, scale = _normalizer(candidates)
    normalized = [np.clip((_feature_vector(item) - lo) / scale, -2, 2)
                  for item in candidates]

    def quality(item):
        geometry = item.get("geometry", [48, 25, 0, 0, 0, 0, 0])
        height, width = geometry[:2]
        return -(abs(height - 48) + abs(width - 25) + item.get("frame_height_std", 0) * 2)

    first = max(range(len(candidates)), key=lambda i: (quality(candidates[i]),
                                                        candidates[i].get("source", "")))
    selected_indices = [first]
    remaining = set(range(len(candidates))) - {first}
    while len(selected_indices) < selected_count and remaining:
        best_index, best_score = None, -float("inf")
        for index in remaining:
            feature_distance = min(float(np.linalg.norm(normalized[index] - normalized[j]))
                                   for j in selected_indices)
            shape_distance = min(_template_distance(candidates[index], candidates[j])
                                 for j in selected_indices)
            source_bonus = 0.0 if any(candidates[index].get("video") == candidates[j].get("video")
                                     for j in selected_indices) else 0.35
            score = 0.65 * feature_distance + 0.25 * shape_distance + source_bonus
            score += 0.10 * max(0.0, quality(candidates[index])) / 50.0
            if score > best_score:
                best_index, best_score = index, score
        selected_indices.append(best_index)
        remaining.remove(best_index)
    return [candidates[index] for index in selected_indices]


def _baseline_for_frames(frames, key):
    valid = [frame[key] for frame in frames if frame.get(key)]
    if not valid:
        return {"sample_frames": 0}
    heights = [[item["height"] for item in frame] for frame in valid]
    xs = [[item["x"] for item in frame] for frame in valid]
    ys = [[item["y"] for item in frame] for frame in valid]
    return {
        "sample_frames": len(valid),
        "height": robust_summary(np.asarray(heights).reshape(-1)),
        "height_dispersion": robust_summary([np.std(row) for row in heights]),
        "position_x": [robust_summary([row[i] for row in xs]) for i in range(14)],
        "position_y": [robust_summary([row[i] for row in ys]) for i in range(14)],
    }


def build_geometry_baseline(frame_records):
    return {
        "description": "Threshold ScanとLuma Searchの正常geometry基準",
        "threshold_scan": _baseline_for_frames(frame_records, "threshold_geometry"),
        "luma_search": _baseline_for_frames(frame_records, "luma_geometry"),
    }


def write_json(path, labels, candidates, frame_records, target=SELECTED_PER_DIGIT,
               candidate_minimum=CANDIDATE_MINIMUM, attempts=0, profile_id=None):
    selected = {digit: select_elite(candidates[digit], target) for digit in "0123456789"}
    data = {
        "format": "tapo_osd_glyph_templates_v1",
        "builder": "EliteFontBuilder",
        "builder_version": "1.3.0",
        "profile_id": profile_id,
        "label_source": "user-confirmed OSD strings from recordings",
        "label_assistant": "tapo_osd_seed.json for provisional labels; user confirmation for ambiguous frames",
        "candidate_minimum_per_digit": candidate_minimum,
        "selected_per_digit": target,
        "max_candidates_per_video_per_digit": MAX_CANDIDATES_PER_VIDEO_PER_DIGIT,
        "min_same_video_time_gap_seconds": MIN_SAME_VIDEO_TIME_GAP,
        "selection_axes": ["background_median", "dark_ratio", "bright_ratio",
                           "local_contrast", "edge_ratio", "glyph_height",
                           "glyph_width", "glyph_area", "glyph_x", "top_width",
                           "middle_width", "bottom_width", "template_hamming_distance",
                           "source_video", "source_time"],
        "selection_strategy": "quality_seed_plus_max_min_diversity",
        "attempts": attempts,
        "geometry_baseline": build_geometry_baseline(frame_records),
        "labels": labels,
        "digits": {
            digit: {
                "sample_count": len(selected[digit]),
                "candidate_count": len(candidates[digit]),
                "templates": [item["matrix"] for item in selected[digit]],
                "template_sources": [item["source"] for item in selected[digit]],
                "template_metadata": [
                    {key: value for key, value in item.items() if key != "matrix"}
                    for item in selected[digit]
                ],
            }
            for digit in "0123456789"
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_existing(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels", {})
    candidates = defaultdict(list)
    for digit in "0123456789":
        bucket = data.get("digits", {}).get(digit, {})
        matrices = bucket.get("templates", [])
        sources = bucket.get("template_sources", [])
        metadata = bucket.get("template_metadata", [])
        for index, matrix in enumerate(matrices):
            item = dict(metadata[index]) if index < len(metadata) else {}
            item.update({"matrix": matrix,
                         "source": sources[index] if index < len(sources) else "existing",
                         "method": "existing"})
            item.setdefault("appearance", item.get("feature", [0, 0, 0, 0, 0])[:5])
            item.setdefault("geometry", item.get("feature", [48, 25, 0, 0, 0, 0, 0])[5:12])
            item.setdefault("feature", item.get("appearance", []) + item.get("geometry", []))
            candidates[digit].append(item)
    return labels, candidates


def valid_label(answer):
    if len(answer) != 14 or not answer.isdigit():
        return False
    try:
        datetime.strptime(answer, "%Y%m%d%H%M%S")
    except ValueError:
        return False
    return True


def load_label_templates(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {digit: [np.asarray(matrix, dtype=np.uint8)
                    for matrix in data["digits"][digit]["templates"]]
            for digit in "0123456789"}


def suggest_label(glyphs, templates):
    digits, margins = [], []
    for glyph in glyphs:
        digit, _distance, margin = recognize(glyph, templates)
        digits.append(digit)
        margins.append(margin)
    return "".join(digits), min(margins)


def suggest_seed_label(glyphs, seed):
    digits, margins = [], []
    for glyph in glyphs:
        digit, _distance, margin = recognize_seed(glyph, seed)
        digits.append(digit)
        margins.append(margin)
    return "".join(digits), min(margins)


def main():
    parser = argparse.ArgumentParser(description="Build a diversity-selected Elite20 font JSON")
    parser.add_argument("--input-dir", type=Path, help="録画データフォルダ")
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_NAME))
    parser.add_argument("--candidate-minimum", type=int, default=CANDIDATE_MINIMUM,
                        help="数字ごとに集める候補数（default: 35）")
    parser.add_argument("--selected-per-digit", type=int, default=SELECTED_PER_DIGIT,
                        help="数字ごとの最終Elite数（default: 20）")
    parser.add_argument("--round-size", type=int, default=ROUND_SIZE)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="既存JSONを候補として使わない")
    parser.add_argument("--auto-label-font", type=Path,
                        help="高信頼フレームの仮ラベルに使う既存font JSON")
    parser.add_argument("--auto-label-min-margin", type=float, default=220.0,
                        help="このmargin以上なら確認なしで仮ラベル採用（default: 220）")
    parser.add_argument("--seed-data", type=Path, default=DEFAULT_SEED,
                        help="初回ラベル候補に使う粗い構造Seed Data JSON")
    parser.add_argument("--seed-min-margin", type=float, default=0.08,
                        help="Seedの仮ラベルを自動採用する最小margin（default: 0.08）")
    parser.add_argument("--no-seed", action="store_true",
                        help="Seed Dataによる仮ラベル提案を無効にする")
    parser.add_argument("--profile-id",
                        help="生成するFontを紐づけるProfile ID")
    parser.add_argument("--profile-file", type=Path,
                        default=BASE / "tapo_profiles.json",
                        help="Profile定義JSON")
    args = parser.parse_args()
    profile = None
    if args.profile_id:
        try:
            profile = load_profile_by_id(args.profile_file, args.profile_id,
                                         require_font=False)
        except ProfileError as exc:
            raise SystemExit(str(exc)) from exc
    if args.candidate_minimum < args.selected_per_digit or args.selected_per_digit < 1:
        raise SystemExit("candidate-minimumはselected-per-digit以上にしてください")
    rng = random.Random(args.seed)
    label_templates = (load_label_templates(args.auto_label_font)
                       if args.auto_label_font else None)
    seed_data = None
    if not args.no_seed and args.seed_data.exists():
        try:
            seed_data = load_seed(args.seed_data)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Seed Dataを読めません: {exc}")

    labels, candidates = {}, defaultdict(list)
    if args.output.exists() and not args.fresh:
        answer = input(f"{args.output}を候補として引き継ぎますか？ [Y/n] ").strip().lower()
        if answer not in {"", "y", "yes"}:
            args.fresh = True
        else:
            try:
                labels, candidates = load_existing(args.output)
            except Exception as exc:
                raise SystemExit(f"既存JSONを読めません: {exc}")

    folder = (args.input_dir.expanduser() if args.input_dir else
              Path(input("録画データフォルダのパス: ").strip()).expanduser())
    videos = sorted(folder.rglob("*.mp4"))
    if not videos:
        raise SystemExit("MP4が見つかりません。録画フォルダを確認してください。")
    sample_dir = args.output.parent / "tapo_training_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    per_video_digit = Counter()
    last_times = defaultdict(list)
    frame_records = []
    attempts = 0

    while attempts < args.max_attempts:
        counts = {digit: len(candidates[digit]) for digit in "0123456789"}
        if all(counts[digit] >= args.candidate_minimum for digit in "0123456789"):
            write_json(args.output, labels, candidates, frame_records,
                       args.selected_per_digit, args.candidate_minimum, attempts,
                       args.profile_id)
            print(f"完成: {args.output}")
            print("候補数:", counts)
            print("最終Elite数:", {digit: args.selected_per_digit for digit in "0123456789"})
            return

        for _ in range(args.round_size):
            if attempts >= args.max_attempts:
                break
            missing = [digit for digit in "0123456789"
                       if len(candidates[digit]) < args.candidate_minimum]
            eligible = [video for video in videos
                        if any(per_video_digit[(video.name, digit)] < MAX_CANDIDATES_PER_VIDEO_PER_DIGIT
                               for digit in missing)]
            if not eligible:
                eligible = videos
            attempts += 1
            video = rng.choice(eligible)
            cap = cv2.VideoCapture(str(video))
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1.0)
            cap.release()
            frame, seconds = choose_video_frame(video, rng.uniform(0, max(0.1, duration - 0.1)))
            if frame is None:
                continue
            geometry = geometry_for_profile(profile)
            roi = geometry["roi"]
            strip = frame[roi["y"]:roi["y"] + roi["height"],
                          roi["x"]:roi["x"] + roi["width"]]
            gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            result = extract_glyphs(gray, profile)
            if result is None:
                continue

            label_path = sample_dir / f"sample_{len(labels):05d}_{video.stem}_{seconds:.2f}s.png"
            cv2.imwrite(str(label_path), strip)
            print(f"\n[{attempts}/{args.max_attempts}] {label_path}")
            answer = None
            if label_templates is not None:
                suggestion, suggestion_margin = suggest_label(result["glyphs"], label_templates)
                if valid_label(suggestion) and suggestion_margin >= args.auto_label_min_margin:
                    answer = suggestion
                    print(f"仮ラベル自動採用: {answer} (margin={suggestion_margin:.1f})")
                else:
                    print(f"要確認: 仮ラベル={suggestion} margin={suggestion_margin:.1f}")
            elif seed_data is not None:
                suggestion, suggestion_margin = suggest_seed_label(result["glyphs"], seed_data)
                if valid_label(suggestion) and suggestion_margin >= args.seed_min_margin:
                    answer = suggestion
                    print(f"Seed仮ラベル自動採用: {answer} (margin={suggestion_margin:.3f})")
                else:
                    print(f"要確認: Seed仮ラベル={suggestion} margin={suggestion_margin:.3f}")
            if answer is None:
                answer = input("OSD正解文字列 (YYYYMMDDHHMMSS / Enter=skip / q=abort): ").strip()
            if answer.lower() == "q":
                write_json(args.output, labels, candidates, frame_records,
                           args.selected_per_digit, args.candidate_minimum, attempts,
                           args.profile_id)
                raise SystemExit("ユーザー指定でabortしました。途中候補を保存しました。")
            if not valid_label(answer):
                continue

            key = label_path.name
            labels[key] = {"osd": answer, "video": str(video), "seconds": seconds,
                           "threshold": result["threshold"]}
            gray_geometry = luma_geometry(gray, profile)
            frame_records.append({
                "source": key, "video": video.name, "seconds": seconds,
                "threshold_geometry": result["components"],
                "luma_geometry": gray_geometry,
            })
            added_digits = set()
            for slot, digit in enumerate(answer):
                if digit in added_digits or len(candidates[digit]) >= args.candidate_minimum:
                    continue
                if per_video_digit[(video.name, digit)] >= MAX_CANDIDATES_PER_VIDEO_PER_DIGIT:
                    continue
                if any(abs(seconds - old) < MIN_SAME_VIDEO_TIME_GAP
                       for old in last_times[(video.name, digit)]):
                    continue
                geometry = result["components"][slot]
                feature = result["appearance"] + [geometry["height"], geometry["width"],
                           geometry["area"], geometry["x"], geometry["top_width"],
                           geometry["middle_width"], geometry["bottom_width"]]
                item = {
                    "digit": digit, "appearance": result["appearance"],
                    "geometry": [geometry["height"], geometry["width"], geometry["area"],
                                 geometry["x"], geometry["top_width"],
                                 geometry["middle_width"], geometry["bottom_width"]],
                    "feature": feature, "frame_height_std": result["height_std"],
                    "threshold": result["threshold"], "video": video.name,
                    "seconds": seconds, "source": f"{video.name}@{seconds:.2f}s",
                    "matrix": result["glyphs"][slot].tolist(),
                }
                candidates[digit].append(item)
                per_video_digit[(video.name, digit)] += 1
                last_times[(video.name, digit)].append(seconds)
                added_digits.add(digit)
            write_json(args.output, labels, candidates, frame_records,
                       args.selected_per_digit, args.candidate_minimum, attempts,
                       args.profile_id)
            print("候補数:", {digit: len(candidates[digit]) for digit in "0123456789"})

    write_json(args.output, labels, candidates, frame_records,
               args.selected_per_digit, args.candidate_minimum, attempts,
               args.profile_id)
    counts = {digit: len(candidates[digit]) for digit in "0123456789"}
    raise SystemExit(f"候補収集を終了しました。max-attempts到達: {counts}")


if __name__ == "__main__":
    main()
