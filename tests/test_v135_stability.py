import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import auto_tapo_split
from build_tapo_osd_glyph_templates import write_json
from tapo_osd_common import detect_osd_stalls


class V135StabilityTests(unittest.TestCase):
    def setUp(self):
        auto_tapo_split._RESERVED_OUTPUTS.clear()

    def tearDown(self):
        auto_tapo_split._RESERVED_OUTPUTS.clear()

    @staticmethod
    def _record(seconds, timestamp, value="20250101000000", margin=100):
        return {
            "video_seconds": seconds,
            "timestamp": timestamp,
            "osd_digits": value,
            "formatted": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "margin": margin,
        }

    def test_stall_is_accumulated_across_small_samples(self):
        origin = datetime(2025, 1, 1)
        records = [
            {"video_seconds": seconds, "osd_digits": "20250101000000",
             "formatted": origin.strftime("%Y-%m-%d %H:%M:%S")}
            for seconds in (0.0, 1.0, 2.0, 4.0, 6.0)
        ]
        records.append({"video_seconds": 7.0, "osd_digits": "20250101000001",
                        "formatted": (origin + timedelta(seconds=1)).strftime(
                            "%Y-%m-%d %H:%M:%S")})
        stalls = detect_osd_stalls(records)
        self.assertEqual(len(stalls), 1)
        self.assertEqual(stalls[0]["video_start_seconds"], 0.0)
        self.assertEqual(stalls[0]["video_end_seconds"], 6.0)
        self.assertEqual(stalls[0]["duration_seconds"], 6.0)

    def test_short_same_osd_span_is_not_a_stall(self):
        origin = datetime(2025, 1, 1)
        records = [
            {"video_seconds": seconds, "osd_digits": "20250101000000",
             "formatted": origin.strftime("%Y-%m-%d %H:%M:%S")}
            for seconds in (0.0, 1.0, 4.9)
        ]
        self.assertEqual(detect_osd_stalls(records), [])

    def test_parallel_output_names_are_reserved_before_ffmpeg_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "20250101_000000_1.0.mov"
            first = auto_tapo_split.unique_path(base)
            second = auto_tapo_split.unique_path(base)
            self.assertNotEqual(first, second)
            self.assertEqual(second.name, "20250101_000000_1.0_part02.mov")

    def test_failed_duration_does_not_publish_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "record.md"
            with self.assertRaises(RuntimeError):
                auto_tapo_split.commit_index(index, "should not publish", -1.1)
            self.assertFalse(index.exists())
            auto_tapo_split.commit_index(index, "completed", 0.0)
            self.assertEqual(index.read_text(encoding="utf-8"), "completed")

    def test_full_lane_internal_observation_is_characterized(self):
        origin = datetime(2025, 1, 1)
        before = (0.0, origin)
        after = (10.0, origin + timedelta(seconds=10))
        normal = [self._record(5.0, origin + timedelta(seconds=5))]
        wild = [self._record(5.0, origin + timedelta(hours=1))]
        self.assertEqual(
            len(auto_tapo_split._remove_transient_full_lane_ocr(
                normal, before, after)), 1)
        self.assertEqual(
            auto_tapo_split._remove_transient_full_lane_ocr(wild, before, after),
            [],
        )

    def test_incomplete_font_stays_partial_until_completion(self):
        def candidate(digit):
            matrix = [[0] * 40 for _ in range(64)]
            geometry = [48, 25, 300, 100, 10, 20, 10]
            return {
                "digit": digit,
                "appearance": [100, 0.1, 0.2, 30, 0.05],
                "geometry": geometry,
                "feature": [100, 0.1, 0.2, 30, 0.05] + geometry,
                "frame_height_std": 0.0,
                "threshold": 235,
                "video": "sample.mp4",
                "seconds": 1.0,
                "source": f"sample.mp4@1.00s:{digit}",
                "matrix": matrix,
            }

        candidates = {digit: [candidate(digit)] for digit in "0123456789"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "font.json"
            write_json(output, {}, candidates, [], target=1,
                       candidate_minimum=1, attempts=1, profile_id="test")
            self.assertFalse(output.exists())
            partial = Path(str(output) + ".partial")
            self.assertTrue(partial.exists())
            json.loads(partial.read_text(encoding="utf-8"))

            write_json(output, {}, candidates, [], target=1,
                       candidate_minimum=1, attempts=1, profile_id="test",
                       complete=True)
            self.assertTrue(output.exists())
            self.assertFalse(partial.exists())
            json.loads(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
