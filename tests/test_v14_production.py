#!/usr/bin/env python3
"""Small regression tests for the v1.4 acquisition/planner changes."""

import unittest

from detect_tapo_time_jumps import plan_ranges


class BatchPlannerTests(unittest.TestCase):
    def test_overlap_and_containment_merge(self):
        planned = plan_ranges([
            {"start": 590, "end": 620},
            {"start": 610, "end": 640},
            {"start": 600, "end": 615},
        ])
        self.assertEqual(planned, [{"start": 590.0, "end": 640.0,
                                    "source_indices": [0, 2, 1]}])

    def test_overlap_chain_merges(self):
        planned = plan_ranges([
            {"start": 895, "end": 915},
            {"start": 910, "end": 940},
            {"start": 935, "end": 970},
        ])
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0]["start"], 895.0)
        self.assertEqual(planned[0]["end"], 970.0)
        self.assertEqual(sorted(planned[0]["source_indices"]), [0, 1, 2])

    def test_disjoint_ranges_stay_separate(self):
        planned = plan_ranges([
            {"start": 590, "end": 620},
            {"start": 900, "end": 930},
        ])
        self.assertEqual(len(planned), 2)

    def test_passes_are_planned_independently(self):
        five_second = plan_ranges([
            {"start": 590, "end": 620},
            {"start": 610, "end": 640},
        ])
        one_second = plan_ranges([
            {"start": 590, "end": 600},
            {"start": 599, "end": 605},
        ])
        self.assertEqual(len(five_second), 1)
        self.assertEqual(len(one_second), 1)
        # The production caller invokes the planner once per pass, so these
        # different sampling intervals can never be merged together.


if __name__ == "__main__":
    unittest.main()
