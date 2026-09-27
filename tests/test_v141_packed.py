#!/usr/bin/env python3
"""Regression tests for the v1.4.1 packed Hamming implementation."""

import unittest
from pathlib import Path

import numpy as np

from tapo_osd_common import load_templates
from tapo_osd_recognizer import (
    _packed_templates,
    _prepared_templates,
    _shifted_variants,
    recognize,
)


def reference_matrix(glyph, templates):
    """The v1.4.0 uint8 comparison, kept only as a test oracle."""
    digits, stacks, bank = _prepared_templates(templates)
    shifted = _shifted_variants(glyph)
    if bank is not None:
        return np.count_nonzero(
            shifted[:, None, None, :, :] != bank[None, :, :, :, :],
            axis=(3, 4),
        )
    return [np.count_nonzero(
        shifted[:, None, :, :] != stack[None, :, :, :],
        axis=(2, 3),
    ) for stack in stacks]


def packed_matrix(glyph, templates):
    _digits, packed_stacks, packed_bank = _packed_templates(templates)
    shifted = _shifted_variants(glyph)
    packed_shifted = np.packbits(
        shifted.reshape(shifted.shape[0], -1), axis=1, bitorder="big"
    )
    popcount = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)
    if packed_bank is not None:
        return popcount[np.bitwise_xor(
            packed_shifted[:, None, None, :], packed_bank[None, :, :, :]
        )].sum(axis=-1, dtype=np.int32)
    return [popcount[np.bitwise_xor(
        packed_shifted[:, None, :], stack[None, :, :]
    )].sum(axis=-1, dtype=np.int32) for stack in packed_stacks]


def result_from_matrix(matrix, digits):
    if isinstance(matrix, list):
        best_by_digit = [int(np.min(item)) for item in matrix]
    else:
        best_by_digit = [int(np.min(matrix[:, index, :]))
                         for index in range(matrix.shape[1])]
    scores = sorted(zip(best_by_digit, digits))
    best, digit = scores[0]
    margin = scores[1][0] - best if len(scores) > 1 else None
    return digit, best, margin


class PackedHammingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = load_templates(
            Path(__file__).parents[1] / "tapo_osd_glyph_templates.json"
        )

    def assert_matrix_equal(self, glyph, templates):
        old = reference_matrix(glyph, templates)
        new = packed_matrix(glyph, templates)
        if isinstance(old, list):
            self.assertEqual(len(old), len(new))
            for expected, actual in zip(old, new):
                np.testing.assert_array_equal(expected, actual)
        else:
            np.testing.assert_array_equal(old, new)

    def test_all_templates_and_random_glyphs_have_identical_distances(self):
        rng = np.random.default_rng(20260927)
        glyphs = [glyph for digit in sorted(self.templates)
                  for glyph in self.templates[digit]]
        glyphs.extend(rng.integers(0, 2, size=(20, 64, 40), dtype=np.uint8))
        for glyph in glyphs:
            self.assert_matrix_equal(glyph, self.templates)
            self.assertEqual(
                recognize(glyph, self.templates),
                result_from_matrix(packed_matrix(glyph, self.templates),
                                   tuple(sorted(self.templates))),
            )

    def test_uneven_template_counts_keep_legacy_behavior(self):
        templates = {"0": self.templates["0"][:3],
                     "1": self.templates["1"][:5]}
        glyph = self.templates["0"][0]
        self.assert_matrix_equal(glyph, templates)
        self.assertEqual(recognize(glyph, templates),
                         result_from_matrix(packed_matrix(glyph, templates),
                                            tuple(sorted(templates))))


if __name__ == "__main__":
    unittest.main()
