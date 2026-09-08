"""Bootstrap Seed Data matching for the first-run font builder.

Seed Data is deliberately a coarse structural description, not a font.  It
is used only to suggest a label for a newly extracted glyph; ambiguous labels
are still sent to the user for confirmation.
"""

import json
from pathlib import Path

import numpy as np


def _coarse_features(matrix):
    """Return the extended 56-dimensional structural feature vector."""
    binary = np.asarray(matrix, dtype=np.float32) > 0
    height, width = binary.shape
    rows = np.array_split(binary, 8, axis=0)
    cols = np.array_split(binary, 5, axis=1)
    grid = np.asarray([[cell.mean() for cell in np.array_split(row, 5, axis=1)]
                       for row in rows], dtype=np.float32).reshape(-1)
    row_occupancy = np.asarray([row.mean() for row in rows], dtype=np.float32)
    column_occupancy = np.asarray([col.mean() for col in cols], dtype=np.float32)

    ys, xs = np.where(binary)
    if len(xs):
        lo, hi = int(ys.min()), int(ys.max())
        span = max(1, hi - lo + 1)
        widths = []
        for start_ratio, end_ratio in ((0.0, 1 / 3),
                                       (1 / 3, 2 / 3),
                                       (2 / 3, 1.0)):
            y0 = lo + int(span * start_ratio)
            y1 = lo + max(1, int(span * end_ratio))
            band = binary[y0:min(height, y1), :]
            band_xs = np.where(band)[1]
            widths.append(float(band_xs.max() - band_xs.min() + 1) / width
                          if len(band_xs) else 0.0)
    else:
        widths = [0.0, 0.0, 0.0]
    return np.concatenate((grid, row_occupancy, column_occupancy,
                           np.asarray(widths, dtype=np.float32)))


def _coarse_grid_features(matrix):
    """Return the compact 40-dimensional coarse-grid feature vector."""
    binary = np.asarray(matrix, dtype=np.float32) > 0
    rows = np.array_split(binary, 8, axis=0)
    return np.asarray([[cell.mean() for cell in np.array_split(row, 5, axis=1)]
                       for row in rows], dtype=np.float32).reshape(-1)


def _mask_features(mask):
    return np.asarray([[int(char) for char in row] for row in mask],
                      dtype=np.float32).reshape(-1)


def load_seed(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "tapo_osd_bootstrap_seed_v1":
        raise ValueError("unsupported Seed Data format")
    prototypes = {
        digit: np.asarray(item["prototype"], dtype=np.float32)
        for digit, item in data.get("digits", {}).items()
        if len(item.get("prototype", [])) in {40, 56}
    }
    for digit, item in data.get("digits", {}).items():
        mask = item.get("coarse_mask")
        if digit not in prototypes and isinstance(mask, list) and len(mask) == 8:
            if all(isinstance(row, str) and len(row) == 5 and
                   set(row) <= {"0", "1"} for row in mask):
                prototypes[digit] = _mask_features(mask)
    if set(prototypes) != set("0123456789"):
        raise ValueError("Seed Data must contain prototypes for digits 0-9")
    return {"raw": data, "prototypes": prototypes}


def recognize_seed(matrix, seed):
    """Return (digit, best_distance, margin) for a normalized glyph matrix."""
    features = (_coarse_grid_features(matrix)
                if len(next(iter(seed["prototypes"].values()))) == 40
                else _coarse_features(matrix))
    scores = sorted((float(np.linalg.norm(features - prototype)), digit)
                    for digit, prototype in seed["prototypes"].items())
    best, digit = scores[0]
    margin = scores[1][0] - best if len(scores) > 1 else None
    return digit, best, margin
