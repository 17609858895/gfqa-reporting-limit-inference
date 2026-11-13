"""Shared geometry for the manuscript workflow and its editable source."""
from pathlib import Path

WIDTH, HEIGHT = 100, 25
ICON = Path(__file__).resolve().parents[1] / "assets/monitoring_vignette.png"
ICON_BOX = (1.6, 7.5, 12.8, 12.8)

# Each rectangle is x, y, width, height, semantic role.
LANES = [
    (32.2, 13.1, 67.0, 11.2, "benchmark"),
    (32.2, 0.7, 67.0, 11.2, "archive"),
]
NODES = [
    (15.8, 3.5, 13.0, 18.0, "input"),
    (33.5, 14.3, 10.7, 6.6, "benchmark"),
    (46.8, 14.3, 12.3, 6.6, "benchmark"),
    (61.7, 14.3, 10.2, 6.6, "benchmark"),
    (74.5, 14.3, 23.4, 6.6, "benchmark"),
    (33.5, 1.9, 12.8, 6.6, "archive"),
    (48.8, 1.9, 10.3, 6.6, "archive"),
    (61.7, 1.9, 21.2, 6.6, "archive"),
    (85.5, 1.9, 12.4, 6.6, "priority"),
]
# label, x, y, relative font size, weight, horizontal alignment
LABELS = [
    ("Freshwater\nobservations", 8.0, 4.5, 1.0, "normal", "center"),
    ("Interval labels", 22.3, 19.5, 1.0, "normal", "center"),
    ("Detected", 22.3, 13.6, 0.88, "normal", "center"),
    ("Non-detect", 22.3, 6.6, 0.88, "normal", "center"),
    ("Model benchmark", 33.5, 22.45, 1.07, "bold", "left"),
    ("22 base\nlearners", 38.85, 17.6, 1.0, "normal", "center"),
    ("Out-of-fold\npredictions", 52.95, 17.6, 1.0, "normal", "center"),
    ("CSL /\nD-CSL", 66.8, 17.6, 1.0, "normal", "center"),
    ("Held-out stations\nKnown-truth tests", 86.2, 17.6, 1.0, "normal", "center"),
    ("Archive inference", 33.5, 10.05, 1.07, "bold", "left"),
    ("Station\ncross-fitting", 39.9, 5.2, 1.0, "normal", "center"),
    ("LGB-Tobit", 53.95, 5.2, 1.0, "normal", "center"),
    ("Exceedance probability\nThreshold prevalence\nWithin-station trends", 72.3, 5.2, 0.9, "normal", "center"),
    ("Follow-up\npriorities", 91.7, 5.2, 1.0, "normal", "center"),
]
CONNECTORS = [
    (14.4, 12.5, 15.8, 12.5),
    (30.1, 17.6, 33.5, 17.6),
    (30.1, 5.2, 33.5, 5.2),
    (44.2, 17.6, 46.8, 17.6),
    (59.1, 17.6, 61.7, 17.6),
    (71.9, 17.6, 74.5, 17.6),
    (46.3, 5.2, 48.8, 5.2),
    (59.1, 5.2, 61.7, 5.2),
    (82.9, 5.2, 85.5, 5.2),
    (25.2, 9.8, 19.3, 9.8),
]
SEGMENTS = [(28.8, 12.5, 30.1, 12.5), (30.1, 5.2, 30.1, 17.6), (25.2, 9.2, 25.2, 10.4)]
DETECTED = (22.3, 16.2)


def scheme(palette):
    from matplotlib.colors import to_rgb
    return {
        "input": to_rgb(palette[4]),
        "benchmark": to_rgb(palette[1]),
        "archive": to_rgb(palette[2]),
        "priority": to_rgb(palette[7]),
    }
