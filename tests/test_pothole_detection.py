"""
Unit tests for pothole candidate extraction and overlay visualization.
"""

import numpy as np
import pytest
from src.image_utils import generate_synthetic_road_sample
from src.preprocessing import to_grayscale, preprocess_pipeline
from src.edge_detection import detect_canny
from src.pothole_detection import (
    apply_morphology,
    extract_candidate_regions,
    draw_candidate_overlays,
)
from src.evaluation import compute_candidate_summary, build_analysis_json


@pytest.fixture
def synthetic_road():
    """Deterministic synthetic road with pothole."""
    return generate_synthetic_road_sample("single_deep_pothole", seed=99)


def test_apply_morphology():
    bin_map = np.zeros((50, 50), dtype=np.uint8)
    bin_map[20:25, 20:25] = 255
    closed = apply_morphology(bin_map, kernel_size=3, operation="close")
    assert closed.shape == (50, 50)
    assert closed.dtype == np.uint8

    dilated = apply_morphology(bin_map, kernel_size=3, operation="dilate")
    assert np.count_nonzero(dilated) >= np.count_nonzero(bin_map)


def test_extract_candidate_regions(synthetic_road):
    prep = preprocess_pipeline(synthetic_road)
    edges = detect_canny(prep["smoothed"], low_threshold=40, high_threshold=120)
    closed = apply_morphology(edges, kernel_size=7, operation="close", iterations=2)

    candidates, mask = extract_candidate_regions(
        closed,
        original_rgb=synthetic_road,
        gray_image=prep["gray"],
        min_area=100,
        max_area=50000,
        roi_top_ratio=0.1,
    )

    assert isinstance(candidates, list)
    assert mask.shape == (synthetic_road.shape[0], synthetic_road.shape[1])
    assert len(candidates) >= 1

    first = candidates[0]
    assert "bbox" in first
    assert "saliency_score" in first
    assert "severity" in first
    assert len(first["bbox"]) == 4
    x, y, w, h = first["bbox"]
    assert w > 0 and h > 0
    assert 0 <= x < synthetic_road.shape[1]
    assert 0 <= y < synthetic_road.shape[0]


def test_extract_candidate_regions_empty_image():
    blank = np.zeros((100, 100), dtype=np.uint8)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    candidates, mask = extract_candidate_regions(blank, rgb)
    assert len(candidates) == 0
    assert np.count_nonzero(mask) == 0


def test_draw_candidate_overlays(synthetic_road):
    candidates = [{
        "id": 1,
        "bbox": (100, 100, 50, 40),
        "area": 2000,
        "saliency_score": 85.5,
        "severity": "Moderate (Medium Anomaly)",
        "contour": None,
    }]
    overlay = draw_candidate_overlays(synthetic_road, candidates)
    assert overlay.shape == synthetic_road.shape
    assert overlay.dtype == np.uint8


def test_evaluation_summary_and_json(synthetic_road):
    candidates = [{
        "id": 1,
        "bbox": (100, 100, 50, 40),
        "area": 2000,
        "saliency_score": 85.5,
        "severity": "Moderate (Medium Anomaly)",
        "contour": None,
    }]
    summary = compute_candidate_summary(candidates)
    assert summary["total_candidates"] == 1
    assert summary["severity_counts"]["Moderate"] == 1

    json_str = build_analysis_json(
        image_name="test_road.png",
        image_shape=synthetic_road.shape,
        pipeline_params={"canny_low": 50},
        detector_metrics={"canny": {"runtime_ms": 2.1}},
        candidates=candidates,
    )
    assert "POTHOLE AI" in json_str
    assert "Moderate" in json_str


def test_extract_candidate_regions_reversed_parameters(synthetic_road):
    prep = preprocess_pipeline(synthetic_road)
    edges = detect_canny(prep["smoothed"])
    closed = apply_morphology(edges, kernel_size=5)

    # Inverted min/max area and aspect ratios
    candidates, mask = extract_candidate_regions(
        closed,
        original_rgb=synthetic_road,
        gray_image=prep["gray"],
        min_area=50000,
        max_area=100,
        min_aspect_ratio=4.0,
        max_aspect_ratio=0.2,
    )
    assert isinstance(candidates, list)
    assert mask.shape == (synthetic_road.shape[0], synthetic_road.shape[1])

