"""
Unit tests for image utilities and evaluation/reporting functions.
"""

import os
import tempfile
import numpy as np
import pytest
from src.image_utils import (
    numpy_to_png_bytes,
    load_image_from_path,
    save_image,
    create_comparison_montage,
    generate_synthetic_road_sample,
)
from src.evaluation import (
    compute_edge_metrics,
    compute_candidate_summary,
    generate_comparison_table,
    build_analysis_json,
)
from src.pothole_detection import draw_candidate_overlays


def test_numpy_to_png_bytes_rgb_and_gray():
    rgb = np.full((50, 50, 3), 120, dtype=np.uint8)
    png_rgb = numpy_to_png_bytes(rgb)
    assert isinstance(png_rgb, bytes)
    assert len(png_rgb) > 0
    # PNG signature check: \x89PNG\r\n\x1a\n
    assert png_rgb.startswith(b"\x89PNG")

    gray = np.full((50, 50), 80, dtype=np.uint8)
    png_gray = numpy_to_png_bytes(gray)
    assert isinstance(png_gray, bytes)
    assert png_gray.startswith(b"\x89PNG")


def test_save_and_load_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_out.png")
        arr = np.full((40, 60, 3), 150, dtype=np.uint8)
        arr[10:30, 20:40] = [200, 50, 20]

        saved = save_image(arr, test_path)
        assert saved is True
        assert os.path.exists(test_path)

        loaded = load_image_from_path(test_path)
        assert loaded.shape == (40, 60, 3)
        assert loaded.dtype == np.uint8


def test_synthetic_road_scenarios():
    scenarios = ["single_deep_pothole", "multiple_depressions", "wet_road_pothole"]
    for sc in scenarios:
        img = generate_synthetic_road_sample(sc, width=320, height=240, seed=42)
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8
        # Ensure image has non-zero intensity
        assert np.mean(img) > 10.0


def test_create_comparison_montage():
    w, h = 100, 80
    orig = np.full((h, w, 3), 50, dtype=np.uint8)
    enh = np.full((h, w), 80, dtype=np.uint8)
    canny = np.zeros((h, w), dtype=np.uint8)
    canny[20:60, 20:60] = 255
    sobel = canny.copy()
    lap = canny.copy()
    overlay = orig.copy()

    montage = create_comparison_montage(
        original=orig,
        enhanced=enh,
        canny=canny,
        sobel=sobel,
        laplacian=lap,
        overlay=overlay,
        target_size=(100, 80),
    )
    assert montage.shape[2] == 3
    # 2 rows of 80 + 32 banner = 112 per row -> total height = 224
    # 3 cols of 100 = 300
    assert montage.shape[0] == 224
    assert montage.shape[1] == 300


def test_draw_candidate_overlays_grayscale_input():
    gray = np.full((100, 100), 50, dtype=np.uint8)
    cands = [{
        "id": 1,
        "bbox": (20, 20, 30, 30),
        "area": 900,
        "saliency_score": 75.0,
        "severity": "Moderate",
        "contour": None,
    }]
    overlay = draw_candidate_overlays(gray, cands)
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8


def test_evaluation_json_serialization_with_numpy_structures():
    edge_map = np.zeros((50, 50), dtype=np.uint8)
    edge_map[10:20, 10:20] = 255

    metrics = compute_edge_metrics(edge_map, runtime_ms=3.45)
    assert metrics["edge_density_pct"] > 0.0
    assert metrics["runtime_ms"] == 3.45

    detector_results = {
        "canny": {
            "name": "Canny",
            "edge_map": edge_map,  # ndarray
            "execution_time_ms": 3.45,
            "edge_density_pct": 4.0,
        }
    }

    candidates = [{
        "id": 1,
        "bbox": (10, 10, 10, 10),
        "area": 100,
        "saliency_score": 70.0,
        "contour": np.array([[[10, 10]], [[10, 20]]]),  # ndarray
        "severity": "Low",
    }]

    json_str = build_analysis_json(
        image_name="test.png",
        image_shape=(50, 50, 3),
        pipeline_params={"canny_low": 50},
        detector_metrics=detector_results,
        candidates=candidates,
    )
    assert isinstance(json_str, str)
    assert "test.png" in json_str
    assert "Canny" in json_str
