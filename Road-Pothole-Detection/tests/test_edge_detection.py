"""
Unit tests for edge detection algorithms (Canny, Sobel, Laplacian).
"""

import numpy as np
import pytest
from src.edge_detection import (
    detect_canny,
    detect_sobel,
    detect_laplacian,
    calculate_edge_density,
    run_all_detectors,
)


@pytest.fixture
def synthetic_edge_image():
    """Create a 120x120 synthetic grayscale image with sharp circle and rectangle edges."""
    img = np.full((120, 120), 50, dtype=np.uint8)
    # Bright rectangle in center
    img[30:90, 30:90] = 200
    return img


def test_detect_canny(synthetic_edge_image):
    edges = detect_canny(synthetic_edge_image, low_threshold=50, high_threshold=150)
    assert edges.shape == synthetic_edge_image.shape
    assert edges.dtype == np.uint8
    # Binary output: only 0 or 255
    unique_vals = set(np.unique(edges))
    assert unique_vals.issubset({0, 255})
    # Since there is a sharp box, edge pixels must be detected
    assert np.count_nonzero(edges) > 0


def test_detect_sobel(synthetic_edge_image):
    edge_map, grad_x, grad_y = detect_sobel(
        synthetic_edge_image, ksize=3, threshold=40, binary=True
    )
    assert edge_map.shape == synthetic_edge_image.shape
    assert grad_x.shape == synthetic_edge_image.shape
    assert grad_y.shape == synthetic_edge_image.shape
    assert edge_map.dtype == np.uint8
    assert np.count_nonzero(edge_map) > 0

    # Non-binary magnitude test
    mag_continuous, _, _ = detect_sobel(synthetic_edge_image, binary=False)
    assert mag_continuous.shape == synthetic_edge_image.shape


def test_detect_laplacian(synthetic_edge_image):
    edge_map = detect_laplacian(
        synthetic_edge_image, ksize=3, threshold=30, binary=True
    )
    assert edge_map.shape == synthetic_edge_image.shape
    assert edge_map.dtype == np.uint8
    assert np.count_nonzero(edge_map) > 0


def test_calculate_edge_density():
    # 50% edges test
    half_edges = np.zeros((10, 10), dtype=np.uint8)
    half_edges[:5, :] = 255
    density = calculate_edge_density(half_edges)
    assert density == 50.0

    # 0% edges test
    zero_edges = np.zeros((10, 10), dtype=np.uint8)
    assert calculate_edge_density(zero_edges) == 0.0


def test_run_all_detectors(synthetic_edge_image):
    results = run_all_detectors(synthetic_edge_image)
    assert "canny" in results
    assert "sobel" in results
    assert "laplacian" in results

    for key, data in results.items():
        assert "edge_map" in data
        assert "execution_time_ms" in data
        assert "edge_density_pct" in data
        assert data["execution_time_ms"] >= 0.0
        assert 0.0 <= data["edge_density_pct"] <= 100.0


def test_detect_canny_reversed_and_bounds(synthetic_edge_image):
    # Reversed thresholds: low=180, high=40 should be auto-sorted
    edges_rev = detect_canny(synthetic_edge_image, low_threshold=180, high_threshold=40)
    assert edges_rev.shape == synthetic_edge_image.shape
    assert np.count_nonzero(edges_rev) > 0

    # Out of bounds: negative and > 255
    edges_oob = detect_canny(synthetic_edge_image, low_threshold=-50, high_threshold=350)
    assert edges_oob.shape == synthetic_edge_image.shape


def test_detect_operators_invalid_ksize(synthetic_edge_image):
    # Sobel with invalid even ksize
    edge_sobel, _, _ = detect_sobel(synthetic_edge_image, ksize=4)
    assert edge_sobel.shape == synthetic_edge_image.shape

    # Laplacian with invalid ksize
    edge_lap = detect_laplacian(synthetic_edge_image, ksize=6)
    assert edge_lap.shape == synthetic_edge_image.shape

