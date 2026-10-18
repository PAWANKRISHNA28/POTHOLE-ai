"""
Edge Detection Module for Road Pothole Detection.

Implements and benchmarks:
- Canny Edge Detection (multi-stage optimal edge detection with hysteresis)
- Sobel Edge Detection (first-order directional gradient magnitude)
- Laplacian Edge Detection (second-order isotropic derivative operator)

Includes metric calculation and side-by-side comparison utilities.
"""

import time
from typing import Any, Dict, Tuple
import cv2
import numpy as np


def detect_canny(
    image_gray: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = True,
) -> np.ndarray:
    """
    Apply Canny Edge Detector.

    Args:
        image_gray: Grayscale input image (uint8).
        low_threshold: Lower threshold for hysteresis.
        high_threshold: Upper threshold for hysteresis.
        aperture_size: Sobel aperture size (3, 5, or 7).
        l2_gradient: If True, uses L2 norm for gradient magnitude calculation.

    Returns:
        Binary edge map (uint8, values 0 or 255).
    """
    if aperture_size not in (3, 5, 7):
        aperture_size = 3

    t1 = max(0, min(255, int(low_threshold)))
    t2 = max(0, min(255, int(high_threshold)))
    low, high = min(t1, t2), max(t1, t2)

    edges = cv2.Canny(
        image_gray,
        threshold1=low,
        threshold2=high,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )
    return edges


def detect_sobel(
    image_gray: np.ndarray,
    ksize: int = 3,
    scale: float = 1.0,
    delta: float = 0.0,
    threshold: int = 40,
    binary: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Sobel Edge Detector (First-order derivative).

    Args:
        image_gray: Grayscale input image (uint8).
        ksize: Aperture size (must be 1, 3, 5, or 7).
        scale: Optional scale factor for computed derivative values.
        delta: Optional delta value added to results.
        threshold: Threshold for binarization if binary is True.
        binary: If True, applies thresholding to return binary edges.

    Returns:
        (edge_map, grad_x, grad_y)
    """
    if ksize not in (1, 3, 5, 7):
        ksize = 3

    # Compute horizontal and vertical gradients with float64 to prevent overflow
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)

    # Compute Euclidean gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize magnitude to 0-255 range
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude_norm = (magnitude / max_val * 255.0).astype(np.uint8)
    else:
        magnitude_norm = np.zeros_like(image_gray, dtype=np.uint8)

    if binary:
        th = max(0, min(255, int(threshold)))
        _, edge_map = cv2.threshold(magnitude_norm, th, 255, cv2.THRESH_BINARY)
    else:
        edge_map = magnitude_norm

    # Convert directional gradients for visualization
    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)

    return edge_map, abs_x, abs_y


def detect_laplacian(
    image_gray: np.ndarray,
    ksize: int = 3,
    scale: float = 1.0,
    delta: float = 0.0,
    threshold: int = 35,
    binary: bool = True,
) -> np.ndarray:
    """
    Apply Laplacian Edge Detector (Second-order derivative).

    Args:
        image_gray: Grayscale input image (uint8).
        ksize: Aperture size used to compute second-derivative filters (1, 3, 5, or 7).
        scale: Optional scale factor.
        delta: Optional delta value.
        threshold: Threshold for binarization if binary is True.
        binary: If True, returns thresholded binary edge map.

    Returns:
        Edge map (uint8).
    """
    if ksize not in (1, 3, 5, 7):
        ksize = 3

    # Laplacian of Gaussian (LoG) is sensitive to noise, so input should be smoothed
    lap = cv2.Laplacian(image_gray, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
    abs_lap = np.absolute(lap)

    max_val = np.max(abs_lap)
    if max_val > 0:
        lap_norm = (abs_lap / max_val * 255.0).astype(np.uint8)
    else:
        lap_norm = np.zeros_like(image_gray, dtype=np.uint8)

    if binary:
        th = max(0, min(255, int(threshold)))
        _, edge_map = cv2.threshold(lap_norm, th, 255, cv2.THRESH_BINARY)
    else:
        edge_map = lap_norm

    return edge_map


def calculate_edge_density(edge_map: np.ndarray) -> float:
    """
    Calculate the percentage of pixels that represent edges in an edge map.
    """
    total_pixels = edge_map.size
    if total_pixels == 0:
        return 0.0
    edge_pixels = np.count_nonzero(edge_map > 0)
    return round((edge_pixels / total_pixels) * 100.0, 2)


def run_all_detectors(
    image_gray: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    sobel_ksize: int = 3,
    sobel_threshold: int = 40,
    laplacian_ksize: int = 3,
    laplacian_threshold: int = 35,
) -> Dict[str, Dict[str, Any]]:
    """
    Execute Canny, Sobel, and Laplacian edge detectors, timing each and
    computing qualitative and quantitative comparison metrics.
    """
    results: Dict[str, Dict[str, Any]] = {}

    # Canny
    t0 = time.perf_counter()
    canny_map = detect_canny(image_gray, low_threshold=canny_low, high_threshold=canny_high)
    canny_time = (time.perf_counter() - t0) * 1000.0

    results["canny"] = {
        "name": "Canny Edge Detection",
        "edge_map": canny_map,
        "execution_time_ms": round(canny_time, 2),
        "edge_density_pct": calculate_edge_density(canny_map),
        "parameters": {
            "low_threshold": canny_low,
            "high_threshold": canny_high,
            "aperture_size": 3,
            "l2_gradient": True,
        },
        "description": "Multi-stage detector with Gaussian smoothing, gradient calculation, non-maximum suppression, and hysteresis thresholding.",
        "strengths": "Sharp, single-pixel thin boundaries; high noise suppression; low false positive rate.",
        "weaknesses": "Higher computational overhead; sensitive to global threshold selection in varying illumination.",
    }

    # Sobel
    t0 = time.perf_counter()
    sobel_map, sobel_x, sobel_y = detect_sobel(
        image_gray, ksize=sobel_ksize, threshold=sobel_threshold, binary=True
    )
    sobel_time = (time.perf_counter() - t0) * 1000.0

    results["sobel"] = {
        "name": "Sobel Edge Detection",
        "edge_map": sobel_map,
        "grad_x": sobel_x,
        "grad_y": sobel_y,
        "execution_time_ms": round(sobel_time, 2),
        "edge_density_pct": calculate_edge_density(sobel_map),
        "parameters": {
            "ksize": sobel_ksize,
            "threshold": sobel_threshold,
            "scale": 1.0,
        },
        "description": "First-order discrete differentiation operator computing horizontal and vertical gradient vectors.",
        "strengths": "Fast computation; directional sensitivity; useful for capturing orientation of road cracks and borders.",
        "weaknesses": "Thicker, blurred edges; more susceptible to high-frequency road grain noise.",
    }

    # Laplacian
    t0 = time.perf_counter()
    lap_map = detect_laplacian(
        image_gray, ksize=laplacian_ksize, threshold=laplacian_threshold, binary=True
    )
    lap_time = (time.perf_counter() - t0) * 1000.0

    results["laplacian"] = {
        "name": "Laplacian Edge Detection",
        "edge_map": lap_map,
        "execution_time_ms": round(lap_time, 2),
        "edge_density_pct": calculate_edge_density(lap_map),
        "parameters": {
            "ksize": laplacian_ksize,
            "threshold": laplacian_threshold,
            "scale": 1.0,
        },
        "description": "Second-order isotropic derivative detecting zero-crossings corresponding to rapid intensity changes.",
        "strengths": "Rotationally invariant (isotropic); sensitive to localized depression cavities and rapid transitions.",
        "weaknesses": "High sensitivity to noise without strong pre-filtering; double-edge artifacts.",
    }

    return results
