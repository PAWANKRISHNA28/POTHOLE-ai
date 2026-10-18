"""
Evaluation and Metrics Module for Edge Detection and Candidate Analysis.

Calculates edge statistics, candidate distributions, detector trade-off matrices,
and generates standardized JSON export reports for reproducible research.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import numpy as np


def compute_edge_metrics(edge_map: np.ndarray, runtime_ms: float) -> Dict[str, Any]:
    """
    Compute quantitative performance metrics for a single edge detector output.
    """
    total_pixels = int(edge_map.size)
    edge_pixels = int(np.count_nonzero(edge_map > 0))
    density_pct = round((edge_pixels / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0
    mean_intensity = round(float(np.mean(edge_map)), 2)

    return {
        "total_pixels": total_pixels,
        "edge_pixels": edge_pixels,
        "edge_density_pct": density_pct,
        "mean_intensity": mean_intensity,
        "runtime_ms": round(runtime_ms, 2),
    }


def compute_candidate_summary(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize candidate regions by count, area footprint, and severity classes.
    """
    count = len(candidates)
    if count == 0:
        return {
            "total_candidates": 0,
            "severity_counts": {"High": 0, "Moderate": 0, "Low": 0},
            "mean_area": 0.0,
            "max_area": 0,
            "min_area": 0,
            "avg_saliency": 0.0,
        }

    areas = [c["area"] for c in candidates]
    scores = [c["saliency_score"] for c in candidates]

    high_c = sum(1 for c in candidates if "High" in c.get("severity", ""))
    mod_c = sum(1 for c in candidates if "Moderate" in c.get("severity", ""))
    low_c = sum(1 for c in candidates if "Low" in c.get("severity", ""))

    return {
        "total_candidates": count,
        "severity_counts": {
            "High": high_c,
            "Moderate": mod_c,
            "Low": low_c,
        },
        "mean_area": round(float(np.mean(areas)), 1),
        "median_area": round(float(np.median(areas)), 1),
        "min_area": int(min(areas)),
        "max_area": int(max(areas)),
        "avg_saliency": round(float(np.mean(scores)), 1),
    }


def generate_comparison_table(
    detector_results: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Compile a comparative benchmark table across Canny, Sobel, and Laplacian.
    """
    rows = []
    for key, data in detector_results.items():
        rows.append({
            "Detector": data.get("name", key.capitalize()),
            "Runtime (ms)": data.get("execution_time_ms", 0.0),
            "Edge Density (%)": data.get("edge_density_pct", 0.0),
            "Mathematical Foundation": "Multi-stage Hysteresis" if key == "canny" else ("First-order Gradient Magnitude" if key == "sobel" else "Second-order Zero-Crossing"),
            "Noise Robustness": "High (Built-in Gaussian)" if key == "canny" else ("Moderate (Averaging kernel)" if key == "sobel" else "Low (Sensitive to second derivative)"),
            "Edge Boundary Quality": "Single-pixel thin" if key == "canny" else ("Diffused / Gradient" if key == "sobel" else "Double-edge response"),
        })
    return rows


def _json_serialize_fallback(obj: Any) -> Any:
    """Robust fallback for JSON serialization of numpy arrays and scalar types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return str(obj)


def build_analysis_json(
    image_name: str,
    image_shape: tuple,
    pipeline_params: Dict[str, Any],
    detector_metrics: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> str:
    """
    Assemble and serialize a comprehensive analysis report into formatted JSON.
    """
    serializable_candidates = []
    for c in candidates:
        cand_copy = {k: v for k, v in c.items() if k != "contour" and not isinstance(v, np.ndarray)}
        serializable_candidates.append(cand_copy)

    serializable_detectors = {}
    for det_key, det_info in detector_metrics.items():
        if isinstance(det_info, dict):
            serializable_detectors[det_key] = {
                k: v for k, v in det_info.items()
                if not isinstance(v, np.ndarray) and k not in ("edge_map", "grad_x", "grad_y")
            }
        else:
            serializable_detectors[det_key] = det_info

    report = {
        "project": "POTHOLE AI — Road Pothole Detection Using Edge Enhancement",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_metadata": {
            "name": image_name,
            "width": int(image_shape[1]),
            "height": int(image_shape[0]),
            "channels": int(image_shape[2]) if len(image_shape) > 2 else 1,
        },
        "pipeline_parameters": pipeline_params,
        "detector_benchmarks": serializable_detectors,
        "candidate_summary": compute_candidate_summary(candidates),
        "candidates": serializable_candidates,
        "academic_disclaimer": (
            "Candidate regions are localized distress hypotheses generated via traditional "
            "edge enhancement and morphological rules. Edge detection alone does not establish "
            "confirmed classification of potholes. Results require human verification or "
            "downstream deep learning validation."
        ),
    }

    return json.dumps(report, indent=2, default=_json_serialize_fallback)
