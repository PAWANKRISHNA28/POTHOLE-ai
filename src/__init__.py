"""
POTHOLE AI - Source Package Init
"""

from .preprocessing import (
    validate_image,
    to_grayscale,
    enhance_contrast,
    reduce_noise,
    preprocess_pipeline,
)
from .edge_detection import (
    detect_canny,
    detect_sobel,
    detect_laplacian,
    calculate_edge_density,
    run_all_detectors,
)
from .pothole_detection import (
    apply_morphology,
    extract_candidate_regions,
    draw_candidate_overlays,
)
from .image_utils import (
    numpy_to_png_bytes,
    load_image_from_path,
    save_image,
    create_comparison_montage,
    generate_synthetic_road_sample,
)
from .evaluation import (
    compute_edge_metrics,
    compute_candidate_summary,
    generate_comparison_table,
    build_analysis_json,
)

__all__ = [
    "validate_image",
    "to_grayscale",
    "enhance_contrast",
    "reduce_noise",
    "preprocess_pipeline",
    "detect_canny",
    "detect_sobel",
    "detect_laplacian",
    "calculate_edge_density",
    "run_all_detectors",
    "apply_morphology",
    "extract_candidate_regions",
    "draw_candidate_overlays",
    "numpy_to_png_bytes",
    "load_image_from_path",
    "save_image",
    "create_comparison_montage",
    "generate_synthetic_road_sample",
    "compute_edge_metrics",
    "compute_candidate_summary",
    "generate_comparison_table",
    "build_analysis_json",
]
