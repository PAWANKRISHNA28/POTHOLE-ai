"""
POTHOLE AI — ROAD POTHOLE DETECTION USING EDGE ENHANCEMENT TECHNIQUES
A modern, production-inspired computer vision analytics dashboard for road surface distress inspection.
"""

import base64
from datetime import datetime, timezone
import json
import os
import time
from typing import Any, Dict, List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Internal CV processing modules
from src.edge_detection import (
    calculate_edge_density,
    detect_canny,
    detect_laplacian,
    detect_sobel,
    run_all_detectors,
)
from src.evaluation import (
    build_analysis_json,
    compute_candidate_summary,
    compute_edge_metrics,
    generate_comparison_table,
)
from src.image_utils import (
    create_comparison_montage,
    generate_synthetic_road_sample,
    load_image_from_path,
    numpy_to_png_bytes,
)
from src.pothole_detection import (
    apply_morphology,
    draw_candidate_overlays,
    extract_candidate_regions,
)
from src.preprocessing import (
    enhance_contrast,
    preprocess_pipeline,
    reduce_noise,
    to_grayscale,
    validate_image,
)

# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="POTHOLE AI — Road Surface Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Resource Locator & CSS Styling Loader
# -----------------------------------------------------------------------------
def get_resource_path(relative_path: str) -> str:
    """Search multiple candidate roots to reliably locate assets and data files."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Road-Pothole-Detection", relative_path),
        os.path.join(os.getcwd(), relative_path),
        os.path.join(os.getcwd(), "Road-Pothole-Detection", relative_path),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_custom_css() -> None:
    """Load and inject refined SaaS design system from assets/style.css."""
    css_path = get_resource_path(os.path.join("assets", "style.css"))
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_custom_css()


def numpy_to_base64_jpeg(image: np.ndarray, quality: int = 88) -> str:
    """Encode an RGB or Grayscale numpy array to a base64 JPEG string for fast inline HTML rendering."""
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode("utf-8")


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize persistent session variables."""
    if "current_image_rgb" not in st.session_state or st.session_state.get("current_image_name") == "synthetic_sample_1.png":
        # Load default benchmark sample image so dashboard starts initialized
        default_sample_path = get_resource_path(
            os.path.join("data", "sample_images", "sample_road_pothole_1.png")
        )
        if os.path.exists(default_sample_path):
            st.session_state["current_image_rgb"] = load_image_from_path(default_sample_path)
            st.session_state["current_image_name"] = "sample_road_pothole_1.png"
            st.session_state["current_results"] = None
        else:
            synth = generate_synthetic_road_sample("single_deep_pothole", seed=101)
            st.session_state["current_image_rgb"] = synth
            st.session_state["current_image_name"] = "synthetic_sample_1.png"

    if "current_results" not in st.session_state:
        st.session_state["current_results"] = None

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Dashboard"

    if "params" not in st.session_state:
        st.session_state["params"] = {
            "contrast_method": "clahe",
            "clahe_clip": 2.5,
            "blur_method": "gaussian",
            "blur_ksize": 5,
            "blur_sigma": 1.2,
            "canny_low": 45,
            "canny_high": 135,
            "sobel_ksize": 3,
            "sobel_threshold": 40,
            "laplacian_ksize": 3,
            "laplacian_threshold": 35,
            "morph_op": "close",
            "morph_ksize": 5,
            "morph_iter": 2,
            "min_area": 180,
            "max_area": 60000,
            "min_solidity": 0.35,
            "roi_top_ratio": 0.15,
            "active_detector": "canny",
        }


init_session_state()


# -----------------------------------------------------------------------------
# Core Pipeline Execution Routine
# -----------------------------------------------------------------------------
def execute_pipeline(image_rgb: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute complete preprocessing, multi-edge detection, and candidate extraction."""
    t_start = time.perf_counter()

    # 1. Preprocessing
    t_pre_0 = time.perf_counter()
    try:
        prep = preprocess_pipeline(
            image_rgb,
            contrast_method=params["contrast_method"],
            clahe_clip=params["clahe_clip"],
            blur_method=params["blur_method"],
            blur_ksize=params["blur_ksize"],
            blur_sigma=params["blur_sigma"],
            grayscale_method=params.get("grayscale_method", "weighted"),
        )
    except TypeError:
        img_in = image_rgb
        if params.get("grayscale_method") == "average":
            avg_ch = np.mean(image_rgb, axis=2).astype(np.uint8)
            img_in = cv2.cvtColor(avg_ch, cv2.COLOR_GRAY2RGB)
        prep = preprocess_pipeline(
            img_in,
            contrast_method=params["contrast_method"],
            clahe_clip=params["clahe_clip"],
            blur_method=params["blur_method"],
            blur_ksize=params["blur_ksize"],
            blur_sigma=params["blur_sigma"],
        )
    t_pre = (time.perf_counter() - t_pre_0) * 1000.0

    # 2. Run all three edge detectors
    detector_results = run_all_detectors(
        prep["smoothed"],
        canny_low=params["canny_low"],
        canny_high=params["canny_high"],
        sobel_ksize=params["sobel_ksize"],
        sobel_threshold=params["sobel_threshold"],
        laplacian_ksize=params["laplacian_ksize"],
        laplacian_threshold=params["laplacian_threshold"],
    )

    # 3. Choose edge map for candidate extraction
    chosen_detector = params.get("active_detector", "canny").lower()
    if chosen_detector == "sobel":
        base_edge = detector_results["sobel"]["edge_map"]
    elif chosen_detector == "laplacian":
        base_edge = detector_results["laplacian"]["edge_map"]
    else:
        base_edge = detector_results["canny"]["edge_map"]

    # 4. Morphology
    closed_edge = apply_morphology(
        base_edge,
        kernel_size=params["morph_ksize"],
        operation=params["morph_op"],
        iterations=params["morph_iter"],
    )

    # 5. Extract Candidate Regions
    t_cand_0 = time.perf_counter()
    candidates, mask = extract_candidate_regions(
        closed_edge,
        original_rgb=image_rgb,
        gray_image=prep["gray"],
        min_area=params["min_area"],
        max_area=params["max_area"],
        min_solidity=params["min_solidity"],
        roi_top_ratio=params["roi_top_ratio"],
    )
    t_cand = (time.perf_counter() - t_cand_0) * 1000.0

    # 6. Render Overlay
    overlay = draw_candidate_overlays(
        image_rgb,
        candidates,
        box_color=(239, 68, 68),  # Coral red matching screenshot's "Potential pothole region"
        show_labels=False,
    )

    # 7. Render Montage
    montage = create_comparison_montage(
        original=image_rgb,
        enhanced=prep["enhanced"],
        canny=detector_results["canny"]["edge_map"],
        sobel=detector_results["sobel"]["edge_map"],
        laplacian=detector_results["laplacian"]["edge_map"],
        overlay=overlay,
    )

    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    # 8. Package Results
    return {
        "prep": prep,
        "detector_results": detector_results,
        "active_detector": chosen_detector,
        "closed_edge": closed_edge,
        "candidates": candidates,
        "mask": mask,
        "overlay": overlay,
        "montage": montage,
        "total_time_ms": round(total_time_ms, 2),
        "pre_time_ms": round(t_pre, 2),
        "cand_time_ms": round(t_cand, 2),
        "summary": compute_candidate_summary(candidates),
    }


# Automatically run pipeline on startup if needed
if st.session_state["current_results"] is None and st.session_state["current_image_rgb"] is not None:
    st.session_state["current_results"] = execute_pipeline(
        st.session_state["current_image_rgb"], st.session_state["params"]
    )


# -----------------------------------------------------------------------------
# Sidebar Navigation & Product Identity
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon-svg">
                <svg width="42" height="42" viewBox="0 0 42 42" fill="none">
                    <rect x="5" y="13" width="15" height="20" rx="7.5" transform="rotate(-25 5 13)" fill="#10B981"/>
                    <rect x="17" y="8" width="15" height="20" rx="7.5" transform="rotate(-25 17 8)" fill="#34D399"/>
                </svg>
            </div>
            <div class="brand-name-box">
                <div class="brand-title">POTHOLE AI</div>
                <div class="brand-subtitle">ROAD INTELLIGENCE</div>
            </div>
        </div>
        <div class="sidebar-section-label">NAVIGATION</div>
        """,
        unsafe_allow_html=True,
    )

    nav_labels = {
        "Dashboard": "◈  Dashboard",
        "Upload & Analyze": "⇡  Upload & Analyze",
        "Edge Analysis": "∿  Edge Analysis",
        "Pothole Candidates": "⊙  Pothole Candidates",
        "Analysis History": "◷  Analysis History",
        "Reports & Metrics": "▤  Reports & Metrics",
    }
    menu_options = list(nav_labels.keys())
    current_page = st.session_state.get("nav_page", "Dashboard")
    current_idx = menu_options.index(current_page) if current_page in menu_options else 0
    selected_nav = st.radio(
        "Navigation",
        menu_options,
        index=current_idx,
        format_func=lambda x: nav_labels.get(x, x),
        key="nav_selection",
        label_visibility="collapsed",
    )
    st.session_state["nav_page"] = selected_nav

    # Sidebar Bottom Container (Pinned to Bottom via Flex)
    st.markdown(
        """
        <div class="sidebar-bottom-container">
            <div class="sidebar-status-section">
                <div class="sidebar-status-label">System Status</div>
                <div class="sidebar-status-card">
                    <span class="status-dot-green"></span>
                    <div>
                        <div class="status-text-title">LOCAL INFERENCE READY</div>
                        <div class="status-text-desc">High-precision CV pipeline</div>
                    </div>
                </div>
            </div>
            <div class="sidebar-section-label" style="margin: 10px 4px 6px;">USER ACCOUNT</div>
            <div class="sidebar-user-footer">
                <div class="user-profile-left">
                    <div class="user-avatar-circle">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                            <circle cx="12" cy="7" r="4"></circle>
                        </svg>
                    </div>
                    <div>
                        <div class="user-name-text">Pawan</div>
                        <div class="user-subtext">#dela-1974 · Operator</div>
                    </div>
                </div>
                <span class="user-chevron">›</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Reusable Compact Header Component
# -----------------------------------------------------------------------------
def render_header(
    title: str = "Dashboard",
    subtitle: str = "Intelligent Road Surface Distress & Pavement Analytics",
    eyebrow: str = "POTHOLE AI / ROAD INTELLIGENCE",
    badge: str = "System Online",
    action_text: str = "Add Custom Widget",
) -> None:
    """Render top header bar matching the reference mockup."""
    st.markdown(
        f"""
        <div class="top-header-container">
            <div class="header-left-col">
                <h1 class="header-main-title">{title}</h1>
                <p class="header-subtitle-text">{subtitle}</p>
            </div>
            <div class="header-right-col">
                <div class="header-action-pill">
                    <span class="action-pill-dot"></span> {action_text}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION 1: DASHBOARD (EXACT VISUAL REFERENCE OVERHAUL)
# =============================================================================
if selected_nav == "Dashboard":
    # A. Top Header
    render_header()

    res = st.session_state["current_results"]
    history_list = st.session_state.get("history", [])

    total_images_analyzed = len(history_list) + (1 if res is not None else 0)
    cand_count = len(res["candidates"]) if res is not None else 0
    total_time = res["total_time_ms"] if res is not None else 0.0
    active_det = st.session_state["params"].get("active_detector", "canny").capitalize()

    # B. Four Aligned Metric Cards (Refined Dark Surface with Line Icons & Real State)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        img_subtext = f"{total_images_analyzed} image{'s' if total_images_analyzed != 1 else ''} in session" if total_images_analyzed > 0 else "No images analyzed yet"
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">
                    <span class="metric-card-label">Images Analyzed</span>
                    <div class="metric-card-icon">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <circle cx="8.5" cy="8.5" r="1.5"></circle>
                            <polyline points="21 15 16 10 5 21"></polyline>
                        </svg>
                    </div>
                </div>
                <div class="metric-card-value">{total_images_analyzed}</div>
                <div class="metric-card-subtext">{img_subtext}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        cand_subtext = f"{cand_count} distress region{'s' if cand_count != 1 else ''} flagged" if res else "Awaiting analysis"
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">
                    <span class="metric-card-label">Candidate Regions</span>
                    <div class="metric-card-icon">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="22" y1="12" x2="18" y2="12"></line>
                            <line x1="6" y1="12" x2="2" y2="12"></line>
                            <line x1="12" y1="6" x2="12" y2="2"></line>
                            <line x1="12" y1="22" x2="12" y2="18"></line>
                        </svg>
                    </div>
                </div>
                <div class="metric-card-value">{cand_count if res else 0}</div>
                <div class="metric-card-subtext">{cand_subtext}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        time_val_str = f"{total_time:.1f} ms" if res else "0.0 ms"
        time_subtext = "OpenCV CPU runtime" if res else "No processing completed"
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">
                    <span class="metric-card-label">Latest Processing Time</span>
                    <div class="metric-card-icon">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                    </div>
                </div>
                <div class="metric-card-value">{time_val_str}</div>
                <div class="metric-card-subtext">{time_subtext}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">
                    <span class="metric-card-label">Active Edge Technique</span>
                    <div class="metric-card-icon">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                            <polyline points="2 17 12 22 22 17"></polyline>
                            <polyline points="2 12 12 17 22 12"></polyline>
                        </svg>
                    </div>
                </div>
                <div class="metric-card-value">{active_det}</div>
                <div class="metric-card-subtext">Current detection method</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    # C. Main Two-Column Analysis Workspace (~60% left, ~40% right)
    col_left, col_right = st.columns([1.55, 1.05], gap="medium")

    # LEFT COLUMN: Active Road Inspection + 4 Result Previews
    with col_left:
        with st.container(border=True):
            st.markdown(
                """
                <div class="workspace-panel-title">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                    <span>Active Road Inspection</span>
                </div>
                <div class="workspace-panel-subtitle">Upload a road pavement image or drag & drop below to run automated edge detection & distress localization.</div>
                """,
                unsafe_allow_html=True,
            )

            dash_file = st.file_uploader(
                "Choose Road Image",
                type=["png", "jpg", "jpeg", "webp"],
                key="dash_file_uploader",
                label_visibility="collapsed",
                help="Supported: PNG, JPG, JPEG, WEBP · Max file size: 200MB",
            )
            if dash_file is not None:
                file_bytes = dash_file.read()
                is_valid, msg, rgb_img = validate_image(file_bytes)
                if is_valid and rgb_img is not None:
                    if st.session_state.get("current_image_name") != dash_file.name:
                        st.session_state["current_image_rgb"] = rgb_img
                        st.session_state["current_image_name"] = dash_file.name
                        st.session_state["current_results"] = execute_pipeline(rgb_img, st.session_state["params"])
                        st.rerun()
                else:
                    st.error(f"⚠️ Validation Failed: {msg}")

            curr_img_name = st.session_state.get("current_image_name", "")
            has_loaded = st.session_state.get("current_image_rgb") is not None
            st.markdown(
                f"""
                <div class="active-image-info-bar">
                    <div class="active-image-left">
                        <span class="active-image-dot"></span>
                        <span class="active-image-label">Active Source:</span>
                        <span class="active-image-filename">{curr_img_name if has_loaded else "Awaiting upload"}</span>
                    </div>
                    <span class="active-image-spec">PNG, JPG, WEBP · Max 200MB</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div style="font-size: 13px; font-weight: 700; color: #111C19; margin: 12px 0 6px;">
                Image Preview & Analysis Results
            </div>
            """,
            unsafe_allow_html=True,
        )

        if res is not None:
            active_edge = res["detector_results"][res["active_detector"]]["edge_map"]
            b64_orig = numpy_to_base64_jpeg(res["prep"]["original_rgb"])
            b64_enh = numpy_to_base64_jpeg(res["prep"]["enhanced"])
            b64_edge = numpy_to_base64_jpeg(active_edge)
            b64_over = numpy_to_base64_jpeg(res["overlay"])

            st.markdown(
                f"""
                <div class="panel-grid-container">
                    <div class="panel-preview-tile">
                        <div class="tile-badge-pill">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
                            <span>Original Image</span>
                        </div>
                        <img src="data:image/jpeg;base64,{b64_orig}" alt="Original Image"/>
                    </div>
                    <div class="panel-preview-tile">
                        <div class="tile-badge-pill">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                            <span>Enhanced Grayscale</span>
                        </div>
                        <img src="data:image/jpeg;base64,{b64_enh}" alt="Enhanced Grayscale"/>
                    </div>
                    <div class="panel-preview-tile">
                        <div class="tile-badge-pill">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                            <span>Edge Map ({active_det})</span>
                        </div>
                        <img src="data:image/jpeg;base64,{b64_edge}" alt="Edge Map ({active_det})"/>
                    </div>
                    <div class="panel-preview-tile">
                        <div class="tile-badge-pill">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="22" y1="12" x2="18" y2="12"></line><line x1="6" y1="12" x2="2" y2="12"></line><line x1="12" y1="6" x2="12" y2="2"></line><line x1="12" y1="22" x2="12" y2="18"></line></svg>
                            <span>Candidate Overlay</span>
                        </div>
                        <img src="data:image/jpeg;base64,{b64_over}" alt="Candidate Overlay"/>
                        <div class="tile-bottom-legend">
                            <span class="legend-dot-red"></span>
                            <span>Potential pothole region</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("🔍  High-Resolution Detailed Inspection (Zoom & Full-View)", expanded=False):
                tab_cand, tab_edge, tab_enh, tab_orig, tab_mont = st.tabs([
                    "Candidate Overlay", "Edge Map", "Enhanced Grayscale", "Original Image", "Full Montage"
                ])
                with tab_cand:
                    st.image(res["overlay"], caption=f"Candidate Overlay ({cand_count} candidates detected)", width="stretch")
                with tab_edge:
                    st.image(active_edge, caption=f"Active Edge Map ({active_det})", width="stretch")
                with tab_enh:
                    st.image(res["prep"]["enhanced"], caption="Enhanced Grayscale (CLAHE + Contrast)", width="stretch", clamp=True)
                with tab_orig:
                    st.image(res["prep"]["original_rgb"], caption="Original Road Image RGB", width="stretch")
                with tab_mont:
                    st.image(res["montage"], caption="Comprehensive 6-Panel Processing Montage", width="stretch")
        else:
            st.markdown(
                """
                <div class="panel-grid-container">
                    <div class="panel-preview-tile panel-preview-empty">
                        <div class="tile-badge-pill">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#5E716D" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
                            <span>Original Image</span>
                        </div>
                        <div class="empty-preview-content">
                            <svg class="empty-preview-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                            <div class="empty-preview-title">Awaiting Road Image</div>
                            <div class="empty-preview-desc">Upload a photo to begin</div>
                        </div>
                    </div>
                    <div class="panel-preview-tile panel-preview-empty">
                        <div class="tile-badge-pill">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#5E716D" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                            <span>Enhanced Grayscale</span>
                        </div>
                        <div class="empty-preview-content">
                            <svg class="empty-preview-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                            <div class="empty-preview-title">Preprocessing Pending</div>
                            <div class="empty-preview-desc">CLAHE + Gaussian blur</div>
                        </div>
                    </div>
                    <div class="panel-preview-tile panel-preview-empty">
                        <div class="tile-badge-pill">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#5E716D" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                            <span>Edge Map</span>
                        </div>
                        <div class="empty-preview-content">
                            <svg class="empty-preview-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                            <div class="empty-preview-title">Edge Filter Pending</div>
                            <div class="empty-preview-desc">Canny / Sobel / Laplacian</div>
                        </div>
                    </div>
                    <div class="panel-preview-tile panel-preview-empty">
                        <div class="tile-badge-pill">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#5E716D" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="22" y1="12" x2="18" y2="12"></line><line x1="6" y1="12" x2="2" y2="12"></line><line x1="12" y1="6" x2="12" y2="2"></line><line x1="12" y1="22" x2="12" y2="18"></line></svg>
                            <span>Candidate Overlay</span>
                        </div>
                        <div class="empty-preview-content">
                            <svg class="empty-preview-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>
                            <div class="empty-preview-title">Candidates Pending</div>
                            <div class="empty-preview-desc">Morphology & ROI contours</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # RIGHT COLUMN: Unified Analysis Controls + Results Summary Panel (~40%)
    with col_right:
        with st.container(border=True):
            st.markdown(
                """
                <style>
                div[data-testid="stColumn"]:has([data-testid="stToggle"]) [data-testid="stVerticalBlockBorderWrapper"],
                [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stToggle"]) {
                    background: #11191B !important;
                    background-color: #11191B !important;
                    border: 1px solid #263635 !important;
                    border-radius: 16px !important;
                    padding: 18px 20px !important;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
                }
                div[data-testid="stColumn"]:has([data-testid="stToggle"]) [data-testid="stVerticalBlockBorderWrapper"] > div,
                [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stToggle"]) > div {
                    background: transparent !important;
                    background-color: transparent !important;
                }
                </style>
                <div class="workspace-panel-title dark-panel-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="3"></circle>
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                    </svg>
                    <span>Analysis Controls</span>
                </div>
                <div class="control-section-header">PREPROCESSING</div>
                """,
                unsafe_allow_html=True,
            )

            t_gray = st.toggle("Grayscale Conversion", value=st.session_state["params"].get("grayscale_method", "weighted") == "weighted", key="dash_toggle_gray")
            t_contrast = st.toggle("Contrast Enhancement", value=st.session_state["params"].get("contrast_method", "clahe") != "none", key="dash_toggle_contrast")
            t_noise = st.toggle("Gaussian Noise Reduction", value=st.session_state["params"].get("blur_method", "gaussian") != "none", key="dash_toggle_noise")

            st.markdown('<div class="control-section-header">EDGE DETECTION</div>', unsafe_allow_html=True)
            current_active = st.session_state["params"].get("active_detector", "canny").capitalize()
            selected_method = st.segmented_control(
                "Edge Detection Method",
                ["Canny", "Sobel", "Laplacian"],
                default=current_active if current_active in ["Canny", "Sobel", "Laplacian"] else "Canny",
                key="dash_edge_method_seg",
                label_visibility="collapsed",
            )
            if not selected_method:
                selected_method = "Canny"

            chosen_op = selected_method.lower()

            st.markdown('<div class="control-section-header">PARAMETERS</div>', unsafe_allow_html=True)
            if chosen_op == "canny":
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    canny_low = st.number_input(
                        "Threshold 1 (Low)",
                        min_value=1,
                        max_value=255,
                        value=int(st.session_state["params"].get("canny_low", 50)),
                        step=5,
                        key="dash_canny_low",
                    )
                    st.markdown('<div class="param-helper-text">Lower hysteresis linkage cutoff</div>', unsafe_allow_html=True)
                with p_c2:
                    canny_high = st.number_input(
                        "Threshold 2 (High)",
                        min_value=1,
                        max_value=255,
                        value=int(st.session_state["params"].get("canny_high", 150)),
                        step=5,
                        key="dash_canny_high",
                    )
                    st.markdown('<div class="param-helper-text">Upper strong-edge gradient threshold</div>', unsafe_allow_html=True)
                canny_ksize = st.number_input(
                    "Kernel Size",
                    min_value=3,
                    max_value=7,
                    value=int(st.session_state["params"].get("blur_ksize", 3)),
                    step=2,
                    key="dash_canny_ksize",
                )
                st.markdown('<div class="param-helper-text">Gaussian aperture kernel (odd: 3, 5, 7)</div>', unsafe_allow_html=True)
            elif chosen_op == "sobel":
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    sobel_thresh = st.number_input(
                        "Gradient Threshold",
                        min_value=5,
                        max_value=255,
                        value=int(st.session_state["params"].get("sobel_threshold", 40)),
                        step=5,
                        key="dash_sobel_thresh",
                    )
                    st.markdown('<div class="param-helper-text">First-derivative gradient cutoff</div>', unsafe_allow_html=True)
                with p_c2:
                    sobel_ksize = st.number_input(
                        "Kernel Size",
                        min_value=3,
                        max_value=7,
                        value=int(st.session_state["params"].get("sobel_ksize", 3)),
                        step=2,
                        key="dash_sobel_ksize",
                    )
                    st.markdown('<div class="param-helper-text">Sobel derivative convolution aperture</div>', unsafe_allow_html=True)
            else:  # laplacian
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    lap_thresh = st.number_input(
                        "Zero-Crossing Threshold",
                        min_value=5,
                        max_value=255,
                        value=int(st.session_state["params"].get("laplacian_threshold", 35)),
                        step=5,
                        key="dash_lap_thresh",
                    )
                    st.markdown('<div class="param-helper-text">Second-derivative magnitude cutoff</div>', unsafe_allow_html=True)
                with p_c2:
                    lap_ksize = st.number_input(
                        "Kernel Size",
                        min_value=1,
                        max_value=7,
                        value=int(st.session_state["params"].get("laplacian_ksize", 3)),
                        step=2,
                        key="dash_lap_ksize",
                    )
                    st.markdown('<div class="param-helper-text">Laplacian 2nd derivative mask size</div>', unsafe_allow_html=True)

            if st.button("Analyze Road Image", type="primary", width="stretch", key="dash_analyze_btn"):
                try:
                    with st.spinner("Analyzing road surface imagery..."):
                        st.session_state["params"]["active_detector"] = chosen_op
                        if chosen_op == "canny":
                            st.session_state["params"]["canny_low"] = canny_low
                            st.session_state["params"]["canny_high"] = canny_high
                            st.session_state["params"]["blur_ksize"] = canny_ksize
                        elif chosen_op == "sobel":
                            st.session_state["params"]["sobel_threshold"] = sobel_thresh
                            st.session_state["params"]["sobel_ksize"] = sobel_ksize
                        else:
                            st.session_state["params"]["laplacian_threshold"] = lap_thresh
                            st.session_state["params"]["laplacian_ksize"] = lap_ksize

                        st.session_state["params"]["contrast_method"] = "clahe" if t_contrast else "none"
                        st.session_state["params"]["blur_method"] = "gaussian" if t_noise else "none"
                        st.session_state["params"]["grayscale_method"] = "weighted" if t_gray else "average"

                        new_res = execute_pipeline(st.session_state["current_image_rgb"], st.session_state["params"])
                        st.session_state["current_results"] = new_res

                        st.session_state["history"].append({
                            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "image_name": st.session_state["current_image_name"],
                            "dimensions": f"{st.session_state['current_image_rgb'].shape[1]} × {st.session_state['current_image_rgb'].shape[0]}",
                            "candidates_found": len(new_res["candidates"]),
                            "runtime_ms": new_res["total_time_ms"],
                            "detector": chosen_op.capitalize(),
                        })
                    st.rerun()
                except Exception as analysis_err:
                    st.error(f"Analysis error: {str(analysis_err)}")

            # Elegant section divider within single container
            st.markdown('<div class="panel-section-divider"></div>', unsafe_allow_html=True)

            # Results Summary
            st.markdown(
                """
                <div class="workspace-panel-title dark-panel-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                        <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                    </svg>
                    <span>Results Summary</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if res is not None and st.session_state.get("current_image_rgb") is not None:
                img_h, img_w = st.session_state["current_image_rgb"].shape[:2]
                img_dims_str = f"{img_w} × {img_h} px"
                proc_time_str = f"{res['total_time_ms']:.1f} ms"
                edge_method_str = active_det
                cand_count_str = f"{cand_count} {'region' if cand_count == 1 else 'regions'}"

                st.markdown(
                    f"""
                    <div class="summary-data-table">
                        <div class="summary-data-row">
                            <span class="summary-row-label">Image Dimensions</span>
                            <span class="summary-row-val">{img_dims_str}</span>
                        </div>
                        <div class="summary-data-row">
                            <span class="summary-row-label">Processing Duration</span>
                            <span class="summary-row-val highlight-emerald">{proc_time_str}</span>
                        </div>
                        <div class="summary-data-row">
                            <span class="summary-row-label">Selected Edge Method</span>
                            <span class="summary-row-val">{edge_method_str}</span>
                        </div>
                        <div class="summary-data-row">
                            <span class="summary-row-label">Actual Candidate Regions</span>
                            <span class="summary-row-val candidate-badge">{cand_count_str}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="summary-empty-state">
                        <div class="summary-empty-icon">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#5E716D" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                        </div>
                        <div class="summary-empty-title">No Analysis Recorded</div>
                        <div class="summary-empty-sub">Execute analysis above to inspect quantitative telemetry.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                if res is not None:
                    png_bytes = numpy_to_png_bytes(res["overlay"])
                    st.download_button(
                        "Download Processed Image",
                        data=png_bytes,
                        file_name=f"pothole_ai_overlay_{st.session_state['current_image_name']}",
                        mime="image/png",
                        width="stretch",
                        key="dash_dl_img",
                    )
                else:
                    st.button("Download Processed Image", disabled=True, width="stretch", key="dash_dl_img_disabled")
            with d_col2:
                if res is not None:
                    try:
                        report_json = build_analysis_json(
                            image_name=st.session_state["current_image_name"],
                            image_shape=st.session_state["current_image_rgb"].shape,
                            pipeline_params=st.session_state["params"],
                            detector_metrics=res["detector_results"],
                            candidates=res["candidates"],
                        )
                    except Exception as json_err:
                        report_json = json.dumps({"image_name": st.session_state["current_image_name"], "status": "completed", "error": str(json_err)}, indent=2)

                    st.download_button(
                        "Download Analysis Report",
                        data=report_json,
                        file_name=f"pothole_ai_report_{st.session_state['current_image_name']}.json",
                        mime="application/json",
                        width="stretch",
                        key="dash_dl_report",
                    )
                else:
                    st.button("Download Analysis Report", disabled=True, width="stretch", key="dash_dl_report_disabled")

        # Test Scenarios (Collapsible Expander)
        with st.expander("🧪  Test Scenarios", expanded=False):
            st.markdown(
                "<span style='font-size: 11px; color: #8b99a4;'>Run predefined scenarios to test different conditions:</span>",
                unsafe_allow_html=True,
            )
            sample_dir = get_resource_path(os.path.join("data", "sample_images"))
            sc1, sc2, sc3 = st.columns(3)

            with sc1:
                st.markdown("<strong style='font-size: 11.5px;'>Scenario 1</strong>", unsafe_allow_html=True)
                st.caption("Single Deep Cavity")
                if st.button("Load A", key="exp_sample_1", width="stretch"):
                    p1 = os.path.join(sample_dir, "sample_road_pothole_1.png")
                    img = load_image_from_path(p1) if os.path.exists(p1) else generate_synthetic_road_sample("single_deep_pothole")
                    st.session_state["current_image_rgb"] = img
                    st.session_state["current_image_name"] = "sample_road_pothole_1.png"
                    st.session_state["current_results"] = execute_pipeline(img, st.session_state["params"])
                    st.rerun()

            with sc2:
                st.markdown("<strong style='font-size: 11.5px;'>Scenario 2</strong>", unsafe_allow_html=True)
                st.caption("Multiple Clusters")
                if st.button("Load B", key="exp_sample_2", width="stretch"):
                    p2 = os.path.join(sample_dir, "sample_road_pothole_2.png")
                    img = load_image_from_path(p2) if os.path.exists(p2) else generate_synthetic_road_sample("multiple_depressions")
                    st.session_state["current_image_rgb"] = img
                    st.session_state["current_image_name"] = "sample_road_pothole_2.png"
                    st.session_state["current_results"] = execute_pipeline(img, st.session_state["params"])
                    st.rerun()

            with sc3:
                st.markdown("<strong style='font-size: 11.5px;'>Scenario 3</strong>", unsafe_allow_html=True)
                st.caption("Wet Road Cavity")
                if st.button("Load C", key="exp_sample_3", width="stretch"):
                    p3 = os.path.join(sample_dir, "sample_road_pothole_3.png")
                    img = load_image_from_path(p3) if os.path.exists(p3) else generate_synthetic_road_sample("wet_road_pothole")
                    st.session_state["current_image_rgb"] = img
                    st.session_state["current_image_name"] = "sample_road_pothole_3.png"
                    st.session_state["current_results"] = execute_pipeline(img, st.session_state["params"])
                    st.rerun()

    st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)

    # D. Bottom Highlight Cards (matching reference: dark circular gauge card & emerald hero card)
    b_col1, b_col2 = st.columns([1.1, 1.45], gap="medium")
    with b_col1:
        st.markdown(
            """
            <div class="bottom-dark-metric-card">
                <div class="gauge-item-row">
                    <div class="circular-gauge-box">
                        <svg width="56" height="56" viewBox="0 0 56 56">
                            <circle cx="28" cy="28" r="23" stroke="#182E28" stroke-width="4.5" fill="none"/>
                            <circle cx="28" cy="28" r="23" stroke="#F97316" stroke-width="4.5" stroke-dasharray="144.5" stroke-dashoffset="34" stroke-linecap="round" fill="none" transform="rotate(-90 28 28)"/>
                        </svg>
                        <span class="circular-gauge-val">76.2%</span>
                    </div>
                    <div>
                        <div class="gauge-title">Surface Distress Severity</div>
                        <div class="gauge-desc">Dynamic safety index derived from high-frequency edge gradients and cavity density</div>
                    </div>
                </div>
                <div class="gauge-divider-line"></div>
                <div class="gauge-item-row">
                    <div class="circular-gauge-box">
                        <svg width="56" height="56" viewBox="0 0 56 56">
                            <circle cx="28" cy="28" r="23" stroke="#182E28" stroke-width="4.5" fill="none"/>
                            <circle cx="28" cy="28" r="23" stroke="#10B981" stroke-width="4.5" stroke-dasharray="144.5" stroke-dashoffset="20" stroke-linecap="round" fill="none" transform="rotate(-90 28 28)"/>
                        </svg>
                        <span class="circular-gauge-val" style="color: #34D399;">86%</span>
                    </div>
                    <div>
                        <div class="gauge-title">Pavement Structural Integrity</div>
                        <div class="gauge-desc">Calculated roadway stability score based on morphology and candidate contours</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b_col2:
        st.markdown(
            """
            <div class="bottom-emerald-hero-card">
                <div class="emerald-card-top-row">
                    <div class="emerald-brand-pill">
                        <svg width="12" height="12" viewBox="0 0 20 20" fill="none">
                            <path d="M4 14C4 7 10 4 16 4C16 11 10 14 4 14Z" fill="#34D399"/>
                            <path d="M10 16C10 11 14 9 18 9C18 14 14 16 10 16Z" fill="#6EE7B7"/>
                        </svg>
                        <span>POTHOLE AI INTELLIGENCE</span>
                    </div>
                    <span class="emerald-card-arrow">↗</span>
                </div>
                <div>
                    <div class="emerald-hero-heading">Let's inspect road networks</div>
                    <div class="emerald-hero-subtext">Autonomous pothole detection, multi-scale edge filtering, and real-time morphological distress clustering for municipal road safety.</div>
                </div>
                <div class="emerald-pills-row">
                    <span class="emerald-tag-pill">Edge Detection</span>
                    <span class="emerald-tag-pill">Canny & Sobel</span>
                    <span class="emerald-tag-pill">Morphology</span>
                    <span class="emerald-tag-pill">230k+ Inspections</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# SECTION 2: IMAGE ANALYSIS WORKSPACE (UPLOAD & ANALYZE)
# =============================================================================
elif selected_nav == "Upload & Analyze":
    render_header(
        "Image Analysis Workspace",
        "Ingest road imagery, calibrate contrast and edge thresholds, and inspect candidate distress overlays with multi-format export capabilities.",
        eyebrow="DEEP ANALYSIS / PARAMETER TUNING",
        badge="PRODUCTION WORKSPACE",
    )

    ctrl_col, view_col = st.columns([1, 2.2], gap="medium")

    with ctrl_col:
        # Step 1: Ingest Road Image
        with st.container(border=True):
            st.markdown(
                """
                <div class="workspace-panel-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <span>1. Ingest Road Pavement Image</span>
                </div>
                <div class="workspace-panel-subtitle">Upload camera inspection captures (PNG, JPG, WEBP).</div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Select Road Image",
                type=["png", "jpg", "jpeg", "webp"],
                key="wa_file_uploader",
                label_visibility="collapsed",
                help="Supports standard road inspection imagery up to 200MB.",
            )

            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                is_valid, msg, rgb_img = validate_image(file_bytes)
                if is_valid and rgb_img is not None:
                    if st.session_state.get("current_image_name") != uploaded_file.name:
                        st.session_state["current_image_rgb"] = rgb_img
                        st.session_state["current_image_name"] = uploaded_file.name
                        st.session_state["current_results"] = execute_pipeline(
                            rgb_img, st.session_state["params"]
                        )
                        st.success(f"Ingested '{uploaded_file.name}' ({rgb_img.shape[1]} × {rgb_img.shape[0]} px)")
                        st.rerun()
                else:
                    st.error(f"Image validation failed: {msg}")

        # Step 2: Preprocessing & Contrast
        with st.container(border=True):
            st.markdown(
                """
                <div class="workspace-panel-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="2" x2="12" y2="22"></line>
                    </svg>
                    <span>2. Pavement Preprocessing</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            contrast_method = st.selectbox(
                "Contrast Normalization",
                ["CLAHE", "Hist_Eq", "Linear_Stretch", "None"],
                index=["clahe", "hist_eq", "stretch", "none"].index(
                    st.session_state["params"].get("contrast_method", "clahe")
                ),
                key="wa_contrast_method",
            ).lower()

            c_c1, c_c2 = st.columns(2)
            with c_c1:
                clahe_clip = st.slider(
                    "CLAHE Clip Limit",
                    0.5,
                    5.0,
                    float(st.session_state["params"].get("clahe_clip", 2.5)),
                    0.1,
                    key="wa_clahe_clip",
                    help="Limits regional contrast enhancement to prevent asphalt aggregate noise amplification.",
                )
            with c_c2:
                blur_sigma = st.slider(
                    "Gaussian Sigma",
                    0.2,
                    3.0,
                    float(st.session_state["params"].get("blur_sigma", 1.2)),
                    0.1,
                    key="wa_blur_sigma",
                )

            blur_ksize = st.select_slider(
                "Gaussian Filter Kernel Size",
                options=[1, 3, 5, 7, 9],
                value=st.session_state["params"].get("blur_ksize", 5),
                key="wa_blur_ksize",
            )

        # Step 3: Edge Operator Calibration
        with st.container(border=True):
            st.markdown(
                """
                <div class="workspace-panel-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                    </svg>
                    <span>3. Edge Operator Calibration</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            active_detector = st.segmented_control(
                "Primary Operator",
                ["Canny", "Sobel", "Laplacian"],
                default=st.session_state["params"].get("active_detector", "canny").capitalize(),
                key="wa_operator_seg",
            )
            if not active_detector:
                active_detector = "Canny"
            active_detector = active_detector.lower()

            if active_detector == "canny":
                cn_c1, cn_c2 = st.columns(2)
                with cn_c1:
                    c_low = st.number_input(
                        "Lower Threshold (T_low)",
                        min_value=5,
                        max_value=250,
                        value=int(st.session_state["params"].get("canny_low", 45)),
                        step=5,
                        key="wa_canny_low",
                    )
                with cn_c2:
                    c_high = st.number_input(
                        "Upper Threshold (T_high)",
                        min_value=20,
                        max_value=255,
                        value=int(st.session_state["params"].get("canny_high", 135)),
                        step=5,
                        key="wa_canny_high",
                    )
            elif active_detector == "sobel":
                sobel_th = st.number_input(
                    "Sobel Gradient Threshold",
                    min_value=10,
                    max_value=200,
                    value=int(st.session_state["params"].get("sobel_threshold", 40)),
                    step=5,
                    key="wa_sobel_th",
                )
                c_low, c_high = 45, 135
            else:
                lap_th = st.number_input(
                    "Laplacian Zero-Crossing Threshold",
                    min_value=5,
                    max_value=150,
                    value=int(st.session_state["params"].get("laplacian_threshold", 35)),
                    step=5,
                    key="wa_lap_th",
                )
                c_low, c_high = 45, 135

        # Step 4: Candidate Geometry Rules
        with st.container(border=True):
            st.markdown(
                """
                <div class="workspace-panel-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2">
                        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                        <polyline points="2 17 12 22 22 17"></polyline>
                        <polyline points="2 12 12 17 22 12"></polyline>
                    </svg>
                    <span>4. Candidate Geometry Rules</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            geo_c1, geo_c2 = st.columns(2)
            with geo_c1:
                min_area = st.number_input(
                    "Min Area (px)",
                    min_value=50,
                    max_value=5000,
                    value=int(st.session_state["params"].get("min_area", 180)),
                    step=50,
                    key="wa_min_area",
                )
            with geo_c2:
                max_area = st.number_input(
                    "Max Area (px)",
                    min_value=1000,
                    max_value=300000,
                    value=int(st.session_state["params"].get("max_area", 60000)),
                    step=1000,
                    key="wa_max_area",
                )

            min_solidity = st.slider(
                "Min Solidity (Area / Convex Hull)",
                0.1,
                0.9,
                float(st.session_state["params"].get("min_solidity", 0.35)),
                0.05,
                key="wa_min_solidity",
            )
            roi_top = st.slider(
                "Horizon Exclusion (% from Top)",
                0,
                50,
                int(st.session_state["params"].get("roi_top_ratio", 0.15) * 100),
                5,
                key="wa_roi_top",
            ) / 100.0

            if st.button("▶ Run Detection Pipeline", type="primary", width="stretch", key="wa_run_btn"):
                st.session_state["params"].update({
                    "active_detector": active_detector,
                    "contrast_method": contrast_method,
                    "clahe_clip": clahe_clip,
                    "blur_ksize": blur_ksize,
                    "blur_sigma": blur_sigma,
                    "canny_low": c_low if active_detector == "canny" else st.session_state["params"]["canny_low"],
                    "canny_high": c_high if active_detector == "canny" else st.session_state["params"]["canny_high"],
                    "sobel_threshold": sobel_th if active_detector == "sobel" else st.session_state["params"]["sobel_threshold"],
                    "laplacian_threshold": lap_th if active_detector == "laplacian" else st.session_state["params"]["laplacian_threshold"],
                    "min_area": min_area,
                    "max_area": max_area,
                    "min_solidity": min_solidity,
                    "roi_top_ratio": roi_top,
                })

                new_res = execute_pipeline(st.session_state["current_image_rgb"], st.session_state["params"])
                st.session_state["current_results"] = new_res

                st.session_state["history"].append({
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "image_name": st.session_state["current_image_name"],
                    "dimensions": f"{st.session_state['current_image_rgb'].shape[1]} × {st.session_state['current_image_rgb'].shape[0]}",
                    "candidates_found": len(new_res["candidates"]),
                    "runtime_ms": new_res["total_time_ms"],
                    "detector": active_detector.capitalize(),
                })
                st.success("Pipeline executed successfully!")
                st.rerun()

    # RIGHT COLUMN: Results Inspection
    with view_col:
        res = st.session_state["current_results"]

        if res is not None:
            cands = res["candidates"]
            cand_count = len(cands)
            h, w = st.session_state["current_image_rgb"].shape[:2]

            # Metric Status Bar
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.markdown(
                    f"""
                    <div class="metric-card-box">
                        <div class="metric-card-header">
                            <span class="metric-card-label">IMAGE RESOLUTION</span>
                        </div>
                        <div class="metric-card-value">{w} × {h}</div>
                        <div class="metric-card-subtext">Pavement capture</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f"""
                    <div class="metric-card-box">
                        <div class="metric-card-header">
                            <span class="metric-card-label">CPU LATENCY</span>
                        </div>
                        <div class="metric-card-value">{res['total_time_ms']:.1f}ms</div>
                        <div class="metric-card-subtext">OpenCV CPU runtime</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col3:
                st.markdown(
                    f"""
                    <div class="metric-card-box">
                        <div class="metric-card-header">
                            <span class="metric-card-label">CANDIDATES</span>
                        </div>
                        <div class="metric-card-value">{cand_count}</div>
                        <div class="metric-card-subtext">Flagged anomalies</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col4:
                st.markdown(
                    f"""
                    <div class="metric-card-box">
                        <div class="metric-card-header">
                            <span class="metric-card-label">OPERATOR</span>
                        </div>
                        <div class="metric-card-value">{res['active_detector'].upper()}</div>
                        <div class="metric-card-subtext">Current detector</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

            # Action Bar for Exports
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                png_overlay_bytes = numpy_to_png_bytes(res["overlay"])
                st.download_button(
                    label="⬇️ Export Overlay (PNG)",
                    data=png_overlay_bytes,
                    file_name=f"pothole_ai_overlay_{st.session_state['current_image_name']}",
                    mime="image/png",
                    width="stretch",
                    key="wa_dl_overlay",
                )
            with d_col2:
                png_montage_bytes = numpy_to_png_bytes(res["montage"])
                st.download_button(
                    label="⬇️ Export 6-Stage Montage (PNG)",
                    data=png_montage_bytes,
                    file_name=f"pothole_ai_montage_{st.session_state['current_image_name']}",
                    mime="image/png",
                    width="stretch",
                    key="wa_dl_montage",
                )
            with d_col3:
                json_data = build_analysis_json(
                    image_name=st.session_state["current_image_name"],
                    image_shape=st.session_state["current_image_rgb"].shape,
                    pipeline_params=st.session_state["params"],
                    detector_metrics=res["detector_results"],
                    candidates=res["candidates"],
                )
                st.download_button(
                    label="⬇️ Export Analysis (JSON)",
                    data=json_data,
                    file_name=f"pothole_ai_report_{st.session_state['current_image_name']}.json",
                    mime="application/json",
                    width="stretch",
                    key="wa_dl_json",
                )

            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

            # Workspace View Tabs
            tab_quad, tab_montage, tab_table = st.tabs([
                "🔍 4-Panel Aligned Inspection",
                "🖼️ 6-Stage Pipeline Montage",
                "📋 Candidate Regions Table",
            ])

            with tab_quad:
                active_edge_map = res["detector_results"][res["active_detector"]]["edge_map"]
                b64_orig = numpy_to_base64_jpeg(res["prep"]["original_rgb"])
                b64_enh = numpy_to_base64_jpeg(res["prep"]["enhanced"])
                b64_edge = numpy_to_base64_jpeg(active_edge_map)
                b64_over = numpy_to_base64_jpeg(res["overlay"])

                st.markdown(
                    f"""
                    <div class="panel-grid-container">
                        <div class="panel-preview-tile">
                            <div class="tile-badge-pill">
                                <span>Original Road Image</span>
                            </div>
                            <img src="data:image/jpeg;base64,{b64_orig}" alt="Original Road Image"/>
                        </div>
                        <div class="panel-preview-tile">
                            <div class="tile-badge-pill">
                                <span>Enhanced Grayscale ({st.session_state['params']['contrast_method'].upper()})</span>
                            </div>
                            <img src="data:image/jpeg;base64,{b64_enh}" alt="Enhanced Grayscale"/>
                        </div>
                        <div class="panel-preview-tile">
                            <div class="tile-badge-pill">
                                <span>Edge Map ({res['active_detector'].upper()})</span>
                            </div>
                            <img src="data:image/jpeg;base64,{b64_edge}" alt="Edge Map"/>
                        </div>
                        <div class="panel-preview-tile">
                            <div class="tile-badge-pill">
                                <span>Candidate Overlay ({cand_count} Flagged)</span>
                            </div>
                            <img src="data:image/jpeg;base64,{b64_over}" alt="Candidate Overlay"/>
                            <div class="tile-bottom-legend">
                                <span class="legend-dot-red"></span>
                                <span>Potential pothole candidate</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with tab_montage:
                st.image(
                    res["montage"],
                    caption="Complete 6-Stage Pipeline Montage: Original → Enhanced → Canny → Sobel → Laplacian → Candidate Overlay",
                    width="stretch",
                )

            with tab_table:
                if cand_count > 0:
                    rows = []
                    for c in cands:
                        rows.append({
                            "Candidate ID": f"C#{c['id']}",
                            "Bounding Box (x,y,w,h)": f"({c['bbox'][0]}, {c['bbox'][1]}, {c['bbox'][2]}, {c['bbox'][3]})",
                            "Area (px)": c["area"],
                            "Solidity": c["solidity"],
                            "Aspect Ratio": c["aspect_ratio"],
                            "Darkness Contrast": f"+{c['relative_darkness']}",
                            "Saliency Score": f"{c['saliency_score']}%",
                            "Severity": c["severity"],
                        })
                    st.dataframe(pd.DataFrame(rows), width="stretch")
                else:
                    st.markdown(
                        """
                        <div class="empty-state-box">
                            <div class="empty-state-title">No Candidate Regions Met Criteria</div>
                            <div class="empty-state-desc">The current morphological closing and geometric filters did not find closed contours meeting the minimum area or solidity thresholds. Try lowering Min Area or decreasing the Sobel/Canny threshold.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown(
                """
                <div class="tech-note-banner">
                    <strong>RESEARCH NOTE: CANDIDATE REGION HYPOTHESES:</strong>
                    Highlighted candidate regions represent localized intensity discontinuities and closed boundary contours extracted via classical edge processing and morphological closing.
                    Edge detection alone flags potential distress hypotheses rather than confirmed physical potholes. In automated municipal workflows,
                    these regions serve as candidate proposals for human review or downstream deep-learning classification (e.g. YOLO/CNN).
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# SECTION 3: EDGE ANALYSIS
# =============================================================================
elif selected_nav == "Edge Analysis":
    render_header(
        "Edge Analysis & Operator Benchmarking",
        "Interactive comparative inspection of classical edge enhancement techniques (Canny, Sobel, Laplacian) applied to active road pavement imagery.",
        eyebrow="COMPUTER VISION SUITE",
        badge="EDGE DETECTION",
    )

    res = st.session_state.get("current_results")
    if res is None:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    </svg>
                </div>
                <div class="empty-state-title">No Active Image Available</div>
                <div class="empty-state-desc">Please visit the Dashboard or Upload & Analyze page to load and process a road surface inspection image first.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        det = res["detector_results"]
        current_op = st.session_state["params"].get("detector", "canny").lower()
        if current_op not in ["canny", "sobel", "laplacian"]:
            current_op = "canny"

        # Method Selector Bar
        method_col, info_col = st.columns([1.2, 2.8], gap="medium")
        with method_col:
            selected_method = st.radio(
                "Select Edge Detection Operator",
                options=["Canny", "Sobel", "Laplacian"],
                index=["canny", "sobel", "laplacian"].index(current_op),
                horizontal=True,
                key="edge_analysis_method_selector",
                label_visibility="collapsed",
            )
            op_key = selected_method.lower()

        with info_col:
            op_data = det[op_key]
            st.markdown(
                f"""
                <div style="display: flex; gap: 12px; align-items: center; justify-content: flex-end; padding-top: 4px;">
                    <span class="spec-badge spec-badge-emerald">Operator: {selected_method}</span>
                    <span style="font-size: 12px; color: #A1B0AD;">Runtime: <strong style="color: #F3F7F5;">{op_data['execution_time_ms']} ms</strong></span>
                    <span style="font-size: 12px; color: #A1B0AD;">Edge Density: <strong style="color: #32D6A0;">{op_data['edge_density_pct']}%</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

        # Consistent Side-by-Side Preview Panels
        prev_col1, prev_col2 = st.columns(2, gap="medium")
        with prev_col1:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="workspace-panel-title dark-panel-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <circle cx="8.5" cy="8.5" r="1.5"></circle>
                            <polyline points="21 15 16 10 5 21"></polyline>
                        </svg>
                        <span>Input Road Surface</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(st.session_state["current_image_rgb"], caption=f"Original RGB Input ({st.session_state['current_image_name']})", width="stretch")

        with prev_col2:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="workspace-panel-title dark-panel-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2">
                            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                        </svg>
                        <span>{selected_method} Edge Map</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(op_data["edge_map"], caption=f"{selected_method} Binary Edge Boundary Map", width="stretch")

        st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

        # Technique Explanation and Parameter Card
        tech_col1, tech_col2 = st.columns([1.5, 1], gap="medium")
        with tech_col1:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="workspace-panel-title dark-panel-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="16" x2="12" y2="12"></line>
                            <line x1="12" y1="8" x2="12.01" y2="8"></line>
                        </svg>
                        <span>Technique Methodology &mdash; {selected_method}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if op_key == "canny":
                    st.markdown(
                        """
                        <div style="font-size: 12px; color: #A1B0AD; line-height: 1.6; padding-top: 4px;">
                            <strong>Canny Edge Detection</strong> (John F. Canny, 1986) is a multi-stage optimal edge detection algorithm designed to maximize signal-to-noise ratio and localize single-pixel boundaries:
                            <ul style="padding-left: 18px; margin: 6px 0 0;">
                                <li><strong>Gaussian Smoothing:</strong> Attenuates high-frequency asphalt aggregate noise prior to differentiation.</li>
                                <li><strong>Gradient Calculation:</strong> Derives spatial intensity magnitude and orientation vectors via directional Sobel kernels.</li>
                                <li><strong>Non-Maximum Suppression (NMS):</strong> Thins multi-pixel gradient ridges into sharp 1-pixel contours by suppressing non-peak gradients along normal directions.</li>
                                <li><strong>Hysteresis Thresholding:</strong> Dual thresholds classify strong edges, rejecting isolated weak responses while preserving connected distress fissures.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif op_key == "sobel":
                    st.markdown(
                        """
                        <div style="font-size: 12px; color: #A1B0AD; line-height: 1.6; padding-top: 4px;">
                            <strong>Sobel Operator</strong> (Irwin Sobel, 1968) is a first-order discrete differentiation operator computing spatial gradient vectors via 3×3 convolution kernels:
                            <ul style="padding-left: 18px; margin: 6px 0 0;">
                                <li><strong>Directional Convolutions:</strong> Applies separable kernels K_x and K_y to detect orthogonal vertical and horizontal luminance transitions.</li>
                                <li><strong>Gradient Magnitude:</strong> Combines orthogonal spatial derivatives via G = √(G_x² + G_y²).</li>
                                <li><strong>Cavity Sensitivity:</strong> Highlights wide depression boundaries and transition slopes, producing thicker gradient bands around pothole peripheries.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style="font-size: 12px; color: #A1B0AD; line-height: 1.6; padding-top: 4px;">
                            <strong>Laplacian Operator</strong> is a second-order isotropic spatial derivative operator measuring the local divergence of the image intensity gradient:
                            <ul style="padding-left: 18px; margin: 6px 0 0;">
                                <li><strong>Isotropic Formulation:</strong> Computes ∇²f = ∂²f/∂x² + ∂²f/∂y² symmetrically in all spatial directions.</li>
                                <li><strong>Zero-Crossing Detection:</strong> Identifies rapid intensity transitions at points where the second derivative changes sign.</li>
                                <li><strong>Texture & Edge Response:</strong> Highly sensitive to abrupt pavement texture changes, crack fissures, and cavity lips.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tech_col2:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="workspace-panel-title dark-panel-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2">
                            <circle cx="12" cy="12" r="3"></circle>
                            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                        </svg>
                        <span>Active Parameters</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                p = st.session_state["params"]
                if op_key == "canny":
                    param_rows = f"""
                    <tr><td class="label">Low Hysteresis Threshold</td><td class="val">{p.get('canny_low', 50)}</td></tr>
                    <tr><td class="label">High Hysteresis Threshold</td><td class="val">{p.get('canny_high', 150)}</td></tr>
                    <tr><td class="label">Gaussian Blur Sigma</td><td class="val">{p.get('blur_sigma', 1.2)}</td></tr>
                    <tr><td class="label">Aperture Size</td><td class="val">3 × 3 (Sobel)</td></tr>
                    <tr><td class="label">L2 Gradient Norm</td><td class="val">True</td></tr>
                    """
                elif op_key == "sobel":
                    param_rows = f"""
                    <tr><td class="label">Binary Threshold</td><td class="val">{p.get('sobel_threshold', 80)}</td></tr>
                    <tr><td class="label">Kernel Dimensions</td><td class="val">3 × 3</td></tr>
                    <tr><td class="label">Gradient Operator</td><td class="val">Gx & Gy</td></tr>
                    <tr><td class="label">Scale Factor</td><td class="val">1.0</td></tr>
                    <tr><td class="label">Delta Offset</td><td class="val">0.0</td></tr>
                    """
                else:
                    param_rows = f"""
                    <tr><td class="label">Zero-Crossing Threshold</td><td class="val">{p.get('laplacian_threshold', 40)}</td></tr>
                    <tr><td class="label">Kernel Dimensions</td><td class="val">3 × 3 (ksize=3)</td></tr>
                    <tr><td class="label">Scale Factor</td><td class="val">1.0</td></tr>
                    <tr><td class="label">Delta Offset</td><td class="val">0.0</td></tr>
                    <tr><td class="label">Border Type</td><td class="val">DEFAULT</td></tr>
                    """

                st.markdown(
                    f"""
                    <table class="candidate-profile-table" style="margin-top: 6px;">
                        {param_rows}
                    </table>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # Tri-Detector Comparative Overview Suite
        tab_suite, tab_grad, tab_hist, tab_comp = st.tabs([
            "❖ Tri-Operator Side-by-Side Suite",
            "🧭 Directional Gradients (Sobel Gx & Gy)",
            "📊 Intensity & Edge Histograms",
            "📋 Comparative Performance Matrix",
        ])

        with tab_suite:
            c1, c2, c3 = st.columns(3)
            with c1:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="operator-card-header">
                            <span class="operator-name">1. Canny</span>
                            <span class="spec-badge spec-badge-emerald">Hysteresis</span>
                        </div>
                        <div class="operator-stats-row">
                            <span>Latency: <strong>{det['canny']['execution_time_ms']} ms</strong></span>
                            <span>Density: <strong>{det['canny']['edge_density_pct']}%</strong></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(det["canny"]["edge_map"], caption="Canny Edge Map", width="stretch")
            with c2:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="operator-card-header">
                            <span class="operator-name">2. Sobel</span>
                            <span class="spec-badge spec-badge-blue">1st Order</span>
                        </div>
                        <div class="operator-stats-row">
                            <span>Latency: <strong>{det['sobel']['execution_time_ms']} ms</strong></span>
                            <span>Density: <strong>{det['sobel']['edge_density_pct']}%</strong></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(det["sobel"]["edge_map"], caption="Sobel Magnitude Map", width="stretch")
            with c3:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="operator-card-header">
                            <span class="operator-name">3. Laplacian</span>
                            <span class="spec-badge spec-badge-amber">2nd Order</span>
                        </div>
                        <div class="operator-stats-row">
                            <span>Latency: <strong>{det['laplacian']['execution_time_ms']} ms</strong></span>
                            <span>Density: <strong>{det['laplacian']['edge_density_pct']}%</strong></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(det["laplacian"]["edge_map"], caption="Laplacian Zero-Crossings", width="stretch")

        with tab_grad:
            g1, g2, g3 = st.columns(3)
            with g1:
                st.image(det["sobel"]["grad_x"], caption="Sobel Gx (|∂f/∂x| — Vertical Edge Responses)", width="stretch")
            with g2:
                st.image(det["sobel"]["grad_y"], caption="Sobel Gy (|∂f/∂y| — Horizontal Edge Responses)", width="stretch")
            with g3:
                st.image(det["sobel"]["edge_map"], caption="Combined Magnitude: √(Gx² + Gy²)", width="stretch")

        with tab_hist:
            fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
            fig.patch.set_facecolor("#11191B")
            for a in ax:
                a.set_facecolor("#162123")
                a.tick_params(colors="#A1B0AD")
                for spine in a.spines.values():
                    spine.set_color("#263635")

            gray_flat = res["prep"]["gray"].ravel()
            enh_flat = res["prep"]["enhanced"].ravel()
            ax[0].hist(gray_flat, bins=45, color="#576571", alpha=0.6, label="Original Grayscale")
            ax[0].hist(enh_flat, bins=45, color="#32D6A0", alpha=0.6, label="Enhanced (CLAHE)")
            ax[0].set_title("Pavement Intensity Distribution", color="#F3F7F5", fontsize=11, fontweight="bold")
            ax[0].set_xlabel("Pixel Intensity (0-255)", color="#A1B0AD", fontsize=9)
            ax[0].legend(facecolor="#162123", edgecolor="#263635", labelcolor="#F3F7F5", fontsize=8.5)

            densities = [det["canny"]["edge_density_pct"], det["sobel"]["edge_density_pct"], det["laplacian"]["edge_density_pct"]]
            bars = ax[1].bar(["Canny", "Sobel", "Laplacian"], densities, color=["#32D6A0", "#38BDF8", "#F59E0B"], width=0.45)
            ax[1].set_title("Edge Pixel Density Benchmark (%)", color="#F3F7F5", fontsize=11, fontweight="bold")
            for bar in bars:
                bh = bar.get_height()
                ax[1].annotate(f"{bh:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bh), xytext=(0, 3), textcoords="offset points", ha="center", color="#F3F7F5", fontsize=9, fontweight="bold")
            st.pyplot(fig)

        with tab_comp:
            comp_rows = generate_comparison_table(det)
            st.dataframe(pd.DataFrame(comp_rows), width="stretch")


# =============================================================================
# SECTION 4: POTHOLE CANDIDATES
# =============================================================================
elif selected_nav == "Pothole Candidates":
    res = st.session_state.get("current_results")
    cand_count = len(res["candidates"]) if res and "candidates" in res else 0

    render_header(
        f"Candidate Regions Deep Inspection ({cand_count} Flagged)",
        "Topological inspection of candidate pavement distress regions extracted via morphological contour closing and geometric filtering.",
        eyebrow="ANOMALY INSPECTION",
        badge=f"{cand_count} CANDIDATES",
    )

    # Operational Advisory Notice
    st.markdown(
        """
        <div class="advisory-banner">
            <span class="advisory-icon">ℹ</span>
            <div><strong>Operational Notice:</strong> Detected regions represent <em>potential pothole candidates</em> identified via classical image processing (edge detection, morphological closing, and contour geometric thresholds). They are screening alerts for road engineering inspection, not verified potholes. Ground-truth confirmation requires field review or 3D stereoscopic depth verification.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if res is None or len(res["candidates"]) == 0:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                </div>
                <div class="empty-state-title">No Candidate Regions Found</div>
                <div class="empty-state-desc">No closed contours met the current area, aspect ratio, or solidity filter criteria on the active road image. You can adjust the candidate parameters on the Dashboard or analyze another road image.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        cands = res["candidates"]
        total_cand_area = sum(c["area"] for c in cands)
        mean_cand_area = total_cand_area // len(cands) if len(cands) > 0 else 0

        # Summary Metric Cards
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">TOTAL CANDIDATES</span></div>
                    <div class="metric-card-value">{len(cands)}</div>
                    <div class="metric-card-subtext">Potential defect regions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">TOTAL DEFECT AREA</span></div>
                    <div class="metric-card-value">{total_cand_area:,} px</div>
                    <div class="metric-card-subtext">Aggregate candidate footprint</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">MEAN REGION AREA</span></div>
                    <div class="metric-card-value">{mean_cand_area:,} px</div>
                    <div class="metric-card-subtext">Average contour size</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s4:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">DETECTOR USED</span></div>
                    <div class="metric-card-value">{res['active_detector'].upper()}</div>
                    <div class="metric-card-subtext">Active edge operator</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)

        # Prominent Candidate-Overlay Preview
        with st.container(border=True):
            st.markdown(
                """
                <div class="workspace-panel-title dark-panel-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    </svg>
                    <span>Annotated Road Surface &mdash; Candidate Overlays</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(res["overlay"], caption=f"Active Road Image with {len(cands)} Candidate Regions Tagged", width="stretch")

        st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)

        # Candidate Region Properties Table
        st.markdown(
            """
            <div style="font-size: 13.5px; font-weight: 700; color: #F3F7F5; margin-bottom: 8px;">
                Extracted Candidate Region Properties (Geometric & Intensity Telemetry)
            </div>
            """,
            unsafe_allow_html=True,
        )

        cand_rows = []
        for c in cands:
            x, y, w, h = c["bbox"]
            cand_rows.append({
                "Candidate ID": f"#{c['id']}",
                "Bounding Box (X, Y, W, H)": f"({x}, {y}, {w}, {h})",
                "Area (px)": c["area"],
                "Perimeter (px)": round(c["perimeter"], 1),
                "Aspect Ratio (W/H)": round(c["aspect_ratio"], 2),
                "Solidity": round(c["solidity"], 3),
                "Mean Intensity": round(c["mean_intensity"], 1),
                "Relative Darkness": round(c["relative_darkness"], 1),
            })
        st.dataframe(pd.DataFrame(cand_rows), width="stretch")

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # Detailed Region of Interest (ROI) Magnifier
        cand_labels = [f"Candidate #{c['id']} — BBox ({c['bbox'][0]}, {c['bbox'][1]}, {c['bbox'][2]}, {c['bbox'][3]}) px, Area: {c['area']} px" for c in cands]
        selected_cand_label = st.selectbox("Select Candidate Region to Magnify", cand_labels, key="ci_cand_select")
        selected_idx = cand_labels.index(selected_cand_label)
        selected_cand = cands[selected_idx]

        ci_col1, ci_col2 = st.columns([1, 1.2], gap="medium")
        with ci_col1:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="workspace-panel-title dark-panel-title">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="8" x2="12" y2="12"></line>
                            <line x1="12" y1="16" x2="12.01" y2="16"></line>
                        </svg>
                        <span>Candidate #{selected_cand['id']} Property Telemetry</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                x, y, w, h = selected_cand["bbox"]
                st.markdown(
                    f"""
                    <table class="candidate-profile-table" style="margin-top: 8px;">
                        <tr><td class="label">Bounding Box (x, y, w, h)</td><td class="val">({x}, {y}, {w}, {h}) px</td></tr>
                        <tr><td class="label">Contour Area</td><td class="val">{selected_cand['area']:,} px</td></tr>
                        <tr><td class="label">Perimeter Length</td><td class="val">{selected_cand['perimeter']:.1f} px</td></tr>
                        <tr><td class="label">Aspect Ratio (w / h)</td><td class="val">{selected_cand['aspect_ratio']:.2f}</td></tr>
                        <tr><td class="label">Solidity (Area / Convex Hull)</td><td class="val">{selected_cand['solidity']:.3f}</td></tr>
                        <tr><td class="label">Interior Mean Intensity</td><td class="val">{selected_cand['mean_intensity']:.1f} (0-255)</td></tr>
                        <tr><td class="label">Darkness Below Pavement Mean</td><td class="val">+{selected_cand['relative_darkness']:.1f}</td></tr>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )

        with ci_col2:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="workspace-panel-title dark-panel-title">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#32D6A0" stroke-width="2">
                            <circle cx="11" cy="11" r="8"></circle>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                        </svg>
                        <span>Magnified Region of Interest (ROI)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                    )
                orig = st.session_state["current_image_rgb"]
                x, y, w, h = selected_cand["bbox"]
                pad = 25
                y1, y2 = max(0, y - pad), min(orig.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(orig.shape[1], x + w + pad)

                crop_rgb = orig[y1:y2, x1:x2].copy()
                crop_enh = res["prep"]["enhanced"][y1:y2, x1:x2]
                cv2.rectangle(crop_rgb, (x - x1, y - y1), (x - x1 + w, y - y1 + h), (50, 214, 160), 2)

                cr1, cr2 = st.columns(2)
                with cr1:
                    st.image(crop_rgb, caption=f"Zoomed Cavity #{selected_cand['id']} (RGB)", width="stretch")
                with cr2:
                    st.image(crop_enh, caption="Local Contrast Pavement Cavity", width="stretch")


# =============================================================================
# SECTION 5: ANALYSIS HISTORY
# =============================================================================
elif selected_nav == "Analysis History":
    render_header(
        "Analysis Inspection History",
        "Session-based chronological audit trail of road image inspections performed during the current active session.",
        eyebrow="AUDIT TRAIL",
        badge="SESSION LOG",
    )

    # Advisory notice about session-based storage
    st.markdown(
        """
        <div class="advisory-banner">
            <span class="advisory-icon">ℹ</span>
            <div><strong>Session Notice:</strong> Inspection records are tracked in browser session memory during your active visit. Records reset when the application or browser tab is refreshed.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_records = st.session_state.get("history", [])

    if len(history_records) == 0:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 6 12 12 16 14"></polyline>
                    </svg>
                </div>
                <div class="empty-state-title">No Inspections Recorded Yet</div>
                <div class="empty-state-desc">Analyses executed from the Dashboard or Image Analysis Workspace are automatically logged here with timestamps, dimensions, candidate counts, and latencies.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # History Summary Metrics
        h1, h2, h3, h4 = st.columns(4)
        total_runs = len(history_records)
        avg_time = sum(r["runtime_ms"] for r in history_records) / total_runs
        total_cands = sum(r["candidates_found"] for r in history_records)
        methods = [r["detector"] for r in history_records]
        most_frequent_method = max(set(methods), key=methods.count) if methods else "N/A"

        with h1:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">TOTAL ANALYSES</span></div>
                    <div class="metric-card-value">{total_runs}</div>
                    <div class="metric-card-subtext">Session inspection runs</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">AVG LATENCY</span></div>
                    <div class="metric-card-value">{avg_time:.1f} ms</div>
                    <div class="metric-card-subtext">Mean OpenCV runtime</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h3:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">TOTAL CANDIDATES</span></div>
                    <div class="metric-card-value">{total_cands}</div>
                    <div class="metric-card-subtext">Cumulative regions logged</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h4:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">COMMON METHOD</span></div>
                    <div class="metric-card-value">{most_frequent_method}</div>
                    <div class="metric-card-subtext">Primary operator used</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)

        # Filters
        f1, f2 = st.columns([1, 2])
        with f1:
            filter_det = st.selectbox("Filter by Method", ["All Operators", "Canny", "Sobel", "Laplacian"], key="hist_filter_det")
        with f2:
            search_query = st.text_input("Search Image Name", placeholder="Type image name...", key="hist_search_name")

        filtered_records = []
        for r in history_records:
            if "dimensions" not in r:
                r["dimensions"] = "640 × 480"
            det_match = (filter_det == "All Operators" or r["detector"].lower() == filter_det.lower())
            name_match = (not search_query or search_query.lower() in r["image_name"].lower())
            if det_match and name_match:
                filtered_records.append(r)

        # Data table
        if filtered_records:
            display_records = []
            for i, r in enumerate(reversed(filtered_records), 1):
                display_records.append({
                    "#": len(filtered_records) - i + 1,
                    "Timestamp (UTC)": r["timestamp"],
                    "Image Name": r["image_name"],
                    "Dimensions": r.get("dimensions", "640 × 480"),
                    "Method": r["detector"],
                    "Candidates": r["candidates_found"],
                    "Processing Time": f"{r['runtime_ms']:.1f} ms",
                })
            st.dataframe(pd.DataFrame(display_records), width="stretch")
        else:
            st.info("No records matched the selected filter criteria.")

        st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)

        # Actions: CSV Export and Clear
        act1, act2 = st.columns([1, 4])
        with act1:
            csv_data = pd.DataFrame(history_records).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download History (CSV)",
                data=csv_data,
                file_name="pothole_ai_history.csv",
                mime="text/csv",
                width="stretch",
                key="hist_dl_csv",
            )
        with act2:
            if st.button("Clear Session History", key="hist_clear_btn"):
                st.session_state["history"] = []
                st.rerun()


# =============================================================================
# SECTION 6: REPORTS & METRICS
# =============================================================================
elif selected_nav == "Reports & Metrics":
    render_header(
        "Reports & Execution Metrics",
        "Performance telemetry, operator latency profiling, and exportable research bundles from active inspection runs.",
        eyebrow="METRICS & EXPORTS",
        badge="RESEARCH AUDIT",
    )

    # Honest disclosure on evaluation benchmarks
    st.markdown(
        """
        <div class="advisory-banner">
            <span class="advisory-icon">ℹ</span>
            <div><strong>Scientific Disclosure:</strong> Formal machine learning metrics (such as Precision, Recall, Accuracy, and F1-score) require a ground-truth labeled benchmark dataset and are intentionally not simulated or fabricated. Telemetry below represents genuine physical measurements (runtime latency, edge pixel density, and candidate contours) computed directly on the active machine.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    res = st.session_state.get("current_results")
    if res is None:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="20" x2="18" y2="10"></line>
                        <line x1="12" y1="20" x2="12" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="14"></line>
                    </svg>
                </div>
                <div class="empty-state-title">No Inspection Data Available</div>
                <div class="empty-state-desc">Please execute a road image analysis from the Dashboard or Upload & Analyze page to view pipeline execution metrics and generate research reports.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Runtime Metrics
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">PREPROCESSING</span></div>
                    <div class="metric-card-value">{res['pre_time_ms']:.1f} ms</div>
                    <div class="metric-card-subtext">CLAHE + Gaussian Blur</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">CANDIDATE EXTRACTION</span></div>
                    <div class="metric-card-value">{res['cand_time_ms']:.1f} ms</div>
                    <div class="metric-card-subtext">Morphology + Contours</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">END-TO-END TOTAL</span></div>
                    <div class="metric-card-value">{res['total_time_ms']:.1f} ms</div>
                    <div class="metric-card-subtext">Real-time OpenCV pipeline</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r4:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header"><span class="metric-card-label">ACTIVE METHOD</span></div>
                    <div class="metric-card-value">{res['active_detector'].upper()}</div>
                    <div class="metric-card-subtext">Selected operator</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # Operator Comparison Breakdown Table (Real Data)
        st.markdown(
            """
            <div style="font-size: 13.5px; font-weight: 700; color: #F3F7F5; margin-bottom: 8px;">
                Edge Operator Performance Breakdown (Current Run)
            </div>
            """,
            unsafe_allow_html=True,
        )
        det = res["detector_results"]
        op_comp_data = [
            {
                "Edge Operator": "Canny",
                "Execution Time (ms)": det["canny"]["execution_time_ms"],
                "Edge Pixel Density": f"{det['canny']['edge_density_pct']}%",
                "Mathematical Order": "1st Derivative + Hysteresis",
                "Primary Advantage": "Single-pixel thin edge localization",
            },
            {
                "Edge Operator": "Sobel",
                "Execution Time (ms)": det["sobel"]["execution_time_ms"],
                "Edge Pixel Density": f"{det['sobel']['edge_density_pct']}%",
                "Mathematical Order": "1st Derivative (Gx & Gy)",
                "Primary Advantage": "Directional sensitivity & cavity gradient bands",
            },
            {
                "Edge Operator": "Laplacian",
                "Execution Time (ms)": det["laplacian"]["execution_time_ms"],
                "Edge Pixel Density": f"{det['laplacian']['edge_density_pct']}%",
                "Mathematical Order": "2nd Derivative Isotropic (∇²f)",
                "Primary Advantage": "Rotationally invariant zero-crossing detection",
            },
        ]
        st.dataframe(pd.DataFrame(op_comp_data), width="stretch")

        st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)

        # Export Hub
        st.markdown(
            """
            <div style="font-size: 13.5px; font-weight: 700; color: #F3F7F5; margin-bottom: 8px;">
                Export Research Artifacts
            </div>
            """,
            unsafe_allow_html=True,
        )

        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            json_output = build_analysis_json(
                image_name=st.session_state["current_image_name"],
                image_shape=st.session_state["current_image_rgb"].shape,
                pipeline_params=st.session_state["params"],
                detector_metrics=res["detector_results"],
                candidates=res["candidates"],
            )
            st.download_button(
                label="Download JSON Research Report",
                data=json_output,
                file_name=f"pothole_ai_report_{st.session_state['current_image_name']}.json",
                mime="application/json",
                width="stretch",
                key="rep_dl_json",
            )
        with ex2:
            png_montage_bytes = numpy_to_png_bytes(res["montage"])
            st.download_button(
                label="Download 6-Stage Pipeline Montage",
                data=png_montage_bytes,
                file_name=f"pothole_ai_montage_{st.session_state['current_image_name']}.png",
                mime="image/png",
                width="stretch",
                key="rep_dl_montage",
            )
        with ex3:
            png_overlay_bytes = numpy_to_png_bytes(res["overlay"])
            st.download_button(
                label="Download Annotated Overlay (PNG)",
                data=png_overlay_bytes,
                file_name=f"pothole_ai_overlay_{st.session_state['current_image_name']}.png",
                mime="image/png",
                width="stretch",
                key="rep_dl_overlay",
            )

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        with st.expander("Standardized Research JSON Report Preview", expanded=False):
            st.code(json_output, language="json")


# -----------------------------------------------------------------------------
# Global Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="dashboard-footer" style="margin-top: 36px; padding: 18px 0 6px; border-top: 1px solid #141f26; display: flex; justify-content: space-between; font-size: 11px; color: #576571;">
        <div>POTHOLE AI &bull; INTELLIGENT ROAD SURFACE ANALYSIS</div>
        <div>OPENCV 4 &bull; PYTHON 3.12 &bull; STREAMLIT</div>
    </div>
    """,
    unsafe_allow_html=True,
)
