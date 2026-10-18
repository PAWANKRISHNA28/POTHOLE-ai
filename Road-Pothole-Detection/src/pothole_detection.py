"""
Pothole Candidate Detection Module.

Extracts candidate road distress / pothole regions using morphological operations,
contour analysis, and transparent geometric filtering rules.

NOTE ON METHODOLOGY:
Edge detection alone does not provide semantic classification. This module
highlights candidate anomaly regions for inspection rather than confirming
actual potholes. Bounding boxes are candidate markers based on local intensity
gradients and morphology.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


def apply_morphology(
    edge_map: np.ndarray,
    kernel_size: int = 5,
    operation: str = "close",
    iterations: int = 2,
) -> np.ndarray:
    """
    Apply morphological operations to connect fragmented edge contours
    and close interior regions of candidate depressions.

    Supported operations: 'close', 'dilate', 'open', 'gradient'
    """
    k = max(1, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    op = operation.lower().strip()
    if op == "close":
        return cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif op == "dilate":
        return cv2.dilate(edge_map, kernel, iterations=iterations)
    elif op == "open":
        return cv2.morphologyEx(edge_map, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op == "gradient":
        return cv2.morphologyEx(edge_map, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    else:
        return cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def extract_candidate_regions(
    binary_map: np.ndarray,
    original_rgb: np.ndarray,
    gray_image: Optional[np.ndarray] = None,
    min_area: int = 200,
    max_area: int = 80000,
    min_aspect_ratio: float = 0.25,
    max_aspect_ratio: float = 4.0,
    min_solidity: float = 0.35,
    roi_top_ratio: float = 0.2,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Extract candidate pothole contours using geometric and intensity filtering.

    Args:
        binary_map: Closed binary edge or mask map (uint8, 0 or 255).
        original_rgb: Preserved original RGB image for feature extraction.
        gray_image: Grayscale version to calculate local contrast and darkness.
        min_area: Minimum contour pixel area.
        max_area: Maximum contour pixel area.
        min_aspect_ratio: Minimum bounding-box aspect ratio (w/h).
        max_aspect_ratio: Maximum bounding-box aspect ratio (w/h).
        min_solidity: Ratio of contour area to convex hull area (filters chaotic noise).
        roi_top_ratio: Top vertical fraction of image to exclude (e.g. 0.2 excludes sky/horizon).

    Returns:
        (candidates_list, mask_map)
    """
    if min_area > max_area:
        min_area, max_area = max_area, min_area
    if min_aspect_ratio > max_aspect_ratio:
        min_aspect_ratio, max_aspect_ratio = max_aspect_ratio, min_aspect_ratio

    h_img, w_img = binary_map.shape[:2]
    roi_top_px = int(h_img * max(0.0, min(0.9, roi_top_ratio)))

    if gray_image is None:
        gray_image = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    # Find external or hierarchical contours
    contours, _ = cv2.findContours(
        binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: List[Dict[str, Any]] = []
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    # Compute global mean road intensity for contrast reference
    road_sample_y = max(roi_top_px, int(h_img * 0.4))
    road_mean_intensity = float(np.mean(gray_image[road_sample_y:, :]))

    candidate_idx = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Exclude candidates in excluded upper ROI (sky / horizon)
        if y < roi_top_px:
            continue

        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0
        if solidity < min_solidity:
            continue

        # Regional intensity inside the candidate
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        # Shift contour to local ROI coordinates
        shifted_cnt = cnt - np.array([x, y])
        cv2.drawContours(roi_mask, [shifted_cnt], -1, 255, -1)
        gray_roi = gray_image[y : y + h, x : x + w]

        candidate_pixels = gray_roi[roi_mask > 0]
        if candidate_pixels.size > 0:
            local_mean = float(np.mean(candidate_pixels))
            local_std = float(np.std(candidate_pixels))
        else:
            local_mean = road_mean_intensity
            local_std = 0.0

        # Darkness contrast relative to surrounding road
        intensity_diff = road_mean_intensity - local_mean

        # Calculate transparent Candidate Saliency Score (0-100)
        # Combines size appropriateness, compactness/solidity, and shadow/depression contrast
        size_score = min(35.0, (area / 1500.0) * 25.0)
        solidity_score = min(30.0, solidity * 30.0)
        contrast_score = min(35.0, max(0.0, (intensity_diff / 40.0) * 35.0))
        saliency_score = round(min(99.0, max(15.0, size_score + solidity_score + contrast_score)), 1)

        # Severity categorization based on physical area footprint in image
        if area > 10000:
            severity = "High (Large Anomaly)"
        elif area > 2500:
            severity = "Moderate (Medium Anomaly)"
        else:
            severity = "Low (Small Surface Distress)"

        candidate_record: Dict[str, Any] = {
            "id": candidate_idx,
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(area),
            "perimeter": round(float(cv2.arcLength(cnt, True)), 1),
            "aspect_ratio": round(aspect_ratio, 2),
            "solidity": round(solidity, 2),
            "mean_intensity": round(local_mean, 1),
            "relative_darkness": round(intensity_diff, 1),
            "saliency_score": saliency_score,
            "severity": severity,
            "center": (int(x + w / 2), int(y + h / 2)),
            "contour": cnt,
        }

        candidates.append(candidate_record)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        candidate_idx += 1

    # Sort candidates by saliency score descending
    candidates.sort(key=lambda c: c["saliency_score"], reverse=True)

    return candidates, mask


def draw_candidate_overlays(
    original_rgb: np.ndarray,
    candidates: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (239, 68, 68),  # Coral red matching reference screenshot
    show_labels: bool = False,
    fill_alpha: float = 0.15,
) -> np.ndarray:
    """
    Render contours, subtle highlight masks, and candidate boundaries
    on top of a copy of the original road image.
    """
    if len(original_rgb.shape) == 2:
        overlay = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)
    else:
        overlay = original_rgb.copy()
    tint_layer = overlay.copy()

    # Draw semi-transparent fills on contours
    for cand in candidates:
        if "contour" in cand and cand["contour"] is not None:
            cv2.drawContours(tint_layer, [cand["contour"]], -1, box_color, -1)
        else:
            x, y, w, h = cand["bbox"]
            cv2.rectangle(tint_layer, (x, y), (x + w, y + h), box_color, -1)

    cv2.addWeighted(tint_layer, fill_alpha, overlay, 1.0 - fill_alpha, 0, overlay)

    # Draw crisp contour borders and/or bounds
    for cand in candidates:
        if "contour" in cand and cand["contour"] is not None:
            # Draw precise organic outline of defect
            cv2.drawContours(overlay, [cand["contour"]], -1, box_color, 2)
        else:
            x, y, w, h = cand["bbox"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)

        if show_labels:
            x, y, w, h = cand["bbox"]
            c_id = cand["id"]
            score = cand["saliency_score"]
            label_text = f"C#{c_id} ({score}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y1 = max(0, y - text_h - 6)
            badge_y2 = y
            badge_x1 = x
            badge_x2 = min(overlay.shape[1], x + text_w + 8)

            # Draw tag background
            cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (17, 24, 32), -1)
            cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, 1)

            # Draw tag text
            cv2.putText(
                overlay,
                label_text,
                (x + 4, y - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return overlay
