"""
Image Utility and IO Helper Module.

Provides functions for image encoding/decoding, file operations,
montage rendering, and synthetic road sample generation for demonstrations and tests.
"""

import io
import os
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image


def numpy_to_png_bytes(image: np.ndarray) -> bytes:
    """
    Encode an RGB, BGR, or grayscale numpy array into PNG format bytes.
    """
    if len(image.shape) == 2:
        # Grayscale
        success, encoded = cv2.imencode(".png", image)
    elif len(image.shape) == 3:
        # Assume RGB if passing to util from app, convert to BGR for cv2 encoding
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".png", bgr)
    else:
        raise ValueError(f"Unsupported image shape for encoding: {image.shape}")

    if not success:
        raise RuntimeError("Failed to encode image to PNG format.")

    return encoded.tobytes()


def load_image_from_path(filepath: str) -> np.ndarray:
    """
    Safely load an image from disk and return as an RGB numpy array.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file does not exist: {filepath}")

    bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image from {filepath}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an RGB or grayscale numpy array to disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if len(image.shape) == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(output_path, bgr)
    else:
        return cv2.imwrite(output_path, image)


def create_comparison_montage(
    original: np.ndarray,
    enhanced: np.ndarray,
    canny: np.ndarray,
    sobel: np.ndarray,
    laplacian: np.ndarray,
    overlay: np.ndarray,
    target_size: Tuple[int, int] = (400, 300),
) -> np.ndarray:
    """
    Create a 2x3 labeled montage grid of the pipeline stages.
    """
    w, h = target_size

    def prepare_tile(img: np.ndarray, title: str) -> np.ndarray:
        if len(img.shape) == 2:
            tile_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            tile_rgb = img.copy()

        resized = cv2.resize(tile_rgb, (w, h), interpolation=cv2.INTER_AREA)

        # Header banner
        banner_h = 32
        banner = np.full((banner_h, w, 3), 18, dtype=np.uint8)
        cv2.putText(
            banner,
            title,
            (12, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (52, 211, 153),
            1,
            cv2.LINE_AA,
        )

        bordered = np.vstack([banner, resized])
        cv2.rectangle(bordered, (0, 0), (bordered.shape[1] - 1, bordered.shape[0] - 1), (37, 49, 61), 1)
        return bordered

    t1 = prepare_tile(original, "1. Original Road Image")
    t2 = prepare_tile(enhanced, "2. Contrast Enhanced")
    t3 = prepare_tile(canny, "3. Canny Edge Map")
    t4 = prepare_tile(sobel, "4. Sobel Edge Map")
    t5 = prepare_tile(laplacian, "5. Laplacian Edge Map")
    t6 = prepare_tile(overlay, "6. Candidate Regions Overlay")

    row1 = np.hstack([t1, t2, t3])
    row2 = np.hstack([t4, t5, t6])
    montage = np.vstack([row1, row2])

    return montage


def generate_synthetic_road_sample(
    scenario: str = "single_deep_pothole",
    width: int = 640,
    height: int = 480,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate realistic synthetic road images with asphalt texture, aggregate speckles,
    lane markings, and simulated road distress / potholes for academic testing.
    """
    rng = np.random.default_rng(seed)

    # Base asphalt color: dark charcoal gray ~45-65
    base_intensity = 55
    asphalt = rng.normal(loc=base_intensity, scale=12.0, size=(height, width)).clip(20, 100).astype(np.uint8)

    # Add fine aggregate noise
    speckles = rng.choice([0, 25, 45], size=(height, width), p=[0.85, 0.12, 0.03]).astype(np.uint8)
    asphalt = cv2.add(asphalt, speckles)

    # Convert to 3-channel RGB with subtle cool tint
    road_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    road_rgb[:, :, 0] = np.clip(asphalt * 0.95, 0, 255).astype(np.uint8)  # R
    road_rgb[:, :, 1] = np.clip(asphalt * 1.00, 0, 255).astype(np.uint8)  # G
    road_rgb[:, :, 2] = np.clip(asphalt * 1.05, 0, 255).astype(np.uint8)  # B

    # Add road perspective horizon darkening (upper 20% is distance)
    horizon_mask = np.linspace(0.7, 1.0, height)[:, np.newaxis]
    road_rgb = (road_rgb.astype(np.float32) * horizon_mask[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

    # Add faint dashed lane marking on the left edge
    lane_x = int(width * 0.18)
    dash_len = 40
    gap_len = 40
    for y_pos in range(int(height * 0.25), height, dash_len + gap_len):
        cv2.line(road_rgb, (lane_x, y_pos), (lane_x, min(height, y_pos + dash_len)), (180, 180, 160), 3)

    if scenario == "single_deep_pothole":
        # Create an elliptical depression with jagged edges and shadow interior
        center = (int(width * 0.52), int(height * 0.62))
        axes = (int(width * 0.14), int(height * 0.09))
        angle = -8

        # Draw deep dark hollow cavity
        cv2.ellipse(road_rgb, center, axes, angle, 0, 360, (22, 22, 24), -1)

        # Draw interior bottom shadow
        inner_center = (center[0] - 8, center[1] + 6)
        inner_axes = (int(axes[0] * 0.68), int(axes[1] * 0.62))
        cv2.ellipse(road_rgb, inner_center, inner_axes, angle, 0, 360, (12, 12, 14), -1)

        # Draw jagged eroded edge boundary
        pts = cv2.ellipse2Poly(center, axes, angle, 0, 360, 10)
        jitter = rng.integers(-4, 5, size=pts.shape)
        pts_jitter = pts + jitter
        cv2.polylines(road_rgb, [pts_jitter], True, (35, 36, 40), 3)

        # Add light lip/rim highlight on top edge from sunlight
        cv2.ellipse(road_rgb, center, axes, angle, 190, 350, (95, 95, 100), 2)

    elif scenario == "multiple_depressions":
        # Pothole 1 (Primary depression)
        c1 = (int(width * 0.45), int(height * 0.68))
        a1 = (int(width * 0.11), int(height * 0.08))
        cv2.ellipse(road_rgb, c1, a1, 5, 0, 360, (20, 21, 23), -1)
        cv2.ellipse(road_rgb, c1, a1, 5, 180, 360, (90, 92, 95), 2)

        # Pothole 2 (Secondary smaller depression)
        c2 = (int(width * 0.72), int(height * 0.54))
        a2 = (int(width * 0.08), int(height * 0.06))
        cv2.ellipse(road_rgb, c2, a2, -15, 0, 360, (24, 25, 27), -1)
        cv2.ellipse(road_rgb, c2, a2, -15, 190, 350, (88, 90, 93), 2)

        # Connected alligator cracking pattern
        crack_pts = np.array([
            [c1[0] + a1[0], c1[1]],
            [c1[0] + a1[0] + 30, c1[1] - 15],
            [c2[0] - a2[0] - 20, c2[1] + 10],
            [c2[0] - a2[0], c2[1]],
        ], np.int32)
        cv2.polylines(road_rgb, [crack_pts], False, (28, 28, 30), 2)

    elif scenario == "wet_road_pothole":
        # Simulate wet darker asphalt with localized high specular contrast
        road_rgb = (road_rgb.astype(np.float32) * 0.85).astype(np.uint8)

        # Water-filled depression
        c = (int(width * 0.50), int(height * 0.65))
        a = (int(width * 0.13), int(height * 0.09))
        cv2.ellipse(road_rgb, c, a, 0, 0, 360, (15, 18, 22), -1)
        # Specular glint on puddle surface
        cv2.ellipse(road_rgb, (c[0] + 10, c[1] - 5), (int(a[0] * 0.4), int(a[1] * 0.3)), 0, 0, 360, (110, 125, 140), -1)
        cv2.ellipse(road_rgb, c, a, 0, 0, 360, (80, 95, 105), 2)

    # Slight Gaussian smoothing to blend synthetic noise realistically
    road_rgb = cv2.GaussianBlur(road_rgb, (3, 3), 0.6)

    return road_rgb
