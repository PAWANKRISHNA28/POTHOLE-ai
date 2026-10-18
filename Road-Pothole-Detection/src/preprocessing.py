"""
Image Preprocessing Module for Road Pothole Detection.

Provides routines to validate input images, convert to grayscale,
apply contrast enhancement (Histogram Equalization, CLAHE, Linear Stretching),
and perform noise reduction via Gaussian blur or bilateral filtering.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2


def validate_image(
    image: Union[np.ndarray, bytes, Image.Image]
) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    Validate that the incoming image object is readable, non-empty,
    and has valid dimensions and channels.

    Returns:
        (is_valid, message, rgb_numpy_array)
    """
    if image is None:
        return False, "Input image is None.", None

    rgb: Optional[np.ndarray] = None

    try:
        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if decoded is None:
                return False, "Failed to decode image bytes. Unsupported or corrupted file format.", None
            if len(decoded.shape) == 2:
                rgb = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
            elif decoded.shape[2] == 4:
                rgb = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        elif isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))

        elif isinstance(image, np.ndarray):
            if image.size == 0:
                return False, "Image array has zero size.", None
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                if image.shape[2] == 1:
                    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 3:
                    rgb = image.copy()
                elif image.shape[2] == 4:
                    rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    return False, f"Unexpected number of channels: {image.shape[2]}", None
            else:
                return False, f"Invalid image array dimension: {len(image.shape)}", None
        else:
            return False, f"Unsupported image input type: {type(image)}", None

        # Sanity check dimensions
        h, w = rgb.shape[:2]
        if h < 16 or w < 16:
            return False, f"Image dimensions too small ({w}x{h}). Minimum size is 16x16.", None
        if h > 8192 or w > 8192:
            return False, f"Image dimensions too large ({w}x{h}). Maximum supported is 8192x8192.", None

        return True, "Image is valid.", rgb

    except Exception as e:
        return False, f"Image validation error: {str(e)}", None


def to_grayscale(image_rgb: np.ndarray, method: str = "weighted") -> np.ndarray:
    """
    Safely convert an RGB image to single-channel 8-bit grayscale.
    method: 'weighted' for standard OpenCV luminance formula,
            'average' for unweighted RGB channel mean,
            'green' for green channel extraction.
    """
    if len(image_rgb.shape) == 2:
        return image_rgb.astype(np.uint8)
    if method == "average":
        return np.mean(image_rgb, axis=2).astype(np.uint8)
    elif method == "green":
        return image_rgb[:, :, 1].copy()
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Prevents over-amplification of noise while balancing regional illumination.
    """
    if clip_limit <= 0:
        clip_limit = 0.1
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(gray)


def apply_histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalization to spread the intensity range across 0-255.
    """
    return cv2.equalizeHist(gray)


def apply_contrast_stretching(
    gray: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """
    Perform linear contrast stretching clipping top and bottom percentiles.
    """
    p_low = np.percentile(gray, lower_percentile)
    p_high = np.percentile(gray, upper_percentile)
    if p_high <= p_low:
        return gray.copy()
    stretched = np.clip((gray - p_low) * (255.0 / (p_high - p_low)), 0, 255)
    return stretched.astype(np.uint8)


def enhance_contrast(
    gray: np.ndarray,
    method: str = "clahe",
    **kwargs: Any,
) -> np.ndarray:
    """
    Dispatch contrast enhancement using the specified technique.

    Supported methods: 'clahe', 'hist_eq', 'stretch', 'none'
    """
    norm_method = method.lower().strip()
    if norm_method == "clahe":
        clip_limit = kwargs.get("clip_limit", 2.5)
        tile_size = kwargs.get("tile_grid_size", (8, 8))
        return apply_clahe(gray, clip_limit=clip_limit, tile_grid_size=tile_size)
    elif norm_method in ("hist_eq", "histogram", "equalization"):
        return apply_histogram_equalization(gray)
    elif norm_method in ("stretch", "linear"):
        low_p = kwargs.get("lower_percentile", 2.0)
        high_p = kwargs.get("upper_percentile", 98.0)
        return apply_contrast_stretching(gray, lower_percentile=low_p, upper_percentile=high_p)
    elif norm_method == "none":
        return gray.copy()
    else:
        # Default fallback to CLAHE
        return apply_clahe(gray, clip_limit=2.0)


def reduce_noise(
    gray: np.ndarray,
    method: str = "gaussian",
    kernel_size: int = 5,
    sigma: float = 1.2,
) -> np.ndarray:
    """
    Apply smoothing filter to suppress high-frequency road surface texture noise.

    Supported methods: 'gaussian', 'bilateral', 'median', 'none'
    """
    # Enforce odd kernel size >= 1
    k = max(1, kernel_size)
    if k % 2 == 0:
        k += 1

    norm_method = method.lower().strip()
    if norm_method == "gaussian":
        sig = max(0.1, float(sigma))
        return cv2.GaussianBlur(gray, (k, k), sig)
    elif norm_method == "bilateral":
        return cv2.bilateralFilter(gray, d=k, sigmaColor=75, sigmaSpace=75)
    elif norm_method == "median":
        return cv2.medianBlur(gray, k)
    elif norm_method == "none":
        return gray.copy()
    else:
        return cv2.GaussianBlur(gray, (k, k), max(0.1, float(sigma)))


def preprocess_pipeline(
    image_rgb: np.ndarray,
    contrast_method: str = "clahe",
    clahe_clip: float = 2.5,
    blur_method: str = "gaussian",
    blur_ksize: int = 5,
    blur_sigma: float = 1.2,
    grayscale_method: str = "weighted",
) -> Dict[str, np.ndarray]:
    """
    Execute full preprocessing pipeline on an RGB image.

    Returns dictionary with intermediate stages:
        'original_rgb': Original preserved RGB image
        'gray': Initial grayscale conversion
        'enhanced': Contrast-enhanced grayscale image
        'smoothed': Noise-filtered grayscale image ready for edge detection
    """
    gray = to_grayscale(image_rgb, method=grayscale_method)
    enhanced = enhance_contrast(
        gray,
        method=contrast_method,
        clip_limit=clahe_clip,
    )
    smoothed = reduce_noise(
        enhanced,
        method=blur_method,
        kernel_size=blur_ksize,
        sigma=blur_sigma,
    )

    return {
        "original_rgb": image_rgb.copy(),
        "gray": gray,
        "enhanced": enhanced,
        "smoothed": smoothed,
    }
