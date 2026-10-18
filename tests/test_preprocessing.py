"""
Unit tests for the preprocessing module.
"""

import io
import numpy as np
from PIL import Image
import pytest
from src.preprocessing import (
    validate_image,
    to_grayscale,
    apply_clahe,
    apply_histogram_equalization,
    apply_contrast_stretching,
    enhance_contrast,
    reduce_noise,
    preprocess_pipeline,
)


@pytest.fixture
def sample_rgb_image():
    """Create a deterministic synthetic 100x100 RGB image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_gray_image():
    """Create a deterministic synthetic 100x100 grayscale image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (100, 100), dtype=np.uint8)


def test_validate_image_valid_numpy(sample_rgb_image):
    is_valid, msg, rgb = validate_image(sample_rgb_image)
    assert is_valid is True
    assert rgb is not None
    assert rgb.shape == (100, 100, 3)
    assert rgb.dtype == np.uint8


def test_validate_image_pil_object(sample_rgb_image):
    pil_img = Image.fromarray(sample_rgb_image)
    is_valid, msg, rgb = validate_image(pil_img)
    assert is_valid is True
    assert rgb is not None
    assert rgb.shape == (100, 100, 3)


def test_validate_image_bytes(sample_rgb_image):
    pil_img = Image.fromarray(sample_rgb_image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    is_valid, msg, rgb = validate_image(raw_bytes)
    assert is_valid is True
    assert rgb is not None
    assert rgb.shape == (100, 100, 3)


def test_validate_image_invalid_inputs():
    # None
    is_valid, msg, rgb = validate_image(None)
    assert is_valid is False
    assert rgb is None

    # Empty array
    is_valid, msg, rgb = validate_image(np.array([]))
    assert is_valid is False

    # Corrupted bytes
    is_valid, msg, rgb = validate_image(b"not_an_image_file")
    assert is_valid is False

    # Too small dimensions (<16x16)
    tiny = np.zeros((8, 8, 3), dtype=np.uint8)
    is_valid, msg, rgb = validate_image(tiny)
    assert is_valid is False


def test_to_grayscale(sample_rgb_image, sample_gray_image):
    gray = to_grayscale(sample_rgb_image)
    assert len(gray.shape) == 2
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8

    # Already gray
    already_gray = to_grayscale(sample_gray_image)
    assert already_gray.shape == (100, 100)


def test_enhance_contrast_methods(sample_gray_image):
    clahe = enhance_contrast(sample_gray_image, method="clahe", clip_limit=2.0)
    assert clahe.shape == sample_gray_image.shape
    assert clahe.dtype == np.uint8

    hist = enhance_contrast(sample_gray_image, method="hist_eq")
    assert hist.shape == sample_gray_image.shape

    stretch = enhance_contrast(sample_gray_image, method="stretch")
    assert stretch.shape == sample_gray_image.shape

    none = enhance_contrast(sample_gray_image, method="none")
    assert np.array_equal(none, sample_gray_image)


def test_reduce_noise_methods(sample_gray_image):
    gaussian = reduce_noise(sample_gray_image, method="gaussian", kernel_size=5, sigma=1.0)
    assert gaussian.shape == sample_gray_image.shape
    assert gaussian.dtype == np.uint8

    bilateral = reduce_noise(sample_gray_image, method="bilateral", kernel_size=5)
    assert bilateral.shape == sample_gray_image.shape

    median = reduce_noise(sample_gray_image, method="median", kernel_size=5)
    assert median.shape == sample_gray_image.shape


def test_preprocess_pipeline(sample_rgb_image):
    results = preprocess_pipeline(sample_rgb_image)
    assert "original_rgb" in results
    assert "gray" in results
    assert "enhanced" in results
    assert "smoothed" in results

    assert results["smoothed"].shape == (100, 100)
    assert results["original_rgb"].shape == (100, 100, 3)
