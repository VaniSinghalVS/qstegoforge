"""
src/attacks.py
QStegoForge — Robustness Attack Suite
Simulates real-world adversarial attacks against watermarked images to evaluate
the robustness of the NEQR-based steganographic watermarking system.

Attacks implemented:
  1. JPEG Compression      — lossy codec attack (quality 10–95)
  2. Gaussian Noise        — additive white Gaussian noise (sigma 5–50)
  3. Rotation              — affine geometric distortion (angle 1–45°)
  4. Diffusion Removal     — mean/median smoothing to blur LSB patterns
"""

import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from skimage.util import random_noise


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 in [0, 255]."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to single-channel grayscale if needed."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# ---------------------------------------------------------------------------
# Attack 1: JPEG Compression
# ---------------------------------------------------------------------------

def jpeg_compression_attack(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Simulate JPEG lossy compression by encoding to JPEG in-memory and decoding.

    Args:
        image   : Input grayscale or BGR image (uint8).
        quality : JPEG quality factor [1, 95]. Lower = more lossy.

    Returns:
        Compressed-and-decompressed image as numpy uint8 array.

    Notes:
        Quality < 30 typically destroys LSB watermarks entirely.
        Quality 70–95 is where partial watermark survival is evaluated.
    """
    if not 1 <= quality <= 95:
        raise ValueError(f"JPEG quality must be in [1, 95], got {quality}")

    image = _to_uint8(image)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # Encode → decode in-memory (no disk I/O)
    success, encoded = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise RuntimeError("cv2.imencode failed for JPEG compression attack")

    attacked = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    return attacked


# ---------------------------------------------------------------------------
# Attack 2: Gaussian Noise
# ---------------------------------------------------------------------------

def gaussian_noise_attack(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Add additive white Gaussian noise (AWGN) to the image.

    Args:
        image : Input image (uint8).
        sigma : Standard deviation of Gaussian noise in pixel intensity units [0–255].
                Typical test range: 5, 15, 25, 35, 50.

    Returns:
        Noisy image clipped to [0, 255] uint8.

    Notes:
        skimage.random_noise works on float images in [0, 1].
        var = (sigma / 255) ** 2 maps intensity sigma to variance.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    image = _to_uint8(image)
    float_img = image.astype(np.float64) / 255.0
    var = (sigma / 255.0) ** 2

    noisy_float = random_noise(float_img, mode="gaussian", var=var, clip=True)
    noisy = (noisy_float * 255).astype(np.uint8)
    return noisy


# ---------------------------------------------------------------------------
# Attack 3: Rotation
# ---------------------------------------------------------------------------

def rotation_attack(
    image: np.ndarray,
    angle: float = 15.0,
    fill_value: int = 0,
) -> np.ndarray:
    """
    Apply in-plane rotation attack (no cropping — canvas expands to fit).

    Args:
        image      : Input image (uint8).
        angle      : Rotation angle in degrees (counter-clockwise). Range tested: 1–45°.
        fill_value : Pixel value used for background padding (default black = 0).

    Returns:
        Rotated image with same dtype as input. Size may differ from input
        because we expand the canvas rather than cropping, matching typical
        geometric-attack evaluation in watermarking literature.

    Notes:
        After rotation the receiving party must re-align before extraction,
        making this a non-blind geometric attack that tests synchronisation loss.
    """
    image = _to_uint8(image)
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Build rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale=1.0)

    # Compute new bounding dimensions so no content is cropped
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Shift the rotation centre to the new canvas centre
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_value,
    )
    return rotated


# ---------------------------------------------------------------------------
# Attack 4: Diffusion Removal (LSB smoothing)
# ---------------------------------------------------------------------------

def diffusion_removal_attack(
    image: np.ndarray,
    method: str = "median",
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Apply spatial filtering to erase LSB-embedded watermark patterns.

    Two strategies are provided:
      - 'median' : Median filter — highly effective against salt-and-pepper-like
                   LSB patterns while preserving edges. Kernel sizes 3 and 5 are
                   standard benchmarks.
      - 'mean'   : Box/averaging filter — blurs the entire neighbourhood,
                   generally less targeted but useful for comparison.

    Args:
        image       : Input image (uint8 grayscale or BGR).
        method      : 'median' | 'mean'
        kernel_size : Filter kernel size (must be odd positive integer ≥ 3).

    Returns:
        Filtered image as uint8 array.

    Notes:
        In NEQR-LSB steganography the watermark resides in bit-0 of pixel
        values. A single pass of median(3×3) is empirically sufficient to
        degrade detection rate below 50% in many systems.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be an odd integer ≥ 1, got {kernel_size}")

    image = _to_uint8(image)

    if method == "median":
        attacked = cv2.medianBlur(image, kernel_size)
    elif method == "mean":
        attacked = cv2.blur(image, (kernel_size, kernel_size))
    else:
        raise ValueError(f"Unknown diffusion method '{method}'. Choose 'median' or 'mean'.")

    return attacked


# ---------------------------------------------------------------------------
# Convenience: run all attacks with default parameters
# ---------------------------------------------------------------------------

def run_all_attacks(image: np.ndarray) -> dict:
    """
    Apply every attack with its default parameters and return a dict of results.

    Keys:
        'original'           : unchanged input
        'jpeg_q50'           : JPEG quality 50
        'jpeg_q30'           : JPEG quality 30 (heavy compression)
        'gaussian_sigma25'   : Gaussian noise σ=25
        'gaussian_sigma50'   : Gaussian noise σ=50
        'rotation_15deg'     : 15° rotation
        'rotation_30deg'     : 30° rotation
        'diffusion_median3'  : median filter 3×3
        'diffusion_mean3'    : mean filter 3×3

    Args:
        image : Input uint8 image.

    Returns:
        dict mapping attack name → attacked image (numpy uint8 array).
    """
    image = _to_uint8(image)

    results = {
        "original":          image.copy(),
        "jpeg_q50":          jpeg_compression_attack(image, quality=50),
        "jpeg_q30":          jpeg_compression_attack(image, quality=30),
        "gaussian_sigma25":  gaussian_noise_attack(image, sigma=25.0),
        "gaussian_sigma50":  gaussian_noise_attack(image, sigma=50.0),
        "rotation_15deg":    rotation_attack(image, angle=15.0),
        "rotation_30deg":    rotation_attack(image, angle=30.0),
        "diffusion_median3": diffusion_removal_attack(image, method="median", kernel_size=3),
        "diffusion_mean3":   diffusion_removal_attack(image, method="mean",   kernel_size=3),
    }

    print("[attacks] All attacks applied:")
    for name, attacked in results.items():
        print(f"  {name:<25} shape={attacked.shape}  dtype={attacked.dtype}")

    return results


# ---------------------------------------------------------------------------
# Quick self-test (run directly: python src/attacks.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== QStegoForge Attack Suite — self-test ===\n")

    # Create a synthetic 256×256 grayscale test image
    rng = np.random.default_rng(42)
    test_img = rng.integers(0, 256, (256, 256), dtype=np.uint8)

    attacked = run_all_attacks(test_img)
    print(f"\nAll {len(attacked)} variants generated successfully.")

    # Verify shapes / dtypes
    for name, arr in attacked.items():
        assert arr.dtype == np.uint8, f"{name}: expected uint8, got {arr.dtype}"

    print("\nSelf-test PASSED.")