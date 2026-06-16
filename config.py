import os

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR       = os.path.join(BASE_DIR, "data", "input")
WATERMARKED_DIR = os.path.join(BASE_DIR, "data", "watermarked")
TAMPERED_DIR    = os.path.join(BASE_DIR, "data", "tampered")
METRICS_DIR     = os.path.join(BASE_DIR, "results", "metrics")
VIZ_DIR         = os.path.join(BASE_DIR, "results", "visualizations")

# ── Image settings ─────────────────────────────────────────────────────
IMAGE_SIZE           = (64, 64)
COLOR_BITS           = 8

# ── NEQR settings ──────────────────────────────────────────────────────
NEQR_COLOR_QUBITS    = 8
NEQR_POSITION_QUBITS = 12

# ── Cryptography ───────────────────────────────────────────────────────
KYBER_SECURITY_LEVEL = 512
HASH_ALGORITHM       = "sha3_256"

# ── Watermark ──────────────────────────────────────────────────────────
WATERMARK_BITS       = 64
LSB_DEPTH            = 2

# ── Attack simulation ──────────────────────────────────────────────────
JPEG_QUALITY         = 70
GAUSSIAN_NOISE_STD   = 10
ROTATION_ANGLE       = 15

# ── Paper's target thresholds ──────────────────────────────────────────
MIN_PSNR             = 42.0
MIN_SSIM             = 0.95
MIN_DETECTION        = 0.96
