"""
main.py
QStegoForge — Full Pipeline Runner
===================================
Executes the complete quantum-safe steganographic watermarking pipeline:

  1. Load cover image
  2. Generate / load watermark payload
  3. Kyber512 key generation + payload encryption (CRYSTALS-Kyber + SHA-3)
  4. NEQR quantum encoding of the cover image
  5. Watermark embedding via X-gate LSB flips
  6. Simulate adversarial attacks (JPEG, Gaussian noise, rotation, diffusion)
  7. Watermark extraction & verification on each attacked variant
  8. Print structured metrics table (DR, BER, PSNR, SSIM)
  9. Save result images + matplotlib summary chart

Usage:
    python main.py [--cover <path>] [--output-dir <dir>] [--no-plots]

Defaults:
    --cover       assets/cover.png   (256×256 grayscale; auto-generated if absent)
    --output-dir  results/
    --no-plots    flag to skip matplotlib chart (useful on headless servers)
"""

import argparse
import os
import sys
import time
import textwrap
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ── project modules ──────────────────────────────────────────────────────────
# Adjust sys.path so this script works whether launched from repo root or src/
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import (
    IMAGE_SIZE,
    WATERMARK_SIZE,
    NUM_QUBITS,
    KYBER_VARIANT,
    RESULTS_DIR,
    ASSETS_DIR,
)
from src.neqr import NEQREncoder
from src.kyber_crypto import KyberCrypto
from src.embedder import WatermarkEmbedder
from src.verifier import WatermarkVerifier
from src.attacks import run_all_attacks


# ── helpers ──────────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_or_create_cover(cover_path: Path, size: tuple) -> np.ndarray:
    """Load a grayscale cover image; auto-generate a synthetic one if not found."""
    if cover_path.exists():
        img = cv2.imread(str(cover_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to read image at {cover_path}")
        img = cv2.resize(img, size)
        print(f"[main] Cover image loaded: {cover_path}  shape={img.shape}")
    else:
        print(f"[main] Cover image not found at {cover_path}. Generating synthetic test image.")
        rng = np.random.default_rng(2024)
        img = rng.integers(0, 256, (size[1], size[0]), dtype=np.uint8)
        _ensure_dir(cover_path.parent)
        cv2.imwrite(str(cover_path), img)
    return img


def _compute_metrics(original: np.ndarray, attacked: np.ndarray) -> dict:
    """
    Compute PSNR and SSIM between the watermarked original and an attacked variant.
    Both arrays are resized to the same shape before comparison.
    """
    h, w = original.shape[:2]
    attacked_resized = cv2.resize(attacked, (w, h), interpolation=cv2.INTER_LINEAR)

    p = psnr(original, attacked_resized, data_range=255)
    s = ssim(original, attacked_resized, data_range=255)
    return {"psnr": round(p, 2), "ssim": round(s, 4)}


def _print_table(rows: list[dict]) -> None:
    """Pretty-print the results table to stdout."""
    header = f"{'Attack':<24} {'DR%':>6} {'BER%':>6} {'PSNR':>7} {'SSIM':>7} {'Status':<10}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        status = "✓ PASS" if r["detected"] else "✗ FAIL"
        print(
            f"{r['attack']:<24} "
            f"{r['dr']:>6.1f} "
            f"{r['ber']:>6.1f} "
            f"{r['psnr']:>7.2f} "
            f"{r['ssim']:>7.4f} "
            f"{status:<10}"
        )
    print(sep)


def _save_chart(rows: list[dict], output_dir: Path) -> None:
    """Save a 2×2 matplotlib figure summarising all metrics."""
    attacks   = [r["attack"] for r in rows if r["attack"] != "original"]
    dr_vals   = [r["dr"]   for r in rows if r["attack"] != "original"]
    ber_vals  = [r["ber"]  for r in rows if r["attack"] != "original"]
    psnr_vals = [r["psnr"] for r in rows if r["attack"] != "original"]
    ssim_vals = [r["ssim"] for r in rows if r["attack"] != "original"]

    x = np.arange(len(attacks))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("QStegoForge — Robustness Metrics", fontsize=14, fontweight="bold")

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

    def _bar(ax, vals, title, ylabel, color="#4C72B0", hline=None):
        bars = ax.bar(x, vals, color=colors[: len(vals)], edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=35, ha="right", fontsize=8)
        if hline is not None:
            ax.axhline(hline, color="red", linestyle="--", linewidth=1, label=f"threshold={hline}")
            ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    _bar(axes[0, 0], dr_vals,   "Detection Rate (%)",           "DR (%)",   hline=50)
    _bar(axes[0, 1], ber_vals,  "Bit Error Rate (%)",            "BER (%)",  hline=10)
    _bar(axes[1, 0], psnr_vals, "PSNR vs Watermarked (dB)",     "PSNR (dB)", hline=30)
    _bar(axes[1, 1], ssim_vals, "SSIM vs Watermarked",           "SSIM",     hline=0.9)

    plt.tight_layout()
    chart_path = output_dir / "robustness_metrics.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[main] Chart saved → {chart_path}")


def _save_attacked_images(attacked_variants: dict, output_dir: Path) -> None:
    """Write each attacked image to disk."""
    img_dir = _ensure_dir(output_dir / "attacked_images")
    for name, img in attacked_variants.items():
        out = img_dir / f"{name}.png"
        cv2.imwrite(str(out), img)
    print(f"[main] Attacked images saved → {img_dir}/")


# ── main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    cover_path: Path,
    output_dir: Path,
    save_plots: bool = True,
) -> list[dict]:
    t_start = time.perf_counter()
    _ensure_dir(output_dir)

    # ── 1. Load cover image ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  QStegoForge — Quantum-Safe Steganographic Watermarking")
    print("=" * 60)

    cover = _load_or_create_cover(cover_path, size=(IMAGE_SIZE, IMAGE_SIZE))

    # ── 2. Generate watermark payload ────────────────────────────────────────
    print(f"\n[main] Generating {WATERMARK_SIZE}-bit watermark payload …")
    rng = np.random.default_rng(42)
    raw_payload = rng.integers(0, 2, WATERMARK_SIZE, dtype=np.uint8)
    print(f"[main] Payload (first 16 bits): {raw_payload[:16].tolist()}")

    # ── 3. Kyber512 encryption ───────────────────────────────────────────────
    print("\n[main] Running CRYSTALS-Kyber key generation & payload encryption …")
    crypto = KyberCrypto(variant=KYBER_VARIANT)
    public_key, secret_key = crypto.generate_keypair()
    encrypted_payload, shared_secret_enc = crypto.encrypt(public_key, raw_payload)
    print(f"[main] Public key size  : {len(public_key)} bytes")
    print(f"[main] Encrypted payload: {len(encrypted_payload)} bytes")

    # ── 4. NEQR quantum encoding ─────────────────────────────────────────────
    print(f"\n[main] NEQR encoding cover image ({NUM_QUBITS} qubits) …")
    encoder = NEQREncoder(num_qubits=NUM_QUBITS)
    quantum_state = encoder.encode(cover)
    print(f"[main] NEQR circuit depth: {quantum_state.depth()}")

    # ── 5. Watermark embedding ───────────────────────────────────────────────
    print("\n[main] Embedding watermark via X-gate LSB manipulation …")
    embedder = WatermarkEmbedder(encoder=encoder)
    watermarked_img, embedded_circuit = embedder.embed(
        cover_image=cover,
        watermark_bits=encrypted_payload,
    )
    print(f"[main] Watermarked image shape: {watermarked_img.shape}")

    # Save watermarked image
    wm_path = output_dir / "watermarked.png"
    cv2.imwrite(str(wm_path), watermarked_img)
    print(f"[main] Watermarked image saved → {wm_path}")

    # Baseline PSNR/SSIM between cover and watermarked
    baseline_psnr = psnr(cover, watermarked_img, data_range=255)
    baseline_ssim = ssim(cover, watermarked_img, data_range=255)
    print(f"[main] Baseline PSNR (cover vs watermarked): {baseline_psnr:.2f} dB")
    print(f"[main] Baseline SSIM (cover vs watermarked): {baseline_ssim:.4f}")

    # ── 6. Adversarial attacks ───────────────────────────────────────────────
    print("\n[main] Applying adversarial attack suite …")
    attacked_variants = run_all_attacks(watermarked_img)
    _save_attacked_images(attacked_variants, output_dir)

    # ── 7. Watermark extraction & verification ───────────────────────────────
    print("\n[main] Verifying watermark in each attacked variant …")
    verifier = WatermarkVerifier(
        encoder=encoder,
        secret_key=secret_key,
        original_payload=raw_payload,
    )

    rows = []
    for attack_name, attacked_img in attacked_variants.items():
        result = verifier.verify(
            watermarked_image=watermarked_img,
            attacked_image=attacked_img,
            embedded_circuit=embedded_circuit,
        )

        img_metrics = _compute_metrics(watermarked_img, attacked_img)

        row = {
            "attack":   attack_name,
            "dr":       round(result.get("detection_rate", 0.0) * 100, 2),
            "ber":      round(result.get("bit_error_rate", 1.0) * 100, 2),
            "detected": result.get("detected", False),
            "psnr":     img_metrics["psnr"],
            "ssim":     img_metrics["ssim"],
        }

        # Tamper localisation map (if available)
        tamper_map = result.get("tamper_map")
        if tamper_map is not None:
            tm_path = output_dir / f"tamper_{attack_name}.png"
            cv2.imwrite(str(tm_path), (tamper_map * 255).astype(np.uint8))

        rows.append(row)

    # ── 8. Print metrics table ───────────────────────────────────────────────
    _print_table(rows)

    # Summary stats
    pass_count  = sum(1 for r in rows if r["detected"])
    total       = len(rows)
    avg_dr      = np.mean([r["dr"]   for r in rows])
    avg_ber     = np.mean([r["ber"]  for r in rows])
    avg_psnr    = np.mean([r["psnr"] for r in rows])
    avg_ssim    = np.mean([r["ssim"] for r in rows])

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Variants tested   : {total}")
    print(f"  Watermark detected: {pass_count}/{total}")
    print(f"  Avg Detection Rate: {avg_dr:.1f}%")
    print(f"  Avg Bit Error Rate: {avg_ber:.1f}%")
    print(f"  Avg PSNR          : {avg_psnr:.2f} dB")
    print(f"  Avg SSIM          : {avg_ssim:.4f}")
    print(f"  Baseline PSNR     : {baseline_psnr:.2f} dB (cover ↔ watermarked)")
    print(f"  Baseline SSIM     : {baseline_ssim:.4f} (cover ↔ watermarked)")
    elapsed = time.perf_counter() - t_start
    print(f"  Total runtime     : {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # ── 9. Save metrics chart ────────────────────────────────────────────────
    if save_plots:
        _save_chart(rows, output_dir)

    # Persist numeric results as CSV
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w") as fh:
        fh.write("attack,dr_pct,ber_pct,psnr_db,ssim,detected\n")
        for r in rows:
            fh.write(
                f"{r['attack']},{r['dr']},{r['ber']},"
                f"{r['psnr']},{r['ssim']},{int(r['detected'])}\n"
            )
    print(f"[main] Metrics CSV saved → {csv_path}")

    return rows


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QStegoForge — Quantum-Safe Steganographic Watermarking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py
              python main.py --cover assets/lena.png --output-dir results/lena
              python main.py --no-plots
        """),
    )
    parser.add_argument(
        "--cover",
        type=Path,
        default=Path("assets/cover.png"),
        help="Path to cover image (default: assets/cover.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(RESULTS_DIR),
        help="Directory for output images, charts, and CSV (default: results/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib chart generation (useful on headless servers)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        cover_path=args.cover,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
    )