"""
main.py
QStegoForge -- Full Pipeline Runner
====================================
Executes the complete quantum-safe steganographic watermarking pipeline:

  1.  Load (or auto-generate) a cover image
  2.  Kyber512 key generation + SHA-3 watermark derivation
  3.  Watermark embedding via classical LSB (fast) or NEQR circuit (--quantum)
  4.  Baseline PSNR / SSIM  (cover <-> watermarked)
  5.  Apply adversarial attack suite (JPEG, Gaussian, rotation, diffusion)
  6.  Watermark extraction & verification on each attacked variant
  7.  Print structured metrics table (DR, BER, PSNR, SSIM)
  8.  Save result images, tamper maps, matplotlib chart, and CSV

Usage:
    python main.py [--cover <path>] [--output-dir <dir>] [--no-plots] [--quantum]

Flags:
    --cover       Path to cover image (default: data/input/cover.png)
    --output-dir  Results directory   (default: results/)
    --no-plots    Skip matplotlib chart
    --quantum     Use full NEQR quantum circuit simulation (very slow)
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
matplotlib.use("Agg")           # non-interactive backend -- safe on all platforms
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity  as compute_ssim

# ---- ensure project root is on sys.path ------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ---- project modules -------------------------------------------------------
import config
from src.kyber_crypto import (
    generate_keypair,
    encrypt_watermark,
    decrypt_watermark,
    generate_watermark_payload,
    compute_image_hash,
)
from src.embedder import select_embedding_positions
from src.verifier import (
    extract_watermark_from_image,
    compute_detection_rate,
    localize_tampering,
)
from src.attacks import run_all_attacks


# ============================================================================
# Helpers
# ============================================================================

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_or_create_cover(cover_path: Path) -> np.ndarray:
    """Load a grayscale cover image; auto-generate a synthetic one if absent."""
    if cover_path.exists():
        img = cv2.imread(str(cover_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError("Failed to read image at %s" % cover_path)
        img = cv2.resize(img, config.IMAGE_SIZE)
        print("[main] Cover image loaded : %s  shape=%s" % (cover_path, img.shape))
    else:
        print("[main] Cover not found at %s -- generating synthetic test image." % cover_path)
        rng = np.random.default_rng(2024)
        img = rng.integers(
            0, 256,
            (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]),
            dtype=np.uint8,
        )
        _ensure_dir(cover_path.parent)
        cv2.imwrite(str(cover_path), img)
        print("[main] Synthetic cover saved -> %s" % cover_path)
    return img


def _embed_lsb(cover: np.ndarray,
               payload: np.ndarray,
               positions: list) -> np.ndarray:
    """
    Multi-plane LSB embedding (config.LSB_DEPTH planes).
    Each payload bit is written redundantly into LSB_DEPTH least-significant
    bit-planes of its target pixel, improving survival probability under
    mild attacks (Gaussian noise, mild JPEG) that corrupt only bit-0.

    Bit-plane 0 = LSB (bit 0), plane 1 = bit 1, ...
    Pixel value v with bit b embedded at plane p:
        v_new = (v & ~(1 << p)) | (b << p)
    """
    depth = config.LSB_DEPTH          # 2 by default
    watermarked = cover.copy()
    h, w = cover.shape
    for idx, pos in enumerate(positions):
        if idx >= len(payload):
            break
        row, col = pos // w, pos % w
        if row < h and col < w:
            v = int(watermarked[row, col])
            b = int(payload[idx])
            for plane in range(depth):
                mask = 1 << plane
                v = (v & ~mask) | (b << plane)
            watermarked[row, col] = v
    return watermarked


def _extract_lsb_majority(img: np.ndarray,
                           positions: list,
                           n_bits: int) -> np.ndarray:
    """
    Extract watermark bits using majority vote across config.LSB_DEPTH planes.
    Increases robustness: a bit is declared '1' if the majority of the
    LSB_DEPTH planes at that position are '1'.
    """
    depth = config.LSB_DEPTH
    h, w  = img.shape
    bits  = []
    for idx, pos in enumerate(positions):
        if idx >= n_bits:
            break
        row, col = pos // w, pos % w
        if row < h and col < w:
            v      = int(img[row, col])
            votes  = sum((v >> plane) & 1 for plane in range(depth))
            bits.append(1 if votes > depth // 2 else 0)
        else:
            bits.append(0)
    return np.array(bits, dtype=np.uint8)


def _compute_metrics(reference: np.ndarray, target: np.ndarray) -> dict:
    """PSNR and SSIM between reference and target (resized to same shape)."""
    h, w = reference.shape[:2]
    t = cv2.resize(target, (w, h), interpolation=cv2.INTER_LINEAR)
    if np.array_equal(reference, t):
        # Identical images: PSNR is infinite; cap at 100 dB for display
        p = 100.0
    else:
        p = float(compute_psnr(reference, t, data_range=255))
    s = float(compute_ssim(reference, t, data_range=255))
    return {"psnr": round(p, 2), "ssim": round(s, 4)}


def _print_table(rows: list) -> None:
    """Pretty-print the per-attack metrics table to stdout."""
    header = "%-26s %6s %6s %7s %7s  %s" % (
        "Attack", "DR %", "BER %", "PSNR", "SSIM", "Status"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        status = "PASS" if r["detected"] else "FAIL"
        print("%-26s %6.1f %6.1f %7.2f %7.4f  %s" % (
            r["attack"], r["dr"], r["ber"], r["psnr"], r["ssim"], status
        ))
    print(sep)


def _save_chart(rows: list, output_dir: Path) -> None:
    """Save a 2x2 matplotlib bar chart summarising all attack metrics."""
    attack_rows = [r for r in rows if r["attack"] != "original"]
    attacks   = [r["attack"]  for r in attack_rows]
    dr_vals   = [r["dr"]      for r in attack_rows]
    ber_vals  = [r["ber"]     for r in attack_rows]
    psnr_vals = [r["psnr"]    for r in attack_rows]
    ssim_vals = [r["ssim"]    for r in attack_rows]

    x = np.arange(len(attacks))
    palette = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("QStegoForge -- Robustness Metrics", fontsize=14, fontweight="bold")

    def _bar(ax, vals, title, ylabel, hline=None):
        bars = ax.bar(
            x, vals,
            color=palette[: len(vals)],
            edgecolor="black", linewidth=0.5,
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=35, ha="right", fontsize=8)
        if hline is not None:
            ax.axhline(hline, color="red", linestyle="--", linewidth=1,
                       label="threshold=%s" % hline)
            ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                "%.1f" % v,
                ha="center", va="bottom", fontsize=7,
            )

    _bar(axes[0, 0], dr_vals,   "Detection Rate (%)",       "DR (%)",    hline=96)
    _bar(axes[0, 1], ber_vals,  "Bit Error Rate (%)",        "BER (%)",   hline=4)
    _bar(axes[1, 0], psnr_vals, "PSNR vs Watermarked (dB)", "PSNR (dB)", hline=config.MIN_PSNR)
    _bar(axes[1, 1], ssim_vals, "SSIM vs Watermarked",       "SSIM",      hline=config.MIN_SSIM)

    plt.tight_layout()
    chart_path = output_dir / "robustness_metrics.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close()
    print("[main] Chart saved -> %s" % chart_path)


def _save_csv(rows: list, output_dir: Path) -> None:
    csv_path = output_dir / "metrics.csv"
    with open(str(csv_path), "w") as fh:
        fh.write("attack,dr_pct,ber_pct,psnr_db,ssim,detected\n")
        for r in rows:
            fh.write("%s,%.2f,%.2f,%.2f,%.4f,%d\n" % (
                r["attack"], r["dr"], r["ber"],
                r["psnr"], r["ssim"], int(r["detected"]),
            ))
    print("[main] Metrics CSV saved -> %s" % csv_path)


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(
    cover_path: Path,
    output_dir: Path,
    save_plots: bool = True,
    fast_mode: bool = True,
) -> list:
    """
    Execute the full QStegoForge pipeline and return per-attack metric rows.

    Parameters
    ----------
    cover_path : path to the cover image (auto-created if absent)
    output_dir : directory where all outputs are written
    save_plots : whether to save the matplotlib summary chart
    fast_mode  : True  -> classical LSB embedding (milliseconds)
                 False -> full NEQR quantum circuit simulation (minutes)
    """
    t_start = time.perf_counter()

    _ensure_dir(output_dir)
    _ensure_dir(output_dir / "attacked_images")
    _ensure_dir(output_dir / "tamper_maps")
    _ensure_dir(Path(config.METRICS_DIR))
    _ensure_dir(Path(config.VIZ_DIR))

    print("\n" + "=" * 62)
    print("  QStegoForge -- Quantum-Safe Steganographic Watermarking")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Step 1: Load cover image
    # ------------------------------------------------------------------
    print("\n[main] -- Step 1: Load cover image ----------------------------")
    cover = _load_or_create_cover(cover_path)
    h, w  = cover.shape

    # ------------------------------------------------------------------
    # Step 2: Kyber512 key generation
    # ------------------------------------------------------------------
    print("\n[main] -- Step 2: CRYSTALS-Kyber512 key generation ------------")
    pub_key, priv_key = generate_keypair()

    # ------------------------------------------------------------------
    # Step 3: Watermark encryption & payload derivation
    # ------------------------------------------------------------------
    print("\n[main] -- Step 3: Watermark encryption (Kyber KEM + SHA-3) ---")
    ciphertext, shared_secret = encrypt_watermark(pub_key)
    payload   = generate_watermark_payload(shared_secret)   # 64-bit np.uint8 array
    img_hash  = compute_image_hash(cover)
    print("[main] Watermark payload      : %d bits" % len(payload))
    print("[main] Cover image SHA-3      : %s..." % img_hash[:32])

    # Pseudo-random embedding positions (seeded by shared secret)
    n_pixels  = h * w
    positions = select_embedding_positions(shared_secret, n_pixels, len(payload))
    print("[main] Embed positions (first 5): %s" % positions[:5])

    # ------------------------------------------------------------------
    # Step 4: Embedding
    # ------------------------------------------------------------------
    if fast_mode:
        print("\n[main] -- Step 4: Classical LSB embedding (fast mode) ---------")
        watermarked_img = _embed_lsb(cover, payload, positions)
    else:
        print("\n[main] -- Step 4: NEQR quantum circuit embedding (slow) -------")
        from src.neqr import image_to_neqr
        from src.embedder import embed_watermark_in_circuit, circuit_to_image
        print("[main] Encoding cover to NEQR circuit...")
        qc = image_to_neqr(cover)
        print("[main] Circuit: %d qubits, depth %d" % (qc.num_qubits, qc.depth()))
        qc_wm = embed_watermark_in_circuit(qc, payload)
        print("[main] Simulating circuit...")
        watermarked_img = circuit_to_image(qc_wm, h, w)

    print("[main] Watermarked image shape : %s" % str(watermarked_img.shape))

    # Save watermarked image
    wm_path = output_dir / "watermarked.png"
    cv2.imwrite(str(wm_path), watermarked_img)
    print("[main] Watermarked image saved -> %s" % wm_path)

    # Baseline PSNR / SSIM  (cover <-> watermarked)
    baseline_psnr = float(compute_psnr(cover, watermarked_img, data_range=255))
    baseline_ssim = float(compute_ssim(cover, watermarked_img, data_range=255))
    print("[main] Baseline PSNR (cover <-> watermarked): %.2f dB" % baseline_psnr)
    print("[main] Baseline SSIM (cover <-> watermarked): %.4f"    % baseline_ssim)
    print("[main] PSNR >= %.1f dB : %s (%.2f)" % (
        config.MIN_PSNR,
        "PASS" if baseline_psnr >= config.MIN_PSNR else "FAIL",
        baseline_psnr,
    ))
    print("[main] SSIM >= %.2f    : %s (%.4f)" % (
        config.MIN_SSIM,
        "PASS" if baseline_ssim >= config.MIN_SSIM else "FAIL",
        baseline_ssim,
    ))

    # ------------------------------------------------------------------
    # Step 5: Recover expected payload via Kyber decryption
    # ------------------------------------------------------------------
    print("\n[main] -- Step 5: Kyber decryption -> expected payload --------")
    recovered_secret = decrypt_watermark(priv_key, ciphertext)
    expected_payload = generate_watermark_payload(recovered_secret)
    print("[main] Shared-secret recovered : %d bytes" % len(recovered_secret))
    print("[main] Expected payload bits   : %d"       % len(expected_payload))

    # ------------------------------------------------------------------
    # Step 6: Adversarial attack suite
    # ------------------------------------------------------------------
    print("\n[main] -- Step 6: Adversarial attack suite --------------------")
    attacked_variants = run_all_attacks(watermarked_img)

    img_dir = output_dir / "attacked_images"
    for name, img in attacked_variants.items():
        cv2.imwrite(str(img_dir / ("%s.png" % name)), img)
    print("[main] Attacked images saved -> %s/" % img_dir)

    # ------------------------------------------------------------------
    # Step 7: Watermark extraction & verification
    # ------------------------------------------------------------------
    print("\n[main] -- Step 7: Verification across all attacked variants ---")
    rows = []

    for attack_name, attacked_img in attacked_variants.items():
        # Resize back to cover dimensions for LSB extraction
        attacked_resized = cv2.resize(attacked_img, (w, h),
                                      interpolation=cv2.INTER_LINEAR)

        # Extract watermark bits using multi-plane majority vote
        extracted = _extract_lsb_majority(attacked_resized, positions, len(expected_payload))

        # Detection Rate and Bit Error Rate
        dr  = compute_detection_rate(expected_payload, extracted)
        ber = 1.0 - dr

        # Image quality metrics (watermarked <-> attacked)
        img_metrics = _compute_metrics(watermarked_img, attacked_img)

        # Tamper localisation map
        tamper_map    = localize_tampering(
            cover, attacked_resized, positions,
            expected_payload, extracted,
        )
        tamper_pixels = int(np.sum(tamper_map > 0))

        # Save tamper map
        tm_path = output_dir / "tamper_maps" / ("tamper_%s.png" % attack_name)
        cv2.imwrite(str(tm_path), tamper_map)

        detected = dr >= config.MIN_DETECTION

        row = {
            "attack":        attack_name,
            "dr":            round(dr  * 100, 2),
            "ber":           round(ber * 100, 2),
            "detected":      detected,
            "psnr":          img_metrics["psnr"],
            "ssim":          img_metrics["ssim"],
            "tamper_pixels": tamper_pixels,
        }
        rows.append(row)

        print("  [%s] %-22s  DR=%.1f%%  BER=%.1f%%  PSNR=%.2f  SSIM=%.4f  tampered_px=%d" % (
            "PASS" if detected else "FAIL",
            attack_name,
            dr * 100, ber * 100,
            img_metrics["psnr"], img_metrics["ssim"],
            tamper_pixels,
        ))

    # ------------------------------------------------------------------
    # Step 8: Print metrics table
    # ------------------------------------------------------------------
    _print_table(rows)

    pass_count = sum(1 for r in rows if r["detected"])
    total      = len(rows)
    avg_dr     = float(np.mean([r["dr"]   for r in rows]))
    avg_ber    = float(np.mean([r["ber"]  for r in rows]))
    avg_psnr   = float(np.mean([r["psnr"] for r in rows]))
    avg_ssim   = float(np.mean([r["ssim"] for r in rows]))
    elapsed    = time.perf_counter() - t_start

    print("\n" + "=" * 62)
    print("  Summary")
    print("=" * 62)
    print("  Variants tested      : %d"           % total)
    print("  Watermark detected   : %d/%d"        % (pass_count, total))
    print("  Avg Detection Rate   : %.2f%%  (target >= %.0f%%)" % (
        avg_dr, config.MIN_DETECTION * 100))
    print("  Avg Bit Error Rate   : %.2f%%"        % avg_ber)
    print("  Avg PSNR             : %.2f dB  (target >= %.1f dB)" % (
        avg_psnr, config.MIN_PSNR))
    print("  Avg SSIM             : %.4f  (target >= %.2f)" % (
        avg_ssim, config.MIN_SSIM))
    print("  Baseline PSNR        : %.2f dB  (cover <-> watermarked)" % baseline_psnr)
    print("  Baseline SSIM        : %.4f  (cover <-> watermarked)"    % baseline_ssim)
    print("  Total runtime        : %.1f s"        % elapsed)
    print("=" * 62 + "\n")

    # ------------------------------------------------------------------
    # Step 9: Save chart & CSV
    # ------------------------------------------------------------------
    if save_plots:
        _save_chart(rows, output_dir)

    _save_csv(rows, output_dir)

    return rows


# ============================================================================
# CLI entry point
# ============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QStegoForge -- Quantum-Safe Steganographic Watermarking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py
              python main.py --cover data/input/lena.png --output-dir results/lena
              python main.py --no-plots
              python main.py --quantum          # full NEQR simulation (slow)
        """),
    )
    parser.add_argument(
        "--cover",
        type=Path,
        default=Path("data/input/cover.png"),
        help="Path to grayscale cover image (default: data/input/cover.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for all outputs (default: results/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib chart generation",
    )
    parser.add_argument(
        "--quantum",
        action="store_true",
        default=False,
        help="Use full NEQR quantum circuit simulation for embedding (very slow)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        cover_path=args.cover,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
        fast_mode=not args.quantum,
    )