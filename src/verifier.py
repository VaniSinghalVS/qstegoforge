"""
verifier.py — Watermark Extraction and Tamper Localization
Extracts the embedded watermark from a watermarked image,
verifies it against the original payload, and localizes
any tampered regions using spatial mismatch patterns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import matplotlib.pyplot as plt
import config
from src.neqr import preprocess_image, image_to_neqr
from src.kyber_crypto import (
    decrypt_watermark,
    generate_watermark_payload,
    compute_image_hash,
    verify_shared_secrets
)
from src.embedder import select_embedding_positions, circuit_to_image


def extract_watermark_from_image(img: np.ndarray, positions: list) -> np.ndarray:
    """
    Extract watermark bits from a (possibly tampered) image
    by reading the LSB of pixel values at the embedding positions.
    """
    height, width = img.shape
    extracted_bits = []

    for pos in positions:
        row = pos // width
        col = pos % width
        if row < height and col < width:
            pixel_value = int(img[row, col])
            lsb = pixel_value & 1  # extract least significant bit
            extracted_bits.append(lsb)
        else:
            extracted_bits.append(0)

    return np.array(extracted_bits, dtype=np.uint8)


def compute_detection_rate(original_payload: np.ndarray,
                           extracted_payload: np.ndarray) -> float:
    """
    Compute the watermark detection rate as the fraction of bits
    that match between original and extracted payloads.
    """
    if len(original_payload) != len(extracted_payload):
        min_len = min(len(original_payload), len(extracted_payload))
        original_payload  = original_payload[:min_len]
        extracted_payload = extracted_payload[:min_len]

    matches       = np.sum(original_payload == extracted_payload)
    detection_rate = matches / len(original_payload)
    return float(detection_rate)


def localize_tampering(original_img: np.ndarray,
                       watermarked_img: np.ndarray,
                       positions: list,
                       original_payload: np.ndarray,
                       extracted_payload: np.ndarray) -> np.ndarray:
    """
    Produce a tamper localization map by marking pixels where
    the extracted watermark bit does not match the original.
    White = tampered, Black = authentic.
    """
    height, width = original_img.shape
    tamper_map    = np.zeros((height, width), dtype=np.uint8)

    for idx, pos in enumerate(positions):
        if idx >= len(original_payload) or idx >= len(extracted_payload):
            break
        if original_payload[idx] != extracted_payload[idx]:
            row = pos // width
            col = pos % width
            if row < height and col < width:
                tamper_map[row, col] = 255  # mark as tampered

    return tamper_map


def save_tamper_map(tamper_map: np.ndarray, output_path: str):
    """Save the tamper localization map as an image."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, tamper_map)
    print(f"[Verifier] Tamper map saved: {output_path}")


def verify_pipeline(watermarked_path: str, metadata: dict) -> dict:
    """
    Full verification pipeline:
    1. Load watermarked image
    2. Decrypt watermark using Kyber private key
    3. Reconstruct expected payload
    4. Extract actual payload from image LSBs
    5. Compare payloads and compute detection rate
    6. Localize tampered regions
    7. Return verification results
    """
    print("\n=== QStegoForge Verification Pipeline ===\n")

    # Step 1: Load watermarked image
    img = preprocess_image(watermarked_path)
    print(f"[Image]    Loaded: {watermarked_path} → shape {img.shape}")

    # Step 2: Decrypt to recover shared secret
    shared_secret = decrypt_watermark(metadata["private_key"],
                                      metadata["ciphertext"])

    # Step 3: Reconstruct expected payload
    expected_payload = generate_watermark_payload(shared_secret)
    positions        = metadata["positions"]
    print(f"[Kyber]    Shared secret recovered: {len(shared_secret)} bytes")

    # Step 4: Extract actual payload from image
    extracted_payload = extract_watermark_from_image(img, positions)
    print(f"[Extract]  Extracted {len(extracted_payload)} bits from image LSBs")

    # Step 5: Compute detection rate
    detection_rate = compute_detection_rate(expected_payload, extracted_payload)
    print(f"[Metrics]  Detection rate: {detection_rate * 100:.2f}%")

    # Step 6: Compute image hash and check integrity
    current_hash   = compute_image_hash(img)
    hash_match     = (current_hash == metadata["image_hash"])
    print(f"[Hash]     Original hash : {metadata['image_hash'][:32]}...")
    print(f"[Hash]     Current hash  : {current_hash[:32]}...")
    print(f"[Hash]     Hash match    : {hash_match}")

    # Step 7: Tamper localization
    original_img = preprocess_image(watermarked_path)
    tamper_map   = localize_tampering(
        original_img, img, positions,
        expected_payload, extracted_payload
    )
    tamper_pixels = np.sum(tamper_map > 0)
    print(f"[Tamper]   Tampered pixels detected: {tamper_pixels}")

    # Step 8: Save tamper map
    save_tamper_map(tamper_map, "results/visualizations/tamper_map.png")

    is_authentic = detection_rate >= config.MIN_DETECTION

    print(f"\n[Result]   Image is {'AUTHENTIC ✓' if is_authentic else 'TAMPERED ✗'}")
    print("\n=== Verification Complete ===")

    return {
        "detection_rate"   : detection_rate,
        "hash_match"       : hash_match,
        "tamper_pixels"    : int(tamper_pixels),
        "is_authentic"     : is_authentic,
        "expected_payload" : expected_payload,
        "extracted_payload": extracted_payload
    }


if __name__ == "__main__":
    from src.embedder import embed_pipeline

    # Embed first
    test_input       = "data/input/test.png"
    test_watermarked = "data/watermarked/test_watermarked.png"
    metadata         = embed_pipeline(test_input, test_watermarked)

    # Then verify
    results = verify_pipeline(test_watermarked, metadata)

    print(f"\nDetection rate : {results['detection_rate'] * 100:.2f}%")
    print(f"Hash match     : {results['hash_match']}")
    print(f"Tamper pixels  : {results['tamper_pixels']}")
    print(f"Authentic      : {results['is_authentic']}")
