"""
verifier.py — Watermark Extraction and Tamper Localization
Extracts the embedded watermark from a watermarked image,
verifies it against the original payload, and localizes
any tampered regions using spatial mismatch patterns.
Supports patch-tiling and frequency domain mode.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import math
import config
from src.neqr import preprocess_image
from src.kyber_crypto import (
    decrypt_watermark,
    generate_watermark_payload,
    generate_position_seed,
    compute_image_hash,
    verify_shared_secrets
)
from src.embedder import select_embedding_positions

# -------------------------
# Extraction Logic
# -------------------------

def extract_from_patch_spatial(patch: np.ndarray, positions: list, depth: int = None) -> np.ndarray:
    """Extract watermark bits from a 64x64 patch using majority vote across `depth` planes.
    
    When depth=1 (quantum modes), reads only bit-plane 0.
    When depth=config.LSB_DEPTH (classical_spatial), uses majority vote across all planes.
    """
    if depth is None:
        depth = config.LSB_DEPTH
    patch_h, patch_w = patch.shape
    extracted_bits = []

    for pos in positions:
        row = pos // patch_w
        col = pos % patch_w
        if row < patch_h and col < patch_w:
            v = int(patch[row, col])
            votes = sum((v >> plane) & 1 for plane in range(depth))
            extracted_bits.append(1 if votes > depth // 2 else 0)
        else:
            extracted_bits.append(0)

    return np.array(extracted_bits, dtype=np.uint8)

def extract_from_patch_frequency(patch: np.ndarray, positions: list) -> np.ndarray:
    """
    Extract watermark bits from 2D FFT mid-frequency coefficients.
    """
    patch_h, patch_w = patch.shape
    fft_patch = np.fft.fftshift(np.fft.fft2(patch, norm="ortho"))
    
    # Identify mid-frequencies (radius 8 to 24)
    center_y, center_x = patch_h // 2, patch_w // 2
    candidates = []
    
    for y in range(patch_h):
        for x in range(patch_w):
            if y == center_y and x == center_x: continue # skip DC
            # To preserve symmetry, only pick upper half plane (y < center_y) or (y == center_y and x > center_x)
            if y < center_y or (y == center_y and x > center_x):
                r = math.sqrt((y - center_y)**2 + (x - center_x)**2)
                if 8 <= r <= 24:
                    candidates.append((y, x))
    
    # Sort candidates for determinism
    candidates.sort()
    
    # Use the random positions to select from candidates
    # In frequency mode, 'positions' refers to indices within the candidates list
    extracted_bits = []
    delta = 20.0  # quantization step
    
    for pos in positions:
        idx = pos % len(candidates)
        cy, cx = candidates[idx]
        val = np.real(fft_patch[cy, cx])
        
        # Quantization index
        q_idx = math.floor(val / delta + 0.5)
        bit = q_idx % 2
        extracted_bits.append(int(bit))
        
    return np.array(extracted_bits, dtype=np.uint8)


def compute_detection_rate(original_payload: np.ndarray,
                           extracted_payload: np.ndarray) -> float:
    if len(original_payload) != len(extracted_payload):
        min_len = min(len(original_payload), len(extracted_payload))
        original_payload  = original_payload[:min_len]
        extracted_payload = extracted_payload[:min_len]

    matches       = np.sum(original_payload == extracted_payload)
    detection_rate = matches / len(original_payload)
    return float(detection_rate)


def verify_pipeline(watermarked_path: str, metadata: dict, embedding_mode: str = "quantum_spatial_shots_2048") -> dict:
    print(f"\n=== QStegoForge Verification ({embedding_mode}) ===\n")

    img = preprocess_image(watermarked_path)
    height, width = img.shape
    
    patch_size = 64
    n_patches_y = metadata["patch_grid"][0]
    n_patches_x = metadata["patch_grid"][1]
    n_patches = n_patches_y * n_patches_x
    
    # Pad if necessary
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')

    shared_secret = decrypt_watermark(metadata["private_key"], metadata["ciphertext"])
    expected_payload = generate_watermark_payload(shared_secret)
    
    extracted_bits_all = np.zeros_like(expected_payload)
    votes = np.zeros(len(expected_payload), dtype=np.int32)
    
    tamper_map = np.zeros((n_patches_y * patch_size, n_patches_x * patch_size), dtype=np.uint8)

    for py in range(n_patches_y):
        for px in range(n_patches_x):
            patch_index = py * n_patches_x + px
            y0, y1 = py * patch_size, (py + 1) * patch_size
            x0, x1 = px * patch_size, (px + 1) * patch_size
            patch = img[y0:y1, x0:x1]
            
            # Payload subset
            start_idx = (patch_index * 8) % len(expected_payload)
            end_idx = start_idx + 8
            if end_idx <= len(expected_payload):
                expected_patch = expected_payload[start_idx:end_idx]
            else:
                expected_patch = np.concatenate((expected_payload[start_idx:], expected_payload[:end_idx - len(expected_payload)]))
                
            seed = generate_position_seed(shared_secret, patch_index=patch_index)
            
            # Determine extraction method
            if "frequency" in embedding_mode:
                # Dynamically count candidates for the PRNG bound
                candidates_count = 0
                for y in range(patch_size):
                    for x in range(patch_size):
                        if y == patch_size//2 and x == patch_size//2: continue
                        if y < patch_size//2 or (y == patch_size//2 and x > patch_size//2):
                            r = math.sqrt((y - patch_size//2)**2 + (x - patch_size//2)**2)
                            if 8 <= r <= 24:
                                candidates_count += 1
                                
                positions = select_embedding_positions(seed, candidates_count, len(expected_patch))
                extracted_patch = extract_from_patch_frequency(patch, positions)
            else:
                # Spatial mode — match extraction depth to embedding depth
                # classical_spatial embeds into LSB_DEPTH planes; quantum modes embed only plane 0
                extract_depth = config.LSB_DEPTH if "classical_spatial" in embedding_mode else 1
                positions = select_embedding_positions(seed, patch_size*patch_size, len(expected_patch))
                extracted_patch = extract_from_patch_spatial(patch, positions, depth=extract_depth)
            
            # Accumulate votes
            for i, bit in enumerate(extracted_patch):
                actual_idx = (start_idx + i) % len(expected_payload)
                if bit == 1:
                    votes[actual_idx] += 1
                elif bit == 0:
                    votes[actual_idx] -= 1
                    
            # Localize tampering on a per-patch basis
            # If the patch payload does not match the expected patch payload
            patch_dr = compute_detection_rate(expected_patch, extracted_patch)
            if patch_dr < 1.0:
                tamper_map[y0:y1, x0:x1] = 255
                
    # Finalize payload via majority vote
    # Break ties to 0 (conservative default) to avoid silent misclassification of unvoted bits
    extracted_payload = np.where(votes > 0, 1, 0).astype(np.uint8)
    
    detection_rate = compute_detection_rate(expected_payload, extracted_payload)
    is_authentic = detection_rate >= config.MIN_DETECTION

    # Remove padding from tamper map
    tamper_map = tamper_map[:height, :width]
    tamper_pixels = np.sum(tamper_map > 0)
    
    # Compute integrity hash
    img_cropped = img[:height, :width]
    current_hash = compute_image_hash(img_cropped)
    hash_match = (current_hash == metadata["image_hash"])

    return {
        "detection_rate"   : detection_rate,
        "hash_match"       : hash_match,
        "tamper_pixels"    : int(tamper_pixels),
        "is_authentic"     : is_authentic,
        "expected_payload" : expected_payload,
        "extracted_payload": extracted_payload
    }
