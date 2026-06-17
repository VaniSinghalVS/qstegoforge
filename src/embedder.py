"""
embedder.py — Quantum LSB Watermark Embedding
Embeds an encrypted watermark payload into an NEQR quantum circuit
using position-conditioned MCX gates on qubit color registers.
Supports patch-tiling for larger images.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import cv2
import math
import config
from src.neqr import preprocess_image, image_to_neqr, neqr_to_image
from src.kyber_crypto import (
    generate_keypair,
    encrypt_watermark,
    generate_watermark_payload,
    generate_position_seed,
    compute_image_hash
)


def select_embedding_positions(seed: int, n_pixels: int, n_bits: int) -> list:
    """
    Pseudo-randomly select pixel positions for embedding using
    a seed derived from the shared secret.
    """
    rng       = np.random.default_rng(seed)
    positions = rng.choice(n_pixels, size=n_bits, replace=False)
    return positions.tolist()


def embed_watermark_in_circuit(qc: QuantumCircuit, original_img_patch: np.ndarray, payload: np.ndarray, positions: list) -> QuantumCircuit:
    """
    Embed watermark bits into the NEQR circuit's color register
    using position-conditioned MCX gates on the LSB qubit.
    """
    qc_embedded = qc.copy()
    
    n_position = config.NEQR_POSITION_QUBITS
    n_color    = config.NEQR_COLOR_QUBITS
    lsb_qubit  = 0  # color_reg[0] is the LSB
    
    height, width = original_img_patch.shape
    
    # In the NEQR circuit, the qubits are ordered:
    # color_reg (0 to n_color-1) then position_reg (n_color to n_color+n_position-1)
    position_qubits = list(range(n_color, n_color + n_position))
    color_qubit = lsb_qubit

    for i, p in enumerate(positions):
        row = p // width
        col = p % width
        
        # Get original LSB
        original_pixel = original_img_patch[row, col]
        original_lsb = original_pixel & 1
        
        target_bit = payload[i]
        
        if original_lsb != target_bit:
            # We need to flip the LSB at this position
            
            # Apply X gates to position qubits that are 0 for this position (open control)
            for j in range(n_position):
                if not ((p >> j) & 1):
                    qc_embedded.x(position_qubits[j])
            
            # Apply MCX from all position qubits to the color LSB qubit
            qc_embedded.mcx(position_qubits, color_qubit)
            
            # Uncompute the X gates
            for j in range(n_position):
                if not ((p >> j) & 1):
                    qc_embedded.x(position_qubits[j])

    return qc_embedded


def embed_in_patch_frequency(patch: np.ndarray, payload: np.ndarray, seed: int) -> np.ndarray:
    """
    Embed watermark payload into mid-frequency coefficients using classical 2D FFT.
    """
    patch_h, patch_w = patch.shape
    fft_patch = np.fft.fftshift(np.fft.fft2(patch, norm="ortho"))
    
    center_y, center_x = patch_h // 2, patch_w // 2
    candidates = []
    
    for y in range(patch_h):
        for x in range(patch_w):
            if y == center_y and x == center_x: continue
            if y < center_y or (y == center_y and x > center_x):
                r = math.sqrt((y - center_y)**2 + (x - center_x)**2)
                if 8 <= r <= 24:
                    candidates.append((y, x))
                    
    candidates.sort()
    positions = select_embedding_positions(seed, len(candidates), len(payload))
    
    delta = 20.0  # Quantization step size robust to uint8 truncation with ortho norm
    
    for i, pos in enumerate(positions):
        cy, cx = candidates[pos]
        val = np.real(fft_patch[cy, cx])
        target_bit = payload[i]
        
        # Quantize and embed using LSB of quantization index
        q_idx = math.floor(val / delta + 0.5)
        current_bit = q_idx % 2
        
        if current_bit != target_bit:
            if val > q_idx * delta:
                val -= delta
            else:
                val += delta
                
        # Update coefficient and its conjugate symmetric counterpart
        sym_y = 2 * center_y - cy
        sym_x = 2 * center_x - cx
        
        # Preserve imaginary part (should be very small for real images, but just in case)
        imag = np.imag(fft_patch[cy, cx])
        
        fft_patch[cy, cx] = val + 1j * imag
        if sym_y < patch_h and sym_x < patch_w:
            fft_patch[sym_y, sym_x] = val - 1j * imag
            
    # IFFT back to spatial domain
    watermarked_patch = np.fft.ifft2(np.fft.ifftshift(fft_patch), norm="ortho")
    watermarked_patch = np.clip(np.real(watermarked_patch), 0, 255).astype(np.uint8)
    return watermarked_patch


def process_patch(patch: np.ndarray, payload_patch: np.ndarray, patch_index: int, shared_secret: bytes, embedding_mode: str = "quantum_spatial_shots_2048", shots: int = 2048, simulator_mode: str = None) -> np.ndarray:
    """
    Process a single 64x64 patch: NEQR encode -> embed -> decode (or classical FFT).
    """
    patch_h, patch_w = patch.shape
    n_pixels = patch_h * patch_w
    seed = generate_position_seed(shared_secret, patch_index=patch_index)
    
    if "frequency" in embedding_mode:
        return embed_in_patch_frequency(patch, payload_patch, seed)
    
    if "classical_spatial" in embedding_mode:
        depth = config.LSB_DEPTH
        positions = select_embedding_positions(seed, n_pixels, len(payload_patch))
        watermarked_patch = patch.copy()
        for i, pos in enumerate(positions):
            row = pos // patch_w
            col = pos % patch_w
            target_bit = int(payload_patch[i])
            v = int(watermarked_patch[row, col])
            for plane in range(depth):
                mask = 1 << plane
                v = (v & ~mask) | (target_bit << plane)
            watermarked_patch[row, col] = v
        return watermarked_patch

    # Quantum Spatial Mode
    positions = select_embedding_positions(seed, n_pixels, len(payload_patch))
    qc = image_to_neqr(patch)
    print(f"[NEQR] Circuit: {qc.num_qubits} qubits, depth {qc.depth()} (patch {patch_index})")
    qc_watermarked = embed_watermark_in_circuit(qc, patch, payload_patch, positions)

    # simulator_mode kwarg takes explicit precedence; fall back to embedding_mode string
    if simulator_mode is None:
        simulator_mode = "statevector" if "statevector" in embedding_mode else "shots"
    watermarked_patch = neqr_to_image(qc_watermarked, patch_h, patch_w, simulator_mode=simulator_mode, shots=shots)
    return watermarked_patch


def embed_pipeline(image_path: str, output_path: str, embedding_mode: str = "quantum_spatial_shots_2048") -> dict:
    """
    Full embedding pipeline with patch tiling.
    embedding_mode: e.g., 'quantum_spatial_statevector', 'quantum_spatial_shots_2048'
    """
    print(f"\n=== QStegoForge Embedding ({embedding_mode}) ===\n")

    # Step 1: Load image
    img = preprocess_image(image_path)
    height, width = img.shape
    
    # Pad image to multiples of 64
    patch_size = 64
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')
        height, width = img.shape
    
    print(f"[Image]    Loaded & padded: {image_path} -> shape {img.shape}")

    # Step 2: Generate Kyber keys
    pub_key, priv_key = generate_keypair()

    # Step 3: Encrypt watermark and derive payload
    ciphertext, shared_secret = encrypt_watermark(pub_key)
    payload  = generate_watermark_payload(shared_secret)
    
    n_patches_y = height // patch_size
    n_patches_x = width // patch_size
    n_patches = n_patches_y * n_patches_x
    print(f"[Grid]     Image divided into {n_patches_y}x{n_patches_x} = {n_patches} patches")
    
    watermarked_img = np.zeros_like(img)
    
    # Determine simulator config from mode
    simulator_mode = "shots"
    shots = 2048
    if "statevector" in embedding_mode:
        simulator_mode = "statevector"
    elif "shots_" in embedding_mode:
        shots = int(embedding_mode.split("shots_")[-1])
        
    for py in range(n_patches_y):
        for px in range(n_patches_x):
            patch_index = py * n_patches_x + px
            y0, y1 = py * patch_size, (py + 1) * patch_size
            x0, x1 = px * patch_size, (px + 1) * patch_size
            
            patch = img[y0:y1, x0:x1]
            
            # Payload capacity is 8 bits per patch (1 byte)
            # Cycle through payload if needed
            start_idx = (patch_index * 8) % len(payload)
            end_idx = start_idx + 8
            if end_idx <= len(payload):
                payload_patch = payload[start_idx:end_idx]
            else:
                # Wrap around
                payload_patch = np.concatenate((payload[start_idx:], payload[:end_idx - len(payload)]))
                
            watermarked_patch = process_patch(
                patch, payload_patch, patch_index, shared_secret,
                embedding_mode=embedding_mode, shots=shots
            )
            
            watermarked_img[y0:y1, x0:x1] = watermarked_patch
            print(f"[Patch {patch_index+1}/{n_patches}] Processed via {simulator_mode} (shots={shots if simulator_mode=='shots' else 'N/A'})")

    # Step 8: Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, watermarked_img)
    print(f"[Image]    Watermarked image saved: {output_path}")

    # Compute hash on the final watermarked image
    img_hash = compute_image_hash(watermarked_img)

    return {
        "private_key" : priv_key,
        "ciphertext"  : ciphertext,
        "image_hash"  : img_hash,
        "payload"     : payload,
        "patch_grid"  : [n_patches_y, n_patches_x],
        "image_size"  : [height, width]
    }


if __name__ == "__main__":
    test_input  = "data/input/test.png"
    test_output = "data/watermarked/test_watermarked.png"
    
    # Create test input
    os.makedirs("data/input", exist_ok=True)
    cv2.imwrite(test_input, np.zeros((64, 64), dtype=np.uint8))

    metadata = embed_pipeline(test_input, test_output, embedding_mode="quantum_spatial_statevector")
    print(f"\nPrivate key size : {len(metadata['private_key'])} bytes")
    print(f"Payload bits     : {len(metadata['payload'])}")
