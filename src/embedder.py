"""
embedder.py — Quantum LSB Watermark Embedding
Embeds an encrypted watermark payload into an NEQR quantum circuit
using CNOT gate operations on qubit color registers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
import cv2
import config
from src.neqr import preprocess_image, image_to_neqr
from src.kyber_crypto import (
    generate_keypair,
    encrypt_watermark,
    generate_watermark_payload,
    compute_image_hash
)


def select_embedding_positions(shared_secret: bytes, n_pixels: int, n_bits: int) -> list:
    """
    Pseudo-randomly select pixel positions for embedding using
    the shared secret as seed. Attacker cannot reconstruct
    positions without the private key.
    """
    seed      = int.from_bytes(shared_secret[:4], byteorder='big')
    rng       = np.random.default_rng(seed)
    positions = rng.choice(n_pixels, size=n_bits, replace=False)
    return positions.tolist()


def embed_watermark_in_circuit(qc: QuantumCircuit, payload: np.ndarray) -> QuantumCircuit:
    """
    Embed watermark bits into the NEQR circuit's color register
    using X gates on the LSB qubit.
    For each payload bit that is 1, flip the LSB of the color register.
    """
    qc_embedded = qc.copy()
    lsb_qubit   = 0  # qubit 0 = LSB of color register

    for bit in payload:
        if bit == 1:
            qc_embedded.x(lsb_qubit)

    print(f"[Embedder] Embedded {len(payload)} bits via X gates on LSB qubit")
    return qc_embedded


def circuit_to_image(qc: QuantumCircuit, height: int, width: int) -> np.ndarray:
    """
    Simulate the quantum circuit and reconstruct the watermarked
    image from measurement outcomes.
    """
    n_color    = config.NEQR_COLOR_QUBITS
    n_position = config.NEQR_POSITION_QUBITS

    qc_measured = qc.copy()
    qc_measured.measure_all()

    simulator = AerSimulator()
    compiled  = transpile(qc_measured, simulator)
    job       = simulator.run(compiled, shots=2048)
    counts    = job.result().get_counts()

    img = np.zeros((height, width), dtype=np.uint8)

    for bitstring in counts:
        bits          = bitstring.replace(" ", "")
        color_bits    = bits[:n_color]
        position_bits = bits[n_color:n_color + n_position]

        pixel_value = int(color_bits, 2)
        position    = int(position_bits, 2)
        row         = position // width
        col         = position % width

        if row < height and col < width:
            img[row, col] = pixel_value

    return img


def embed_pipeline(image_path: str, output_path: str) -> dict:
    """
    Full embedding pipeline:
    1. Load and preprocess image
    2. Generate Kyber keypair
    3. Encrypt watermark and derive payload
    4. Encode image to NEQR circuit
    5. Embed payload via X gates on LSB qubit
    6. Reconstruct and save watermarked image
    7. Return keys and metadata for verification
    """
    print("\n=== QStegoForge Embedding Pipeline ===\n")

    # Step 1: Load image
    img          = preprocess_image(image_path)
    height, width = img.shape
    print(f"[Image]    Loaded: {image_path} → shape {img.shape}")

    # Step 2: Generate Kyber keys
    pub_key, priv_key = generate_keypair()

    # Step 3: Encrypt watermark and derive payload
    shared_secret, ciphertext = encrypt_watermark(pub_key)
    payload  = generate_watermark_payload(shared_secret)
    img_hash = compute_image_hash(img)
    print(f"[Hash]     Original image SHA-3: {img_hash[:32]}...")

    # Step 4: Select embedding positions
    n_pixels  = height * width
    positions = select_embedding_positions(shared_secret, n_pixels, len(payload))
    print(f"[Positions] First 5 embed positions: {positions[:5]}")

    # Step 5: NEQR encoding
    print("[NEQR]     Encoding image to quantum circuit...")
    qc = image_to_neqr(img)
    print(f"[NEQR]     Circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

    # Step 6: Embed watermark
    qc_watermarked = embed_watermark_in_circuit(qc, payload)

    # Step 7: Reconstruct watermarked image
    print("[NEQR]     Simulating circuit to reconstruct image...")
    watermarked_img = circuit_to_image(qc_watermarked, height, width)

    # Step 8: Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, watermarked_img)
    print(f"[Image]    Watermarked image saved: {output_path}")

    print("\n=== Embedding Complete ===")

    return {
        "private_key" : priv_key,
        "ciphertext"  : ciphertext,
        "image_hash"  : img_hash,
        "positions"   : positions,
        "payload"     : payload
    }


if __name__ == "__main__":
    test_input  = "data/input/test.png"
    test_output = "data/watermarked/test_watermarked.png"

    metadata = embed_pipeline(test_input, test_output)
    print(f"\nPrivate key size : {len(metadata['private_key'])} bytes")
    print(f"Payload bits     : {len(metadata['payload'])}")
    print(f"Embed positions  : {metadata['positions'][:5]}... (first 5)")