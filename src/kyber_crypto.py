"""
kyber_crypto.py — CRYSTALS-Kyber Post-Quantum Cryptography Layer
Handles key generation, watermark encryption via Kyber KEM,
and SHA-3 hashing for the QStegoForge pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import secrets
import numpy as np
from kyber_py.kyber import Kyber512
import config


def generate_keypair():
    """Generate a Kyber512 public/private key pair."""
    public_key, private_key = Kyber512.keygen()
    print(f"[Kyber] Public key size  : {len(public_key)} bytes")
    print(f"[Kyber] Private key size : {len(private_key)} bytes")
    return public_key, private_key


def encrypt_watermark(public_key: bytes) -> tuple:
    """
    Encapsulate a shared secret using the public key.
    The shared secret becomes our watermark encryption key.
    """
    shared_secret, ciphertext = Kyber512.encaps(public_key)
    print(f"[Kyber] Ciphertext size    : {len(ciphertext)} bytes")
    print(f"[Kyber] Shared secret size : {len(shared_secret)} bytes")
    return ciphertext, shared_secret


def decrypt_watermark(private_key: bytes, ciphertext: bytes) -> bytes:
    """Decapsulate and recover the shared secret using the private key."""
    shared_secret = Kyber512.decaps(private_key, ciphertext)    
    return shared_secret


def generate_watermark_payload(shared_secret: bytes) -> np.ndarray:
    """
    Derive a fixed-length watermark bit array from the shared secret
    using SHA-3. This is the actual payload embedded into the image.
    """
    digest = hashlib.new(config.HASH_ALGORITHM, shared_secret).digest()

    all_bits = []
    for byte in digest:
        all_bits.extend([int(b) for b in format(byte, '08b')])

    payload = all_bits[:config.WATERMARK_BITS]
    return np.array(payload, dtype=np.uint8)


def compute_image_hash(img: np.ndarray) -> str:
    """Compute SHA-3 hash of an image array for tamper detection."""
    return hashlib.new(config.HASH_ALGORITHM, img.tobytes()).hexdigest()


def verify_shared_secrets(secret1: bytes, secret2: bytes) -> bool:
    """Constant-time comparison of two shared secrets."""
    return secrets.compare_digest(secret1, secret2)


if __name__ == "__main__":
    print("=== Testing Kyber Crypto Pipeline ===\n")

    pub_key, priv_key = generate_keypair()

    ciphertext, secret_sender = encrypt_watermark(pub_key)

    secret_receiver = decrypt_watermark(priv_key, ciphertext)

    match = verify_shared_secrets(secret_sender, secret_receiver)
    print(f"\n[Kyber] Shared secrets match : {match}")

    payload = generate_watermark_payload(secret_sender)
    print(f"[SHA-3] Watermark payload    : {len(payload)} bits")
    print(f"[SHA-3] First 16 bits        : {payload[:16]}")

    print("\n=== Kyber Crypto: OK ===")
