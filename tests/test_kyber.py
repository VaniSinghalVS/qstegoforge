import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.kyber_crypto import (
    generate_keypair,
    encrypt_watermark,
    decrypt_watermark,
    generate_watermark_payload,
    generate_position_seed,
    verify_shared_secrets
)

def test_kyber_kdf_correct_key():
    """
    Test 1: Correct Key
    Given: Kyber512 key pair (private_key, public_key)
    Action: Encapsulate with public_key -> ciphertext, shared_secret_A
            Decapsulate with private_key -> shared_secret_B
    Assertion: shared_secret_A == shared_secret_B
    Assertion: HKDF-derived payload_A == payload_B
    Assertion: HKDF-derived seed_A == seed_B
    """
    public_key, private_key = generate_keypair()
    
    ciphertext, shared_secret_A = encrypt_watermark(public_key)
    shared_secret_B = decrypt_watermark(private_key, ciphertext)
    
    assert verify_shared_secrets(shared_secret_A, shared_secret_B), "Shared secrets must match"
    
    payload_A = generate_watermark_payload(shared_secret_A)
    payload_B = generate_watermark_payload(shared_secret_B)
    np.testing.assert_array_equal(payload_A, payload_B, "Payloads must match")
    
    seed_A = generate_position_seed(shared_secret_A, patch_index=0)
    seed_B = generate_position_seed(shared_secret_B, patch_index=0)
    assert seed_A == seed_B, "Position seeds must match"

def test_kyber_kdf_incorrect_key():
    """
    Test 2: Incorrect Key
    Given: Two different Kyber512 key pairs (pair1, pair2)
    Action: Decapsulate ciphertext from pair1 with pair2.private_key
    Assertion: Decapsulated secret is different
    Assertion: HKDF-derived payload is statistically independent
    """
    public_key1, private_key1 = generate_keypair()
    public_key2, private_key2 = generate_keypair()
    
    ciphertext, shared_secret_correct = encrypt_watermark(public_key1)
    
    # Decapsulate with wrong key
    shared_secret_wrong = decrypt_watermark(private_key2, ciphertext)
    
    assert not verify_shared_secrets(shared_secret_correct, shared_secret_wrong), "Shared secrets must not match"
    
    payload_correct = generate_watermark_payload(shared_secret_correct)
    payload_wrong = generate_watermark_payload(shared_secret_wrong)
    
    assert not np.array_equal(payload_correct, payload_wrong), "Payloads must not match"
    
    seed_correct = generate_position_seed(shared_secret_correct, patch_index=1)
    seed_wrong = generate_position_seed(shared_secret_wrong, patch_index=1)
    assert seed_correct != seed_wrong, "Position seeds must not match"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
