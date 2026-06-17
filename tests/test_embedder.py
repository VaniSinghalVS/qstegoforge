import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.embedder import embed_pipeline, process_patch, select_embedding_positions
from src.kyber_crypto import generate_keypair, encrypt_watermark, generate_watermark_payload
import cv2

def test_mcx_embedding():
    """
    Test position-conditioned MCX embedding via process_patch.
    We will create a 64x64 black image (all 0s).
    We will embed a payload of all 1s.
    This should result in the LSB of selected pixels being 1.
    """
    original_pos_qubits = config.NEQR_POSITION_QUBITS
    # Use real 64x64 config
    config.NEQR_POSITION_QUBITS = 12
    config.NEQR_COLOR_QUBITS = 8
    
    # 64x64 image
    img = np.zeros((64, 64), dtype=np.uint8)
    
    # Generate keys
    pub, priv = generate_keypair()
    ct, secret = encrypt_watermark(pub)
    
    # Mock a payload of 8 bits all set to 1
    payload = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    
    # Run process_patch using statevector to avoid shot noise issues
    watermarked_patch = process_patch(img, payload, patch_index=0, shared_secret=secret, simulator_mode="statevector")
    
    # The watermarked patch should have 1s at exactly the selected positions
    import src.kyber_crypto as kc
    seed = kc.generate_position_seed(secret, patch_index=0)
    positions = select_embedding_positions(seed, 64*64, len(payload))
    
    for i in range(64):
        for j in range(64):
            pos = i * 64 + j
            if pos in positions:
                assert watermarked_patch[i, j] == 1, f"Pixel at {i},{j} should be 1 but is {watermarked_patch[i,j]}"
            else:
                assert watermarked_patch[i, j] == 0, f"Pixel at {i},{j} should be 0 but is {watermarked_patch[i,j]}"

    config.NEQR_POSITION_QUBITS = original_pos_qubits

if __name__ == "__main__":
    pytest.main(["-v", __file__])
