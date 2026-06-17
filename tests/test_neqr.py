import sys
import os
import numpy as np
from qiskit import QuantumCircuit
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.neqr import image_to_neqr, neqr_to_image

def test_neqr_roundtrip():
    # Let's use a very small toy image for speed: 4x4
    # Note: For this to work, we need to temporarily override config or just use the config as is if it's not too big.
    # config.IMAGE_SIZE is likely 64x64 or 256x256. 
    # NEQR position qubits would be 12 for 64x64.
    # We can just test a small block on the actual 64x64 if we want, but generating the circuit for 64x64 takes a while in python.
    # Let's mock the config in the test for a 2x2 image (4 pixels = 2 position qubits).
    
    original_pos_qubits = config.NEQR_POSITION_QUBITS
    config.NEQR_POSITION_QUBITS = 2
    
    height, width = 2, 2
    test_img = np.array([
        [100, 200],
        [50,  25]
    ], dtype=np.uint8)
    
    # 1. Encode
    qc = image_to_neqr(test_img)
    
    # 2. Decode using statevector
    reconstructed_img = neqr_to_image(qc, height, width, simulator_mode="statevector")
    
    # 3. Assert equality
    np.testing.assert_array_equal(test_img, reconstructed_img)
    
    # Also test shots mode with a very high shot count to guarantee coverage for 4 pixels
    # 4 pixels, so 1024 shots is plenty to see all 4 states
    reconstructed_img_shots = neqr_to_image(qc, height, width, simulator_mode="shots", shots=1024)
    np.testing.assert_array_equal(test_img, reconstructed_img_shots)
    
    # Restore config
    config.NEQR_POSITION_QUBITS = original_pos_qubits

if __name__ == "__main__":
    pytest.main(["-v", __file__])
