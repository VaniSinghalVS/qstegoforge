import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.embedder import embed_pipeline
from src.verifier import verify_pipeline
import cv2

def test_verifier_pipeline_spatial():
    """Test full verify pipeline in classical spatial mode."""
    test_input = "data/input/test_verifier_spatial.png"
    test_output = "data/watermarked/test_verifier_spatial_watermarked.png"
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/watermarked", exist_ok=True)
    
    # 256x512 image so it has exactly 32 patches (256 bits)
    img = np.random.randint(0, 256, (256, 512), dtype=np.uint8)
    cv2.imwrite(test_input, img)
    
    metadata = embed_pipeline(test_input, test_output, embedding_mode="classical_spatial")
    
    # verify
    results = verify_pipeline(test_output, metadata, embedding_mode="classical_spatial")
    
    print(f"Spatial Detection Rate: {results['detection_rate']}")
    assert results["is_authentic"] is True
    assert results["detection_rate"] == 1.0
    assert results["tamper_pixels"] == 0

def test_verifier_pipeline_frequency():
    """Test full verify pipeline in classical frequency mode."""
    test_input = "data/input/test_verifier_freq.png"
    test_output = "data/watermarked/test_verifier_freq_watermarked.png"
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/watermarked", exist_ok=True)
    
    # 256x512 image, with restricted range to avoid clipping when IFFT alters values
    img = np.random.randint(50, 200, (256, 512), dtype=np.uint8)
    cv2.imwrite(test_input, img)
    
    metadata = embed_pipeline(test_input, test_output, embedding_mode="classical_frequency")
    
    # verify
    results = verify_pipeline(test_output, metadata, embedding_mode="classical_frequency")
    
    print(f"Frequency Detection Rate: {results['detection_rate']}")
    assert results["is_authentic"] is True
    assert results["detection_rate"] >= 0.9  # Because FFT/IFFT might have minor rounding differences

if __name__ == "__main__":
    pytest.main(["-v", __file__])
