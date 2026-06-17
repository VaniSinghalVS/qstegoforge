"""
probe_baseline_256.py
Embed into a 256x256 image, verify directly on the watermarked file.
NO attack applied at any point.
Reports: detection_rate, hash_match, tamper_pixels, expected vs extracted payload.
"""
import os, sys, json, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.embedder import embed_pipeline
from src.verifier import verify_pipeline

os.makedirs("data/input", exist_ok=True)
os.makedirs("data/watermarked", exist_ok=True)

# Reproducible 256x256 image (16 patches — full payload covered twice)
np.random.seed(42)
img = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
input_path = "data/input/probe256_input.png"
cv2.imwrite(input_path, img)
print(f"Input image written: {input_path}  shape=(256,256)")

for mode in ["classical_spatial", "classical_frequency", "quantum_spatial_statevector"]:
    print(f"\n{'='*60}")
    print(f"MODE: {mode}")
    print(f"{'='*60}")

    wm_path = f"data/watermarked/probe256_{mode}.png"

    t0 = time.time()
    metadata = embed_pipeline(input_path, wm_path, embedding_mode=mode)
    embed_sec = time.time() - t0
    print(f"Embed time : {embed_sec:.2f}s")

    # Verify on the watermarked image — zero attacks
    t0 = time.time()
    results = verify_pipeline(wm_path, metadata, embedding_mode=mode)
    verify_sec = time.time() - t0
    print(f"Verify time: {verify_sec:.2f}s")

    print(f"\n--- Key results ---")
    print(f"  detection_rate : {results['detection_rate']}")
    print(f"  hash_match     : {results['hash_match']}")
    print(f"  tamper_pixels  : {results['tamper_pixels']}")
    print(f"  is_authentic   : {results['is_authentic']}")
    print(f"\n  expected_payload  : {results['expected_payload']}")
    print(f"  extracted_payload : {results['extracted_payload']}")

    # Bit-level diff
    exp = results['expected_payload']
    ext = results['extracted_payload']
    mismatches = np.where(exp != ext)[0]
    print(f"\n  Mismatch indices ({len(mismatches)} total): {mismatches[:20]}{'...' if len(mismatches)>20 else ''}")

    

print("\nProbe complete.")
