import os
import sys
import json
import time
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.embedder import embed_pipeline
from src.verifier import verify_pipeline
from src.attacker import apply_attack

EMBEDDING_MODES = [
    "classical_spatial",
    "classical_frequency",
    "quantum_spatial_statevector",
]

ATTACKS = ["none", "jpeg", "noise", "rotate", "crop"]

def main():
    with open("datasets/dataset_manifest.json", "r") as f:
        manifest = json.load(f)
        
    results = []
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # We will process a subset if specified, otherwise all 50
    # Allow passing a limit via sys.argv for debugging
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(manifest)
    
    start_time = time.time()
    
    for i, item in enumerate(tqdm(manifest[:limit], desc="Images")):
        domain = item["dataset"]
        image_path = os.path.join("datasets", domain, item["filename"])
        image_name = item["filename"]
        
        for mode in EMBEDDING_MODES:
            watermarked_path = os.path.join(output_dir, f"wm_{mode}_{image_name}")
            
            # 1. Embed
            t0 = time.time()
            try:
                metadata = embed_pipeline(image_path, watermarked_path, embedding_mode=mode)
                embed_time = time.time() - t0
            except Exception as e:
                print(f"Error embedding {image_name} with {mode}: {e}")
                continue
                
            wm_img = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
            
            # 2. Attack & Verify
            for attack in ATTACKS:
                attacked_img = apply_attack(wm_img, attack)
                attacked_path = os.path.join(output_dir, f"attacked_{attack}_{mode}_{image_name}")
                cv2.imwrite(attacked_path, attacked_img)
                
                t1 = time.time()
                try:
                    vr = verify_pipeline(attacked_path, metadata, embedding_mode=mode)
                    verify_time = time.time() - t1
                    
                    results.append({
                        "image_name": image_name,
                        "domain": domain,
                        "embedding_mode": mode,
                        "extraction_mode": mode,  # they are symmetric
                        "attack_type": attack,
                        "detection_rate": vr["detection_rate"],
                        "hash_match": vr["hash_match"],
                        "tamper_pixels": vr["tamper_pixels"],
                        "is_authentic": vr["is_authentic"],
                        "embed_time_sec": embed_time,
                        "verify_time_sec": verify_time
                    })
                except Exception as e:
                    print(f"Error verifying {image_name} with {mode} under {attack}: {e}")
                    
        # Incremental save
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nEvaluation complete in {total_time/3600:.2f} hours.")
    print(f"Results saved to {output_dir}/results.json")

if __name__ == "__main__":
    main()
