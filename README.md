# QStegoForge

**QStegoForge** is a quantum-safe steganographic watermarking system designed for **deepfake and image-tampering detection**. It combines quantum image encoding, post-quantum cryptography, cryptographic hashing, and classical watermark embedding techniques.

> **Academic Research Project**

---

## 🔬 Overview

QStegoForge embeds **cryptographically secure and tamper-evident watermarks** into images through a pipeline combining quantum and classical techniques.

The system integrates:

* **NEQR (Novel Enhanced Quantum Representation)** — quantum image encoding
* **CRYSTALS-Kyber** — post-quantum key encapsulation
* **SHA-3** — cryptographic hashing
* **Position-conditioned MCX-gate LSB embedding** — watermark bit insertion

The primary goal is to investigate a watermarking approach that provides security against both **classical and quantum adversaries**, while supporting forensic detection of deepfakes and image tampering.

---

## 🏗️ System Pipeline

The QStegoForge pipeline consists of the following major stages:

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
NEQR / Classical Image Representation
     │
     ▼
SHA-3 Hash Generation
     │
     ▼
CRYSTALS-Kyber Key Encapsulation
     │
     ▼
Watermark Generation
     │
     ▼
Position-Conditioned MCX / LSB Embedding
     │
     ▼
Watermarked Image
     │
     ▼
Attack / Manipulation
     │
     ▼
Watermark Detection
     │
     ▼
PSNR / SSIM / Detection Rate
```

---

## 🧩 Embedding Modes

QStegoForge supports three evaluated embedding modes:

| Mode                          | Description                                               | Depth |
| ----------------------------- | --------------------------------------------------------- | ----: |
| `classical_spatial`           | LSB embedding in the spatial domain                       |     2 |
| `classical_frequency`         | Watermark embedding in the frequency domain               |     — |
| `quantum_spatial_statevector` | NEQR-based quantum embedding using statevector simulation |     1 |

### Quantum Simulation Note

Shots-based quantum modes were evaluated and subsequently dropped.

At **2048 shots**, the coverage of the full **4096-state space** of a 64×64 NEQR patch is insufficient. This results in a predictable zero-pixel default behavior, producing approximately **60% zero-pixel coverage**, consistent with coupon-collector predictions.

Therefore, **statevector simulation** is used for the quantum evaluation.

---

## 🧪 Evaluation Design

The system was evaluated across multiple image domains, attack scenarios, and performance metrics.

### Image Domains

| Dataset       | Content        |
| ------------- | -------------- |
| **CelebA-HQ** | Human faces    |
| **COCO**      | Natural scenes |
| **IAM**       | Documents      |

### Attack Types

* None / No attack
* Cropping
* JPEG compression
* Noise
* Rotation

### Evaluation Metrics

**Image Quality**

* PSNR
* SSIM

**Watermark Robustness**

* Detection Rate

**Performance**

* Embedding time

---

## 📊 Verified Results

### Image Quality — No Attack

| Embedding Mode                | PSNR (dB) |   SSIM |
| ----------------------------- | --------: | -----: |
| `classical_frequency`         |     47.87 | 0.9943 |
| `classical_spatial`           |     69.67 | 0.9999 |
| `quantum_spatial_statevector` |     78.30 | 1.0000 |

The quantum statevector mode achieved the highest measured PSNR and SSIM among the evaluated modes.

---

### 🛡️ Detection Robustness

* **No-attack detection:** `1.0` for both `classical_spatial` and `quantum_spatial_statevector`.
* **Crop attack survival:** approximately `0.90–0.94` across the evaluated modes.
* **JPEG compression, noise, and rotation without realignment:** detection decreases to approximately `0.50`, which is near-random detection performance.

The behavior under destructive attacks is expected for the evaluated forensic configuration: the watermark is designed to become detectable as compromised when image content is substantially altered rather than being artificially forced to survive every transformation.

> **Rotation note:** Results reported **with realignment** use the known-angle inverse rotation and therefore measure interpolation-artifact robustness rather than general rotation robustness. Results reported **without realignment** represent the actual geometric-desynchronization behavior.

---

## ⚡ Performance

| Mode                          | Mean Embedding Time |
| ----------------------------- | ------------------: |
| `classical_spatial`           |        ~20 ms/image |
| `quantum_spatial_statevector` |        ~263 s/image |

The quantum implementation is currently a **proof-of-concept simulation**. The measured runtime is therefore primarily representative of the current simulation implementation and should not be interpreted as a fundamental limitation of future quantum or optimized implementations.

---

## 📁 Repository Structure

```text
QStegoForge/
│
├── results/
│   ├── results.json
│   │   └── 750 entries — verified detection metrics
│   │
│   └── psnr_ssim.json
│       └── 150 entries — verified image-quality metrics
│
├── evaluate.py
│   └── Detection evaluation pipeline
│
├── compute_psnr_ssim.py
│   └── PSNR/SSIM evaluation pipeline
│
└── ...
```

> ⚠️ **Important:** `results/results.json` and `results/psnr_ssim.json` are **locked ground-truth files** containing the verified evaluation results. Do not overwrite these files.

---

## ⚙️ Requirements

QStegoForge requires:

* **Python 3.x**
* **Qiskit** — quantum circuit construction and simulation
* **CRYSTALS-Kyber** — post-quantum key encapsulation
* **scikit-image** — PSNR/SSIM computation
* **OpenCV** — image processing

### Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd QStegoForge
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

> If a `requirements.txt` file is not currently included, add one containing the exact package versions used for the verified evaluation environment.

---

## 🚀 Usage

### Run Detection Evaluation

```bash
python evaluate.py
```

### Compute Image Quality Metrics

```bash
python compute_psnr_ssim.py
```

The evaluation scripts generate metrics that can be compared against the verified results stored in:

```text
results/results.json
results/psnr_ssim.json
```

---

## 🔐 Security Components

QStegoForge combines multiple security mechanisms:

| Component          | Purpose                                          |
| ------------------ | ------------------------------------------------ |
| **NEQR**           | Quantum representation of image information      |
| **CRYSTALS-Kyber** | Post-quantum key encapsulation                   |
| **SHA-3**          | Cryptographic hashing and integrity support      |
| **MCX Gates**      | Position-conditioned quantum embedding mechanism |
| **LSB Embedding**  | Classical spatial-domain watermark insertion     |

The combination is intended to provide a research framework for investigating **quantum-safe image watermarking and forensic tamper detection**.

---

## 📈 Research Status

**Current status: Evaluation pipeline completed and verified.**

The system has been evaluated across:

* Multiple embedding modes
* Multiple image domains
* Multiple attack types
* Image-quality metrics
* Watermark detection metrics
* Quantum simulation performance

The accompanying research paper is currently **in progress**. All reported figures and numerical results are traced directly to the verified JSON ground-truth files included in the repository.

---

## 📚 Citation

Citation details will be added upon publication.

```text
QStegoForge — A Quantum-Safe Steganographic Watermarking System
for Deepfake and Image-Tampering Detection
```

---

## 👩‍💻 Project

**QStegoForge**
Quantum-safe steganographic watermarking for deepfake and image-tampering detection.

Developed as an **academic research project**.
