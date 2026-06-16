"""
neqr.py — NEQR Quantum Image Encoding & Decoding
Converts classical images into NEQR quantum circuit representations
and back, enabling quantum gate operations on pixel values.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
import cv2
import config


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and resize image to standard size defined in config."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, config.IMAGE_SIZE)
    return img


def image_to_neqr(img: np.ndarray) -> QuantumCircuit:
    """
    Encode a grayscale image into an NEQR quantum circuit.

    NEQR uses two quantum registers:
      - color register  : 8 qubits → stores pixel intensity (0–255)
      - position register: 12 qubits → stores (row, col) as binary (64x64)

    For each pixel, an X gate flips the qubit if the corresponding
    bit in the pixel value or position is 1.
    """
    height, width = img.shape
    n_position = config.NEQR_POSITION_QUBITS   # 12 qubits for position
    n_color    = config.NEQR_COLOR_QUBITS       # 8 qubits for color

    color_reg    = QuantumRegister(n_color,    name='color')
    position_reg = QuantumRegister(n_position, name='position')
    qc = QuantumCircuit(color_reg, position_reg)

    # Put position register in superposition (represents all pixels)
    qc.h(position_reg)

    for row in range(height):
        for col in range(width):
            pixel_value = int(img[row, col])
            position    = row * width + col  # flatten 2D →  1D index

            # Encode position as binary on position register
            pos_bits = format(position, f'0{n_position}b')
            for i, bit in enumerate(pos_bits):
                if bit == '1':
                    qc.x(position_reg[i])

            # Encode pixel color value as binary on color register
            color_bits = format(pixel_value, f'0{n_color}b')
            for i, bit in enumerate(color_bits):
                if bit == '1':
                    qc.x(color_reg[i])

    return qc


def neqr_to_image(qc: QuantumCircuit, height: int, width: int) -> np.ndarray:
    """
    Decode an NEQR quantum circuit back to a classical image.
    Reads the color register state for each pixel position.
    This is a classical simulation of the inverse NEQR operation.
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    # Add measurement to all qubits
    qc_measured = qc.copy()
    qc_measured.measure_all()

    simulator = AerSimulator()
    compiled  = transpile(qc_measured, simulator)
    job       = simulator.run(compiled, shots=1024)
    counts    = job.result().get_counts()

    # Reconstruct image from most frequent measurement outcomes
    img = np.zeros((height, width), dtype=np.uint8)
    n_color    = config.NEQR_COLOR_QUBITS
    n_position = config.NEQR_POSITION_QUBITS

    for bitstring, count in counts.items():
        # Qiskit returns bitstring as: color | position (right to left)
        bits         = bitstring.replace(" ", "")
        color_bits   = bits[:n_color]
        position_bits= bits[n_color:n_color + n_position]

        pixel_value = int(color_bits, 2)
        position    = int(position_bits, 2)

        row = position // width
        col = position % width

        if row < height and col < width:
            img[row, col] = pixel_value

    return img


if __name__ == "__main__":
    import os

    # Quick smoke test with a synthetic image
    test_img = np.random.randint(0, 256, config.IMAGE_SIZE, dtype=np.uint8)
    print("Original image shape:", test_img.shape)
    print("Sample pixel [0,0]:", test_img[0, 0])

    qc = image_to_neqr(test_img)
    print(f"NEQR circuit created: {qc.num_qubits} qubits, {qc.depth()} depth")
    print("NEQR encoding: OK")