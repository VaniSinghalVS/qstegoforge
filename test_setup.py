import qiskit
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

print("Qiskit:", qiskit.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("All basic imports working!")