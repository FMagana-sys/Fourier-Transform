import numpy as np
import cmath 

# TODO: DFT in 1D with two cycles using the definition 

# Vector
x = [1, complex(2, -1), complex(-1), complex(-1, 2) ]

dft_x = np.zeros_like(x)

k = 0
n = 0
N = 0

e = np.exp(-2j * np.pi * k * n / N)