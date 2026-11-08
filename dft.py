import numpy as np

def dft(x):
    """Calculates the Discrete Fourier Transform using the definition"""

    # Create the coefficients array
    dft_x = np.zeros_like(x, dtype= complex)

    N = np.size(x)

    # Fixing k in exponential
    for k in range(N):
        coefficient = 0

        # Fixing x[n] and n in exponential
        for n in range(N):

            # Calculating DFT coefficient
            coefficient = coefficient + x[n] * np.exp(-2j * np.pi * k * n / N)
        
        # Introducing the coefficient value to an DFT array
        dft_x[k] = coefficient

    return dft_x

