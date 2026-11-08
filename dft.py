import numpy as np

def _check_size(x):
    if x.ndim > 1:
        raise ValueError("The vector must be one dimensional")

    if x.size < 1:
        raise ValueError("The vectoe is empty") 

def dft(x):
    """Calculates the Discrete Fourier Transform using the definition"""

    x_array = np.array(x, dtype= complex, ndmax= 1)

    # Create the coefficients array
    dft_x = np.zeros_like(x_array)

    N = np.size(x)

    # Fixing k in exponential
    for k in range(N):
        coefficient = 0

        # Fixing x[n] and n in exponential
        for n in range(N):

            # Calculating DFT coefficient
            coefficient = coefficient + x_array[n] * np.exp(-2j * np.pi * k * n / N)
        
        # Introducing the coefficient value to an DFT array
        dft_x[k] = coefficient

    return dft_x

