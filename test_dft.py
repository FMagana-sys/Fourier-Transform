import pytest
import numpy as np
import dft


# Compare DFT implemented by definition against FFT from numpy

# a - DFT(a) = [1, 1, 1, 1]
a = np.array([1, 0, 0, 0], dtype= complex)
# b - DFT(b) = [4, 0, 0, 0]
b = np.array([1, 1, 1, 1], dtype= complex)
# c - DFT(c) = [1, -j, -1, j]
c = np.array([0, 1, 0, 0,], dtype= complex)
# random 
semilla = 20
real = np.random.default_rng(seed = semilla)
imaginary = np.random.default_rng(seed = semilla)
random = real.standard_normal(size= 4) + imaginary.standard_normal(size=4) * 1j

@pytest.mark.parametrize("array, fft", [(a, np.fft.fft(a)), 
                                           (b, np.fft.fft(b)),
                                           (c, np.fft.fft(c)), 
                                           (random, np.fft.fft(random))],                 
                                        ids = ["DFT of a", "DFT of b", "DFT of c", "DFT of random array"]
)

def test_dft_matches_fft(array, fft):
    np.testing.assert_allclose(dft.dft(array), fft, atol = 1e-12, strict = True)

# Multiple N tests
N = np.arange(10, 100, 10)


