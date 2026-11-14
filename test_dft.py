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

def gen_random(n):
    random_seed = 20
    real = np.random.default_rng(seed = random_seed)
    imaginary = np.random.default_rng(seed = random_seed)
    random_array = real.standard_normal(size = n) + imaginary.standard_normal(size = n) * 1j
    
    return random_array

@pytest.mark.parametrize("array, fft", [(a, np.array([1, 1, 1, 1], dtype= complex)), 
                                           (b, np.array([4, 0, 0, 0], dtype= complex)),
                                           (c, np.array([1, -1j, -1, 1j], dtype= complex))],                 
                                        ids = ["DFT of a", "DFT of b", "DFT of c"]
)

def test_constant_dft_matches_calculated(array, fft):
    np.testing.assert_allclose(dft.dft(array), fft, atol = 1e-12, strict = True)

@pytest.mark.parametrize("n", [10, 20, 30, 40, 50, 70, 80, 90, 100
])
    

def test_random_dft_matches_fft(n):
    array = gen_random(n)
    np.testing.assert_allclose(dft.dft(array), np.fft.fft(array), atol = 1e-12, strict = True)

