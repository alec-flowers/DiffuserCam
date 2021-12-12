import numpy as np
from scipy.signal import convolve2d

def autocorr2d(vals, pad_mode="reflect"):
    """
    Compute 2-D autocorrelation of image via the FFT.

    Parameters
    ----------
    vals : py:class:`~numpy.ndarray`
        2-D image.
    pad_mode : str
        Desired padding. See NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Return
    ------
    autocorr : py:class:`~numpy.ndarray`
    """

    n = vals.shape[0]
    m = vals.shape[1]
    vals_pad = np.pad(vals, [(0,n-1),(0,m-1)], mode=pad_mode)
    vals_fft = np.fft.fft2(vals_pad)
    return np.real(np.fft.fftshift(np.fft.ifft2(vals_fft*np.conjugate(vals_fft))))[:, int(m/2):-int(m/2)]

