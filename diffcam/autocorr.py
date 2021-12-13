import numpy as np
from time import process_time

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

    t1_start = process_time();
    print('Image dimensions', vals.shape)
    height = vals.shape[0];
    width = vals.shape[1];
    pad_width = int(1*vals.shape[1]);
    vals = np.pad(vals,((height, height), (width,width)),pad_mode)
    print('pad dimensions', vals.shape)
    #vals = vals - np.mean(vals)
    valsFT = np.fft.fft2(vals)
    print('FT: done')
    valsAC = np.fft.fftshift(np.fft.ifft2(valsFT * np.conjugate(valsFT))).real
    print('AC : done')
    vals = valsAC
    t1_stop = process_time()
    vals = vals[height: 2*height, width:2*width]
    print('Autocorrelation computing time : ', t1_stop-t1_start)
    print('Matrix size = ', vals.shape)
    print('\n')
    return vals
