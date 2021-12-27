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
    height, width = vals.shape[:2];
    half_height, half_width = height//2, width//2
    pad_width = ((half_height, half_height), (half_width, half_width))

    vals = np.pad(vals, pad_width, pad_mode)
    print('pad dimensions', vals.shape)
    
    valsFT = np.fft.fft2(vals)
    print('FT: done')
    
    vals = np.fft.fftshift(np.fft.ifft2(valsFT * np.conjugate(valsFT))).real
    print('AC : done')
    t1_stop = process_time()
    
    vals = vals[half_height:3*half_height, half_width:3*half_width]
    print('Autocorrelation computing time : ', t1_stop-t1_start)
    print('Matrix size = ', vals.shape)
    print('\n')
    return vals
