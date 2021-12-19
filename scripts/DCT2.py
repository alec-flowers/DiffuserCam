from pycsou.core import LinearOperator
from scipy.fftpack import dctn, idctn
import numpy as np

# scipy.fft.dctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None)
# scipy.fft.idctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None)


class DCT2(LinearOperator):
    #def __init__(self, size : int, type : int, shape : tuple):
    #    self.type = type
    #    self.shape = shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return dctn(x.flatten(), type = 2 , norm = 'ortho').reshape(x.shape)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return idctn(y.flatten(), type = 2 , norm = 'ortho').reshape(y.shape)

class IDCT2(LinearOperator):
    #def __init__(self, size : int, type : int, shape : tuple):
    #    self.type = type
    #    self.shape = shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return idctn(x.flatten(), type = 2 , norm = 'ortho').reshape(x.shape)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return dctn(y.flatten(), type = 2 , norm = 'ortho').reshape(y.shape)


