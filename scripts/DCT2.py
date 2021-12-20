from pycsou.core import LinearOperator
from scipy.fftpack import dctn, idctn, dct
import numpy as np




class DCT2(LinearOperator):
    def __init__(self, shape: tuple ):
        super(DCT2, self).__init__( shape=shape)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return dctn(x.reshape(self.shape), type = 2 , norm = 'ortho').flatten()

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return idctn(yreshape(self.shape), type = 2 , norm = 'ortho').flatten()

class IDCT2(LinearOperator):
    def __init__(self, shape: tuple ):
        self.origshape = shape
        self.size = ( shape[0]*shape[1], shape[0]*shape[1]);
        super(IDCT2, self).__init__(shape=self.size)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return idctn(x.reshape(self.origshape), type = 2, norm = 'ortho').flatten()

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return dctn(y.reshape(self.origshape), type = 2, norm = 'ortho').flatten()

##if _name__ == "__main__": 
#    x = np.array([[1,2,3],[4,5,6]]);
#    print(x)
#    op = DCT2(x.shape);
#    y = op * x;
#    print(y)
#    print(dctn(x, type = 2, norm = 'ortho'));
#    print(dct(x,type = 2, norm = 'ortho'))


