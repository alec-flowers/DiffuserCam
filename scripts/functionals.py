from numbers import Number
from typing import Union
from scipy import fft

from pycsou.core import LinearOperator
from pycsou.core.functional import DifferentiableFunctional
import numpy as np


class DCT2(LinearOperator):

    def __init__(self, shape: tuple ):
        """DCT of type 2 pycsou linear operator.

        Args:
            shape (tuple): The shape of the input of DCT.
        """
        self.origshape = shape
        self.size = (np.prod(shape), np.prod(shape))
        super(DCT2, self).__init__(shape=self.size)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return fft.dctn(x.reshape(self.origshape), type = 2 , norm = 'ortho', workers=-1).flatten()

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return fft.idctn(y.reshape(self.origshape), type = 2 , norm = 'ortho', workers=-1).flatten()

class OptiConvolve2D(LinearOperator):
    def __init__(self, psf:np.ndarray):
        """Optimized Convolve 2D pycsou linear operator. Assumes psf and inputs have 
            all the same size. Manage gray and color images.

        Args:
            psf (np.ndarray): The PSF used as filter for linear inverse problem optimization.
                Takes shape such as (height, width, channels).
                Depending on the situation, it can be good to pass channels = 1 even 
                for gray images and make sure shape has 3 elements.
        """
        
        super(OptiConvolve2D, self).__init__(shape=(psf.size, psf.size), dtype=psf.dtype, lipschitz_cst=np.infty)
        self.origshape = np.array(psf.shape)
        
        offset = np.zeros_like(psf.shape)
        offset[0:2] = self.origshape[0:2]//2
        self.fshape2d = np.array(psf.shape)[0:2] + offset[0:2]*2
        
        offset_conv = offset.copy()
        offset_conv[0:2] -= (self.origshape[0:2] + 1)%2
        offset_corr = offset.copy()
        self.fslice2d_conv = tuple([slice(off, off + sz) for off, sz in zip(offset_conv, psf.shape)])
        self.fslice2d_corr = tuple([slice(off, off + sz) for off, sz in zip(offset_corr, psf.shape)])
        
        self.fft_psf = fft.rfftn(psf, self.fshape2d, axes=(0,1), workers=-1)
        psf_conj = psf[::-1,::-1].conj()
        self.fft_psf_conj = fft.rfftn(psf_conj, self.fshape2d, axes=(0,1), workers=-1)
        
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.reshape(x, self.origshape)
        x = fft.rfftn(x, self.fshape2d, axes=(0,1), workers=-1)
        y = fft.irfftn(x * self.fft_psf, self.fshape2d, axes=(0,1), workers=-1)
        return y[self.fslice2d_conv].flatten()
    
    # Function almost exactly repeated to avoid branching
    def adjoint(self, y: np.ndarray) -> np.ndarray:
        y = np.reshape(y, self.origshape)
        y = fft.rfftn(y, self.fshape2d, axes=(0,1), workers=-1)
        x = fft.irfftn(y * self.fft_psf_conj, self.fshape2d, axes=(0,1), workers=-1)
        return x[self.fslice2d_corr].flatten()


class HuberNorm(DifferentiableFunctional):
    
    def __init__(self, dim: int, delta: float):
        """Functional defining the Huber norm.

        Args:
            dim (int): Dimension of the functional’s domain.
            delta (float): Delta parameter of Huber norm, defines the absolute threshold 
                for the transition between L1 and L2 norm blending.
        """
        self.delta = delta
        super().__init__(dim=dim, data=None, is_linear=False, lipschitz_cst=np.infty, diff_lipschitz_cst=1)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        abs_x = np.abs(x)
        small = abs_x[abs_x <= self.delta]
        large = abs_x[abs_x > self.delta]
        small_sum = np.sum(0.5*small**2)
        large_sum = np.sum(self.delta * (large - self.delta/2))
        return small_sum + large_sum

    def jacobianT(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        greater_ind = np.nonzero(x > self.delta)
        less_ind = np.nonzero(x < -self.delta)
        x[greater_ind] = self.delta
        x[less_ind] = -self.delta
        return x

if __name__ == "__main__":
    h = HuberNorm(5, 2.0)
    a = np.array([2,4,1,1,.5,3, -3, -2,-1,-10])
    val = h(a)
    j = h.jacobianT(a)
    b



