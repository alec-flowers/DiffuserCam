import time

from pycsou.linop.conv import Convolve2D
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import SquaredL2Norm, L2Norm, L1Norm, NonNegativeOrthant
from pycsou.opt.proxalgs import APGD, PDS
from pycsou.linop.diff import Gradient

from scripts.DCT2 import DCT2, IDCT2
from scipy.fftpack import dctn

def lasso(psf, data, n_iter):
    start_time = time.time()

    Hop = Convolve2D(size=data.size, filter=psf, shape=data.shape, method='fft')
    Hop.compute_lipschitz_cst(tol=5e-1)

    l22_loss = (1 / 2) * SquaredL2Loss(dim=Hop.shape[0], data=data.flatten())
    F = l22_loss * Hop
    lambda_ = 0.1
    G = lambda_ * L1Norm(dim=Hop.shape[0])

    apgd = APGD(dim=Hop.shape[1], F=F, G=G, acceleration='CD', verbose=None,
                min_iter=1, max_iter=n_iter, accuracy_threshold=0.0001)

    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    estimate, converged, diagnostics = apgd.iterate()
    print(f"proc time : {time.time() - start_time} s")

    return estimate, converged, diagnostics


def ridge(psf, data, n_iter):
    start_time = time.time()

    Hop = Convolve2D(size=data.size, filter=psf, shape=data.shape, method='fft')
    Hop.compute_lipschitz_cst(tol=5e-1)

    lambda_ = 0.1
    l22_loss = (1 / 2) * SquaredL2Loss(dim=Hop.shape[0], data=data.flatten())
    F = l22_loss * Hop + lambda_ * SquaredL2Norm(dim=Hop.shape[0])

    apgd = APGD(dim=Hop.shape[1], F=F, acceleration='CD', verbose=None,
                min_iter=1, max_iter=n_iter, accuracy_threshold=0.0001)

    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    estimate, converged, diagnostics = apgd.iterate()
    print(f"proc time : {time.time() - start_time} s")

    return estimate, converged, diagnostics


def nnls(psf, data, n_iter):
    start_time = time.time()
    Hop = Convolve2D(size=data.size, filter=psf, shape=data.shape, method='fft')  # Regularisation operator
    Hop.compute_lipschitz_cst(tol=5e-1)

    l22_loss = (1 / 2) * SquaredL2Loss(dim=Hop.shape[0], data=data.flatten())
    F = l22_loss * Hop  # Differentiable function
    G = NonNegativeOrthant(dim=Hop.shape[0])

    apgd = APGD(dim=Hop.shape[1], F=F, G=G, acceleration='CD', verbose=None,
                min_iter=1, max_iter=n_iter, accuracy_threshold=0.0001)  # Initialize APGD
    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    estimate, converged, diagnostics = apgd.iterate()  # Run APGD
    print(f"proc time : {time.time() - start_time} s")

    return estimate, converged, diagnostics

def glasso(psf, data, n_iter):
    '''
    Performs an image reconstruction with a generalized LASSO probleme with type2 DCT : DOES NOT WORK YET
    '''
    start_time = time.time()

    Hop = Convolve2D(size=data.size, filter=psf, shape=data.shape, method='fft')
    Hop.compute_lipschitz_cst(tol=5e-1)

    l22_loss = (1 / 2) * SquaredL2Loss(dim=Hop.shape[0], data=data.flatten())
    IDCT =  IDCT2(shape=Hop.shape) #size = data.size, type=2,  shape=Hop.shape
    IDCT.compute_lipschitz_cst(tol=5e-1)
    F = l22_loss * Hop * IDCT
    lambda_ = 0.1
    #D = DCT2(size = data.size,type = 2)
    G = lambda_ * L1Norm(dim=Hop.shape[0]) 

    apgd = APGD(dim=Hop.shape[1], F=F, G=G, acceleration='CD', verbose=None,
                min_iter=1, max_iter=n_iter, accuracy_threshold=0.0001)

    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    estimate, converged, diagnostics = apgd.iterate()
    estimate['iterand'] = dctn(estimate['iterand'], type = 2, norm = 'ortho')
    print(f"proc time : {time.time() - start_time} s")

    return estimate, converged, diagnostics

def pls(psf, data, n_iter):
    '''
    Performs an image reconstruction with a penalised least squares probleme
    TV + Non-negativity prior
    '''
    start_time = time.time()
    Hop = Convolve2D(size=data.size, filter=psf, shape=data.shape, method='fft') 
    Hop.compute_lipschitz_cst(tol=5e-1)
    D = Gradient(shape = data.shape)
    D.compute_lipschitz_cst(tol=5e-1)

    lambda_ = 0.1
    l22_loss = (1 / 2) * SquaredL2Loss(dim=Hop.shape[0], data=data.flatten())
    F = l22_loss * Hop  # Differentiable function
    G = NonNegativeOrthant(dim=Hop.shape[0])
    H = lambda_ * L1Norm(dim=D.shape[0])

    pds = PDS(dim=Hop.shape[0], F=F, G=G, H=H, K=D, verbose=None,
                min_iter=1, max_iter=n_iter, accuracy_threshold=0.0001)  
    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    estimate, converged, diagnostics = pds.iterate() 
    print(f"proc time : {time.time() - start_time} s")

    return estimate, converged, diagnostics
