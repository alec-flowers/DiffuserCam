import time

from pycsou.linop.conv import Convolve2D
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import SquaredL2Norm, L2Norm, L1Norm, NonNegativeOrthant
from pycsou.opt.proxalgs import APGD, PDS
from pycsou.linop.diff import Gradient

from scripts.functionals import DCT2, HuberNorm, OptiConvolve2D
from scipy.fftpack import dctn, idctn


def lasso(psf, data, n_iter):
    start_time = time.time()

    Hop = OptiConvolve2D(psf)
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
    a = Hop.shape
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
    Performs an image reconstruction with a generalized LASSO probleme with type2 DCT.
    '''

    start_time = time.time()

    dct2 = DCT2(data.shape)
    idct2 = dct2.get_adjointOp()

    Hop = Convolve2D(size=data.size, filter=psf, shape=data.shape, method='fft')
    Hop.compute_lipschitz_cst(tol=5e-1)

    l22_loss = (1 / 2) * SquaredL2Loss(dim=Hop.shape[0], data=data.flatten())
    idct2.compute_lipschitz_cst(tol=5e-1)
    F = l22_loss * Hop * idct2
    lambda_ = 1e-10
    G = lambda_ * L1Norm(dim=Hop.shape[0]) 

    apgd = APGD(dim=Hop.shape[1], F=F, G=G, acceleration='CD', verbose=None,
                min_iter=1, max_iter=n_iter, accuracy_threshold=0.0001)

    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    estimate, converged, diagnostics = apgd.iterate()
    estimate['iterand'] = idct2(estimate['iterand'])
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


def pls_huber(psf, data, n_iter):
    '''
    Performs an image reconstruction with a penalised least squares probleme
    TV + Non-negativity prior + differentiable HuberNorm
    '''
    start_time = time.time()
    Hop = Convolve2D(size=data.size, filter=psf, shape=data.shape, method='fft')
    Hop.compute_lipschitz_cst(tol=5e-1)
    D = Gradient(shape = data.shape)
    D.compute_lipschitz_cst(tol=5e-1)


    lambda_ = 0.01
    delta = 5
    l22_loss = (1 / 2) * SquaredL2Loss(dim=Hop.shape[0], data=data.flatten())
    F = l22_loss * Hop + lambda_ * HuberNorm(dim=D.shape[0], delta=delta) * D  # Differentiable function
    G = NonNegativeOrthant(dim=Hop.shape[0])

    apgd = APGD(dim=Hop.shape[1], F=F, G=G, acceleration='CD', verbose=None,
                min_iter=1, max_iter=n_iter, accuracy_threshold=0.0001)
    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    estimate, converged, diagnostics = apgd.iterate()
    print(f"proc time : {time.time() - start_time} s")

    return estimate, converged, diagnostics