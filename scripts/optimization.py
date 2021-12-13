import time

from pycsou.linop.conv import Convolve2D
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import SquaredL2Norm, L2Norm, L1Norm, NonNegativeOrthant
from pycsou.opt.proxalgs import APGD


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