import time

from pycsou.linop.conv import Convolve2D
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import SquaredL2Norm, L2Norm, L1Norm, NonNegativeOrthant
from pycsou.opt.proxalgs import APGD, PDS
from pycsou.linop.diff import Gradient

from scripts.functionals import DCT2, HuberNorm, OptiConvolve2D
from scipy.fftpack import dctn, idctn

def optimize(method, psf, data, n_iter, lambda_=0.1, delta=1):
    start_time = time.time()
    runner, post_process = get_runner(method, psf, data, n_iter, lambda_, delta)
    print(f"setup time : {time.time() - start_time} s")
    
    start_time = time.time()
    estimate, converged, diagnostics = runner.iterate()
    print(f"proc time : {time.time() - start_time} s")

    post_process(estimate)
    
    return estimate, converged, diagnostics

def get_runner(method, psf, data, n_iter, lambda_, delta):
    Hop = OptiConvolve2D(psf)
    Hop.compute_lipschitz_cst(tol=5e-2)
    
    #====================== LOSS & INIT =======================
    
    loss = (1 / 2) * SquaredL2Loss(dim=Hop.shape[0], data=data.flatten())
    F = loss * Hop
    G = None
    H = None
    D = None
    post_process = lambda estimate: None
    
    #====================== ALGO SETUP =======================
    
    if method == "ridge":
        F += lambda_ * SquaredL2Norm(Hop.shape[0])
    elif method == "lasso":
        G = lambda_ * L1Norm(Hop.shape[0])
    elif method == "glasso":
        dct2 = DCT2(data.shape)
        idct2 = dct2.get_adjointOp()
        idct2.compute_lipschitz_cst(tol=5e-1)
        F *= idct2
        G = lambda_ * L1Norm(Hop.shape[0])
        def post_process(estimate):
            estimate['iterand'] = idct2(estimate['iterand'])
    elif method == "nnls":
        G = lambda_ * NonNegativeOrthant(Hop.shape[0])
    elif method == "pls":
        D = Gradient(shape = data.shape)
        D.compute_lipschitz_cst(tol=5e-1)
        H = lambda_ * L1Norm(dim=D.shape[0])
        G = NonNegativeOrthant(dim=Hop.shape[0])
    elif method == "pls_huber":
        if delta is None:
            raise ValueError("Delta should be defined for huber")
        D = Gradient(shape = data.shape)
        D.compute_lipschitz_cst(tol=5e-1)
        F += lambda_ * HuberNorm(dim=D.shape[0], delta=delta) * D  # Differentiable function
        G = NonNegativeOrthant(dim=Hop.shape[0])
    else:
        raise AttributeError("Reconstruction algorithm not defined.")
    
    #====================== OPTIMIZATION =======================
    
    if method in ["pls"]:
        return PDS(dim=Hop.shape[0], 
                   F=F, G=G, H=H, K=D, 
                   verbose=None,
                   min_iter=1, 
                   max_iter=n_iter, 
                   accuracy_threshold=0.0001),\
                post_process
    else:
        return APGD(dim=Hop.shape[1], 
                    F=F, G=G, 
                    acceleration='CD', 
                    verbose=None,
                    min_iter=1, 
                    max_iter=n_iter, 
                    accuracy_threshold=0.0001),\
                post_process
    