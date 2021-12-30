import time

from pycsou.linop.conv import Convolve2D
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import SquaredL2Norm, L2Norm, L1Norm, NonNegativeOrthant
from pycsou.opt.proxalgs import APGD, PDS
from pycsou.linop.diff import Gradient

from scripts.functionals import DCT2, HuberNorm, OptiConvolve2D

def optimize(method, psf, data, n_iters, dtype, lambda_=0.1, delta=1):
    """Setup and run the required optimization.

    Args:
        method (str): can be "ridge", "lasso", "glasso", "nnls", "pls", "pls_huber"
        psf (np.ndarray): the point spread function.
        data (np.ndarray): the measurement from which the estimate will be computed.
        n_iters (Union(int, list(int))): the maximum number of iteration to run the optimization.
            Alternatively, the list of successive iterations to run and to checkpoint.
        dtype (np.dtype): data type of structure.
        lambda_ (float, optional): Penalty weight in optimization. 
            Defaults to 0.1.
        delta (int, optional): Delta parameter of Huber norm, defines the absolute threshold 
            for the transition between L1 and L2 norm blending. 
            Defaults to 1.

    Returns:
        tuple(list(np.ndarray),list(float)): list of estimates and list of elapsed times. 
            They will be lists of 1 element even though n_iters is a single value. Otherwise
            they will be lists with the same length as n_iters.
    """
    estimates = []
    elapsed_times = []
    
    start_time = time.process_time()
    runner, post_process = get_runner(method, psf, data, sum(n_iters), lambda_, delta)
    print(f"setup time : {time.process_time() - start_time} s")
    
    proc_start_time = time.process_time()
    for n in n_iters:
        for _ in range(n):
            runner.iterand = runner.update_iterand()
            runner.iter += 1
            
        elapsed_time = time.process_time() - start_time
        print(f"proc time... : {time.process_time() - proc_start_time} s")
        
        elapsed_times.append(elapsed_time)
        estimates.append(runner.postprocess_iterand())
    
    # Postprocess
    for i in range(len(estimates)):
        post_process(estimates[i])
        if method == 'pls':
            estimates[i] = estimates[i]['primal_variable']
        else:
            estimates[i] = estimates[i]['iterand']
        estimates[i][estimates[i] < 0] = 0.0
        estimates[i] = estimates[i].reshape(data.shape).astype(dtype).squeeze()
    
    return estimates, elapsed_times

def get_runner(method, psf, data, n_iter, lambda_, delta):
    """Setup the runner and post process function and return it.

    Args:
        method (str): can be "ridge", "lasso", "glasso", "nnls", "pls", "pls_huber"
        delta ([type]): [description]
        psf (np.ndarray): the point spread function.
        data (np.ndarray): the measurement from which the estimate will be computed.
        n_iter (int): the maximum number of iteration to run the optimization.
        lambda_ (float, optional): Penalty weight in optimization. 
            Defaults to 0.1.
        delta (int, optional): Delta parameter of Huber norm, defines the absolute threshold 
            for the transition between L1 and L2 norm blending. 
            Defaults to 1.
        
    Raises:
        ValueError: method is not correct.
        ValueError: huber needs non-None delta.

    Returns:
        tuple(runner, function): Returns a pycsou runner full setup + a post process function 
            in case data needs it after optimization.
    """
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
        raise ValueError("Reconstruction algorithm not defined.")
    
    #====================== OPTIMIZATION METHOD =======================
    
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
    