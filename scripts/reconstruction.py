"""
This script will load the PSF data and raw measurement for the reconstruction
that can implement afterwards.
This scripts implements the Tikhonov Regularization and the reconstruction uses the
APGD algorithm.

```bash
python scripts/reconstruction.py --psf_fp data/psf/diffcam_rgb.png \
--data_fp data/raw_data/thumbs_up_rgb.png\
--gray
```

"""

import os
import time
import pathlib as plib
import click
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from diffcam.io import load_data
from diffcam.util import DATAPATH, RECONSTRUCTIONPATH

from scripts.optimization import lasso, ridge, nnls

def reconstruction(
    psf_fp,
    data_fp,
    algo,
    n_iter,
    downsample,
    disp,
    flip,
    gray,
    bayer,
    bg,
    rg,
    gamma,
    save,
    plot,
    single_psf,
):
    psf, data = load_data(
        psf_fp=psf_fp,
        data_fp=data_fp,
        downsample=downsample,
        bayer=bayer,
        blue_gain=bg,
        red_gain=rg,
        plot=plot,
        flip=flip,
        gamma=gamma,
        gray=gray,
        single_psf=single_psf,
    )

    if disp < 0:
        disp = None
    if save:
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        save = RECONSTRUCTIONPATH / str(algo + '_' + str(n_iter) + timestamp + '.png')
    
    if algo == "ridge":
        estimate, converged, diagnostics = ridge(psf, data, n_iter)
    elif algo == "lasso":
        estimate, converged, diagnostics = lasso(psf, data, n_iter)
    elif algo == "nnls":
        estimate, converged, diagnostics = nnls(psf, data, n_iter)

    estimate = estimate['iterand'].reshape(data.shape)

    plt.figure()
    plt.imshow(estimate, cmap='gray')
    if save:
        plt.savefig(save, format='png')
        print(f"Files saved to : {save}")
    if plot:
        plt.show()


if __name__ == "__main__":

    psf_fp = str(DATAPATH) + '/psf/diffcam_rgb.png'
    data_fp = str(DATAPATH) + '/raw_data/thumbs_up_rgb.png'
    algo = 'nnls'
    n_iter = 50
    gray = True
    downsample = 4
    disp = 50
    flip = False
    bayer = False
    bg = None
    rg = None
    gamma = None
    save = True
    plot = True
    single_psf = False

    reconstruction(
        psf_fp,
        data_fp,
        algo,
        n_iter,
        downsample,
        disp,
        flip,
        gray,
        bayer,
        bg,
        rg,
        gamma,
        save,
        plot,
        single_psf)
