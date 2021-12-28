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
from diffcam.plot import plot_image

from scripts.optimization import lasso, ridge, nnls, glasso, pls, pls_huber

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

    if gray:
        psf = psf[:, :, np.newaxis]
        data = data[:, :, np.newaxis]

    estimates = []
    for i in range(psf.shape[2]):
        psf_c = psf[:, :, i]
        data_c = data[:, :, i]
        if algo == "ridge":
            estimate, converged, diagnostics = ridge(psf_c, data_c, n_iter)
            estimates.append(estimate['iterand'].reshape(data.shape[0:2]))
        elif algo == "lasso":
            estimate, converged, diagnostics = lasso(psf_c, data_c, n_iter)
            estimates.append(estimate['iterand'].reshape(data.shape[0:2]))
        elif algo == "nnls":
            estimate, converged, diagnostics = nnls(psf_c, data_c, n_iter)
            estimates.append(estimate['iterand'].reshape(data.shape[0:2]))
        elif algo == "glasso":
            estimate, converged, diagnostics = glasso(psf_c, data_c, n_iter)
            estimates.append(estimate['iterand'].reshape(data.shape[0:2]))
        elif algo == "pls":
            estimate, converged, diagnostics = pls(psf_c, data_c, n_iter)
            estimates.append(estimate['primal_variable'].reshape(data.shape[0:2]))
        elif algo == "pls_huber":
            estimate, converged, diagnostics = pls_huber(psf_c, data_c, n_iter)
            estimates.append(estimate['iterand'].reshape(data.shape[0:2]))

    plt.figure()
    if gray:
        reconstruction = np.array(estimates).squeeze()
        plt.imshow(reconstruction, cmap='gray')
    else:
        reconstruction = np.array(estimates).swapaxes(0,2).swapaxes(0,1)
        #plt.imshow(reconstruction)
        ax = plot_image(reconstruction)

    if save:
        plt.savefig(save, format='png')
        print(f"Files saved to : {save}")
    if plot:
        plt.show()


if __name__ == "__main__":

    psf_fp = str(DATAPATH) + '/psf/psf_rgb_ours.png'
    data_fp = str(DATAPATH) + '/our_images/diffuser/img8_rgb.png'
    algo = 'lasso'
    n_iter = 100
    gray = False
    downsample = 4
    disp = 50
    flip = False
    bayer = False
    bg = None
    rg = None
    gamma = None
    save = False
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
