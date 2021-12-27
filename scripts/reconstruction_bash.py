"""
This script will load the PSF data and raw measurement for the reconstruction
that can implement afterwards.
This scripts implements the Tikhonov Regularization and the reconstruction uses the
APGD algorithm.

```bash
python scripts/reconstruction_bash.py --psf_fp data/psf/diffcam_rgb.png \
--data_fp data/our_images/img1_rgb.png \
--algo ridge --gray --n_iter 500 --plot
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

from scripts.optimization import lasso, ridge, nnls, glasso, pls, pls_huber

@click.command()
@click.option(
    "--psf_fp",
    type=str,
    help="File name for recorded PSF.",
)
@click.option(
    "--data_fp",
    type=str,
    help="File name for raw measurement data.",
)
@click.option("--algo",
              type=str,
              help="Name of reconstruction algorithm.",
)
@click.option(
    "--n_iter",
    type=int,
    default=500,
    help="Number of iterations.",
)
@click.option(
    "--downsample",
    type=float,
    default=4,
    help="Downsampling factor.",
)
@click.option(
    "--disp",
    default=50,
    type=int,
    help="How many iterations to wait for intermediate plot/results. Set to negative value for no intermediate plots.",
)
@click.option(
    "--flip",
    is_flag=True,
    help="Whether to flip image.",
)
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save intermediate and final reconstructions.",
)
@click.option(
    "--gray",
    is_flag=True,
    help="Whether to perform construction with grayscale.",
)
@click.option(
    "--bayer",
    is_flag=True,
    help="Whether image is raw bayer data.",
)
@click.option(
    "--plot",
    is_flag=True,
    help="Whether to no plot.",
)
@click.option(
    "--bg",
    type=float,
    help="Blue gain.",
)
@click.option(
    "--rg",
    type=float,
    help="Red gain.",
)
@click.option(
    "--gamma",
    default=None,
    type=float,
    help="Gamma factor for plotting.",
)
@click.option(
    "--single_psf",
    is_flag=True,
    help="Same PSF for all channels (sum) or unique PSF for RGB.",
)

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
        estimate = estimate['iterand'].reshape(data.shape)
    elif algo == "lasso":
        estimate, converged, diagnostics = lasso(psf, data, n_iter)
        estimate = estimate['iterand'].reshape(data.shape)
    elif algo == "nnls":
        estimate, converged, diagnostics = nnls(psf, data, n_iter)
        estimate = estimate['iterand'].reshape(data.shape)
    elif algo == "glasso":
        estimate, converged, diagnostics = glasso(psf, data, n_iter)
        estimate = estimate['iterand'].reshape(data.shape)
    elif algo == "pls":
        estimate, converged, diagnostics = pls(psf, data, n_iter)
        estimate = estimate['primal_variable'].reshape(data.shape)
    elif algo == "pls_huber":
        estimate, converged, diagnostics = pls_huber(psf, data, n_iter)
        estimate = estimate['iterand'].reshape(data.shape)



    plt.figure()
    plt.imshow(estimate, cmap='gray')
    if save:
        plt.savefig(save, format='png')
        print(f"Files saved to : {save}")
    if plot:
        plt.show()

    #return estimate


if __name__ == "__main__":
    reconstruction()

    """

    psf_fp = rf'{str(DATAPATH)}{os.sep}psf{os.sep}diffcam_rgb.png'
    data_fp = rf'{str(DATAPATH)}{os.sep}raw_data{os.sep}thumbs_up_rgb.png'
    algo = 'ridge'
    n_iter = 5
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
        single_psf) """



