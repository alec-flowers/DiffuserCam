"""

```bash

python scripts/evaluation_bash.py --data our_images --psf_fp data/psf/diffcam_rgb.png --algo ridge --n_iter 500 --gray --plot --save

```
"""

import os
import cv2
import time
import glob
import click
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from diffcam.plot import plot_image
from diffcam.io import load_psf, load_image
from diffcam.metric import mse, psnr, ssim, lpips
from diffcam.util import DATAPATH, RECONSTRUCTIONPATH, print_image_info, resize, rgb2gray

from scripts.optimization import lasso, ridge, nnls, glasso, pls, pls_huber


@click.command()
@click.option(
    "--data",
    type=str,
    default='our_images',
    help="Specify the data set to use."
)
@click.option(
    "--psf_fp",
    type=str,
    help="File name for recorded PSF.",
)
@click.option(
    "--n_files",
    type=int,
    default=None,
    help="Evaluates first n files. None yields all",
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
def evaluate(data,
             n_files,
             psf_fp,
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
             dtype=np.float32,
             bg_pix=(5, 25),
             ):
    assert data is not None

    # determining data paths
    diffuser_dir = os.path.join(str(DATAPATH), data, "diffuser")
    lensed_dir = os.path.join(str(DATAPATH), data, "lensed")

    # specifying filetype
    if data == 'our_images':
        filetype = 'png'
    else:
        filetype = 'tif'

    # list of all files
    files = glob.glob(diffuser_dir + f"/*.{filetype}")
    if n_files:
        files = files[:n_files]
    files = [os.path.basename(fn) for fn in files]
    print("\nNumber of files : ", len(files))

    # Load PSF
    assert os.path.isfile(psf_fp)
    psf, background = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=True,
        bg_pix=bg_pix,
        return_bg=True,
        flip=flip,
        bayer=bayer,
        blue_gain=bg,
        red_gain=rg,
        dtype=dtype,
        single_psf=single_psf)

    print_image_info(psf)
    if gray:
        psf = rgb2gray(psf)

    print("\nLooping through files...")
    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    height_crops = (180, 420)
    width_crops = (290, 640)

    for fn in files:

        bn = os.path.basename(fn).split(".")[0]
        print(f"\n-----------------------------\n Evaluating {bn}...\n-----------------------------")

        # prepare file paths
        lenseless_fp = os.path.join(diffuser_dir, fn)

        if data == 'our_images':
            lensed_fp = os.path.join(lensed_dir, "_".join([fn.split("_")[0], 'original.png']))
        else:
            lensed_fp = os.path.join(lensed_dir, fn)

        # load ground truth image
        lensed = load_image(lensed_fp, flip=flip, bayer=bayer, blue_gain=bg, red_gain=rg)
        lensed = np.array(lensed, dtype=dtype)

        # load and process raw measurement
        lenseless = load_image(lenseless_fp, flip=flip, bayer=bayer, blue_gain=bg, red_gain=rg)
        lenseless = np.array(lenseless, dtype=dtype)

        lenseless -= background
        lenseless = np.clip(lenseless, a_min=0, a_max=lenseless.max())

        if data == 'our_images':
            if lenseless.shape != psf.shape:
                # in DiffuserCam dataset, images are already reshaped
                lenseless = resize(lenseless, 1 / downsample)
        lenseless /= np.linalg.norm(lenseless.ravel())

        if gray:
            lenseless = rgb2gray(lenseless)
            lensed = rgb2gray(lensed)

        # reconstruct image
        if disp < 0:
            disp = None
        if save:
            timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
            save = RECONSTRUCTIONPATH / str(bn.split("_")[0] + '_' + algo + '_' + str(n_iter) + timestamp + '.png')
            save_uncropped = RECONSTRUCTIONPATH / str('uncropped_' + bn.split("_")[0] + '_' + algo + '_' + str(n_iter) + timestamp + '.png')

        if algo == "ridge":
            estimate, converged, diagnostics = ridge(psf, lenseless, n_iter)
            estimate = estimate['iterand'].reshape(lenseless.shape)
        elif algo == "lasso":
            estimate, converged, diagnostics = lasso(psf, lenseless, n_iter)
            estimate = estimate['iterand'].reshape(lenseless.shape)
        elif algo == "nnls":
            estimate, converged, diagnostics = nnls(psf, lenseless, n_iter)
            estimate = estimate['iterand'].reshape(lenseless.shape)
        elif algo == "glasso":
            estimate, converged, diagnostics = glasso(psf, lenseless, n_iter)
            estimate = estimate['iterand'].reshape(lenseless.shape)
        elif algo == "pls":
            estimate, converged, diagnostics = pls(psf, lenseless, n_iter)
            estimate = estimate['primal_variable'].reshape(lenseless.shape)
        elif algo == "pls_huber":
            estimate, converged, diagnostics = pls_huber(psf, lenseless, n_iter)
            estimate = estimate['iterand'].reshape(lenseless.shape)
        else:
            estimate = None
            raise AttributeError("Reconstruction algorithm not defined.")

        # save and plot un-cropped reconstruction?
        ax = plot_image(estimate, gamma=gamma)
        ax.set_title("Uncropped reconstruction")
        if save:
            plt.savefig(save_uncropped, format='png')
            print(f"\nFiles saved to : {save_uncropped}")

        estimate = estimate[height_crops[0]:height_crops[1], width_crops[0]:width_crops[1]]
        lensed = cv2.resize(lensed, (estimate.shape[::-1]), interpolation=cv2.INTER_CUBIC)  # TODO: Check this!

        print("\nGround truth shape:", lensed.shape)
        print("Reconstruction shape:", estimate.shape)

        # save and plot reconstruction?
        ax = plot_image(estimate, gamma=gamma)
        ax.set_title("Reconstructed")
        if save:
            plt.savefig(save, format='png')
            print(f"\nFiles saved to : {save}")

        # plot images
        if plot:
            # psf
            ax = plot_image(psf, gamma=gamma)
            ax.set_title("PSF")

            # diffusercam image
            ax = plot_image(lenseless, gamma=gamma)
            ax.set_title("Raw data")

            # lensed image
            ax = plot_image(lensed, gamma=gamma)
            ax.set_title("Ground truth")
            plt.show()

        mse_scores.append(mse(lensed, estimate))
        psnr_scores.append(psnr(lensed, estimate))
        lpips_scores.append(lpips(lensed, estimate, gray=gray))  # TODO: Check if grayscale implementation is correct!

        # ssim function changed in order to account for grayscale input
        if gray:
            ssim_scores.append(ssim(lensed, estimate, channel_axis=None))
        else:
            ssim_scores.append(ssim(lensed, estimate))

        print(f"\nMES: {mse_scores[-1]}")
        print(f"\nPSNR: {psnr_scores[-1]}")
        print(f"\nSSIM: {ssim_scores[-1]}")
        print(f"\nLPIPS: {lpips_scores[-1]}")

    print("\n-----------------------------\n Average scores \n-----------------------------")
    print("\nMSE (avg)", np.mean(mse_scores))
    print("PSNR (avg)", np.mean(psnr_scores))
    print("SSIM (avg)", np.mean(ssim_scores))
    print("LPIPS (avg)", np.mean(lpips_scores))


if __name__ == '__main__':
    evaluate()
