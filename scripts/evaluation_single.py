"""

"""

import os
import time
import glob
import click
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from diffcam.io import load_psf
from diffcam.metric import mse, psnr, ssim, lpips
from reconstruction import reconstruction
from diffcam.plot import plot_image

from diffcam.mirflickr import postprocess
from diffcam.io import load_image
from scripts.optimization import lasso, ridge, nnls, glasso, pls, pls_huber

from diffcam.util import DATAPATH, RECONSTRUCTIONPATH, print_image_info, resize, rgb2gray


def evaluate(psf_fp,
             data_fp,
             ground_truth_fp,
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
             bg_pix=(5,25),
             ):

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
        single_psf=single_psf,
    )
    print_image_info(psf)

    assert os.path.isfile(ground_truth_fp)
    assert os.path.isfile(data_fp)

    # load ground truth image #TODO: Not sure if this should be processed in some way?
    lensed = load_image(ground_truth_fp, flip=flip, bayer=bayer, blue_gain=bg, red_gain=rg)
    lensed = np.array(lensed, dtype=dtype)

    # load and process raw measurement
    lenseless = load_image(data_fp, flip=flip, bayer=bayer, blue_gain=bg, red_gain=rg)
    lenseless = np.array(lenseless, dtype=dtype)

    lenseless -= background
    lenseless = np.clip(lenseless, a_min=0, a_max=lenseless.max())
    if lenseless.shape != psf.shape:
        # in DiffuserCam dataset, images are already reshaped
        lenseless = resize(lenseless, 1 / downsample) #TODO: this fucks up MirFlickr data
    lenseless /= np.linalg.norm(lenseless.ravel())

    if gray:
        psf = rgb2gray(psf)
        lenseless = rgb2gray(lenseless)
        lensed = rgb2gray(lensed)

    # reconstruct image
    if disp < 0:
        disp = None
    if save:
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        save = RECONSTRUCTIONPATH / str(algo + '_' + str(n_iter) + timestamp + '.png')

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

    # TODO: reshape reconstruction in a clever way

    estimate = estimate[:, :]

    # show figures
    plt.figure()
    plt.imshow(estimate, cmap='gray')
    if save:
        plt.savefig(save, format='png')
        print(f"Files saved to : {save}")

    if plot:
        ax = plot_image(lenseless, gamma=gamma)
        ax.set_title("Raw data")
        ax = plot_image(estimate, gamma=gamma)
        ax.set_title("Reconstructed")
        ax = plot_image(lensed, gamma=gamma)
        ax.set_title("Ground truth")
        plt.show()

    print("\nMSE", mse(lensed, estimate))
    print("PSNR", psnr(lensed, estimate))
    print("SSIM", ssim(lensed, estimate, channel_axis=None))
    #print("LPIPS", lpips(lensed, estimate)) #TODO: bug when trying lpips score


if __name__ == '__main__':

    algo = 'ridge'
    n_iter = 500
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

    # MirFlickr
    #psf_fp = rf'{str(DATAPATH)}{os.sep}psf{os.sep}diffcam_rgb.png'
    #data_fp = rf'{str(DATAPATH)}{os.sep}MirFlickr{os.sep}diffuser{os.sep}im170.tif'
    #ground_truth_fp = rf'{str(DATAPATH)}{os.sep}MirFlickr{os.sep}lensed{os.sep}im170.tif'

    # Our data
    psf_fp = rf'{str(DATAPATH)}{os.sep}psf{os.sep}diffcam_rgb.png'
    data_fp = rf'{str(DATAPATH)}{os.sep}our_images{os.sep}diffuser{os.sep}img2_rgb.png'
    ground_truth_fp = rf'{str(DATAPATH)}{os.sep}our_images{os.sep}lensed{os.sep}img2_original.png'

    evaluate(psf_fp,
             data_fp,
             ground_truth_fp,
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
             single_psf
             )
