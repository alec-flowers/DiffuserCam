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
             bg_pix=(5,25),
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
    print("Number of files : ", len(files))

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
    if gray:
        psf = rgb2gray(psf)
    ax = plot_image(psf, gamma=gamma)
    ax.set_title("PSF")

    print("\nLooping through files...")
    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    for fn in files:
        # prepare file paths
        lenseless_fp = os.path.join(diffuser_dir, fn)

        if data == 'our_images':
            lensed_fp = os.path.join(lensed_dir, ("_").join([fn.split("_")[0], 'original.png']))
        else:
            lensed_fp = os.path.join(lensed_dir, fn)

        # load ground truth image #TODO: Not sure if this should be processed in some way
        lensed = load_image(lensed_fp, flip=flip, bayer=bayer, blue_gain=bg, red_gain=rg)
        lensed = np.array(lensed, dtype=dtype)

        # load and process raw measurement
        lenseless = load_image(lenseless_fp, flip=flip, bayer=bayer, blue_gain=bg, red_gain=rg)
        lenseless = np.array(lenseless, dtype=dtype)

        lenseless -= background
        lenseless = np.clip(lenseless, a_min=0, a_max=lenseless.max())

        if data == 'our_images': #TODO: Not sure about this
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

        estimate = estimate[200:420, 290:620] #Manually set at the moment #TODO: change cropping
        #resize(lenseless, 1 / downsample) #TODO: resize to lensed image shape (to compute metrics)

        # plot images
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

        bn = os.path.basename(fn).split(".")[0]
        print(f"\n{bn}")

        mse_scores.append(mse(lensed, estimate))
        psnr_scores.append(psnr(lensed, estimate))
        if gray:
            ssim_scores.append(ssim(lensed, estimate, channel_axis=None)) #TODO: this was changed to be able to use grayscale images
        else:
            ssim_scores.append(ssim(lensed, estimate))
        #lpips_scores.append(lpips(lensed, estimate)) #TODO: bug in lpips score

        print(mse_scores[-1])
        print(psnr_scores[-1])
        print(ssim_scores[-1])
        #print(lpips_scores[-1]) #TODO: lpips

    print("\nMSE (avg)", np.mean(mse_scores))
    print("PSNR (avg)", np.mean(psnr_scores))
    print("SSIM (avg)", np.mean(ssim_scores))
    #print("LPIPS (avg)", np.mean(lpips_scores)) #TODO: lpips



if __name__ == '__main__':
    data = 'our_images'
    n_files = None # None yields all :-)
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

    psf_fp = rf'{str(DATAPATH)}{os.sep}psf{os.sep}diffcam_rgb.png'

    evaluate(data,
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
             single_psf
             )
