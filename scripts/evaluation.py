import os
import cv2
import time
import glob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from time import process_time

from diffcam.plot import plot_image
from diffcam.io import load_psf, load_image
from diffcam.metric import mse, psnr, ssim, lpips, LogMetrics
from diffcam.util import DATAPATH, RECONSTRUCTIONPATH, print_image_info, resize, rgb2gray

from scripts.optimization import optimize


def evaluate(data,
             n_files,
             psf_fp,
             algo,
             lambda_,
             delta,
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
    log = LogMetrics()
    log.add_param('data', data)
    log.add_param('psf_fp', psf_fp)
    log.add_param('algo', algo)
    log.add_param('n_iter', n_iter)
    log.add_param('gray', gray)
    log.add_param('lambda', lambda_)
    log.add_param('delta', delta)

    #===================== FILENAME MANAGEMENT ===================
    
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
        # TODO change back...
        files = [files[n_files]]
    files = [os.path.basename(fn) for fn in files]
    print("\nNumber of files : ", len(files))

    #===================== LOADING PSF ======================
    assert os.path.isfile(psf_fp)
    psf, background = load_psf(psf_fp,
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
        psf = rgb2gray(psf)[:, :, np.newaxis]

    

    
    #===================== CUSTOM CROP SIZE  ======================
    # height crops determined from ADMM reconstructions
    height_crops = {'img1_rgb': (179, 407),
                    'img3_rgb': (179, 395),
                    'img4_rgb': (181, 385),
                    'img5_rgb': (183, 414),
                    'img6_rgb': (179, 399),
                    'img7_rgb': (183, 414),
                    'img8_rgb': (181, 410)}

    # width crops determined from ADMM reconstructions
    width_crops = {'img1_rgb': (328, 669),
                   'img3_rgb': (317, 681),
                   'img4_rgb': (315, 683),
                   'img5_rgb': (416, 585),
                   'img6_rgb': (323, 679),
                   'img7_rgb': (416, 585),
                   'img8_rgb': (326, 671)}
    
    #===================== PER IMAGE FILE COMPUTATIONS ======================
    
    print("\nLooping through files...")
    for fn in files:
        bn = os.path.basename(fn).split(".")[0]
        print(f"\n-----------------------------\n Evaluating {bn}...\n-----------------------------")

        #===================== IMAGES LOADING AND PREPROCESS ======================
        lenseless_fp = os.path.join(diffuser_dir, fn)
        log.add_img_param('lenseless_fp', lenseless_fp)
        if data == 'our_images':
            lensed_fp = os.path.join(lensed_dir, "_".join([fn.split("_")[0], 'original.png']))
        else:
            lensed_fp = os.path.join(lensed_dir, fn)
        log.add_img_param('lensed_fp', lensed_fp)

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
            log.add_img_param('recon_fp', save)
            log.add_img_param('ucrop_recon_fp', save_uncropped)

        # =============================================================================
        # ==================== INVERSE ESTIMATE =======================================
        
        start = process_time()
        estimate, _, _ = optimize(algo, psf, lenseless, n_iter, lambda_, delta)
        stop = process_time()
        log.add_img_param("process_time", stop - start)

        if algo == 'pls':
            estimate = estimate['primal_variable'].reshape(lenseless.shape)
        else:
            estimate = estimate['iterand'].reshape(lenseless.shape)

        # ========================== POSTPROCESS PLOTTING/SAVING ===============================

        # save and plot un-cropped reconstruction?
        ax, uncropped_img = plot_image(estimate, gamma=gamma, return_image=True)
        ax.set_title("Uncropped reconstruction")
        if save:
            # plt.savefig(save_uncropped, format='png')
            cv2.imwrite(str(save_uncropped), uncropped_img * 255) # Plotting the image only
            print(f"\nFiles saved to : {save_uncropped}")

        estimate = estimate[height_crops[bn][0]:height_crops[bn][1], width_crops[bn][0]:width_crops[bn][1]]

        log.add_img_param("est_min", estimate.min())
        log.add_img_param("est_max", estimate.max())

        new_shape = estimate.shape[:2][::-1]
        lensed = cv2.resize(lensed, new_shape, interpolation=cv2.INTER_NEAREST)  # TODO: Check this!

        print("\nGround truth shape:", lensed.shape)
        print("Reconstruction shape:", estimate.shape)

        # save and plot reconstruction?
        ax, cropped_img = plot_image(estimate, gamma=gamma, return_image=True)
        ax.set_title("Reconstructed")

        #TODO when saving images need to not save axes and title, want just the image itself
        if save:
            # plt.savefig(save, format='png')
            cv2.imwrite(str(save), cropped_img * 255)
            npy_save = RECONSTRUCTIONPATH / str(bn.split("_")[0] + '_' + algo + '_' + str(n_iter) + timestamp + '.npy')
            np.save(npy_save, estimate)
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

        log.calculate_metrics(lensed, estimate)
        log.save_metrics(bn)
        log.print_metrics()

    log.save_metric_list()
    log.print_average_metrics()

    if save:
        log.save_logs()

    return log


if __name__ == '__main__':
    data = 'our_images'
    n_files = 3          # None yields all :-)
    algo = 'ridge'
    n_iter = 10
    gray = False
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
    lambda_ = 0.1
    delta = 1

    psf_fp = rf'{str(DATAPATH)}{os.sep}psf{os.sep}psf_rgb_ours.png'
    
    evaluate(data,
             n_files,
             psf_fp,
             algo,
             lambda_,
             delta,
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
