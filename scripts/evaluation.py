import os
import cv2
import time
import glob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from diffcam.plot import plot_image
from diffcam.io import load_psf, load_image
from diffcam.admm import ADMM
from diffcam.metric import mse, psnr, ssim, lpips, LogMetrics
from diffcam.util import DATAPATH, RECONSTRUCTIONPATH, print_image_info, resize, rgb2gray

from scripts.optimization import optimize

#===================== CUSTOM CROP SIZE  ======================
# height crops determined from ADMM reconstructions
HEIGHT_CROPS = {'img1_rgb': (179, 407),
                'img3_rgb': (179, 395),
                'img4_rgb': (181, 385),
                'img5_rgb': (183, 414),
                'img6_rgb': (179, 399),
                'img7_rgb': (183, 414),
                'img8_rgb': (181, 410)}

# width crops determined from ADMM reconstructions
WIDTH_CROPS = {'img1_rgb': (328, 669),
                'img3_rgb': (317, 681),
                'img4_rgb': (315, 683),
                'img5_rgb': (416, 585),
                'img6_rgb': (323, 679),
                'img7_rgb': (416, 585),
                'img8_rgb': (326, 671)}


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
    if isinstance(n_iter, int):
        n_iters = [n_iter]
    elif isinstance(n_iter, list) and isinstance(n_iter[0], int):
        n_iters = n_iter
    else:
        raise ValueError("n_iter should be an int value or a non empty list of int")
    n_iter = None
    
    log = LogMetrics()
    log.add_param('data', data)
    log.add_param('psf_fp', psf_fp)
    log.add_param('algo', algo)
    log.add_param('gray', gray)
    log.add_param('lambda', lambda_)
    log.add_param('delta', delta)
    log.add_param('n_iter', n_iters)
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
        files = files[:n_files]
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
    
    #===================== PER IMAGE FILE COMPUTATIONS ======================
    
    print("\nLooping through files...")
    for fn in files:
        bn = os.path.basename(fn).split(".")[0]
        print(f"\n-----------------------------\n Evaluating {bn}...\n-----------------------------")

        #===================== IMAGES LOADING AND PREPROCESS ==================
    
        lenseless_fp = os.path.join(diffuser_dir, fn)
        if data == 'our_images':
            lensed_fp = os.path.join(lensed_dir, "_".join([fn.split("_")[0], 'original.png']))
        else:
            lensed_fp = os.path.join(lensed_dir, fn)
    
        lensed, lenseless = load_lensed_lenseless(lensed_fp, lenseless_fp,  
                                                psf, background, bn,
                                                gray, data, flip, bayer, 
                                                bg, rg, downsample, dtype)

        # ==================== INVERSE ESTIMATE ===============================
        
        estimates, elapsed_times = inverse_estimate(algo, psf, lenseless, n_iters, lambda_, delta, dtype)
        print("Total computation time:", elapsed_times[-1],"s")
        # =============== POSTPROCESS + PLOTTING/SAVING =======================
        
        # Loop, in case we have more than one n_iter (to checkpoint optimization through various steps)
        total_iter = 0
        for n, estimate, elapsed_time in zip(n_iters, estimates, elapsed_times):
            
            total_iter += n
            ax, uncropped_img = plot_image(estimate, gamma=gamma, return_image=True)
            ax.set_title("Uncropped reconstruction")

            cropped_estimate = estimate[HEIGHT_CROPS[bn][0]:HEIGHT_CROPS[bn][1], WIDTH_CROPS[bn][0]:WIDTH_CROPS[bn][1]]
            
            ax, cropped_img = plot_image(cropped_estimate, gamma=gamma, return_image=True)
            ax.set_title("Reconstructed")
            
            print("Reconstruction shape:", cropped_estimate.shape)
            
            if save:
                save_results(algo, uncropped_img, cropped_img, cropped_estimate, gray, total_iter, bn, log)
            
            # plot images
            if plot:
                show_results(psf, lensed, lenseless, gamma)
                
            log.add_iter_param("process_time", elapsed_time)
            log.add_iter_param('lenseless_fp', lenseless_fp)
            log.add_iter_param('lensed_fp', lensed_fp)
            log.add_iter_param("est_min", cropped_estimate.min())
            log.add_iter_param("est_max", cropped_estimate.max())
            log.calculate_metrics(lensed, cropped_estimate)
            log.save_metrics(total_iter)
            log.print_metrics()

        log.save_metric_list(bn)
    if save:
        log.save_logs()

    return log

def load_lensed_lenseless(lensed_fp, lenseless_fp,  
                          psf, background, bn,
                          gray, data, flip, bayer, 
                          bg, rg, downsample, dtype):

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
    new_shape = (WIDTH_CROPS[bn][1] - WIDTH_CROPS[bn][0], HEIGHT_CROPS[bn][1] - HEIGHT_CROPS[bn][0])
    lensed = cv2.resize(lensed, new_shape, interpolation=cv2.INTER_AREA)
    
    return lensed, lenseless

def inverse_estimate(algo, psf, lenseless, n_iters, lambda_, delta, dtype):
    estimates = []
    elapsed_times = []
    
    if algo == "admm":
        start_time = time.process_time()
        recon = ADMM(psf.squeeze())
        recon.set_data(lenseless.squeeze())
        print(f"setup time : {time.process_time() - start_time} s")

        proc_start_time = time.process_time()
        for n in n_iters:
            for _ in range(n):
                recon._update()
            elapsed_time = time.process_time() - start_time
            print(f"proc time... : {time.process_time() - proc_start_time} s")
            elapsed_times.append(elapsed_time)
            estimates.append(recon._form_image().squeeze())
    else:
        estimates, elapsed_times = optimize(algo, psf, lenseless, n_iters, dtype, lambda_, delta)
        
    
    return estimates, elapsed_times

def save_results(algo, uncropped_img, cropped_img, cropped_estimate, gray, n_iter, bn, log):     
    timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
    save = RECONSTRUCTIONPATH / str(bn.split("_")[0] + '_' + algo + '_' + str(n_iter) + timestamp + '.png')
    save_uncropped = RECONSTRUCTIONPATH / str('uncropped_' + bn.split("_")[0] + '_' + algo + '_' + str(n_iter) + timestamp + '.png')
    log.add_iter_param('recon_fp', save)
    log.add_iter_param('ucrop_recon_fp', save_uncropped)
    
    if gray:
        cv2.imwrite(str(save_uncropped), uncropped_img * 255)
        cv2.imwrite(str(save), cropped_img * 255)
    else:
        cv2.imwrite(str(save_uncropped), cv2.cvtColor(uncropped_img, cv2.COLOR_RGB2BGR) * 255)
        cv2.imwrite(str(save), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)* 255)
    npy_save = RECONSTRUCTIONPATH / str(bn.split("_")[0] + '_' + algo + '_' + str(n_iter) + timestamp + '.npy')
    np.save(npy_save, cropped_estimate)
    print(f"\nFiles saved to : {save_uncropped}")
    print(f"\nFiles saved to : {save}")

def show_results(psf, lensed, lenseless, gamma):
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

if __name__ == '__main__':
    data = 'our_images'
    n_files = 1          # None yields all :-)
    algo = 'admm'
    n_iter = [3, 1, 1]
    gray = False
    downsample = 4
    disp = 50
    flip = False
    bayer = False
    bg = None
    rg = None
    gamma = None
    save = True
    plot = False
    single_psf = False
    lambda_ = 0.0001
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
