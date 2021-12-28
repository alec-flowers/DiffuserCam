import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import lpips as lpips_lib
import torch
from diffcam.util import LOGPATH
import pickle
import os
from datetime import datetime


def mse(true, est, normalize=True):
    if normalize:
        true /= true.max()
        est /= est.max()
    return mean_squared_error(image0=true, image1=est)


def psnr(true, est, normalize=True):
    if normalize:
        true /= true.max()
        est /= est.max()
    return peak_signal_noise_ratio(image_true=true, image_test=est)


def ssim(true, est, normalize=True, channel_axis=2): #changed to account for grayscale input
    if normalize:
        true /= true.max()
        est /= est.max()
    return structural_similarity(im1=true, im2=est, channel_axis=channel_axis)


def lpips(true, est, normalize=True, gray=False):
    # https://github.com/richzhang/PerceptualSimilarity
    if normalize:
        true /= true.max()
        est /= est.max()
    loss_fn = lpips_lib.LPIPS(net="alex", verbose=False)
    if gray: #TODO: Check this thing! Not sure :-)
        true = torch.from_numpy(true[np.newaxis,].copy()).float()
        est = torch.from_numpy(est[np.newaxis,].copy()).float()
    else:
        true = torch.from_numpy(np.transpose(true, axes=(2, 0, 1))[np.newaxis,].copy())
        est = torch.from_numpy(np.transpose(est, axes=(2, 0, 1))[np.newaxis,].copy())

    return loss_fn.forward(true, est).squeeze().item()


class LogMetrics():
    def __init__(self):
        self.save = {}

        self.mse_scores = []
        self.psnr_scores = []
        self.ssim_scores = []
        self.lpips_scores = []

    def add_param(self, key, value):
        self.save[key] = value

    def calculate_metrics(self, lensed, estimate):
        self.mse_scores.append(mse(lensed, estimate))
        self.psnr_scores.append(psnr(lensed, estimate))
        # self.lpips_scores.append(lpips(lensed, estimate, gray=self.save['gray']))  # TODO: Check if grayscale implementation is correct!

        # ssim function changed in order to account for grayscale input
        if self.save['gray']:
            self.ssim_scores.append(ssim(lensed, estimate, channel_axis=None))
        else:
            self.ssim_scores.append(ssim(lensed, estimate))

    def save_metrics(self, photo):
        self.add_param(photo, {"mse": self.mse_scores[-1],
                            "psnr":self.psnr_scores[-1],
                            "ssim":self.ssim_scores[-1],
                            # "lpips":self.lpips_scores[-1]
                            })

    def print_metrics(self):
        print(f"\nMSE: {self.mse_scores[-1]}")
        print(f"\nPSNR: {self.psnr_scores[-1]}")
        print(f"\nSSIM: {self.ssim_scores[-1]}")
        # print(f"\nLPIPS: {self.lpips_scores[-1]}")

    def save_metric_list(self):
        self.add_param("mse", self.mse_scores)
        self.add_param("psnr", self.psnr_scores)
        self.add_param("ssim", self.ssim_scores)
        #self.add_param("lpips", self.lpips_scores)

    def print_average_metrics(self):
        print("\n-----------------------------\n Average scores \n-----------------------------")
        print("\nMSE (avg)", np.mean(self.mse_scores))
        print("PSNR (avg)", np.mean(self.psnr_scores))
        print("SSIM (avg)", np.mean(self.ssim_scores))
        #print("LPIPS (avg)", np.mean(self.lpips_scores))

    def save_logs(self):
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        path = os.path.join(str(LOGPATH), f"{self.save['algo']}{timestamp}")
        print(f"Logs saved to: {path}")
        with open(path+".pkl", 'wb') as f:
            pickle.dump(self.save, f, pickle.HIGHEST_PROTOCOL)
