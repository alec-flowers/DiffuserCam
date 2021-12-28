from scripts.reconstruction import reconstruction
from diffcam.util import DATAPATH
from diffcam.metric import mse, psnr, ssim, lpips
import numpy as np
import os

class TestSuite():
    def __init__(self, parameters):

    def run(self):


    def save_results(self):
        pass

class LogMetrics():
    def __init__(self, gray):
        self.save = {}
        self.gray = gray

        self.mse_scores = []
        self.psnr_scores = []
        self.ssim_scores = []
        self.lpips_scores = []

    def record_metrics(self):
        self.mse_scores = []
        self.psnr_scores = []
        self.ssim_scores = []
        self.lpips_scores = []

    def add_metrics(self, lensed, estimate):
        self.mse_scores.append(mse(lensed, estimate))
        self.psnr_scores.append(psnr(lensed, estimate))
        if self.gray:
            self.ssim_scores.append(
                ssim(lensed, estimate, channel_axis=None))  # TODO: this was changed to be able to use grayscale images
        else:
            self.ssim_scores.append(ssim(lensed, estimate))
        # lpips_scores.append(lpips(lensed, estimate)) #TODO: bug in lpips score

    def save_metrics(self, photo):
        self.save[photo] = {"mse": self.mse_scores,
                            "psnr":self.psnr_scores,
                            "ssim":self.ssim_scores,
                            "lpips":self.lpips_scores}

    def print_metric(self):
        print(self.mse_scores[-1])
        print(self.psnr_scores[-1])
        print(self.ssim_scores[-1])
        #print(lpips_scores[-1]) #TODO: lpips

    def print_average(self):
        print("\nMSE (avg)", np.mean(self.mse_scores))
        print("PSNR (avg)", np.mean(self.psnr_scores))
        print("SSIM (avg)", np.mean(self.ssim_scores))
        # print("LPIPS (avg)", np.mean(lpips_scores)) #TODO: lpips

if __name__ == "__main__":
    parameters = {
        "data" : ['our_images'],
        "n_files" : [None], # None yields all :-)
        "algo" : ['ridge'],
        "n_iter" : [500],
        "gray" : [True],
        "downsample" : [4],
        "disp" : [50],
        "flip" : [False],
        "bayer" : [False],
        "bg" : [None],
        "rg" : [None],
        "gamma" : [None],
        "save" : [True],
        "plot" : [True],
        "single_psf" : [False],
        "psf_fp" : [rf'{str(DATAPATH)}{os.sep}psf{os.sep}diffcam_rgb.png'],
    }