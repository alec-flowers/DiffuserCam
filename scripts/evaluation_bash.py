"""

```bash

python scripts/evaluation_bash.py --psf_fp data/psf/diffcam_rgb.png --algo ridge --n_iter 500 --gray --plot --save

```
"""

import click
from evaluation import evaluate
import numpy as np

# TODO Understand why using bash version cause a wrong error.

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
@click.option(
    "--algo",
    type=str,
    help="Name of reconstruction algorithm.",
)

@click.option(
    "--lambda_",
    type=float,
    default=0.1,
    help="Lambda value for regularization.",
)

@click.option(
    "--delta",
    type=float,
    default=1,
    help="Delta for huber metric.",
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

def local_evaluate(data,
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
    evaluate(data, n_files, psf_fp, algo, lambda_, delta, n_iter, 
             downsample, disp, flip, gray, bayer, bg, rg, gamma, 
             save, plot, single_psf, dtype=dtype, bg_pix=bg_pix,)
    

if __name__ == '__main__':
    local_evaluate()
