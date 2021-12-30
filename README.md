# DiffuserCam

### _Lensless imaging with a Raspberry Pi and a piece of tape!_

This codebase was forked from [LCAV\DiffuserCam](https://github.com/LCAV/DiffuserCam). 

## Running our Code

You can run a showcase of our image reconstruction:  
[jupyter notebook](running_models_and_pulling_data.ipynb).

1. To run a command line version of our code use `evaluation_bash.py`


## Files
These are the files that we changed in the course of working on the project. 

**scripts**
* `evaluation.py` - controls loading in all the proper files and data, then compute using methods of `optimization.py` `admm.py`, collecting outputs and saving them.
* `evaluation_bash.py` - Bash version of evaluation using click
* `optimization.py` - We use the [Pyscou](https://github.com/matthieumeo/pycsou) framework to build regularized linear inverse problems and solve them using APGD or PDS. 
* `functionals.py` - We defined a DCT2 functional and an OptiConvolve2D functional.
* `run.py` - This file automates multiple runs of evalution() with a flexible system of adding parameters to lists. 

**notebooks**
* `running_models_and_pulling_data.ipynb` - example notebook of how to run our code and pull data from our logs.
* `plots_old.ipynb` - old plotting notebook for report plots
* `benchmark_custom_convolve2d.ipynb` - you can test the performance our custom Convolve2D class on your hardware.

**diffcam**
* `utils.py` - added utility function load pickle and filepath management.
* `metric.py` - changed various metrics to make them color friendly. Also added a logging class that is used in 
evaluation.py to save data in a dictionary structure for use later.

## Data

You can find in:
* `DiffuserCam/data/our_images/lensed/` the original images dataset.
* `DiffuserCam/data/our_images/images/` the raw acquisition of the dataset.
* `DiffuserCam/data/our_images/diffuser/` the color corrected dataset.
* `DiffuserCam/data/psf/psf_ours.png` the raw acquisition of the point spread function.
* `DiffuserCam/data/psf/psf_rgb_ours.png` the color corrected psf.


