from diffcam.util import DATAPATH
from scripts.evaluation import evaluate
import os


def get_max(p):
    length = []
    for key, value in p.items():
        length.append(len(value))
    return max(length)


def get_current_run(i):
    current = {}
    for key, value in parameters.items():
        if len(value) <= i:
            current[key] = value[-1]
        else:
            current[key] = value[i]
    return current


def multiple_runs(parameters):
    log_list = []
    for i in range(get_max(parameters)):
        current = get_current_run(i)
        log = evaluate(**current)  # could save list of dicts but currently are saving a dict for every run


if __name__ == "__main__":
    parameters = {
        "data": ['our_images'],
        "n_files": [None],
        "algo": ["glasso"],
        "lambda_": [.0000001],
        "delta": [1],
        "n_iter": [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100]],
        "gray": [False],
        "downsample": [4],
        "disp": [50],
        "flip": [False],
        "bayer": [False],
        "bg": [None],
        "rg": [None],
        "gamma": [None],
        "save": [True],
        "plot": [False],
        "single_psf": [False],
        "psf_fp": [rf'{str(DATAPATH)}{os.sep}psf{os.sep}psf_rgb_ours.png'],
    }
    multiple_runs(parameters)
