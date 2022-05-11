import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genpareto


def save_distrib_plot(save_path, ratio_list, shape, location, scale):
    max_singular_ratio = 60
    histogram_bin_freq = 2
    pdf_density = 500

    x = np.linspace(0, max_singular_ratio, pdf_density)  # TODO: Fixed value?
    fitted_data = genpareto.pdf(x, shape, loc=location, scale=scale)

    plt.figure()
    plt.hist(ratio_list, bins=range(
        0, max_singular_ratio+1, histogram_bin_freq), density=True)
    plt.plot(x, fitted_data, 'r-')

    plt.savefig(save_path)


def save_config(save_path, config, verbose=True):
    with open(save_path, 'w') as f:
        json.dump(config, f)

    if verbose:
        print("Config file saved to:", save_path)


def save_ratio_list(save_path, ratio_list, verbose=True):
    np.save(save_path, ratio_list)

    if verbose:
        print(f"Ratio list: length={len(ratio_list)}")
        print("Ratio values saved to:", save_path)
