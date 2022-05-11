import os

import numpy as np
from scipy.stats import genpareto

from utils.math import generate_ratios_list
from utils.save import *
from utils.spargair import generate_positions, get_distance


def shd_and_pareto(save_path, higrid_path, music, pos):
    music_name = music.split('/')[-1].split('.')[0]

    dist = get_distance(pos)

    pos_folder_name = f"{pos[0]}{pos[1]}{pos[2]}"
    print(f"Positioning to {pos_folder_name} (distance: {dist:.3f} m)")

    # ==================================================
    # Generate singular value ratios list

    ratios_list = generate_ratios_list(higrid_path, music, pos)

    # ==================================================
    # Fit to Generalized Pareto Distribution

    shape, location, scale = genpareto.fit(np.array(ratios_list))
    print(f"Found parameters: {shape=}, {location=}, {scale=}")
    
    # ==================================================
    # Saves
    
    # Make sure save folder exists
    if not os.path.exists(os.path.join(save_path, pos_folder_name)):
        os.makedirs(os.path.join(save_path, pos_folder_name))

    savepath_woext = os.path.join(save_path, pos_folder_name, f"{music_name}")

    save_ratio_list(savepath_woext + '.npy', ratios_list)

    # Save details and fitting results to a json file
    config = dict(
                sndfile = music_name,
                pos_grid = pos,
                dist_mic = dist,
                pareto_params = dict(
                    shape = shape,
                    scale = scale,
                    location = location
                )
            )
    save_config(savepath_woext + '.json', config)

    # Plot and save histogram and fitted pdf
    save_distrib_plot(savepath_woext + '.png',
                      ratios_list, shape, location, scale)

def main():
    from multiprocessing import Pool
    N_THREADS = 80

#    save_path = '/mnt/d/dev/higrid/results/single_source_take3'
#    higrid_path = '/mnt/d/dev/higrid/higrid'
    save_path = './results/dgx_threadpool_trial/'
    higrid_path = './higrid/'

    music_signals = ['music/mahler_vl1a_6.wav', 'music/mahler_vl1b_6.wav',
                     'music/mahler_vl2a_6.wav', 'music/mahler_vl2b_6.wav']

    src_grid = generate_positions()

    # for music in music_signals:
    #     for pos in src_grid:
    #         shd_and_pareto(save_path, higrid_path, music, pos)

    #====================
    with Pool(processes=N_THREADS) as pool:
        args_list = []
        for music in music_signals:
            for pos in src_grid:
                # shd_and_pareto(save_path, higrid_path, music, pos)
                args_list.append((save_path, higrid_path, music, pos))

        pool.starmap(shd_and_pareto, args_list)

if __name__ == "__main__":
    main()
