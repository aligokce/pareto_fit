from matplotlib.pyplot import sca
import numpy as np
import higrid as hg
import os


def main():
    """
       1a.   Select a number of source directions from the measurement grid
             The example below is for positions ar 45, 135, 225, 315 degrees azimuth in the horizontal plane
    """

    # TODO: Set instances according to that in the Evalutaion chapter on Olgun's paper
    # These are sample points in 3D space, x and y: 0 to 6
    # Thus, varying distances will be set from here, see todo below
    # testinstance = set([(3, 2, 2)]) # 0.5m, single source
    testinstance = set([(3, 2, 2), (3, 4, 2), (2, 3, 2), (4, 3, 2)]) # 0.5m x 4 src, phi = [90, 270, 0, 180] degrees, correspondingly [from Olgun's paper Evaluation chapter] 
    # testinstance = set([(1, 1, 2)]) # distxyz=(0.3*2, 0.5*2, 0) => 1.17 m
    # testinstance = set([(1, 1, 2), (5, 1, 2), (5, 5, 2), (1, 5, 2)])
    # testinstance = set([(1, 1, 2), (5, 1, 2), (5, 5, 2), (1, 5, 2)])

    """1b.   Uncomment lines below to create emulations for testing the algorithm using the measurement grid"""

    # mg = hg.measgrid()
    # testinstance = hg.selectset(mg, 4, 3, 8, np.pi / 4, 1)[0][0]

    """""""# 1c.   Compose an emulated scene..."""

    """i) ...using near-coherent music signals"""
    data_folder = '/mnt/d/dev/higrid/higrid'
    os.chdir(data_folder)

    # [deprecated] xTODO: Do not need multiple sounds but varying distances for comparable parameter estimation from fitting to Pareto distribution
    # Note: May need multiple sounds, look at the Evaluation chapter on Olgun's paper for details [add: is it relevant to pareto distribution fitting?]
    # music_signals = ['music/mahler_vl1a_6.wav']
    music_signals = ['music/mahler_vl1a_6.wav', 'music/mahler_vl1b_6.wav', 'music/mahler_vl2a_6.wav', 'music/mahler_vl2b_6.wav'] 

    # [deprecated] xTODO: hg.emulate.emulatescene is enough for a single sound
    sg = hg.composescene(
        music_signals,
        testinstance, (0, 192000)) # 48000 Hz * 4 sec [checked from Olgun's paper]

    """ii) OR comment above and uncomment below to use speech signals"""
    # sg = hg.composescene(
    #    ['speech/male1.wav', 'speech/female2.wav', 'speech/male3.wav', 'speech/female4.wav'],
    #    testinstance[0][0], (0, 192000))

    """1d.   Alternatively, comment everything above and uncomment the following to use real recordings."""
    # sg = hg.realrec(getcwd() + '/data/sdata/real/', 'Quartet', 192000)

    """
        Convert 32 channels to 25 SHD coeffs
    """
    from higrid.higridestimate import preprocessinput
    
    Ndec = 4 # SHD order
    NFFT = 1024
    olap = 16  # TODO: unchecked
    Pnm, P = preprocessinput(sg, Ndec, NFFT, olap) # return: SHD, STFT


    """
        Calculate spatial correlation matrix
    """
    # maxnum = 1000
    Fs = 48000
    # tLevel = 3,
    # dpdflag = True
    fL = 2608.
    fH = 5216.
    # thr = 6

    # === From higrid.higridestimate.binstoprocess_dpd
    # --binstoprocess_dpd(P, fL, fH, Fs, NFFT, 25, 4, 3, thr)
    from higrid.Microphone import EigenmikeEM32

    Jnu = 25
    Jtau = 4
    Ndec = 3 # TODO: 3 WTF? It was 4 before, looks like specific to DPD test
    em32 = EigenmikeEM32().returnAsStruct()
    fimin = int(round(fL / Fs * NFFT))
    fimax = int(round(fH / Fs * NFFT))

    # W, Y = getWY(em32, Ndec)
    from higrid.dpd import getBmat, getAnm

    Bmat = getBmat(em32, fimin, fimax + Jnu, NFFT, Fs, Ndec)
    Anm = getAnm(P, em32, Bmat, fimin, fimax + Jnu, Ndec) # Note: (Ndec + 1)**2 = 25 for Ndec = 4
    

    """
        Calculate singular values ratios
    """

    # === From higrid.dpd.dpd
    def singular_ratio(Anm, Ndec, find, tind, Jtau, Jnu):
        anm = np.matrix(np.zeros(((Ndec + 1) ** 2, 1), dtype=complex))
        Ra = np.matrix(np.zeros(((Ndec + 1) ** 2, (Ndec + 1) ** 2), dtype=complex))

        # TODO: Convert bin selection to all
        for fi in range(find, find + Jnu):
            for ti in range(tind, tind + Jtau):
                for ind in range((Ndec + 1) ** 2):
                    anm[ind, 0] = Anm[ind][ti, fi]
                Ra += (anm * anm.H)
        Ra = Ra / (Jtau * Jnu)
        S = np.linalg.svd(Ra, compute_uv=False) # sorted, it looks like from hg.dpd.dpd()
        ratio = S[0] / S[1]
        return ratio


    import tqdm
    ratio_list = []
    # === From higrid.higridestimate.binstoprocess_dpd

    imax = Anm[0].shape[0]

    # idx = []
    # idy = []

    for tind in tqdm.trange(imax - Jtau, desc="DPD bin selection     "):
        for find in range(fimin, fimax):
            ratio = singular_ratio(Anm, Ndec, find, tind, Jtau, Jnu)
            ratio_list.append(ratio)
            

    """
        Fit ratio histogram to a Pareto distribution
    """
    import scipy
    shape, location, scale = scipy.stats.genpareto.fit(np.array(ratio_list)) # c for shape, then loc and scale
    print(f"Found parameters: {shape=}, {location=}, {scale=}")


    """
        Plot histogram and fitted pdf
    """
    x = np.linspace(0, max(ratio_list), 100) # TODO: Fixed value?
    fitted_data = scipy.stats.genpareto.pdf(x, shape, loc=location, scale=scale)

    import matplotlib.pyplot as plt
    plt.hist(ratio_list, density=True)
    plt.plot(x, fitted_data, 'r-')
    plt.show()


    """
        THETASK: Find a distance - distribution fit relation between
    """
    pass

if __name__=='__main__':
    # os.chdir('/home/aligokce/dev/higrid')
    main()