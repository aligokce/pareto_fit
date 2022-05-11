import numpy as np
# import tqdm
from numba import jit

from higrid.dpd import getBmat, getAnm
# from higrid import composescene
from higrid.higridestimate import preprocessinput
from higrid.Microphone import EigenmikeEM32
from utils.higrid import composescene

# ==================================================
# Constants

Ndec = 4  # SHD order
NFFT = 1024
olap = 16  # TODO: unchecked - in paper 75% but is it for this?
Fs = 48000
fL = 2608.
fH = 5216.
Jnu = 25
Jtau = 4
Ndec_spcorr = 4  # TODO: 3 WTF? It was 4 before, looks like specific to DPD test

dur_scene = 4  # seconds, 4 s [checked from Olgun's paper]

# ==================================================
# Create mic specific response equalisation matrix

em32 = EigenmikeEM32().returnAsStruct()
fimin = int(round(fL / Fs * NFFT))
fimax = int(round(fH / Fs * NFFT))

Bmat = getBmat(em32, fimin, fimax + Jnu, NFFT, Fs, Ndec)


@jit(nopython=True)
def spatial_corr(Anm, Ndec, find, tind, Jtau, Jnu):
    """
    From higrid.dpd.dpd
    """
    anm = np.zeros(((Ndec + 1) ** 2, 1), dtype=np.complex128)
    Ra = np.zeros(((Ndec + 1) ** 2, (Ndec + 1) ** 2), dtype=np.complex128)

    for fi in range(find, find + Jnu):
        for ti in range(tind, tind + Jtau):
            for ind in range((Ndec + 1) ** 2):
                anm[ind, 0] = Anm[ind][ti, fi]
            # Ra += (anm @ anm.H)
            Ra += (anm @ anm.conj().T)

    Ra = Ra / (Jtau * Jnu)
    return Ra

def singular_ratio(Anm, Ndec, find, tind, Jtau, Jnu):
    '''
    From higrid.dpd.dpd
    '''
    Ra = spatial_corr(Anm, Ndec, find, tind, Jtau, Jnu)

    S = np.linalg.svd(Ra, compute_uv=False)  # Returns sorted
    ratio = S[0] / S[1]
    return ratio

def generate_ratios_list(higrid_path, music, pos):
    # ==================================================
    # Compose scene from AIR IR and anechoic sound

    sg = composescene([music], set([pos]), higrid_path,
                      (0, Fs * dur_scene), datafromhigrid=True)

    # ==================================================
    # Calculate STFT

    # TODO: This is slow!
    _, P = preprocessinput(sg, Ndec, NFFT, olap)  # return: SHD, STFT

    # ==================================================
    # Calculate SHD coefficients for each TF-bin

    # TODO: This is too slow!
    Anm = getAnm(P, em32, Bmat, fimin, fimax + Jnu, Ndec) 

    # ==================================================
    # Find singular ratios

    ratio_list = []

    imax = Anm[0].shape[0]

    # TODO: This is painfully slow!
    # for tind in tqdm.trange(imax - Jtau, desc="DPD bin selection     "):
    for tind in range(imax - Jtau):
        for find in range(fimin, fimax):
            ratio = singular_ratio(Anm, Ndec_spcorr, find, tind, Jtau, Jnu)
            ratio_list.append(ratio)

    return ratio_list
