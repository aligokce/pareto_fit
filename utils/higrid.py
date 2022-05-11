import numpy as np

from higrid.utils import wavread
from higrid.emulate import emptyscene, emulatescene, combinescene

def composescene(filelist, dirset, higridpath, samples=(0, 96000), roomstr='ii-s05', datafromhigrid=True):
    """
    Compose an emulated scene using a number of anechoic sound signals and measured AIRs

    :param filelist: List of files to be used
    :param dirset: Set containing tuples with (X, Y, Z) as the AIR indices
    :param samples: Start and end points of samples to be prococessed as a tuple (sstart, send)
    :param roomstr: Used to select from a specific directory (default is 'ii-s05' as we only provided AIRs for that room)
    :return: 32 channels of audio from an emulated em32 recording.
    """

    drset = dirset.copy()
    assert len(filelist) == len(drset)
    numsamp = samples[1] - samples[0]
    sgo = emptyscene((32, numsamp))
    for item in filelist:
        dr = drset.pop()
        drtxt = str(dr[0]) + str(dr[1]) + str(dr[2])
        if datafromhigrid:
            snd = wavread(higridpath + '/data/sdata/anechoic/' + item)
        else:
            snd = wavread(item)
        snd = snd[0].reshape((snd[0].shape[0]))[samples[0]:samples[1]]
        gain = np.sqrt((dr[0]-3.0)**2 + (dr[1]-3.0)**2 + ((dr[2]-2.0)*0.6)**2)
        sg = emulatescene(snd, gain, higridpath +
                          '/data/rdata/' + roomstr + '/' + drtxt)
        sgo = combinescene(sgo, sg)
    return sgo