"""
Author: Seyed Hamidreza Mohammadi
This file is part of the shamidreza/uniselection software.
Please refer to the LICENSE provided alongside the software (which is GPL v2,
http://www.gnu.org/licenses/gpl-2.0.html). 

This file includes the code for some utility function.
"""

from scipy.io.wavfile import read as wread
from os.path import exists
from scipy.fftpack import dct, idct
import numpy as np
try:
    from matplotlib import pyplot as pp
except:
    print 'Could not load matplotlib'


# change corpus_path to refer to your local version of the repository
# It assumes there are 'wav', 'lab', and 'pm' directories are available in data
corpus_path = '/Users/hamid/Code/gitlab/voice-conversion/src/lib/arctic/cmu_us_slt_arctic'

def read_wav(wav_fname):
    if not exists(wav_fname):
        raise IOError, 'The following file does not exist: ' + wav_fname
    fs, wav = wread(wav_fname)
    return fs, wav


def read_lab(lab_fname):
    if not exists(lab_fname):
        raise IOError, 'The following file does not exist: ' + lab_fname
    f = open(lab_fname, 'r')
    times = [0.0]
    lab = []
    for line in f:
        line = line[:-1]
        if line == '#':
            continue
        pars = line.split(' ')
        times.append(float(pars[0]))
        lab.append(pars[-1])

    return times, lab


def read_pm(pm_fname):
    if not exists(pm_fname):
        raise IOError, 'The following file does not exist: ' + pm_fname
    f = open(pm_fname, 'r')
    times = []
    cnt = 0
    for line in f:
        line = line[:-1]
        if cnt < 6:
            cnt += 1
            continue
        pars = line.split(' ')
        times.append(float(pars[0]))
    return times

def compute_cepstrum(wav_frame):
    spectrum = np.log(np.abs(np.fft.fft(wav_frame)))
    cep = dct(spectrum, norm='ortho')
    return cep


def get_filenames(file_extension):
    fnames = []
    #from glob import iglob
    # for fname in iglob(corpus_path+'/'+file_extension+'/*.'+file_extension):
    # fnames.append(fname.split('/')[-1].split('.')[0])
    for i in range(100):
        fnames.append('arctic_b' + str(i + 1).zfill(4))
    return fnames
