"""
Thi file is part of uniselection software.
Please refer to the LICENSE provided along the software (which is GPL v2,
http://www.gnu.org/licenses/gpl-2.0.html). 

This file includes the code for reading in the speech corpus (tested with
CMU-Arctic corpus) and extracting the unit information needed as part of 
the unit-selection engine.
"""
import numpy as np
from scipy.io.wavfile import read as wread
from os.path import exists

# change corpus_path to refer to your local version of the repository
# It assumes there are 'wav', 'lab', and 'pm' directories are available in data
corpus_path = '/Users/hamid/Code/gitlab/voice-conversion/src/lib/arctic/cmu_us_slt_arctic'
# phoneme group info., for improving the search
phoneme_group = {'phone':'class'}

def read_wav(wav_fname):
    if not exists(wav_fname):
        raise IOError, 'The following file does not exist: ' + wav_fname
    fs, wav = wread(wav_fname)
    return fs, wav

def read_lab(lab_fname):
    if not exists(lab_fname):
        raise IOError, 'The following file does not exist: ' + lab_fname
    f=open(lab_fname, 'r')
    times = [0.0]
    lab = []
    for line in f:
        line = line[:-1]
        if line == '#':
            continue
        pars=line.split(' ')
        times.append(float(pars[0]))
        lab.append(pars[-1])
        
    return times, lab

def read_pm(pm_fname):
    if not exists(pm_fname):
        raise IOError, 'The following file does not exist: ' + pm_fname
    f=open(pm_fname, 'r')
    times = []
    cnt = 0
    for line in f:
        line = line[:-1]
        if cnt < 6:
            cnt += 1
            continue
        pars=line.split(' ')
        times.append(float(pars[0]))
    return times

def extract_info(times, labs, wav, fs):
    diphones = []
    
    return units

def compute_cepstrum(wav_frame):
    return cep

def get_filenames(file_extension):
    from glob import iglob
    fnames = []
    for fname in iglob(corpus_path+'/'+file_extension+'/*.'+file_extension):
        fnames.append(fname.split('/')[-1].split('.')[0])
    return fnames

if __name__ == "__main__":
    fnames = get_filenames('lab')
    units = []
    for fname in fnames:
        times, labs = read_lab(corpus_path+'/lab/'+fname+'.lab')
        fs, wav = read_wav(corpus_path+'/wav/'+fname+'.wav')

        cur_units = extract_info(times, labs, wav, fs)
        pass