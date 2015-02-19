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
def read_hts_dur(hts_dur_file):
    f=open(hts_dur_file, 'r')
    time = [0.0]
    phonemes = []
    while True:
        f.readline();f.readline();f.readline();f.readline();f.readline()
        line=f.readline()
        if line == '':
            break
        inx=line.find('duration')
        num_frames=int(line[inx+9:inx+12].split(' ')[0])
        st=line.find('-')
        en=line.find('+')
        phoneme = line[st+1:en]
        check = False
        orig = phoneme
        if orig == 'd' or orig == 't' or orig == 'p' or orig == 'b' \
           or orig == 'k' or orig == 'g' or orig == 'ch' or orig == 'jh': 
            check = True
        #if check: # if affreicative (needs to be split to closure and burst)
            #phoneme_closure = phoneme +  'c'
            #num_frames_closure = num_frames // 2
            #num_frames -= num_frames_closure
            #phonemes.append(cmuclosure_to_worldbet[phoneme_closure])
            #phonemes.append(cmuclosure_to_worldbet[phoneme])
            #time.append(num_frames_closure+time[-1])
            #time.append(num_frames+time[-1])
	if True: # if affreicative (needs to be split to closure and burst)
            #phoneme_closure = phoneme +  'c'
            #num_frames_closure = num_frames // 2
            #num_frames -= num_frames_closure
            #phonemes.append(cmuclosure_to_worldbet[phoneme_closure])
            phonemes.append(phoneme)#cmuclosure_to_worldbet[phoneme])
            #time.append(num_frames_closure+time[-1])
            time.append(num_frames+time[-1])

            #print '***'
            #print phonemes[-2], time[-2]
            #print phonemes[-1], time[-1]


        else:
            phonemes.append(phoneme)#cmuclosure_to_worldbet[phoneme])
            time.append(num_frames+time[-1])
            #print phonemes[-1], time[-1]
    duration = time[-1]
    for i in range(len(time)):
        time[i] *= (0.005)
    #for i in range(len(phonemes)):
        #print phonemes[i], time[i], time[i+1]
    
    value = np.array(phonemes, dtype=unicode)
    time = (np.array(time))#.astype(np.int32)
    return time, value

def read_hts_pit(hts_pit_file):
    f = open(hts_pit_file, 'rb')
    pit = np.zeros(100000)
    cnt = 0
    import struct
    while True:
	x = f.read(4)
	if x == '':
	    break	   
	pit[cnt] = struct.unpack('f', x)[0]
	cnt += 1
    pit=pit[:cnt]
    f.close()
    pit = np.exp(pit)
    #pit[pit!=0] = 48000.0/pit[pit!=0]
    time_pit = np.linspace(0, cnt*(0.005), pit.shape[0])
    pit_no_zero = pit[pit!=0]
    time_pit_no_zero = time_pit[pit!=0]#.astype(np.int32)
    #pit_track = track.TimeValue(time_pit_no_zero, pit_no_zero, 16000, int(duration*(0.005)*16000+1))
    vox_val = []
    vox_time = [0]
    in_voiced_region = False
    for i in range(pit.shape[0]):
	if pit[i] > 0 and not in_voiced_region:
	    vox_time.append(time_pit[i-1])
	    vox_val.append(0)
	    in_voiced_region = True
	if pit[i] == 0 and in_voiced_region:
	    vox_time.append(time_pit[i-1])
	    vox_val.append(1)
	    in_voiced_region = False
    return time_pit_no_zero, pit_no_zero, np.array(vox_time), np.array(vox_val, np.int32)
    
def read_hts_for(hts_for_file):
    f = open(hts_for_file, 'rb')
    frm = np.zeros((10000,8))
    cnt = 0
    import struct
    while True:
	for j in range(8):
	    x = f.read(4)
	    if x == '':
		break	   
	    frm[cnt, j] = struct.unpack('f', x)[0]
	if x == '':
	    break
	cnt += 1
    frm=frm[:cnt,:]
    f.close()
    
    time = np.linspace(0, cnt*(0.005), frm.shape[0])
    
    return time, frm
