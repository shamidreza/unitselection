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
from collections import namedtuple
from scipy.fftpack import dct, idct
try:
    from matplotlib import pyplot as pp
except:
    print 'Could not load matplotlib'
# change corpus_path to refer to your local version of the repository
# It assumes there are 'wav', 'lab', and 'pm' directories are available in data
corpus_path = '/Users/hamid/Code/gitlab/voice-conversion/src/lib/arctic/cmu_us_slt_arctic'
# phoneme group info., for improving the search
phoneme_category = {'aa':'vowel_mid', #bot
                    'ae':'vowel_mid', #bat
                    'ah':'vowel_mid', #but
                    'ax':'vowel_mid', #but? (not available in ARPABET)
                    'ao':'vowel_mid', #bought
                    'aw':'vowel_mid', #bout
                    'ay':'vowel_mid', #bay
                    'eh':'vowel_front', # bet
                    'er':'vowel_front', # bird
                    'ey':'vowel_front', # bait
                    'ih':'vowel_front', #bit
                    'iy':'vowel_front', #beat
                    'ow':'vowel_back', #boat
                    'oy':'vowel_back', #boy
                    'uh':'vowel_back', #put
                    'uw':'vowel_back', #bood
                    'b':'plosive',
                    'p':'plosive',
                    'k':'plosive',
                    'g':'plosive',
                    't':'plosive',
                    'd':'plosive',
                    'w':'glide',
                    'y':'glide',
                    'l':'liquid',
                    'r':'liquid',
                    'dh':'fricative',#that
                    'th':'fricative',#think
                    's':'fricative',#sue
                    'z':'fricative',#zoo
                    'sh':'fricative',#she
                    'zh':'fricative',#vision
                    'v':'fricative',#van
                    'f':'fricative',#fan
                    'ch':'affricative',#chalk
                    'jh':'affricative',#jam
                    'h':'whisper',#ham
                    'hh':'whisper',#ham
                    'm':'nasal', #map
                    'n':'nasal', #nap
                    'ng':'nasal', #sing
                    'pau':'pau'
                    }

# each unit is stored like this:
# type: is it a demiphone? phone_L, or phone_R
# name: for demiphone: 'f_ih' for phone: 'ih'
# left_phone: left phoneme
# right_phone: right phoneme
# left_phone_category: The category (nasal, fricative, etc) of the left phone
# right_phone_category: The category (nasal, fricative, etc) of the right phone
# filename: wave filename
# starting_sample: the starting sample of the unit in the waveform
# ending_sample: the ending sample of the unit in the waveform
# overlap_starting_sample: the starting sample of the overlap unit in the waveform
# overlap_ending_sample: the ending sample of the overlap unit in the waveform
# left_CEP: cepstrum vector of the furthest left frame
# right_CEP: cepstrum vector of the furthest right frame
# unit_id: a unique ID assigned to the unit

Unit = namedtuple("Unit", "LR phone left_phone right_phone \
left_phone_category right_phone_category filename starting_sample \
ending_sample overlap_starting_sample overlap_ending_sample left_CEP \
right_CEP unit_id")

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

def extract_info(lab_path, wav_path, start_uid, file_number):
    times, labs = read_lab(lab_path)
    fs, wav = read_wav(wav_path)
    units = []
    if 0:
        # compute demiphones
        for i in range(1,len(labs)-2):
            id = labs[i]+'_'+labs[i+1]
            left_phone = labs[i-1]
            right_phone = labs[i+2]
            left_phone_cat = phoneme_category[left_phone]
            right_phon_cat = phoneme_category[right_phone]
            starting_sample = int(fs * (times[i]+times[i+1])/2)
            ending_sample = int(fs * (times[i+1]+times[i+2])/2)
            overlap_starting_sample = starting_sample - int(fs * (times[i+1]-times[i])/2)
            overlap_ending_sample = ending_sample + int(fs * (times[i+2]-times[i+1])/2)
            left_CEP = compute_cepstrum(wav[starting_sample:starting_sample+int(0.025*fs)])[1:21]
            right_CEP = compute_cepstrum(wav[ending_sample-int(0.025*fs):ending_sample])[1:21]
    
            cur_unit = Unit(type='demiphone', id=id, 
                            left_phone=left_phone, 
                            right_phone=right_phone,
                            left_phone_category=left_phone_cat, 
                            right_phone_category=right_phon_cat,
                            filename=wav_path, 
                            starting_sample=starting_sample,
                            ending_sample=ending_sample, 
                            overlap_starting_sample=overlap_starting_sample,
                            overlap_ending_sample=overlap_ending_sample,
                            left_CEP=left_CEP, right_CEP=right_CEP)
            units.append(cur_unit)
    for i in range(1,len(labs)-1):
        if 1: # compute left phones
            phone = labs[i]#+'_'+'*'
            left_phone = labs[i-1]
            right_phone = labs[i+1]
            left_phone_cat = phoneme_category[left_phone]
            right_phon_cat = phoneme_category[right_phone]
            starting_sample = int(fs * (times[i]))
            ending_sample = int(fs * (times[i]+times[i+1])/2)
            overlap_starting_sample = starting_sample - int(fs * (times[i+1]-times[i])/2)
            overlap_ending_sample = ending_sample + int(fs * (times[i+1]-times[i])/2)
            left_CEP = compute_cepstrum(wav[starting_sample:starting_sample+int(0.025*fs)])[1:21]
            right_CEP = compute_cepstrum(wav[ending_sample-int(0.025*fs):ending_sample])[1:21]
    
            cur_unit = Unit(LR='L', phone=phone, 
                            left_phone=left_phone, 
                            right_phone=right_phone,
                            left_phone_category=left_phone_cat, 
                            right_phone_category=right_phon_cat,
                            filename=file_number, 
                            starting_sample=starting_sample,
                            ending_sample=ending_sample, 
                            overlap_starting_sample=overlap_starting_sample,
                            overlap_ending_sample=overlap_ending_sample,
                            left_CEP=left_CEP, right_CEP=right_CEP, unit_id=start_uid+i*2)
            units.append(cur_unit)
        if 1: # compute right phones
            phone = labs[i]#+'_'+'*'
            left_phone = labs[i-1]
            right_phone = labs[i+1]
            left_phone_cat = phoneme_category[left_phone]
            right_phon_cat = phoneme_category[right_phone]
            starting_sample = int(fs * (times[i]+times[i+1])/2)
            ending_sample = int(fs * (times[i+1]))
            overlap_starting_sample = starting_sample - int(fs * (times[i+1]-times[i])/2)
            overlap_ending_sample = ending_sample + int(fs * (times[i+1]-times[i])/2)
            left_CEP = compute_cepstrum(wav[starting_sample:starting_sample+int(0.025*fs)])[1:21]
            right_CEP = compute_cepstrum(wav[ending_sample-int(0.025*fs):ending_sample])[1:21]
    
            cur_unit = Unit(LR='R', phone=phone, 
                            left_phone=left_phone, 
                            right_phone=right_phone,
                            left_phone_category=left_phone_cat, 
                            right_phone_category=right_phon_cat,
                            filename=file_number, 
                            starting_sample=starting_sample,
                            ending_sample=ending_sample, 
                            overlap_starting_sample=overlap_starting_sample,
                            overlap_ending_sample=overlap_ending_sample,
                            left_CEP=left_CEP, right_CEP=right_CEP,unit_id=start_uid+i*2+1)
            units.append(cur_unit)
          
    return units

def compute_cepstrum(wav_frame):
    spectrum=np.log(np.abs(np.fft.fft(wav_frame)))
    cep = dct(spectrum, norm='ortho')
    return cep

def get_filenames(file_extension):
    fnames = []
    #from glob import iglob
    #for fname in iglob(corpus_path+'/'+file_extension+'/*.'+file_extension):
        #fnames.append(fname.split('/')[-1].split('.')[0])
    for i in range(100):
        fnames.append('arctic_b'+str(i+1).zfill(4))
    return fnames

if __name__ == "__main__":
    fnames = get_filenames('lab')
    units = np.zeros(100000, 'object')
    cnt = 0
    file_counter = 0
    for fname in fnames:
        print 'Analyzing ' + fname
        cur_units = extract_info(corpus_path+'/lab/'+fname+'.lab', 
                                 corpus_path+'/wav/'+fname+'.wav',
                                 file_counter*500,
                                 file_counter)
        for j in xrange(len(cur_units)):
            units[cnt] = cur_units[j]
            cnt += 1
        file_counter += 1
    units = units[:cnt]
    import pickle
    f=open('units.pkl','w+')
    pickle.dump(units, f)
    pickle.dump(fnames, f)
    f.flush()
    f.close()
    print 'successfully pickled units in units.pkl!'