"""
Author: Seyed Hamidreza Mohammadi
This file is part of the shamidreza/uniselection software.
Please refer to the LICENSE provided alongside the software (which is GPL v2,
http://www.gnu.org/licenses/gpl-2.0.html). 

This file includes the code for reading in the speech corpus (tested with
CMU-Arctic corpus) and extracting the unit information needed as part of 
the unit-selection engine.
"""
import numpy as np
from collections import namedtuple
from utils import *

# phoneme group info., for improving the search
phoneme_category = {'aa': 'vowel_mid',  # bot
                    'ae': 'vowel_mid',  # bat
                    'ah': 'vowel_mid',  # but
                    'ax': 'vowel_mid',  # but? (not available in ARPABET)
                    'ao': 'vowel_mid',  # bought
                    'aw': 'vowel_mid',  # bout
                    'ay': 'vowel_mid',  # bay
                    'eh': 'vowel_front',  # bet
                    'er': 'vowel_front',  # bird
                    'ey': 'vowel_front',  # bait
                    'ih': 'vowel_front',  # bit
                    'iy': 'vowel_front',  # beat
                    'ow': 'vowel_back',  # boat
                    'oy': 'vowel_back',  # boy
                    'uh': 'vowel_back',  # put
                    'uw': 'vowel_back',  # bood
                    'b': 'plosive',
                    'p': 'plosive',
                    'k': 'plosive',
                    'g': 'plosive',
                    't': 'plosive',
                    'd': 'plosive',
                    'w': 'glide',
                    'y': 'glide',
                    'l': 'liquid',
                    'r': 'liquid',
                    'dh': 'fricative',  # that
                    'th': 'fricative',  # think
                    's': 'fricative',  # sue
                    'z': 'fricative',  # zoo
                    'sh': 'fricative',  # she
                    'zh': 'fricative',  # vision
                    'v': 'fricative',  # van
                    'f': 'fricative',  # fan
                    'ch': 'affricative',  # chalk
                    'jh': 'affricative',  # jam
                    'h': 'whisper',  # ham
                    'hh': 'whisper',  # ham
                    'm': 'nasal',  # map
                    'n': 'nasal',  # nap
                    'ng': 'nasal',  # sing
                    'pau': 'pau'
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

Unit_old = namedtuple("Unit", "LR phone left_phone right_phone \
left_phone_category right_phone_category filename starting_sample \
ending_sample overlap_starting_sample overlap_ending_sample left_CEP \
right_CEP pit unit_id")

class Unit:
    def __init__(self, LR, phone, left_phone, right_phone, \
                 left_phone_category, right_phone_category, filename, starting_sample, \
                 ending_sample, overlap_starting_sample, overlap_ending_sample, left_CEP, \
                 right_CEP, pit, unit_id):
        self.LR = LR
        self.phone = phone
        self.left_phone = left_phone
        self.right_phone = right_phone
        self.left_phone_category = left_phone_category
        self.right_phone_category = right_phone_category
        self.filename = filename
        self.starting_sample = starting_sample
        self.ending_sample = ending_sample
        self.overlap_starting_sample = overlap_starting_sample
        self.overlap_ending_sample = overlap_ending_sample
        self.left_CEP = left_CEP
        self.right_CEP = right_CEP
        self.pit = pit
        self.unit_id = unit_id
        
def extract_info(lab_path, wav_path, start_uid, file_number):
    times, labs = read_lab(lab_path)
    fs, wav = read_wav(wav_path)
    units = []
    if 0:
        # compute demiphones
        for i in range(1, len(labs) - 2):
            id = labs[i] + '_' + labs[i + 1]
            left_phone = labs[i - 1]
            right_phone = labs[i + 2]
            left_phone_cat = phoneme_category[left_phone]
            right_phon_cat = phoneme_category[right_phone]
            starting_sample = int(fs * (times[i] + times[i + 1]) / 2)
            ending_sample = int(fs * (times[i + 1] + times[i + 2]) / 2)
            overlap_starting_sample = starting_sample - \
                int(fs * (times[i + 1] - times[i]) / 2)
            overlap_ending_sample = ending_sample + \
                int(fs * (times[i + 2] - times[i + 1]) / 2)
            left_CEP = compute_cepstrum(
                wav[starting_sample:starting_sample + int(0.025 * fs)])[1:21]
            right_CEP = compute_cepstrum(
                wav[ending_sample - int(0.025 * fs):ending_sample])[1:21]

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
        if 1:  # compute left phones
            phone = labs[i]+'_L'  # +'_'+'*'
            left_phone = labs[max(i-1,0)]##labs[i - 1]
            right_phone = labs[min(i+1,len(labs)-1)]##labs[i + 1]
            left_phone_cat = phoneme_category[left_phone]
            right_phone_cat = phoneme_category[right_phone]
            starting_sample = int(fs * (times[i]))
            ending_sample = int(fs * (times[i] + times[i + 1]) / 2)
            overlap_starting_sample = starting_sample - \
                int(fs * (times[i + 1] - times[i]) / 2)
            overlap_ending_sample = ending_sample + \
                int(fs * (times[i + 1] - times[i]) / 2)
            left_CEP = compute_cepstrum(
                wav[starting_sample:starting_sample + int(0.025 * fs)])[1:21]
            right_CEP = compute_cepstrum(
                wav[max(0,ending_sample - int(0.025 * fs)):ending_sample])[1:21]

            if ending_sample-starting_sample > 400:
                pit = compute_f0(wav[starting_sample:ending_sample],16000)
            else:
                pit = compute_f0(wav[overlap_starting_sample:overlap_ending_sample],16000)

            cur_unit = Unit(LR='L', phone=phone,
                            left_phone=left_phone,
                            right_phone=right_phone,
                            left_phone_category=left_phone_cat,
                            right_phone_category=right_phone_cat,
                            filename=file_number,
                            starting_sample=starting_sample,
                            ending_sample=ending_sample,
                            overlap_starting_sample=overlap_starting_sample,
                            overlap_ending_sample=overlap_ending_sample,
                            left_CEP=left_CEP, right_CEP=right_CEP, pit=pit, unit_id=start_uid + i * 2)
            units.append(cur_unit)
        if 1:  # compute right phones
            phone = labs[i]+'_R'   # +'_'+'*'
            left_phone = labs[max(i-1,0)]
            right_phone = labs[min(i+1,len(labs)-1)]
            left_phone_cat = phoneme_category[left_phone]
            right_phone_cat = phoneme_category[right_phone]
            starting_sample = int(fs * (times[i] + times[i + 1]) / 2)
            ending_sample = int(fs * (times[i + 1]))
            overlap_starting_sample = starting_sample - \
                int(fs * (times[i + 1] - times[i]) / 2)
            overlap_ending_sample = ending_sample + \
                int(fs * (times[i + 1] - times[i]) / 2)
            left_CEP = compute_cepstrum(
                wav[starting_sample:starting_sample + int(0.025 * fs)])[1:21]
            right_CEP = compute_cepstrum(
                wav[max(0,ending_sample - int(0.025 * fs)):ending_sample])[1:21]
            if ending_sample-starting_sample > 400:
                pit = compute_f0(wav[starting_sample:ending_sample],16000)
            else:
                pit = compute_f0(wav[overlap_starting_sample:overlap_ending_sample],16000)

            cur_unit = Unit(LR='R', phone=phone,
                            left_phone=left_phone,
                            right_phone=right_phone,
                            left_phone_category=left_phone_cat,
                            right_phone_category=right_phone_cat,
                            filename=file_number,
                            starting_sample=starting_sample,
                            ending_sample=ending_sample,
                            overlap_starting_sample=overlap_starting_sample,
                            overlap_ending_sample=overlap_ending_sample,
                            left_CEP=left_CEP, right_CEP=right_CEP, pit=pit, unit_id=start_uid + i * 2 + 1)
            units.append(cur_unit)

    return units

def read_input_lab(lab_path, pit_path):
    ##times, labs = read_lab(lab_path)
    times, labs = read_hts_dur(lab_path)
    tp, p, = read_hts_pit_withzero(pit_path)
    units = []
    times_units = [0.0]
    for i in range(len(labs)):
        if 1:  # compute left phones
            phone = labs[i]+'_L'  # +'_'+'*'
            left_phone = labs[max(i-1,0)]##labs[i - 1]
            right_phone = labs[min(i+1,len(labs)-1)]##labs[i + 1]
            left_phone_cat = phoneme_category[left_phone]
            right_phone_cat = phoneme_category[right_phone]
            starting_sample = None
            ending_sample = None
            overlap_starting_sample = None
            overlap_ending_sample = None
            left_CEP = None
            right_CEP = None
            ix = times[i]+(times[i+1]-times[i])/4.0
            pit=p[np.abs(tp-ix).argmin()]
            
            file_number = None
            
            cur_unit = Unit(LR='L', phone=phone,
                            left_phone=left_phone,
                            right_phone=right_phone,
                            left_phone_category=left_phone_cat,
                            right_phone_category=right_phone_cat,
                            filename=file_number,
                            starting_sample=starting_sample,
                            ending_sample=ending_sample,
                            overlap_starting_sample=overlap_starting_sample,
                            overlap_ending_sample=overlap_ending_sample,
                            left_CEP=left_CEP, right_CEP=right_CEP, pit=pit, unit_id=None)
            units.append(cur_unit)
            times_units.append(times[i]+(times[i+1]-times[i])/2.0)
        if 1:  # compute right phones
            phone = labs[i]+'_R'   # +'_'+'*'
            left_phone = labs[max(i-1,0)]
            right_phone = labs[min(i+1,len(labs)-1)]
            left_phone_cat = phoneme_category[left_phone]
            right_phone_cat = phoneme_category[right_phone]
            starting_sample = None
            ending_sample = None
            overlap_starting_sample = None
            overlap_ending_sample = None
            left_CEP = None
            right_CEP = None
            file_number = None
            ix = times[i]+3.0*(times[i+1]-times[i])/4.0
            pit=p[np.abs(tp-ix).argmin()]
            
            cur_unit = Unit(LR='R', phone=phone,
                            left_phone=left_phone,
                            right_phone=right_phone,
                            left_phone_category=left_phone_cat,
                            right_phone_category=right_phone_cat,
                            filename=file_number,
                            starting_sample=starting_sample,
                            ending_sample=ending_sample,
                            overlap_starting_sample=overlap_starting_sample,
                            overlap_ending_sample=overlap_ending_sample,
                            left_CEP=left_CEP, right_CEP=right_CEP, pit=pit, unit_id=None)
            units.append(cur_unit)
            times_units.append(times[i+1])

    return units, times_units



if __name__ == "__main__":
    fnames = get_filenames('lab')
    units = np.zeros(100000, 'object')
    cnt = 0
    file_counter = 0
    for fname in fnames:
        print 'Analyzing ' + fname
        cur_units = extract_info(corpus_path + '/lab/' + fname + '.lab',
                                 corpus_path + '/wav/' + fname + '.wav',
                                 file_counter * 500,
                                 file_counter)
        for j in xrange(len(cur_units)):
            units[cnt] = cur_units[j]
            cnt += 1
        file_counter += 1
    units = units[:cnt]
    import pickle
    f = open('units.pkl', 'w+')
    pickle.dump(units, f)
    pickle.dump(fnames, f)
    f.flush()
    f.close()
    print 'successfully pickled units in units.pkl!'
