"""
Author: Seyed Hamidreza Mohammadi
This file is part of the shamidreza/uniselection software.
Please refer to the LICENSE provided alongside the software (which is GPL v2,
http://www.gnu.org/licenses/gpl-2.0.html). 

This file includes the code for synthesizing speech as output by getting the 
unit sequence as input.

input:
1- unit sequence
2- target duration (not yet)
3- target pitch (not yet)
4- target formants (not yet)
output:
1- speech
"""
from utils import *
from extract_unit_info import *
import copy

def concatenate_units_nooverlap(units, fnames):
    wavs = np.zeros((16000*10),dtype=np.int16)
    cur = 0
    i = 0
    while True:
        st = units[i].starting_sample
        st_ov = units[i].overlap_starting_sample

        en = 0
        j = i
        for j in range(i, units.shape[0]-1):
            if units[j].unit_id != units[j+1].unit_id-1:
                break
        en= units[j].ending_sample
        en_ov= units[j].overlap_ending_sample
        wav_name=corpus_path+'/wav/'+fnames[units[i].filename]+'.wav'
        fs, wav = read_wav(wav_name)
        wavs[cur:cur+(en-st)] = wav[st:en]
        cur += (en-st)
                
        i = j + 1
        if i >= units.shape[0]:
            break
    return wavs[:cur]
    
    
def concatenate_units_overlap(units, fnames, overlap=0.2):
    wavs = np.zeros((16000*10),dtype=np.int16)
    wavs_debug = np.zeros((16000*10,units.shape[0]),dtype=np.int16)
    cur = 0
    i = 0
    while True:
        st = units[i].starting_sample
        st_ov = units[i].overlap_starting_sample

        en = 0
        j = i
        for j in range(i, units.shape[0]-1):
            if units[j].unit_id != units[j+1].unit_id-1:
                break
        en= units[j].ending_sample
        en_ov= units[j].overlap_ending_sample
        wav_name=corpus_path+'/wav/'+fnames[units[i].filename]+'.wav'
        fs, wav = read_wav(wav_name)
        cur_wav = copy.deepcopy(wav[st-int(overlap*abs(st_ov-st)):en+int(overlap*abs(en_ov-en))])
        cur_wav[:int(overlap*abs(st_ov-st))] *= np.linspace(0.0,1.0,int(overlap*abs(st_ov-st)))
        cur_wav[-int(overlap*abs(en_ov-en)):] *= np.linspace(1.0,0.0,int(overlap*abs(en_ov-en)))
        if cur-int(overlap*abs(st_ov-st)) < 0:
            wavs[:cur-int(overlap*abs(st_ov-st))+cur_wav.shape[0]] += \
                cur_wav[-(cur-int(overlap*abs(st_ov-st))+cur_wav.shape[0]):]
        else:
            wavs[cur-int(overlap*abs(st_ov-st)):cur-int(overlap*abs(st_ov-st))+cur_wav.shape[0]] += cur_wav
        cur += (en-st)
                
        i = j + 1
        if i >= units.shape[0]:
            break
    return wavs[:cur]

def concatenate_units_duration_overlap(units, fnames, times, overlap=0.2):
    wavs = np.zeros((16000*10),dtype=np.int16)
    wavs_debug = np.zeros((16000*10,units.shape[0]),dtype=np.int16)
    cur = 0
    i = 0
    while True:
        st = units[i].starting_sample
        st_ov = units[i].overlap_starting_sample

        en = 0
        j = i
        for j in range(i, units.shape[0]-1):
            if units[j].unit_id != units[j+1].unit_id-1:
                break
        en= units[j].ending_sample
        en_ov= units[j].overlap_ending_sample
        wav_name=corpus_path+'/wav/'+fnames[units[i].filename]+'.wav'
        fs, wav = read_wav(wav_name)
        cur_wav = copy.deepcopy(wav[st-int(overlap*abs(st_ov-st)):en+int(overlap*abs(en_ov-en))])
        cur_wav[:int(overlap*abs(st_ov-st))] *= np.linspace(0.0,1.0,int(overlap*abs(st_ov-st)))
        cur_wav[-int(overlap*abs(en_ov-en)):] *= np.linspace(1.0,0.0,int(overlap*abs(en_ov-en)))
        if cur-int(overlap*abs(st_ov-st)) < 0:
            wavs[:cur-int(overlap*abs(st_ov-st))+cur_wav.shape[0]] += \
                cur_wav[-(cur-int(overlap*abs(st_ov-st))+cur_wav.shape[0]):]
        else:
            wavs[cur-int(overlap*abs(st_ov-st)):cur-int(overlap*abs(st_ov-st))+cur_wav.shape[0]] += cur_wav
        cur += (en-st)
                
        i = j + 1
        if i >= units.shape[0]:
            break
    return wavs[:cur]
    
    

        
        
        