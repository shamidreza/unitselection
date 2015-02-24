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
    wavs = np.zeros((16000*30),dtype=np.int16)
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
    wavs = np.zeros((16000*30),dtype=np.int16)
    wavs_debug = np.zeros((16000*30,units.shape[0]),dtype=np.int16)
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
    
def pit2gci(pit_fname):
    times, pits, vox_times, vox_vals = read_hts_pit(pit_fname)
    ##pits += 50##
    gcis = np.zeros((10000))
    #gcis2 = np.zeros((1000000), dtype=np.uint32)

    cur = 0
    cur_pi = 0
    cur_vi = 0
    cnt = 0
    mean_p = pits.mean()
    std_p = pits.std()

    while True:
        if vox_vals[cur_vi] == 0: # unvoiced
            period = 1.0/(mean_p+np.random.normal()*std_p)
        else: # voiced
            period = (1.0/pits[cur_pi])
        print cur, period*16000
        cur+=period
        if cur > times[-1]:
            break
        gcis[cnt] = cur
        #gcis2[cur*16000] = 1
        # find next closest cur_vi        
        for i in xrange(5):
            if vox_times[cur_vi+i]<cur and vox_times[cur_vi+i+1]>cur:
                cur_vi = cur_vi+i
                break
        if vox_vals[cur_vi]: # if voiced, find next closest cur_pi
            closest_p = 10000000
            closest_pi = 10000000
            for i in xrange(5):
                if abs(1.0*times[min(cur_pi+i,pits.shape[0]-1)]-cur) < closest_p:
                    closest_p = abs(1.0*times[min(cur_pi+i,pits.shape[0]-1)]-cur)
                    closest_pi = cur_pi+i
                    
            assert closest_p != 10000000
            cur_pi = closest_pi
        
        cnt += 1
    gcis=gcis[:cnt]
    return gcis

def _select_gci_range(gcis, st, en):
    first_gci = 1000000
    
    for i in range(gcis.shape[0]):
        if gcis[i] > st:
            first_gci = i
            break
    assert  first_gci != 1000000
    last_gci = 1000000
    for j in range(first_gci, gcis.shape[0]):
        if gcis[j] > en:
            last_gci = j-1
            break
    if last_gci == 1000000:
        last_gci = gcis.shape[0]-1
    return first_gci, last_gci
        
def _psola(output_gcis, input_gcis, input_wav):
    num_input_frames = input_gcis.shape[0]-2
    num_output_frames = output_gcis.shape[0]-2
    out_wav = np.zeros((output_gcis[-1]-output_gcis[0]))
    out_wav_debug = np.zeros((output_gcis[-1]-output_gcis[0], 1000))

    for i in range(1, output_gcis.shape[0]-1):
        sample_out = (output_gcis[i]-output_gcis[0])/float(output_gcis[-1]-output_gcis[0])
        #sample_inp = input_gcis[i]/float(input_gcis[-1]-input_gcis[0])
        #sample_out = 1+int(sample_out*num_output_frames)
        sample_inp = 1+int(sample_out*num_input_frames)
        left_input_size = input_gcis[sample_inp]-input_gcis[sample_inp-1]
        left_output_size = output_gcis[i]-output_gcis[i-1]
        left_inp = input_wav[input_gcis[sample_inp-1]:input_gcis[sample_inp]]
        left_out = np.zeros(left_output_size)
        left_out[-1*min(left_input_size, left_output_size):] = \
            copy.deepcopy(left_inp[-1*min(left_input_size, left_output_size):])
        ##left_out *= np.linspace(0.0, 1.0, left_out.shape[0])
        left_out[-1*min(left_input_size, left_output_size):] *= \
            np.linspace(0.0, 1.0, min(left_input_size, left_output_size))
        right_input_size = input_gcis[sample_inp+1]-input_gcis[sample_inp]
        right_output_size = output_gcis[i+1]-output_gcis[i]
        right_inp = input_wav[input_gcis[sample_inp]:input_gcis[sample_inp+1]]
        right_out = np.zeros(right_output_size)
        right_out[:min(right_output_size,right_input_size)] = \
            copy.deepcopy(right_inp[:min(right_output_size,right_input_size)])
        ##right_out *= np.linspace(1.0, 0.0, right_out.shape[0])
        right_out[:min(right_output_size,right_input_size)] *= \
            np.linspace(1.0, 0.0, min(right_output_size,right_input_size))

        out_wav[output_gcis[i-1]-output_gcis[0]:output_gcis[i+1]-output_gcis[0]] += np.r_[left_out, right_out]
        out_wav_debug[output_gcis[i-1]-output_gcis[0]:output_gcis[i+1]-output_gcis[0], i-1] = np.r_[left_out, right_out]
    if 0: ## vis
        for j in range(output_gcis.shape[0]-2):
            pp.plot(out_wav_debug[:,j])
        pp.show()
    return out_wav
def concatenate_units_psola_nooverlap(units, fnames, times, gcis):
    wavs = np.zeros((16000*30),dtype=np.int16)
    wavs_debug = np.zeros((16000*30,units.shape[0]),dtype=np.int16)
    cur = 0
    i = 0
    cnt = 0
    while True:
        st = units[i].starting_sample

        en = 0
        j = i
        cur_dur = 0
        for j in range(i, units.shape[0]-1): # find consecutive
            if units[j].unit_id != units[j+1].unit_id-1:
                break
            cur_dur += (times[j+1]-times[j])

        cur_dur += (times[j+1]-times[j])
        #if j//2-1>=0:
        #    cur_dur += (times[1+j//2]-times[j//2])//2

        first_gci, last_gci = _select_gci_range(gcis, cur, cur+cur_dur)
        en= units[j].ending_sample
        en_ov= units[j].overlap_ending_sample
        wav_name=corpus_path+'/wav/'+fnames[units[i].filename]+'.wav'
        fs, wav = read_wav(wav_name)
        pm_name=corpus_path+'/pm/'+fnames[units[i].filename]+'.pm'
        cur_gcis = read_pm(pm_name)
        cur_gcis = np.array(cur_gcis)
        cur_gcis *= 16000.0
        #cur_wav = copy.deepcopy(wav[st:en])
        cur_first_gci, cur_last_gci = _select_gci_range(cur_gcis, st, en)
        cur_wav=_psola(gcis[first_gci:last_gci+1], cur_gcis[cur_first_gci:cur_last_gci+1], wav)
        wavs[gcis[first_gci]:gcis[last_gci]] += cur_wav.astype(np.int16)
        wavs_debug[gcis[first_gci]:gcis[last_gci], cnt] += cur_wav.astype(np.int16)
        #assert cur_dur == cur_wav.shape[0]
        #cur += (en-st)
        cur += (gcis[last_gci]-gcis[first_gci])        
        i = j + 1
        cnt += 1
        if i >= units.shape[0]:
            break
    if 0: ## vis
        for j in range(cnt):
            pp.plot(wavs_debug[:cur,j])
        pp.show()
    return wavs[:cur]
    