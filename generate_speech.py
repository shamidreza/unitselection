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

def hanning2(t0l, t0r):
    w = np.ones(t0l+t0r, dtype=np.float64)
    w[:t0l] = 0.5 - 0.5 * np.cos(np.pi * np.arange(t0l) / (t0l - 1)) if t0l > 1 else 0.5
    w[t0l:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(t0r) / (t0r - 1)) if t0r > 1 else 0.5
    return w
def hanning_mod2(t0l, t0r, alpha):
    w = np.ones(t0l+t0r, dtype=np.float64)
    n = int(np.round(t0l * alpha * 2))
    w[:n] = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / (n - 1))
    n = int(np.round(t0r * alpha * 2))
    w[-n:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(n) / (n - 1))
    return w
def encode_har(s, F0, t0l, t0r, Fs):
    def toeplitz(c):
        c = np.asarray(c).ravel()
        r = c.conjugate()
        vals = np.concatenate((r[-1:0:-1], c))
        a, b = np.ogrid[:len(c), len(r)-1:-1:-1]
        return vals[a + b]
    TwoPiJ = 2j * np.pi
    numH = Fs // (2 * F0) # sinusoids UP to this Fs (but not necessarily the actual Fs as below)
    DC = 1 # after Xiaochuan Niu
    if DC:
        l = np.arange(numH + 1)
    else:
        l = np.arange(numH) + 1
    n = np.arange(t0l + t0r) - t0l
    B = np.exp((TwoPiJ * F0 / Fs) * np.dot(n[:,np.newaxis], l[:,np.newaxis].T))
    w = hanning2(t0l, t0r)
    R = np.dot(np.diag(w), B)
    t = np.dot(np.conj(R.T), R[:,0])
    b = np.dot(np.conj(R.T), w * s)
    H = np.conj(np.linalg.solve(0.5 * toeplitz(t), b).T)
    if 0:
        shat = np.dot(B, np.conj(H.T)).real
        print(np.mean((s-shat)**2)**0.5) # RMS
        from matplotlib import pyplot as pp
        pp.plot(s)
        pp.plot(shat)
        pp.show()
    if DC:
        return H[1:] # don't return the DC component, just use it during fitting
    else:
        return H

    
def decode_har(H, F0, t0l, t0r, Fs):
    TwoPiJ = 2j * np.pi

    l = np.arange(len(H)) + 1 # no DC
    n = np.arange(t0l + t0r) - t0l
    B = np.exp((TwoPiJ * F0 / Fs) * np.dot(n[:,np.newaxis], l[:,np.newaxis].T))
    #s=real(exp((j*2*pi*F0/Fs).*((-t0l:t0r-1)'*(1:length(H))))*H');
    #s=real(exp((j*2*pi*F0/Fs).*((-t0l:t0r-1)'*(linspace(1,round(22050/2/F0),length(H)))))*H');
    return np.dot(B, np.conj(H.T)).real
def warp_har(H, inp_frm, out_frm, F0, t0l, t0r, Fs):
    M = np.abs(H)
    P = np.angle(H)
    P = np.unwrap(P)
    w1 = np.arange(1, len(H) + 1) * F0
    w2 = np.interp(w1, out_frm , inp_frm )
    M[:] = np.interp(w2, w1, M)
    P[:] = np.interp(w2, w1, P) # linear interpolation of unwrapped phase
    Hnew = M * np.exp(P * 1j)
    return Hnew
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
    
def pit2gci(times, pits, vox_times, vox_vals):
    #times, pits, vox_times, vox_vals = read_hts_pit(pit_fname)
    #pits += 20##
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
        #print cur, period*16000
        cur+=period
        if cur > times[-1]:
            break
        gcis[cnt] = cur
        #gcis2[cur*16000] = 1
        # find next closest cur_vi        
        for i in xrange(5):
            if cur_vi+i+1 < len(vox_times):
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

def units2gci(units, fnames):
    cur = 0
    i = 0
    cnt = 0
    gcis = []
    while True:
        st = units[i].starting_sample
        j = i
        for j in range(i, units.shape[0]-1): # find consecutive
            if units[j].unit_id != units[j+1].unit_id-1:
                break

        en= units[j].ending_sample
      
        pm_name=corpus_path+'/pm/'+fnames[units[i].filename]+'.pm'
        cur_gcis = read_pm(pm_name)
        cur_gcis = np.array(cur_gcis)
        cur_gcis *= 16000.0
        #cur_wav = copy.deepcopy(wav[st:en])
        cur_first_gci, cur_last_gci = _select_gci_range(cur_gcis, st, en)
        if not gcis:
            start_from = 0
        else:
            start_from = gcis[-1]
        gcis = gcis + (cur_gcis[cur_first_gci:cur_last_gci+1]-cur_gcis[cur_first_gci]+start_from).tolist()[1:]
        cur += (cur_gcis[cur_last_gci]-cur_gcis[cur_first_gci])        
        i = j + 1
        cnt += 1
        if i >= units.shape[0]:
            break
    return gcis
def units2dur(units, fnames):
    cur = 0
    i = 0
    times = [0.0]
    while True:
        st=units[i].starting_sample
        en= units[i].ending_sample
        times.append(times[-1]+en-st)
        i += 1
        if i >= units.shape[0]:
            break
    return times

def units2for(units, fnames, times, for_time, for_val):
    new_for_time = np.zeros(100000)
    new_for_val = np.zeros((100000,for_val.shape[1]))
    cur_new = 0
    cur = 0
    i = 0
    while True:
        st=units[i].starting_sample
        en= units[i].ending_sample
        ust = times[i]
        uen = times[i+1]
        ust_nearest = np.abs(ust-for_time).argmin()
        uen_nearest = np.abs(uen-for_time).argmin()
        st_nearest = cur_new
        en_nearest = st_nearest + (en-st)/80#framesize
        for k in range(for_val.shape[1]):
            new_for_val[st_nearest:en_nearest,k] = \
                np.interp(np.linspace(0.0,1.0,en_nearest-st_nearest),
                          np.linspace(0.0,1.0,uen_nearest-ust_nearest),
                          for_val[ust_nearest:uen_nearest,k])
        new_for_time[st_nearest:en_nearest] = new_for_time[cur_new-1] + 80 +\
            np.arange(en_nearest-st_nearest)*80
        cur_new += en_nearest-st_nearest
        i += 1
        if i >= units.shape[0]:
            break
    new_for_val = new_for_val[:cur_new,:]
    new_for_time = new_for_time[:cur_new]
    return new_for_time, new_for_val
def _select_gci_range(gcis, st, en):
    first_gci = 1000000
    
    for i in range(gcis.shape[0]):
        if gcis[i] > st:
            first_gci = i
            break
    assert first_gci != 1000000
    last_gci = 1000000
    for j in range(first_gci, gcis.shape[0]):
        if gcis[j] > en:
            last_gci = j-1
            break
    if last_gci == 1000000:
        last_gci = gcis.shape[0]-1
    return first_gci, last_gci
        
def _psola(output_gcis, input_gcis, input_wav):
    output_gcis = (output_gcis).astype(np.int32)
    input_gcis = (input_gcis).astype(np.int32)
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
        # linear
        #left_out[-1*min(left_input_size, left_output_size):] *= \
            #np.linspace(0.0, 1.0, min(left_input_size, left_output_size))
        # hanning
        left_out[-1*min(left_input_size, left_output_size):] *= \
            np.hanning(min(left_input_size, left_output_size)*2)[:min(left_input_size, left_output_size)]
        #left_out *= \
            #np.hanning(left_out.shape[0]*2)[:left_out.shape[0]]
            #np.linspace(0.0, 1.0, min(left_input_size, left_output_size))
        right_input_size = input_gcis[sample_inp+1]-input_gcis[sample_inp]
        right_output_size = output_gcis[i+1]-output_gcis[i]
        right_inp = input_wav[input_gcis[sample_inp]:input_gcis[sample_inp+1]]
        right_out = np.zeros(right_output_size)
        right_out[:min(right_output_size,right_input_size)] = \
            copy.deepcopy(right_inp[:min(right_output_size,right_input_size)])
        ##right_out *= np.linspace(1.0, 0.0, right_out.shape[0])
        # linear
        #right_out[:min(right_output_size,right_input_size)] *= \
            #np.linspace(1.0, 0.0, min(right_output_size,right_input_size))
        # hanning
        right_out[:min(right_output_size,right_input_size)] *= \
            np.hanning(min(right_output_size,right_input_size)*2)[min(right_output_size,right_input_size):]
        #right_out *= \
            #np.hanning(right_out.shape[0]*2)[right_out.shape[0]:]
       
            #np.linspace(1.0, 0.0, min(right_output_size,right_input_size))
            
        if 1: # true psola
            out_wav[output_gcis[i-1]-output_gcis[0]:output_gcis[i+1]-output_gcis[0]] += np.r_[left_out, right_out]
            out_wav_debug[output_gcis[i-1]-output_gcis[0]:output_gcis[i+1]-output_gcis[0], i-1] = np.r_[left_out, right_out]
        else: # only right
            out_wav[output_gcis[i]-output_gcis[0]:output_gcis[i+1]-output_gcis[0]] = right_out
            out_wav_debug[output_gcis[i]-output_gcis[0]:output_gcis[i+1]-output_gcis[0], i-1] = right_out
        
            
    if 0: ## vis
        ax=pp.subplot(311)
        pp.plot(out_wav)
        pp.plot(output_gcis-output_gcis[0], np.ones(output_gcis.shape[0])*2000, '*')

        pp.subplot(312,sharex=ax)
        for j in range(output_gcis.shape[0]-2):
            pp.plot(out_wav_debug[:,j])
        pp.plot(output_gcis-output_gcis[0], np.ones(output_gcis.shape[0])*2000, '*')
        pp.subplot(313,sharex=ax)
        pp.plot(input_wav[input_gcis[0]:input_gcis[-1]])
        pp.plot(input_gcis-input_gcis[0], np.ones(input_gcis.shape[0])*2000, '*')
        pp.show()
    return out_wav
def _psola_har(output_gcis, input_gcis, input_wav):
    output_gcis = (output_gcis).astype(np.int32)
    input_gcis = (input_gcis).astype(np.int32)
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
        # linear
        #left_out[-1*min(left_input_size, left_output_size):] *= \
            #np.linspace(0.0, 1.0, min(left_input_size, left_output_size))
        # hanning
        left_out[-1*min(left_input_size, left_output_size):] *= \
            np.hanning(min(left_input_size, left_output_size)*2)[:min(left_input_size, left_output_size)]
        #left_out *= \
            #np.hanning(left_out.shape[0]*2)[:left_out.shape[0]]
            #np.linspace(0.0, 1.0, min(left_input_size, left_output_size))
        right_input_size = input_gcis[sample_inp+1]-input_gcis[sample_inp]
        right_output_size = output_gcis[i+1]-output_gcis[i]
        right_inp = input_wav[input_gcis[sample_inp]:input_gcis[sample_inp+1]]
        right_out = np.zeros(right_output_size)
        right_out[:min(right_output_size,right_input_size)] = \
            copy.deepcopy(right_inp[:min(right_output_size,right_input_size)])
        ##right_out *= np.linspace(1.0, 0.0, right_out.shape[0])
        # linear
        #right_out[:min(right_output_size,right_input_size)] *= \
            #np.linspace(1.0, 0.0, min(right_output_size,right_input_size))
        # hanning
        right_out[:min(right_output_size,right_input_size)] *= \
            np.hanning(min(right_output_size,right_input_size)*2)[min(right_output_size,right_input_size):]
        #right_out *= \
            #np.hanning(right_out.shape[0]*2)[right_out.shape[0]:]
       
            #np.linspace(1.0, 0.0, min(right_output_size,right_input_size))
            
        if 1: # true psola
            t0l = left_out.shape[0]
            t0r = right_out.shape[0]
            f0 = 2.0*16000.0/(t0l+t0r)
            har=encode_har(np.r_[left_out, right_out], f0, t0l, t0r, 16000)
            ww=decode_har(har, f0, t0l, t0r, 16000)*hanning_mod2(t0l, t0r, 0.25)
            out_wav[output_gcis[i-1]-output_gcis[0]:output_gcis[i+1]-output_gcis[0]] += ww
            out_wav_debug[output_gcis[i-1]-output_gcis[0]:output_gcis[i+1]-output_gcis[0], i-1] = ww
        else: # only right
            out_wav[output_gcis[i]-output_gcis[0]:output_gcis[i+1]-output_gcis[0]] = right_out
            out_wav_debug[output_gcis[i]-output_gcis[0]:output_gcis[i+1]-output_gcis[0], i-1] = right_out
        
            
    if 0: ## vis
        ax=pp.subplot(311)
        pp.plot(out_wav)
        pp.plot(output_gcis-output_gcis[0], np.ones(output_gcis.shape[0])*2000, '*')

        pp.subplot(312,sharex=ax)
        for j in range(output_gcis.shape[0]-2):
            pp.plot(out_wav_debug[:,j])
        pp.plot(output_gcis-output_gcis[0], np.ones(output_gcis.shape[0])*2000, '*')
        pp.subplot(313,sharex=ax)
        pp.plot(input_wav[input_gcis[0]:input_gcis[-1]])
        pp.plot(input_gcis-input_gcis[0], np.ones(input_gcis.shape[0])*2000, '*')
        pp.show()
    return out_wav

def _psola_har_warp(output_gcis, input_gcis, input_wav, src_frm, trg_frm):
    output_gcis = (output_gcis).astype(np.int32)
    input_gcis = (input_gcis).astype(np.int32)
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
        # linear
        #left_out[-1*min(left_input_size, left_output_size):] *= \
            #np.linspace(0.0, 1.0, min(left_input_size, left_output_size))
        # hanning
        left_out[-1*min(left_input_size, left_output_size):] *= \
            np.hanning(min(left_input_size, left_output_size)*2)[:min(left_input_size, left_output_size)]
        #left_out *= \
            #np.hanning(left_out.shape[0]*2)[:left_out.shape[0]]
            #np.linspace(0.0, 1.0, min(left_input_size, left_output_size))
        right_input_size = input_gcis[sample_inp+1]-input_gcis[sample_inp]
        right_output_size = output_gcis[i+1]-output_gcis[i]
        right_inp = input_wav[input_gcis[sample_inp]:input_gcis[sample_inp+1]]
        right_out = np.zeros(right_output_size)
        right_out[:min(right_output_size,right_input_size)] = \
            copy.deepcopy(right_inp[:min(right_output_size,right_input_size)])
        ##right_out *= np.linspace(1.0, 0.0, right_out.shape[0])
        # linear
        #right_out[:min(right_output_size,right_input_size)] *= \
            #np.linspace(1.0, 0.0, min(right_output_size,right_input_size))
        # hanning
        right_out[:min(right_output_size,right_input_size)] *= \
            np.hanning(min(right_output_size,right_input_size)*2)[min(right_output_size,right_input_size):]
        #right_out *= \
            #np.hanning(right_out.shape[0]*2)[right_out.shape[0]:]
       
            #np.linspace(1.0, 0.0, min(right_output_size,right_input_size))
            
        if 1: # true psola
            t0l = left_out.shape[0]
            t0r = right_out.shape[0]
            f0 = 2.0*16000.0/(t0l+t0r)
            har=encode_har(np.r_[left_out, right_out], f0, t0l, t0r, 16000)
            if trg_frm.shape[0] != 0:
                cur_src_frm = np.r_[0.0]
                for k in range(4):
                    #cur_src_frm=np.r_[cur_src_frm, src_frm[int(src_frm.shape[0]*i/float(output_gcis.shape[0])),k]-\
                                    #src_frm[int(src_frm.shape[0]*i/float(output_gcis.shape[0])),k+4]/2.0 ]
                    #cur_src_frm=np.r_[cur_src_frm, src_frm[int(src_frm.shape[0]*i/float(output_gcis.shape[0])),k]+\
                                    #src_frm[int(src_frm.shape[0]*i/float(output_gcis.shape[0])),k+4]/2.0 ]
                    cur_src_frm=np.r_[cur_src_frm, src_frm[int(src_frm.shape[0]*i/float(output_gcis.shape[0])),k]]
                    
                cur_src_frm=np.r_[cur_src_frm, 8000.0]
                for k in range(cur_src_frm.shape[0]-1):
                    if cur_src_frm[k+1] < cur_src_frm[k]:
                        cur_src_frm[k+1] = cur_src_frm[k]
                cur_trg_frm = np.r_[0.0]
                for k in range(4):
                    #cur_trg_frm=np.r_[cur_trg_frm, trg_frm[int(trg_frm.shape[0]*i/float(output_gcis.shape[0])),k]-\
                                    #trg_frm[int(trg_frm.shape[0]*i/float(output_gcis.shape[0])),k+4]/2.0 ]
                    #cur_trg_frm=np.r_[cur_trg_frm, trg_frm[int(trg_frm.shape[0]*i/float(output_gcis.shape[0])),k]+\
                                    #trg_frm[int(trg_frm.shape[0]*i/float(output_gcis.shape[0])),k+4]/2.0 ]
                    cur_trg_frm=np.r_[cur_trg_frm, trg_frm[int(trg_frm.shape[0]*i/float(output_gcis.shape[0])),k]]
                    
                cur_trg_frm=np.r_[cur_trg_frm, 8000.0]
                for k in range(cur_trg_frm.shape[0]-1):
                    if cur_trg_frm[k+1] < cur_trg_frm[k]:
                        cur_trg_frm[k+1] = cur_trg_frm[k]
            else:
                cur_src_frm = np.r_[0.0]
                for k in range(4):
                    cur_src_frm=np.r_[cur_src_frm, src_frm[int(src_frm.shape[0]*i/float(output_gcis.shape[0])),k]-\
                                    src_frm[int(src_frm.shape[0]*i/float(output_gcis.shape[0])),k+4]/2.0 ]
                    cur_src_frm=np.r_[cur_src_frm, src_frm[int(src_frm.shape[0]*i/float(output_gcis.shape[0])),k]+\
                                    src_frm[int(src_frm.shape[0]*i/float(output_gcis.shape[0])),k+4]/2.0 ]
                cur_src_frm=np.r_[cur_src_frm, 8000.0]
                for k in range(cur_src_frm.shape[0]-1):
                    if cur_src_frm[k+1] < cur_src_frm[k]:
                        cur_src_frm[k+1] = cur_src_frm[k]
                cur_trg_frm = cur_src_frm.copy()
                           
            har2=warp_har(har, cur_src_frm, cur_trg_frm, f0, t0l, t0r, 16000)
            ww=decode_har(har2, f0, t0l, t0r, 16000)*hanning_mod2(t0l, t0r, 0.25)
            #sm=np.sum(ww**2)
            #ww/=np.abs(ww).max()##$
            #ww*=10000.0
            if np.abs(ww).max() > 20000:
                ww/= 2
            out_wav[output_gcis[i-1]-output_gcis[0]:output_gcis[i+1]-output_gcis[0]] += ww
            out_wav_debug[output_gcis[i-1]-output_gcis[0]:output_gcis[i+1]-output_gcis[0], i-1] = ww
        else: # only right
            out_wav[output_gcis[i]-output_gcis[0]:output_gcis[i+1]-output_gcis[0]] = right_out
            out_wav_debug[output_gcis[i]-output_gcis[0]:output_gcis[i+1]-output_gcis[0], i-1] = right_out
        
            
    if 0: ## vis
        ax=pp.subplot(311)
        pp.plot(out_wav)
        pp.plot(output_gcis-output_gcis[0], np.ones(output_gcis.shape[0])*2000, '*')

        pp.subplot(312,sharex=ax)
        for j in range(output_gcis.shape[0]-2):
            pp.plot(out_wav_debug[:,j])
        pp.plot(output_gcis-output_gcis[0], np.ones(output_gcis.shape[0])*2000, '*')
        pp.subplot(313,sharex=ax)
        pp.plot(input_wav[input_gcis[0]:input_gcis[-1]])
        pp.plot(input_gcis-input_gcis[0], np.ones(input_gcis.shape[0])*2000, '*')
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
    if 1: ## vis
        for j in range(cnt):
            pp.plot(wavs_debug[:cur,j])
        pp.show()
    return wavs[:cur]
    
def concatenate_units_psola_overlap(units, fnames, times, gcis, overlap=0.1):
    wavs = np.zeros((16000*30),dtype=np.int16)
    wavs_debug = np.zeros((16000*30,units.shape[0]),dtype=np.int16)
    cur = 0
    i = 0
    cnt = 0
    while True:
        st = units[i].starting_sample-int(overlap*(units[i].starting_sample-units[i].overlap_starting_sample))
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
        st_ov = int(overlap*(units[i].starting_sample-units[i].overlap_starting_sample))
        en_ov = int(overlap*(units[i].overlap_ending_sample-units[i].ending_sample))

        cur_ov = cur-st_ov
        if cur_ov < 0:
            cur_ov = 0
            st_ov = 0
        cur_dur_ov = cur_dur + en_ov

        first_gci, last_gci = _select_gci_range(gcis, cur, cur+cur_dur)
        first_gci_ov, last_gci_ov = _select_gci_range(gcis, cur_ov, cur+cur_dur_ov)

        en= units[j].ending_sample+en_ov
        wav_name=corpus_path+'/wav/'+fnames[units[i].filename]+'.wav'
        fs, wav = read_wav(wav_name)
        pm_name=corpus_path+'/pm/'+fnames[units[i].filename]+'.pm'
        cur_gcis = read_pm(pm_name)
        cur_gcis = np.array(cur_gcis)
        cur_gcis *= 16000.0
        #cur_wav = copy.deepcopy(wav[st:en])
        cur_first_gci, cur_last_gci = _select_gci_range(cur_gcis, st, en)
        cur_wav=_psola(gcis[first_gci_ov:last_gci_ov+1], cur_gcis[cur_first_gci:cur_last_gci+1], wav)
        
        wavs[gcis[first_gci_ov]:gcis[last_gci_ov]] += cur_wav.astype(np.int16)
        wavs_debug[gcis[first_gci]:gcis[last_gci], cnt] +=\
            cur_wav[gcis[first_gci]-gcis[first_gci_ov]:cur_wav.shape[0]-\
                    (gcis[last_gci_ov]-gcis[last_gci])].astype(np.int16)
        wavs_debug[gcis[first_gci_ov]:gcis[first_gci], cnt] +=\
            cur_wav[:gcis[first_gci]-gcis[first_gci_ov]].astype(np.int16) *\
            np.linspace(0.0, 1.0, gcis[first_gci]-gcis[first_gci_ov])
        wavs_debug[gcis[last_gci]:gcis[last_gci_ov], cnt] +=\
            cur_wav[cur_wav.shape[0]-(gcis[last_gci_ov]-gcis[last_gci]):].astype(np.int16) *\
            np.linspace(1.0, 0.0, gcis[last_gci_ov]-gcis[last_gci])
        
        #assert cur_dur == cur_wav.shape[0]
        #cur += (en-st)
        cur += (gcis[last_gci]-gcis[first_gci])        
        i = j + 1
        cnt += 1
        if i >= units.shape[0]:
            break
    if 1: ## vis
        for j in range(cnt):
            pp.plot(wavs_debug[:cur,j])
        pp.show()
    return wavs[:cur]
    
def concatenate_units_psola_har_overlap(units, fnames, old_times, times, gcis, trg_frm_time, trg_frm_val, overlap=0.5):
    wavs = np.zeros((16000*30),dtype=np.int16)
    wavs_debug = np.zeros((16000*30,units.shape[0]),dtype=np.int16)
    cur = 0
    i = 0
    cnt = 0
    frm1 =np.zeros((100000,4))
    frm2 =np.zeros((100000,4))
    frm_cnt = 0
    while True:
        st = units[i].starting_sample-int(overlap*(units[i].starting_sample-units[i].overlap_starting_sample))
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
        st_ov = int(overlap*(units[i].starting_sample-units[i].overlap_starting_sample))
        en_ov = int(overlap*(units[i].overlap_ending_sample-units[i].ending_sample))

        cur_ov = cur-st_ov
        if cur_ov < 0:
            cur_ov = 0
            st_ov = 0
        cur_dur_ov = cur_dur + en_ov

        first_gci, last_gci = _select_gci_range(gcis, cur, cur+cur_dur)
        first_gci_ov, last_gci_ov = _select_gci_range(gcis, cur_ov, cur+cur_dur_ov)

        en= units[j].ending_sample+en_ov
        wav_name=corpus_path+'/wav/'+fnames[units[i].filename]+'.wav'
        fs, wav = read_wav(wav_name)
       
        pm_name=corpus_path+'/pm/'+fnames[units[i].filename]+'.pm'
        cur_gcis = read_pm(pm_name)
        cur_gcis = np.array(cur_gcis)
        cur_gcis *= 16000.0
        #cur_wav = copy.deepcopy(wav[st:en])
        cur_first_gci, cur_last_gci = _select_gci_range(cur_gcis, st, en)
        ftime, fval = get_formant(wav, 16000)
        ftime *= 16000
        inp_frm_st = np.abs(cur_gcis[cur_first_gci]-ftime).argmin()
        inp_frm_en = np.abs(cur_gcis[cur_last_gci]-ftime).argmin()
        out_frm_st = np.abs(gcis[first_gci_ov]-trg_frm_time).argmin()
        out_frm_en = np.abs(gcis[last_gci_ov]-trg_frm_time).argmin()
        #cur_wav=_psola_har(gcis[first_gci_ov:last_gci_ov+1], cur_gcis[cur_first_gci:cur_last_gci+1], wav)
        #pp.plot(fval[inp_frm_st:inp_frm_en,:4],'b')
        #pp.plot(trg_frm_val[out_frm_st:out_frm_en,:4],'g')
        #pp.show()
        LEN = (out_frm_en-out_frm_st)
        if inp_frm_st+LEN > fval.shape[0]:
            LEN = fval.shape[0] - inp_frm_st
        if 1: # align for
            ust = old_times[i]
            uen = old_times[j+1]
            ust_nearest = np.abs(ust-trg_frm_time).argmin()
            uen_nearest = np.abs(uen-trg_frm_time).argmin()
            st_nearest = frm_cnt
            en_nearest = st_nearest + (en-st)/80#framesize
            new_for_val = np.zeros((LEN, 8))
            for k in range(trg_frm_val.shape[1]):
                new_for_val[:, k] = \
                    np.interp(np.linspace(0.0,1.0,LEN),
                              np.linspace(0.0,1.0,uen_nearest-ust_nearest),
                              trg_frm_val[ust_nearest:uen_nearest,k])
        

        #LEN = (inp_frm_en-inp_frm_st)
        #if LEN > (out_frm_en-out_frm_st):
            #LEN = (out_frm_en-out_frm_st)

        frm1[frm_cnt:frm_cnt+LEN,:] = fval[inp_frm_st:inp_frm_st+LEN,:4]
        #frm1[frm_cnt:frm_cnt+LEN,:] = (new_for_val[:, :4]-new_for_val[:, :4].mean(0))+fval[inp_frm_st:inp_frm_st+LEN,:4].mean(0)
        #frm2[frm_cnt:frm_cnt+LEN,:] = new_for_val[:, :4]
        frm2[frm_cnt:frm_cnt+LEN,:] = fval[inp_frm_st:inp_frm_st+LEN,:4]#(fval[inp_frm_st:inp_frm_st+LEN,:4]-fval[inp_frm_st:inp_frm_st+LEN,:4].mean(0))+new_for_val[:, :4].mean(0)
        frm2[frm_cnt:frm_cnt+LEN,:][frm2[frm_cnt:frm_cnt+LEN,:]<0]=0.0
        cur_wav = _psola_har_warp(gcis[first_gci_ov:last_gci_ov+1],
                                  cur_gcis[cur_first_gci:cur_last_gci+1],
                                  wav,
                                  frm1[frm_cnt:frm_cnt+LEN,:],
                                  frm2[frm_cnt:frm_cnt+LEN,:])
        wavs[gcis[first_gci_ov]:gcis[last_gci_ov]] += cur_wav.astype(np.int16)
        wavs_debug[gcis[first_gci]:gcis[last_gci], cnt] +=\
            cur_wav[gcis[first_gci]-gcis[first_gci_ov]:cur_wav.shape[0]-\
                    (gcis[last_gci_ov]-gcis[last_gci])].astype(np.int16)
        wavs_debug[gcis[first_gci_ov]:gcis[first_gci], cnt] +=\
            cur_wav[:gcis[first_gci]-gcis[first_gci_ov]].astype(np.int16) *\
            np.linspace(0.0, 1.0, gcis[first_gci]-gcis[first_gci_ov])
        wavs_debug[gcis[last_gci]:gcis[last_gci_ov], cnt] +=\
            cur_wav[cur_wav.shape[0]-(gcis[last_gci_ov]-gcis[last_gci]):].astype(np.int16) *\
            np.linspace(1.0, 0.0, gcis[last_gci_ov]-gcis[last_gci])
        
        #assert cur_dur == cur_wav.shape[0]
        #cur += (en-st)
        frm_cnt += LEN

        cur += (gcis[last_gci]-gcis[first_gci])        
        i = j + 1
        cnt += 1
        print i, units.shape[0]
        if i >= units.shape[0]:
            break
    if 1: ## vis
        frm1 = frm1[:frm_cnt,:]
        frm2 = frm2[:frm_cnt,:]
        pp.plot(frm1,'b')
        pp.plot(frm2,'g')
        pp.show()

        for j in range(cnt):
            pp.plot(wavs_debug[:cur,j])
        pp.show()
    return wavs[:cur]
    
if __name__ == "__main__":
    fname = 'arctic_a0007'
    lab_name=corpus_path+'/lab/'+fname+'.lab'
    wav_name=corpus_path+'/wav/'+fname+'.wav'
    pm_name=corpus_path+'/pm/'+fname+'.pm'
    inp_gcis=read_pm(pm_name)
    inp_gcis = np.array(inp_gcis)
    time, pit = gci2pit(inp_gcis)
    #times, pits, vox_times, vox_vals = pit_nozero_2_pit_vox(time, pit)
    times=time
    pits=pit
    vox_times=[0.0, time[-1]]
    vox_vals = [1.0]
    inp_gcis=pit2gci(times, pits, vox_times, vox_vals)
    out_gcis=pit2gci(times, pits, vox_times, vox_vals)
    inp_gcis *= 16000
    out_gcis *= 16000

    wav_name=corpus_path+'/wav/'+fname+'.wav'
    fs, wav = read_wav(wav_name)
    out_wav = _psola(out_gcis, inp_gcis, wav)
    out_wav = out_wav.astype(np.int16)
    #pp.plot(out_wav);pp.show()
    
    from scipy.io.wavfile import write as wwrite
    wwrite('out.wav', 16000, out_wav)
    print 'successfully saved out.wav'