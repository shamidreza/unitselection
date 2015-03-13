"""
Author: Seyed Hamidreza Mohammadi
This file is part of the shamidreza/uniselection software.
Please refer to the LICENSE provided alongside the software (which is GPL v2,
http://www.gnu.org/licenses/gpl-2.0.html). 

This file includes the code for putting all the pieces together.
"""

from utils import *
from extract_unit_info import *
from search import *
from generate_speech import *

if __name__ == "__main__":
    if 0: # test pit2gci
        pit_file='/Users/hamid/Code/hts/HTS-demo_CMU-ARCTIC-SLT2/gen/qst001/ver1/2mix/2/alice01.lf0'
        target_gci = pit2gci(pit_file)
        
        
    if 1: # test read_dur,pit,for methods
        dur_file='/Users/hamid/Code/hts/HTS-demo_CMU-ARCTIC-SLT2/gen/qst001/ver1/2mix/2/alice01.dur'
        for_file='/Users/hamid/Code/hts/HTS-demo_CMU-ARCTIC-SLT2/gen/qst001/ver1/2mix/2/alice01.for'
        pit_file='/Users/hamid/Code/hts/HTS-demo_CMU-ARCTIC-SLT2/gen/qst001/ver1/2mix/2/alice01.lf0'

        #a=read_hts_for(for_file)
        #b=read_hts_pit(pit_file)
        #c=read_hts_dur(dur_file)

    fname = 'arctic_a0001'
    lab_name=corpus_path+'/lab/'+fname+'.lab'
    wav_name=corpus_path+'/wav/'+fname+'.wav'
    pm_name=corpus_path+'/pm/'+fname+'.pm'

    ##target_units = load_input(lab_name)
    #times, labs = read_lab(lab_name)
    ##tmp_units=extract_info(lab_name, wav_name, 0,0)
    times, pits, vox_times, vox_vals = read_hts_pit(pit_file)
    frm_time, frm_val = read_hts_for(for_file)
    gcis=pit2gci(times, pits, vox_times, vox_vals)
    tmp_units, times=read_input_lab(dur_file, pit_file)
    #tmp_units = tmp_units[128:140]##
    
    target_units = np.zeros(len(tmp_units), 'object')
    for j in xrange(len(tmp_units)):
        target_units[j] = tmp_units[j]
    if 0:
        units, fnames=load_units()
        units = units[:int(units.shape[0]*(100.0/100.0))]
        best_units_indice=search(target_units, units,limit=20)
        best_units = units[best_units_indice]
        f=open('tmp2.pkl','w+')
        import pickle
        pickle.dump(best_units,f)
        pickle.dump(fnames,f)
        f.flush()
        f.close()
    else:
        f=open('tmp2.pkl','r')
        import pickle
        best_units=pickle.load(f)
        fnames=pickle.load(f)
        #best_units = best_units[128:140]##
        f.close()
    for i in xrange(target_units.shape[0]):
        print target_units[i].phone, best_units[i].phone, best_units[i].unit_id
    #wavs=concatenate_units_overlap(best_units, fnames)
    #gcis = gcis[(gcis>times[128]) * (gcis<times[140])]
    #gcis -= times[128]
    frm_time, frm_val = units2for(best_units, fnames, times, frm_time, frm_val)
    gcis=units2gci(best_units, fnames)##$
    gcis = np.array(gcis)
    ##$gcis *= 16000
    gcis = gcis.astype(np.uint32)
    times=units2dur(best_units, fnames)##$
    times = np.array(times)
    ##$times *= 16000
    times = times.astype(np.uint32)
    
    
    #times = times[128:141]##
    #aa=times[0]##
    #for i in range(len(times)):##
        #times[i] -= aa##
    #frm_time *= 16000
    wavs=concatenate_units_psola_har_overlap(best_units, fnames, times, gcis, frm_time, frm_val, overlap=0.2)
    #wavs=concatenate_units_nooverlap(best_units, fnames)
    ftime, fval = get_formant(wavs, 16000)
    from scipy.io.wavfile import write as wwrite
    wwrite('out.wav', 16000, wavs)
    print 'successfully saved out.wav'