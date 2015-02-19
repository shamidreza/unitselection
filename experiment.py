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
    fname = 'arctic_a0001'
    lab_name=corpus_path+'/lab/'+fname+'.lab'
    wav_name=corpus_path+'/wav/'+fname+'.wav'
    ##target_units = load_input(lab_name)
    times, lans = read_lab(lab_name)
    tmp_units=extract_info(lab_name, wav_name, 0,0)
    target_units = np.zeros(len(tmp_units), 'object')
    for j in xrange(len(tmp_units)):
        target_units[j] = tmp_units[j]
        
    units, fnames=load_units()
    units = units[:int(units.shape[0]*(100.0/100.0))]
    best_units_indice=search(target_units, units,limit=20)
    best_units = units[best_units_indice]
    for i in xrange(target_units.shape[0]):
        print target_units[i].phone, best_units[i].phone, best_units[i].unit_id
    #wavs=concatenate_units_overlap(best_units, fnames)
    wavs=concatenate_units_duration_overlap(best_units, fnames, times)

    from scipy.io.wavfile import write as wwrite
    wwrite('out.wav', 16000, wavs)
    print 'successfully saved out.wav'