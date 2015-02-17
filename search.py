"""
This file is part of the shamidreza/uniselection software.
Please refer to the LICENSE provided alongside the software (which is GPL v2,
http://www.gnu.org/licenses/gpl-2.0.html). 

This file includes the code for searching through the units to find the best consecutive units.
The input is a .lab file. (A seperate text2normalizedtext2phone module is needed 
to process raw text, which is not included in this software.)
"""
from extract_unit_info import *

def load_units():
    import pickle
    f=open('units.pkl','r')
    units=pickle.load(f)
    fnames=pickle.load(f)
    f.close()
    return units, fnames

def load_input(inp_name):
    times, labs = read_lab(inp_name)
    units = []
    for i in range(1, len(labs) - 1):
        if 1:  # compute left phones
            phone = labs[i]+'_L'  # +'_'+'*'
            left_phone = labs[i - 1]
            right_phone = labs[i + 1]
            left_phone_cat = phoneme_category[left_phone]
            right_phone_cat = phoneme_category[right_phone]
            stating_sample=0
            ending_sample=int(16000*(times[i+1]-times[i])/2)
            cur_unit = Unit(LR='L', phone=phone,
                            left_phone=left_phone,
                            right_phone=right_phone,
                            left_phone_category=left_phone_cat,
                            right_phone_category=right_phone_cat,
                            filename=None,
                            starting_sample=stating_sample,
                            ending_sample=ending_sample,
                            overlap_starting_sample=None,
                            overlap_ending_sample=None,
                            left_CEP=None, right_CEP=None, unit_id=None)
            units.append(cur_unit)
        if 1:  # compute right phones
            phone = labs[i]+'_R'   # +'_'+'*'
            left_phone = labs[i - 1]
            right_phone = labs[i + 1]
            left_phone_cat = phoneme_category[left_phone]
            right_phon_cat = phoneme_category[right_phone]
            stating_sample=0
            ending_sample=int(16000*(times[i+1]-times[i])/2)
            cur_unit = Unit(LR='R', phone=phone,
                            left_phone=left_phone,
                            right_phone=right_phone,
                            left_phone_category=left_phone_cat,
                            right_phone_category=right_phone_cat,
                            filename=None,
                            starting_sample=stating_sample,
                            ending_sample=ending_sample,
                            overlap_starting_sample=None,
                            overlap_ending_sample=None,
                            left_CEP=None, right_CEP=None, unit_id=None)
            units.append(cur_unit)

    units_np = np.zeros(len(units), 'object')
    for i in range(len(units)):
        units_np[i] = units[i]
    return units_np
    
if __name__ == "__main__":
    fname = 'arctic_a0001'
    lab_name=corpus_path+'/lab/'+fname+'.lab'
    wav_name=corpus_path+'/wav/'+fname+'.wav'
    target_units = load_input(lab_name)
    units, fnames=load_units()
    a=0