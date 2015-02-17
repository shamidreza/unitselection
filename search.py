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
    
def target_cost(target_unit, unit):
    pass
def joint_cost(target_unit1, target_unit2):
    pass
def search(target_units, all_units, limit=20):
    # viterbi search through the units
    target_score = np.zeros((target_units.shape[0], limit))
    target_indice = np.zeros((target_units.shape[0], limit), dtype=np.uint)
    # compute target costs
    for t in range(target_units.shape[0]):
        cur_distances = np.zeros(all_units.shape[0])
        for j in range(all_units.shape[0]):
            cur_distances[j] = target_cost(target_unit[t], all_units[j])
        cur_indice = cur_distances.argsort()[-limit,:][::-1]
        target[t, :] = cur_indice
        target_score[t, :] = cur_distances[cur_indice]
    # compute joint costs
    score = np.zeros((target_units.shape[0], limit))
    path = np.zeros((target_units.shape[0], limit), dtype=np.uint)

    score[0, :] = target_score[0, :]
    
    for t in xrange(1,target_units.shape[0]):
        for i in xrange(limit): #from
            score_min = -1.0
            score_imin = 10000000.0
            for j in xrange(limit): # to
                tmp_cost = score[t-1, i] + \
                    joint_cost(all_units[target_indice[i]], all_units[target_indice[j]])
                if tmp_cost < score_min:
                    score_min = tmp_cost
                    score_imin = i
            score[t, j] = score_min + target_score[t-1, :]
            path[t, j] = score_imin
    
    best_units_indice = np.zeros(target_units.shape[0], dtype='object')
    best_units_indice[-1] = score[-1,:].argmin()
    for t in xrange(target_units.shape[0]-2, -1, -1):
        best_units_indice[t] = path[t+1, best_units_indice[t+1]]
    return best_units_indice
                
if __name__ == "__main__":
    fname = 'arctic_a0001'
    lab_name=corpus_path+'/lab/'+fname+'.lab'
    wav_name=corpus_path+'/wav/'+fname+'.wav'
    target_units = load_input(lab_name)
    units, fnames=load_units()
    a=0