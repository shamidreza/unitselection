"""
Author: Seyed Hamidreza Mohammadi
This file is part of the shamidreza/uniselection software.
Please refer to the LICENSE provided alongside the software (which is GPL v2,
http://www.gnu.org/licenses/gpl-2.0.html). 

This file includes the code for searching through the units to find the best consecutive units.
The input is a .lab file. (A seperate text2normalizedtext2phone module is needed 
to process raw text, which is not included in this software.)
"""
from utils import *
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
            left_phone = labs[i-1]
            right_phone = labs[i+1]
            left_phone_cat = phoneme_category[labs[i-1]]
            right_phone_cat = phoneme_category[labs[i+1]]
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
            left_phone = labs[i-1]
            right_phone = labs[i+1]
            left_phone_cat = phoneme_category[labs[i-1]]
            right_phon_cat = phoneme_category[labs[i+1]]
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
    cost = 1.0
    if 'L' in unit.phone:
    #if 1:
        cost += 0.3 * (target_unit.left_phone==unit.left_phone)
        cost += 0.1 * (target_unit.left_phone_category==unit.left_phone_category)
        cost += 0.03 * (target_unit.right_phone==unit.right_phone)
        cost += 0.01 * (target_unit.right_phone_category==unit.right_phone_category)
    if 'R' in unit.phone:
        cost += 0.03 * (target_unit.right_phone==unit.right_phone)
        cost += 0.01 * (target_unit.right_phone_category==unit.right_phone_category)
        cost += 0.3 * (target_unit.left_phone==unit.left_phone)
        cost += 0.1 * (target_unit.left_phone_category==unit.left_phone_category)
    cost *= (target_unit.phone == unit.phone)

    return -cost
def joint_cost(target_unit1, target_unit2):
    cost = 0.0
    if target_unit1.phone.find('_L')!=-1:
        cost = (1.0*(target_unit1.phone.split('_')[0]==target_unit2.phone.split('_')[0]))
        if cost:
            cost += 0.5 if (target_unit1.unit_id == target_unit2.unit_id-1) else 0.0

    elif target_unit1.phone.find('_R')!=-1:
        cost = (3.0*(target_unit1.phone.split('_')[0]==target_unit2.left_phone))
        if cost:
            cost += 2.0 if (target_unit1.unit_id == target_unit2.unit_id-1) else 0.0
            
    else:
        raise AssertionError


    #+\
            #0.1*(target_unit1.right_phone_category==target_unit2.left_phone_category))
    cost -= 0.001*np.mean((target_unit1.right_CEP - target_unit2.left_CEP)**2)
    #cost = -10.0 if cost==0.0 else cost
    #cost = 10.0 if (cost != -10.0) and (target_unit1.unit_id == target_unit2.unit_id-1) else cost
    return -cost
def search(target_units, all_units, limit=20):
    # viterbi search through the units
    target_score = np.zeros((target_units.shape[0], limit))
    target_indice = np.zeros((target_units.shape[0], limit), dtype=np.uint)
    # compute target costs
    for t in range(target_units.shape[0]):
        cur_distances = np.zeros(all_units.shape[0])
        for j in range(all_units.shape[0]):
            cur_distances[j] = target_cost(target_units[t], all_units[j])
        cur_indice = cur_distances.argsort()[:limit]
        target_indice[t, :] = cur_indice
        target_score[t, :] = cur_distances[cur_indice]
    
    # compute joint costs
    score = np.zeros((target_units.shape[0], limit))
    path = np.zeros((target_units.shape[0], limit), dtype=np.uint)

    score[0, :] = target_score[0, :]
    
    for t in xrange(1,target_units.shape[0]):
        for j in xrange(limit): # to
            score_imin = 1000000
            score_min = 10000000.0
            for i in xrange(limit): # from
                tmp_cost = score[t-1, i] + \
                    joint_cost(all_units[target_indice[t-1, i]], all_units[target_indice[t, j]])
                if tmp_cost < score_min:
                    score_min = tmp_cost
                    score_imin = i
            assert score_imin != 1000000
            score[t, j] = score_min + target_score[t, j]
            path[t, j] = score_imin
    
    best_units_indice = np.zeros(target_units.shape[0], dtype=np.uint)
    best_units_indice[-1] = score[-1,:].argmin()
    for t in xrange(target_units.shape[0]-2, -1, -1):
        best_units_indice[t] = path[t+1, best_units_indice[t+1]]
    for t in xrange(target_units.shape[0]):
        best_units_indice[t] = target_indice[t, best_units_indice[t]]
    
    return best_units_indice
                
if __name__ == "__main__":
    fname = 'arctic_a0001'
    lab_name=corpus_path+'/lab/'+fname+'.lab'
    wav_name=corpus_path+'/wav/'+fname+'.wav'
    ##target_units = load_input(lab_name)
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
    a=0