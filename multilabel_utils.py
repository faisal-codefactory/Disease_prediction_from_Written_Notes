import numpy as np
import pandas as pd

#Aug28 Sheraz
#This file is for Multilable classification using mimic dataset.
#No other method is available here.

def return_topk_diagnosis_from_predictions(pred_tensor,topk_icd_list,df_map,**kwargs):
    # topk, return_prob, return_diagnosis       
    if not 'topk' in kwargs.keys():
        topk=5
    else:
        topk= kwargs['topk']

    if 'return_prob' in kwargs.keys():
        return_prob =  kwargs['return_prob']
    else:
        return_prob = False     
        
    if 'return_diagnosis' in kwargs.keys():
        return_diagnosis =  kwargs['return_diagnosis']
    else:
        return_diagnosis = False
              
    #print(f"Parameters: topk:{topk}  return_prob:{return_prob}  return_diagnosis:{return_diagnosis}")

    if ( not return_prob and  not return_diagnosis):
        #print(f" 1 activated")
        tmp = pred_tensor.reshape(-1)
        tmp = np.argpartition(tmp, -topk )[-topk:]
        topkICDPreds = []
        for idx in tmp:
            icd =  topk_icd_list[idx].strip()
    #print(icd)
            topkICDPreds.append(icd)
        return topkICDPreds
        #return return_topk_diagnosis_from_predictions_in_icd9(pred_tensor, topk_icd_list, topk)
    elif  (return_prob and not return_diagnosis):
        #print(f" 2 activated")
       # topkICDPreds = return_topk_diagnosis_from_predictions_in_icd9(pred_tensor, topk_icd_list, topk)
        prob = pred_tensor.reshape(-1)
        indices = np.argpartition(prob, -topk )[-topk:]
        #print(indices)
        topkICDPreds = []
        topkprobs = []  # to store the topk highest probabilities
        for idx in indices:
            icd =  topk_icd_list[idx].strip()
            topkICDPreds.append(icd)
            topkprobs.append(prob[idx])   
        #print(topkICDPreds)
        #print(topkprobs)
        prob_dict = dict(zip(topkICDPreds,topkprobs))
        return prob_dict
    elif  ( not return_prob and return_diagnosis):
        #print(f" 3 activated")
        prob = pred_tensor.reshape(-1)
        indices = np.argpartition(prob, -topk )[-topk:]
        topkICDPreds = []
        for idx in indices:
            icd =  topk_icd_list[idx].strip()
            topkICDPreds.append(icd)
        try:
            predictions = df_map.loc[df_map['ICD9_CODE'].isin(topkICDPreds)]
            text_preds = predictions['LONG_TITLE'].tolist()
            if "OTHER" in topkICDPreds:
                idx = topkICDPreds.index("OTHER")
                text_preds.insert(idx,"OTHER")
            text_dict = dict(zip(topkICDPreds,text_preds))
            return text_dict
        except:
            print (f"exception occurred in diagnoses calculation.")
            return
    elif (return_prob and return_diagnosis):
        #print(f" 4 activated")
        prob = pred_tensor.reshape(-1)
        indices = np.argpartition(prob, -topk )[-topk:]
        topkICDPreds = []
        topkprobs = []  # to store the topk highest probabilities
        for idx in indices:
            icd =  topk_icd_list[idx].strip()
            topkICDPreds.append(icd)
            topkprobs.append(prob[idx])
        try:
            predictions = df_map.loc[df_map['ICD9_CODE'].isin(topkICDPreds)]
            text_preds = predictions['LONG_TITLE'].tolist()
            if "OTHER" in topkICDPreds:
                idx = topkICDPreds.index("OTHER")
                text_preds.insert(idx,"OTHER")
            text_dict = dict(zip(topkICDPreds,text_preds))
            prob_dict = dict(zip(topkICDPreds,topkprobs))
            return prob_dict, text_dict
        except:
            print (f"exception occurred in diagnoses calculation.")
            return
