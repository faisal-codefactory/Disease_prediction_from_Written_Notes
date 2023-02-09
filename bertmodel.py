import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
import torch
from multilabel_utils import return_topk_diagnosis_from_predictions


#Aug29 Sheraz
#This class serve as boilerplate for loading the transformer model in FastAPI
class DiaTransModel:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.df_map = pd.read_csv("../assets/D_ICD_DIAGNOSES_WITH_ICD10CSV.csv", sep=',')
        with open("../assets/top_codes_orderbylabel.txt", 'r') as file_handle:
            top101ICD9_list = file_handle.readlines()
        self.top101ICD9_list = top101ICD9_list
        
        clfr= MultiLabelClassificationModel('bert', "../outputs/checkpoint-1350-epoch-2", num_labels=101, 
                                      args={'train_batch_size':8,
                                            'gradient_accumulation_steps':16,
                                            'learning_rate': 5e-5,
                                            'num_train_epochs': 2,
                                            'max_seq_length': 250,
                                            'reprocess_input_data': True,
                                            'overwrite_output_dir': True,
                                            'save_optimizer_and_scheduler': True})
        #clfr = clfr.eval()
        self.clfr = clfr#.to(self.device)
        
    def predict(self, text):
        
        if (not isinstance(text, list)):
           textlist = [text]
           
        _, probs = self.clfr.predict(textlist)
        #prob_dict,text_dict = return_topk_diagnosis_from_predictions(probs,self.top101ICD9_list,self.df_map, topk=5, return_prob=True, return_diagnosis=False)
        prob_dict = return_topk_diagnosis_from_predictions(probs,self.top101ICD9_list,self.df_map, topk=5, return_prob=True, return_diagnosis=False)
        return_list = []
        for key in prob_dict.keys():
            a = f"ICD9_code: {key}  Probability:{prob_dict[key]}\n"
            #a = f"ICD9_code: {key}  Diagnoses:{text_dict[key]}  Probability:{prob_dict[key]}\n"
            return_list.append(a)
        return return_list    
        
model = DiaTransModel()

def get_model():
    return model   
        
        
    