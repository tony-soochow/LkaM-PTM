from torch.utils.data import Dataset
from manage.read_document import read_txt
from manage.feature import get_aa_id,get_sequence_feature
import torch
import numpy as np
import re
class data_process(Dataset):
    def __init__(self, data_path, phase, file_name,sequence_center, windowsize,  PTM_name,tokenizer,  is_test   , train_structure_feature,val_structure_feature,test_structure_feature ):
        if phase.split(':')[0] == 'Train_data' and is_test ==True:
            sequences, labels,_,_ = read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequences_used_data =sequences
            self.labels_used_data =labels
            sequences_2, _,uniprot_ID ,site= read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequence_feature = get_sequence_feature(sequences_2 , uniprot_ID,site, windowsize,PTM_name )
            self.structure_feature=train_structure_feature
            pass
        if phase == 'Val_data' and is_test ==True:
            sequences, labels,_,_ = read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequences_used_data =sequences
            self.labels_used_data =labels
            sequences_2, _,uniprot_ID ,site= read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequence_feature = get_sequence_feature(sequences_2 , uniprot_ID,site, windowsize,PTM_name )
            self.structure_feature=val_structure_feature
            pass
        elif phase.split(':')[0] == 'Test_data' and is_test ==True:
            sequences, labels,_,_ = read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequences_used_data = sequences
            self.labels_used_data = labels
            sequences_2, _,uniprot_ID ,site= read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequence_feature = get_sequence_feature(sequences_2 , uniprot_ID,site, windowsize,PTM_name )
            self.structure_feature = test_structure_feature
        self.length = len(self.sequences_used_data)
        self.tokenizer=tokenizer
    def __getitem__(self, idx):
        seq = [token for token in re.sub(r"[UZOB*]", "X", self.sequences_used_data[idx].rstrip('*'))]
        max_len = len(seq)
        encoded = self.tokenizer.encode_plus(' '.join(seq), add_special_tokens=True, padding='max_length', return_token_type_ids=False, pad_to_max_length=True,truncation=True, max_length=max_len, return_tensors='pt')
        self.input_ids = encoded['input_ids'].flatten()
        self.attention_mask = encoded['attention_mask'].flatten()
        return (torch.tensor(self.sequence_feature[idx]),
                torch.tensor(get_aa_id(self.sequences_used_data[idx])),
                torch.tensor(self.structure_feature[idx, :]),
                self.input_ids.clone().detach(),
                self.attention_mask.clone().detach(),
                torch.tensor(self.labels_used_data[idx]))
    def __len__(self):
        return self.length