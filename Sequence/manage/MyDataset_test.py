from torch.utils.data import Dataset
from manage.read_document import read_txt
from manage.feature import get_aa_id,get_sequence_feature
import torch
import numpy as np
import re
class MyDataset_test(Dataset):
    def __init__(self, data_path, phase, file_name,sequence_center, windowsize,  PTM_name,tokenizer,  is_test   , train_structure_feature,val_structure_feature,test_structure_feature ):
        if phase.split(':')[0] == 'Train_data' and is_test ==True:
            sequences, labels,_,_ = read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequences_used_data =sequences
            self.labels_used_data =labels
            sequences_2, _,uniprot_ID ,site= read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequence_feature = get_sequence_feature(sequences_2 , uniprot_ID,site, windowsize,PTM_name )
            self.structure_feature=train_structure_feature
            # self.ppi_feature=get_ppi_features(uniprot_ID,f'PPI/{PTM_name}/ppi_train_feature.npy',f'PPI/{PTM_name}/train_id_ppi.txt')
            # np.save('train_ss.npy', sequences_ss)
            pass
        if phase == 'Val_data' and is_test ==True:
            sequences, labels,_,_ = read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequences_used_data =sequences
            self.labels_used_data =labels
            sequences_2, _,uniprot_ID ,site= read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequence_feature = get_sequence_feature(sequences_2 , uniprot_ID,site, windowsize,PTM_name )
            self.structure_feature=val_structure_feature
            # self.ppi_feature=get_ppi_features(uniprot_ID,f'PPI/{PTM_name}/train_embeddings.npy',f'PPI/{PTM_name}/train_id_ppi.txt')
            # np.save('train_ss.npy', sequences_ss)
            pass
        elif phase.split(':')[0] == 'Test_data' and is_test ==True:
            sequences, labels,_,_ = read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequences_used_data = sequences
            self.labels_used_data = labels
            sequences_2, _,uniprot_ID ,site= read_txt(data_path + file_name, sequence_center,windowsize)
            self.sequence_feature = get_sequence_feature(sequences_2 , uniprot_ID,site, windowsize,PTM_name )
            self.structure_feature = test_structure_feature
            # self.ppi_feature=get_ppi_features(uniprot_ID,f'PPI/{PTM_name}/ppi_test_feature.npy',f'PPI/{PTM_name}/test_id_ppi.txt')

            # np.save('test_ss.npy', sequences_ss)
        self.length = len(self.sequences_used_data)
        self.tokenizer=tokenizer
    def __getitem__(self, idx):
        seq = [token for token in re.sub(r"[UZOB*]", "X", self.sequences_used_data[idx].rstrip('*'))]
        max_len = len(seq)
        encoded = self.tokenizer.batch_encode_plus(' '.join(seq), add_special_tokens=True, padding='max_length', return_token_type_ids=False, pad_to_max_length=True,truncation=True, max_length=max_len, return_tensors='pt')
        self.input_ids = encoded['input_ids'].flatten()
        self.attention_mask = encoded['attention_mask'].flatten()
        # edge_index1 = torch.tensor(edge_index(self.sequences_used_data[idx]))  # 生成原始 edge_index
        # print(torch.tensor(edge_index(self.sequences_used_data[idx])).size())
        # print(len(self.sequences_used_data[idx]),len(seq),self.input_ids.shape,self.attention_mask.shape,len(self.labels_used_data))
        return (torch.tensor(self.sequence_feature[idx]),
                torch.tensor(get_aa_id(self.sequences_used_data[idx])),
                torch.tensor(self.structure_feature[idx, :]),
                # torch.tensor(self.ppi_feature[idx,:]),
                self.input_ids.clone().detach(),
                self.attention_mask.clone().detach(),
                # torch.tensor(self.blo[idx]),
                # self.datas[idx],
                torch.tensor(self.labels_used_data[idx]))
    def __len__(self):
        return self.length