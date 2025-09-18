import os
from manage.read_document import read_txt
from manage.manage_data import get_structure_feature
import pickle
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F
from structure_model import Stru_model, test
from torch.utils.data import Dataset
import tqdm
structure_data='Datasets/Crotonylation/pdb_files'#PDB structure,添加自己需要的pdb结构文件存储的路径。
device_set="cpu"
Atom_target="NZ"#Atom_赖氨酸（Lys）侧链中的氮原子
train_data_address='/home/admin641/lmj/PTM-CMGMS-main/Datasets/Crotonylation/train.txt'
test_data_address='/home/admin641/lmj/PTM-CMGMS-main/Datasets/Crotonylation/test.txt'
PTM_name=train_data_address.split('/')[-2]


if len(structure_data)!=0:
    train_and_test_structure_feature=get_structure_feature(train_data_address,test_data_address,structure_data,Atom_target)
elif len(structure_data)==0:
    train_and_test_structure_feature_file_path = f"Datasets/{PTM_name}/{PTM_name}_train_and_test_structure_feature.pkl"
    with open(train_and_test_structure_feature_file_path, 'rb') as f:
        train_and_test_structure_feature = pickle.load(f)#(17287,114),(12494,114)


train_structure_feature=read_txt(train_data_address, train_and_test_structure_feature)#(13950,116)
test_structure_feature=read_txt(test_data_address, train_and_test_structure_feature)#(4828,116)
train_structure_feature_shuffle=train_structure_feature.sample(frac=1, random_state=1)#(13950,116)

class data_process(Dataset):
    def __init__(self, data  ):
        self.feature=data
        self.length = data.shape[0]
    def __getitem__(self, idx):
        return (  torch.tensor(self.feature.iloc[idx, 4:], dtype=torch.float32  ),
                torch.tensor(self.feature.iloc[idx,  1   ])   )
    def __len__(self):
        return self.length

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
device = torch.device(device_set)
path = Path(f'./{PTM_name}_result/')
writer = SummaryWriter(path)
setup_seed(1)
batch_size=128
lr=0.001
n_epoch=20
w1=1/8
margin_1=10
dim_feature=112//4
data_loaders = {}
data_loaders["Train_data"] = DataLoader(data_process(train_structure_feature), batch_size=batch_size,pin_memory=True, shuffle=False)
data_loaders["Test_data"] = DataLoader(data_process(test_structure_feature), batch_size=batch_size,pin_memory=True, shuffle=False)

model = Stru_model().to(device)
loss_function = nn.BCELoss(reduction='sum')
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
for epoch in range(1, n_epoch + 1):
    model.train()
    train_loss = 0.0
    tbar = tqdm(enumerate(data_loaders["Train_data"]), disable=False, total=len(data_loaders["Train_data"]))
    for idx, batch in tbar:
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device).float()
        outputs = model(features)
        loss = loss_function(outputs.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

for _p in ['Train_data']:
    Comparative_learning_feature_train = test(model, data_loaders[_p], loss_function, device,dim_feature)
    Comparative_learning_feature_train=  Comparative_learning_feature_train.cpu().numpy()
    with open('./%s/Comparative_learning_feature_dim%s_train1.pkl'%(str(path.name),str(dim_feature)), 'wb') as f:
        pickle.dump(Comparative_learning_feature_train, f)

for _p in ['Test_data']:
    Comparative_learning_feature_test = test(model, data_loaders[_p], loss_function, device,dim_feature)
    Comparative_learning_feature_test=  Comparative_learning_feature_test.cpu().numpy()
    with open('./%s/Comparative_learning_feature_dim%s_test1.pkl'%(str(path.name),str(dim_feature)), 'wb') as f:
        pickle.dump(Comparative_learning_feature_test, f)
