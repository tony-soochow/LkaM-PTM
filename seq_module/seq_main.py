from datetime import datetime
import torch
import numpy as np
import random
import os 
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from manage.data_process import data_process
from manage.read_document import read_txt
from model_MS import combine_model, test
from transformers import AutoModel, AutoTokenizer
import pickle

device_set="cuda:0"
data_path = 'Datasets/Succinylation/'#Crotonylation,Nitrosylation,Succinylation,Acetylation
windowsize=15
sequence_center =15#原始数据中k的位置

PTM_name = data_path.split('/')[-2]
length_sequence=2*windowsize+1
start = datetime.now()
print('start at ', start)
device = torch.device(device_set)
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

def save_model(model, optimizer, epoch, loss, file_path):
    """保存模型和优化器的状态"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, file_path)

phase=0
K_FOLD=5
best_metrics_list = []
val_metrics_list=[]
for fold in range(K_FOLD):
    print(f'---FOLD:{fold}----')
    sequences, labels,_,_ = read_txt(data_path + f"train_fold_{fold}.txt", sequence_center,windowsize)
    with open(f'Codes/Multi-granularity-Structure/{PTM_name}_result/structure_feature_train_{fold}.pkl' ,'rb') as f:
        train_structure_feature = pickle.load(f)
    with open(f'Codes/Multi-granularity-Structure/{PTM_name}_result/structure_feature_val_{fold}.pkl' ,'rb') as f:
        val_structure_feature = pickle.load(f)
    with open(f'Codes/Multi-granularity-Structure/{PTM_name}_result/structure_feature_test.pkl' ,'rb') as f:
        test_structure_feature = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained('pro_bert', do_lower_case=False, use_fast=False)
    BERT_encoder = AutoModel.from_pretrained('pro_bert', local_files_only=True, output_attentions=False).to(device)
    n_epoch=20
    embedding_dim=128
    Bilstm_output_feature_size=169
    Bilstm_layer_number=4
    n_layers=4
    n_head=8
    d_k=128
    CNN_out_channels=64
    conv1_kernel_size=2
    conv1_padding=4
    batch_size=32
    lr= 0.00002
    MLP_input_dim= int(length_sequence)
    dmodel=embedding_dim
    max_len = 2 * windowsize + 1
    vocab_size = 21
    setup_seed(2024)
    file_names = {
    'Train_data': f'train_fold_{fold}.txt',
    'Val_data': f'valid_fold_{fold}.txt',
    'Test_data': 'test.txt'
    }
    model = combine_model(BERT_encoder,dmodel, Bilstm_output_feature_size, Bilstm_layer_number, n_layers, n_head,d_k, CNN_out_channels, conv1_kernel_size,conv1_padding  ,MLP_input_dim ,max_len  , vocab_size  ,device)
    model = model.to(device)
    loss_function = nn.BCELoss(reduction='sum')
    data_loaders = {phase_name: DataLoader(data_process(data_path, phase_name,file_name, sequence_center, windowsize, PTM_name,tokenizer,True    ,train_structure_feature,val_structure_feature,test_structure_feature   ),
                                                                            batch_size=batch_size,
                                                                            pin_memory=True,
                                                                            shuffle=False)
                                                                             for phase_name, file_name in file_names.items()}

    test_performance = np.zeros((1, 8))
    train_acc_record = []
    train_pre_labels = []
    val_acc_record = []
    val_pre_labels = []
    test_pre_labels = []
    train_losses = []
    val_losses = []
    best_acc=0
    val_acc=0
    best_combine_features=None
    best_repres=None
    best_metrics=None
    val_metrics=None
    for epoch in range(1, n_epoch + 1):
        if epoch % 5 == 0:
            lr = lr / 5
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        tbar = tqdm(enumerate(data_loaders['Train_data']),disable=False,total=len(data_loaders['Train_data']))
        for idx, (*x,y) in tbar:
            model.train()
            x[0] = (x[0].float()).to(device)
            x[2] = (x[2].float()).to(device)
            x[3] = x[3].to(device)
            x[4] = x[4].to(device)
            y = (y.float()).to(device)
            class_fenlei, _,_,_= model(*x)
            loss = loss_function(class_fenlei.squeeze(),y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for _p in ['Train_data']:
            performance, repres_list, label_list, _, _,train_true_label,train_pre_label,_,_  = test(model, data_loaders[_p], loss_function, device, True)
            if _p == 'Train_data':
                train_acc_record.append(performance["else"][0].cpu().numpy())
                train_pre_labels.append(train_pre_label)
                train_losses.append(performance["loss"].cpu().numpy())
        for _p in ['Val_data']:
            performance, repres_list, label_list, _, _,val_true_label,val_pre_label,_,_  = test(model, data_loaders[_p], loss_function, device, True)
            test_performance[phase] = performance["else"]
            current_acc = performance["else"][0]
            if current_acc > val_acc:
                val_acc = current_acc
                val_metrics = {  
                    'fold': fold,
                    'ACC': performance["else"][0].cpu().numpy(),
                    'Precision': performance["else"][1].cpu().numpy(),
                    'Sensitivity': performance["else"][2].cpu().numpy(),
                    'Specificity': performance["else"][3].cpu().numpy(),
                    'F1': performance["else"][4].cpu().numpy(),
                    'AUC': performance["else"][5].cpu().numpy(),
                    'MCC': performance["else"][6].cpu().numpy(),
                    'AUPR': performance["else"][7].cpu().numpy(),
                }
            if _p == 'Val_data':
                val_acc_record.append(performance["else"][0].cpu().numpy())
                val_pre_labels.append(val_pre_label)
                val_losses.append(performance["loss"].cpu().numpy())
        for _p in ['Test_data']:
            metric, repres_list_test, label_list, roc, aupr, test_true_label, pre_label,combine_features,all_emb = test(model, data_loaders[_p], loss_function, device, True)
            test_performance[phase] = metric["else"]
            current_acc = metric["else"][0]
            if current_acc > best_acc:
                best_acc = current_acc
                best_repres=repres_list_test
                best_emb=all_emb
                best_combine_features=combine_features
                best_metrics = {  
                    'fold': fold,
                    'ACC': metric["else"][0].cpu().numpy(),
                    'Precision': metric["else"][1].cpu().numpy(),
                    'Sensitivity': metric["else"][2].cpu().numpy(),
                    'Specificity': metric["else"][3].cpu().numpy(),
                    'F1': metric["else"][4].cpu().numpy(),
                    'AUC': metric["else"][5].cpu().numpy(),
                    'MCC': metric["else"][6].cpu().numpy(),
                    'AUPR': metric["else"][7].cpu().numpy(),
                }
            if _p == 'Test_data':
                test_pre_labels.append(pre_label)
        dataFrame_test = pd.DataFrame(test_performance, index=[epoch], columns=["ACC", "Precision", "Sensitivity", "Specificity", "F1", "AUC", "MCC", "AUPR"])
    best_metrics_list.append(best_metrics)
    val_metrics_list.append(val_metrics)
best_metrics_df = pd.DataFrame(best_metrics_list)
best_metrics_df.to_csv(f'{PTM_name}_result/best_metrics_5fold_d5.csv',float_format='%.4f', index=False)
