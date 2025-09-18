
import torch
import os
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import DSSP
import pandas as pd
def get_structure_feature(train_address,test_address,structure_data_adaress,Atom_target):
    uniprot_ID_train,modify_site_train,_= read_txt(train_address)
    uniprot_ID_test, modify_site_test, _= read_txt(test_address)
    protein_dict = {}
    for protein_id, site in zip(uniprot_ID_train, modify_site_train):
        if protein_id in protein_dict:
            protein_dict[protein_id].append(site)
        else:
            protein_dict[protein_id] = [site]
    for protein_id, site in zip(uniprot_ID_test, modify_site_test):
        if protein_id in protein_dict:
            protein_dict[protein_id].append(site)
        else:
            protein_dict[protein_id] = [site]
    for protein_id, site in protein_dict.items():
        protein_dict[protein_id] = list(set(site))
    protein_dict_site = {key: list(map(int, value)) for key, value in protein_dict.items()}#3484
    train_and_test_structure_feature=[]
    count=0
    for key in protein_dict_site:
        count+=1
        if count%100==0:
            print("%s_of_%s"%(int(count), len(protein_dict_site)   ))

        pdb_name = f"{structure_data_adaress}/{key}.pdb"
        if os.path.exists(pdb_name):
            file_path=pdb_name
            # print('find')
        else:
            file_path=False
        # print(file_path)
        modify_site_pdb = protein_dict_site[key]
        for i in range(len(modify_site_pdb)):
            modify_site_pdb[i] -= 1
        if file_path!=False:
            pdb_document = file_path
            p = PDBParser()
            structure = p.get_structure("1", pdb_document)
            model = structure[0]
            dssp = DSSP(model, pdb_document, dssp='/home/admin641/anaconda3/envs/DDI/bin/mkdssp')
            feature,modify_site_pdb_shunxu= process_dssp_and_pdb(dssp, pdb_document, modify_site_pdb,Atom_target)#(len(modify_site_pdb),112),(*)——112列特征+1列site
            # print(feature.shape,'---',np.array(modify_site_pdb_shunxu).shape)
            arrayB = np.array(key).repeat( feature.shape[0], axis=0).reshape( feature.shape[0] , 1)
            arrayC = np.array(modify_site_pdb_shunxu).reshape(-1, 1)
            result = np.concatenate((arrayB, arrayC), axis=1)
            feature_ndarray=feature.numpy()
            df = pd.concat([pd.DataFrame(result), pd.DataFrame(feature_ndarray)], axis=1)#(*,114)——112列特征+1列ID+1列site
            train_and_test_structure_feature.append(df)
            # print(df.shape,'----',np.array(train_and_test_structure_feature).shape)
    
    train_and_test_structure_feature = pd.concat(train_and_test_structure_feature, ignore_index=True)#（16083,114）(2701,114)
    # print(train_and_test_structure_feature.shape)
    return train_and_test_structure_feature



def read_txt(address):
    sequences = []
    labels = []
    AlphaFold_pdb_document_adderss=[]
    modify_site=[]
    uniprot_ID=[]
    with open(address, 'r') as file:
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split(' ')
            modify_site.append(list[3])
            uniprot_ID.append(list[2])
            labels.append(int(list[1]))
    return  uniprot_ID,modify_site,labels

def pdb_split(line):
    atom_type = "CNOS$"
    aa_trans_DICT = {
        'ALA': 'A', 'CYS': 'C', 'CCS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'MSE': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', }
    aa_type = "ACDEFGHIKLMNPQRSTVWY$"
    Atom_order=int(line[6:11].strip())-1
    atom=line[11:16].strip()
    amino=line[16:21].strip()
    AA_order=int(line[22:28].strip())-1
    x=line[28:38].strip()
    y=line[38:46].strip()
    z=line[46:54].strip()
    atom_single_name=line.strip()[-1]
    atom_single_name_vec = np.zeros(len(atom_type))
    atom_single_name_vec[atom_type.find(atom_single_name)] = 1
    AA_single_name_vec = np.zeros(len(aa_type))
    AA_single_name_vec[   aa_type.find(   aa_trans_DICT[ amino ] )  ] = 1
    atom_feature_combine= np.concatenate(( atom_single_name_vec.reshape(1, -1)   , AA_single_name_vec.reshape(1, -1)),axis=1)
    return atom,amino,AA_order, Atom_order, float(x),float(y),float(z)  ,atom_feature_combine


def Amino_acid_granularity(coord_all_AA_tensor, ALL_AA_feature_combine, modify_site_pdb,dseq=3,dr=10,dlong=5,k=10 ):
    modify_site_pdb_AA_combine_node_edge_feature = []
    for AA in  modify_site_pdb:
        AA_index=AA
        point = coord_all_AA_tensor[AA_index]
        expanded_point = point.unsqueeze(0).expand(coord_all_AA_tensor.shape[0], -1)
        differences = coord_all_AA_tensor - expanded_point
        distances_square = torch.sum(differences ** 2, dim=1)
        dist = (torch.sqrt(distances_square)).unsqueeze(0)
        nodes = dist.shape[1]
        adj = torch.zeros((1, nodes))
        E = torch.zeros(( nodes, 2*dseq-1 +1+1+1+1  ))
        _, indices = torch.topk(dist, k=k + 1, largest=False)
        knn = indices[0][1:]
        for j in range(nodes):
            not_edge = True
            dij_seq = abs(AA_index - j)
            if dij_seq < dseq and dist[0][j] < dr/2 :
                E[j][0 - 1 + AA_index - j + dseq] = 1
                not_edge = False
            if dist[0][j] < dr and dij_seq >= dlong:
                E[j][0 - 1 + 2 * dseq] = 1
                not_edge = False
            if j in knn and dij_seq >= dlong:
                E[j][0 + 2 * dseq] = 1
                not_edge = False
            if not_edge:
                continue
            adj[0][j] = 1
            E[j][0 + 1 + 2 * dseq] = dij_seq
            E[j][0 + 2 + 2 * dseq] = dist[0][j]
        EDGE_feature_sum = torch.matmul(adj, E[:,0:7])#(1,7)
        EDGE_feature_mean= torch.matmul(adj, E[:,7:])/((adj == 1).sum().item() )#(1,)
        aggregate_EDGE_feature = torch.cat([EDGE_feature_sum, EDGE_feature_mean], dim=1)
        aggregate_node_feature_1=torch.matmul(adj,     torch.from_numpy(ALL_AA_feature_combine[:,0:5]  ).to(torch.float32)       )
        aggregate_node_feature_1=aggregate_node_feature_1/((adj == 1).sum().item() )
        aggregate_node_feature_2=torch.matmul(adj,     torch.from_numpy(ALL_AA_feature_combine [:,5:]  ).to(torch.float32)      )
        aggregate_node_feature = torch.cat([aggregate_node_feature_1, aggregate_node_feature_2], dim=1)
        combine_node_edge_feature = torch.cat([aggregate_node_feature, aggregate_EDGE_feature], dim=1)
        modify_site_pdb_AA_combine_node_edge_feature.append(combine_node_edge_feature)
    modify_site_pdb_AA_combine_node_edge_feature=torch.cat(modify_site_pdb_AA_combine_node_edge_feature, dim=0)
    return modify_site_pdb_AA_combine_node_edge_feature


def Atom_granularity(coord_all_Atom_tensor, ALL_atom_feature_combine, modify_site_pdb,modify_site_pdb_NZ_index,dseq=3,dr=10,dlong=5,k=10 ):
    modify_site_pdb_ATOM_combine_node_edge_feature = []
    for AA in  modify_site_pdb:
        atom_index=modify_site_pdb_NZ_index[AA]
        point = coord_all_Atom_tensor[atom_index]
        expanded_point = point.unsqueeze(0).expand(coord_all_Atom_tensor.shape[0], -1)
        differences = coord_all_Atom_tensor - expanded_point
        distances_square = torch.sum(differences ** 2, dim=1)
        dist = (torch.sqrt(distances_square)).unsqueeze(0)
        nodes = dist.shape[1]
        adj = torch.zeros((1, nodes))
        E = torch.zeros(( nodes, 2*dseq-1 +1+1+1+1  ))
        _, indices = torch.topk(dist, k=k + 1, largest=False)
        knn = indices[0][1:]
        for j in range(nodes):
            not_edge = True
            dij_seq = abs(atom_index - j)
            if dij_seq < dseq and dist[0][j] < dr/2 :
                E[j][0 - 1 + atom_index - j + dseq] = 1
                not_edge = False
            if dist[0][j] < dr and dij_seq >= dlong:
                E[j][0 - 1 + 2 * dseq] = 1
                not_edge = False
            if j in knn and dij_seq >= dlong:
                E[j][0 + 2 * dseq] = 1
                not_edge = False
            if not_edge:
                continue
            adj[0][j] = 1
            E[j][0 + 1 + 2 * dseq] = dij_seq
            E[j][0 + 2 + 2 * dseq] = dist[0][j]
        EDGE_feature_sum = torch.matmul(adj, E[:,0:7])
        EDGE_feature_mean= torch.matmul(adj, E[:,7:])/((adj == 1).sum().item() )
        aggregate_EDGE_feature = torch.cat([EDGE_feature_sum, EDGE_feature_mean], dim=1)
        aggregate_node_feature=torch.matmul(adj, (torch.from_numpy(ALL_atom_feature_combine)).to(torch.float32)    )
        ATOM_combine_node_edge_feature = torch.cat([aggregate_node_feature, aggregate_EDGE_feature], dim=1)
        modify_site_pdb_ATOM_combine_node_edge_feature.append(ATOM_combine_node_edge_feature)
    modify_site_pdb_ATOM_combine_node_edge_feature=torch.cat(modify_site_pdb_ATOM_combine_node_edge_feature, dim=0)
    return modify_site_pdb_ATOM_combine_node_edge_feature


def process_dssp_and_pdb(dssp,pdb_document,modify_site_pdb,Atom_target):
    aa_type = "ACDEFGHIKLMNPQRSTVWY$"
    SS_type = "HBEGITS-"
    dssp_feature = []
    for i in range(len(dssp)):
        SS_vec = np.zeros(8)
        SS=dssp.property_list[i][2]
        SS_vec[SS_type.find(SS)] = 1
        PHI = dssp.property_list[i][4]
        PSI = dssp.property_list[i][5]
        ASA = dssp.property_list[i][3]
        aa_name_onehot=np.zeros(21)
        aa_name=dssp.property_list[i][1]
        if aa_name in aa_type:
            aa_name_onehot[aa_type.find(aa_name)] = 1
        else:
            aa_name_onehot[20] = 1
        feature1= np.concatenate(   (np.array([PHI, PSI, ASA]), SS_vec))
        feature2=aa_name_onehot
        feature=np.concatenate(   (feature1,feature2 ))
        dssp_feature.append( feature     )
    dssp_feature = np.array(dssp_feature)
    angle = dssp_feature[:, 0:2]
    ASA_SS = dssp_feature[:, 2:]
    radian = angle * (np.pi / 180)
    ALL_AA_dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis=1)
    ALL_atom_feature_combine = np.array([]).reshape(0, 5 + 21)
    coord_all_Atom = []
    coord_all_AA = [] 
    modify_site_pdb_NZ_index = {}
    with open(pdb_document, 'r') as f:
        filt_atom = 'CA'
        for line in f:
            kind = line[:6].strip()
            if kind not in ['ATOM']:
                continue
            atom, amino, amino_order, atom_order, x, y, z  ,atom_feature_combine= pdb_split(line)
            ALL_atom_feature_combine=np.vstack([ALL_atom_feature_combine, atom_feature_combine])
            if atom == filt_atom:
                coord_all_AA.append([x, y, z])
            coord_all_Atom.append([x, y, z])
            if  atom==Atom_target  and  (amino_order in modify_site_pdb):
                modify_site_pdb_NZ_index[amino_order]=atom_order
        coord_all_AA_tensor=torch.FloatTensor(coord_all_AA)
        coord_all_Atom_tensor = torch.FloatTensor(coord_all_Atom)
    if len(modify_site_pdb_NZ_index) == len(modify_site_pdb):#K的数量，如果数据集没有欠采样则相同
        modify_site_pdb_aggregate_atom_node_and_edge_feature = Atom_granularity(coord_all_Atom_tensor, ALL_atom_feature_combine,modify_site_pdb, modify_site_pdb_NZ_index, dseq=3, dr=10, dlong=5, k=10 )
        modify_site_pdb_aggregate_AA_node_and_edge_feature = Amino_acid_granularity(coord_all_AA_tensor,ALL_AA_dssp_feature ,modify_site_pdb, dseq=3,dr=10, dlong=5, k=10)

        total_complete_structure= torch.from_numpy(  np.mean(ALL_AA_dssp_feature, axis=0, keepdims=True) )
        total_complete_structure_used=total_complete_structure.repeat(  modify_site_pdb_aggregate_AA_node_and_edge_feature.shape[0]   , 1)

        final_feature=torch.cat((total_complete_structure_used, modify_site_pdb_aggregate_AA_node_and_edge_feature, modify_site_pdb_aggregate_atom_node_and_edge_feature), dim=1)
        used_modify_site_pdb=modify_site_pdb
    elif len(modify_site_pdb_NZ_index) != len(modify_site_pdb)   and len(modify_site_pdb_NZ_index) !=0:
        modify_site_pdb_exist=list(modify_site_pdb_NZ_index.keys())
        modify_site_pdb_not_exist = [num for num in modify_site_pdb if num not in modify_site_pdb_exist]

        modify_site_pdb_aggregate_atom_node_and_edge_feature = Atom_granularity(coord_all_Atom_tensor, ALL_atom_feature_combine,modify_site_pdb_exist, modify_site_pdb_NZ_index, dseq=3, dr=10, dlong=5, k=10 )
        modify_site_pdb_aggregate_AA_node_and_edge_feature = Amino_acid_granularity(coord_all_AA_tensor,ALL_AA_dssp_feature ,modify_site_pdb_exist, dseq=3,dr=10, dlong=5, k=10)

        total_complete_structure= torch.from_numpy(  np.mean(ALL_AA_dssp_feature, axis=0, keepdims=True) )
        total_complete_structure_used=total_complete_structure.repeat(  modify_site_pdb_aggregate_AA_node_and_edge_feature.shape[0]   , 1)
        final_feature_modify_site_pdb_exist=torch.cat((total_complete_structure_used, modify_site_pdb_aggregate_AA_node_and_edge_feature, modify_site_pdb_aggregate_atom_node_and_edge_feature), dim=1)

        final_modify_site_pdb_not_exist = torch.zeros((len(modify_site_pdb_not_exist), 112))
        final_feature=torch.cat((final_feature_modify_site_pdb_exist, final_modify_site_pdb_not_exist), dim=0)
        used_modify_site_pdb=modify_site_pdb_exist+modify_site_pdb_not_exist
    elif len(modify_site_pdb_NZ_index) == 0:
        final_feature = torch.zeros((len(modify_site_pdb), 112))
        used_modify_site_pdb = modify_site_pdb
    return final_feature,used_modify_site_pdb