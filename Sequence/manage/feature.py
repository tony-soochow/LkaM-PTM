import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import os
from torch_geometric.utils import dense_to_sparse
import torch
from torch_geometric.data import Data
from manage.feps import get_FEPS_features
def get_sequence_feature(aa_list ,uniprot_ID  ,site, windowsize,PTM_name ):
    str = ''
    sequence_feature=[]
    sequence_BLOSUMs=[]
    sequence_sss=[]
    for sequence_index in range(len(aa_list)):
        # feps=get_FEPS_features(aa_list[sequence_index])
        # print(feps.shape)
        sequence_BLOSUM=BLOSUM62(aa_list[sequence_index])#(31,20)
        sequence_one=one_hot(aa_list[sequence_index])
        # sequence_ss=dssp(aa_list[sequence_index],uniprot_ID[sequence_index],site[sequence_index], windowsize,PTM_name)#(31,9)
        seq_feature=np.hstack((sequence_BLOSUM,sequence_one))#(31,50)
        # sequence_feature.append( feps )
        sequence_BLOSUMs.append(seq_feature)
        # print(np.array(sequence_feature).shape)
        # sequence_sss.append(sequence_ss)
    return sequence_BLOSUMs
# def get_sequence_feature(aa_list ,MLP_feature,   max_ratios   ):
#     str = ''
#     motif_list=[      max_ratios[i]["char"]   for i in range(len(max_ratios)  )    ]
#     motif=str.join(motif_list)
#     motif_sequence_AAI14_BLOSUM62=AAI14_BLOSUM62(motif)#(win,34)
#     motif_sequence_AAI14_BLOSUM62_difference = np.zeros(  (  len(aa_list),  len(max_ratios)   ))
#     for sequence_index in range(len(aa_list)):
#         sequence_AAI14_BLOSUM=AAI14_BLOSUM62(aa_list[sequence_index])
#         list_feature=[]
#         for i in range(sequence_AAI14_BLOSUM.shape[0]):
#             list_feature.append(  ( max_ratios[i]["ratio"] )  *   np.sqrt(  np.sum(  np.square (sequence_AAI14_BLOSUM[i]  - motif_sequence_AAI14_BLOSUM62[i]  )  )  ) )
#         motif_sequence_AAI14_BLOSUM62_difference[sequence_index]=list_feature
#     sequence_feature=motif_sequence_AAI14_BLOSUM62_difference#(len_13950/4828,31)
#     return sequence_feature


def get_aa_id(aa_string):
    token2index = {}
    AA = list('$ACDEFGHIKLMNPQRSTWYV')
    for i in range(21):
        token2index[AA[i]] = i
    aa_string = list(aa_string)
    for i in range(len(aa_string)):#windows31
        if aa_string[i] not in AA:
            aa_string[i] = "$"
            pass
        pass
    seq_id = np.array( [token2index[residue] for residue in aa_string])#(31,)一维整数组
    return seq_id

def AAI14_BLOSUM62(gene):
    with open("Codes/Multi-scale-Sequence/manage/AAI.txt") as f:
        records = f.readlines()[1:]
    AAI = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AAI.append(array)
    AAI = np.array(
        [float(AAI[i][j]) for i in range(len(AAI)) for j in range(len(AAI[i]))]).reshape((14, 21))
    AAI = AAI.transpose()
    GENE_BE = {}
    AA = 'ACDEFGHIKLMNPQRSTWYV$'
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)#31
    gene_array = np.zeros((n, AAI.shape[1]))#（31,14）
    for i in range(n):
        if gene[i] in AA:
            gene_array[i] = AAI[(GENE_BE[gene[i]])]
        else:
            gene_array[i] = AAI[(GENE_BE["$"])]
    with open("Codes/Multi-scale-Sequence/manage/blosum62.txt") as f:
        records = f.readlines()[1:]
    blosum62 = []
    for i in records:
        array = i.rstrip().split() if i.rstrip() != '' else None
        blosum62.append(array)
    blosum62 = np.array(
        [float(blosum62[i][j]) for i in range(len(blosum62)) for j in range(len(blosum62[i]))]).reshape((20, 21))
    blosum62 = blosum62.transpose()
    GENE_BE = {}
    AA = 'ARNDCQEGHILKMFPSTWYV$'
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)#31
    gene_array_1= np.zeros((n, 20))#(31,20)
    for i in range(n):
        if gene[i] in AA:
            gene_array_1[i] = blosum62[(GENE_BE[gene[i]])]

        else:
            gene_array_1[i] = blosum62[(GENE_BE["$"])]
    feature=np.hstack((gene_array,gene_array_1))#(31,34)
    return feature

def one_hot(gene):
    GENE_BE = {}
    AA = 'ARNDCQEGHILKMFPSTWYVX'
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)#31
    one_hot_encoding = np.zeros((n, len(AA)))
    for i in range(n):
        if gene[i] in AA:
            one_hot_encoding[i, GENE_BE[gene[i]]] = 1
        else:
            one_hot_encoding[i, GENE_BE["X"]] = 1
    return one_hot_encoding

def BLOSUM62(gene):
    with open("Codes/Multi-scale-Sequence/manage/blosum62.txt") as f:
        records = f.readlines()[1:]
    blosum62 = []
    for i in records:
        array = i.rstrip().split() if i.rstrip() != '' else None
        blosum62.append(array)
    blosum62 = np.array(
        [float(blosum62[i][j]) for i in range(len(blosum62)) for j in range(len(blosum62[i]))]).reshape((20, 21))
    blosum62 = blosum62.transpose()
    GENE_BE = {}
    AA = 'ARNDCQEGHILKMFPSTWYVX'
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)#31
    gene_array_1= np.zeros((n, 20))#(31,20)
    for i in range(n):
        if gene[i] in AA:
            gene_array_1[i] = blosum62[(GENE_BE[gene[i]])]
        else:
            gene_array_1[i] = blosum62[(GENE_BE["X"])]
    return gene_array_1
def dssp(gene,id_pdb,site, windowsize,PTM_name):
    GENE_BE = {}
    AA = 'ARNDCQEGHILKMFPSTWYV$'
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    DSSP_MAP = {'H': 0, 'B': 1,'E': 2,  'G': 3, 'I': 4,'T': 5,'S': 6, '-': 7}
    n = len(gene)  # 序列长度
    dssp_array = np.zeros((n, len(DSSP_MAP)+1))  # 存储 DSSP 结构编码
    pdb_file = f"Datasets/{PTM_name}/pdb_files/{id_pdb}.pdb"
    
    # 检查 pdb 文件是否存在
    if os.path.exists(pdb_file):
        structure = PDBParser().get_structure(id_pdb, pdb_file)
        model = structure[0]
        dssp = DSSP(model, pdb_file, dssp="/home/admin641/anaconda3/envs/lmj_torch/bin/mkdssp")
        value = dssp.property_list

        dssp_structures=np.array(value)[(site-windowsize):(site+windowsize+1), 2]
        if dssp_structures.shape[0]<n:
            if site<windowsize:
                for i in range(n-dssp_structures.shape[0],n):
                    dssp_code = dssp_structures[i]
                    if dssp_code in DSSP_MAP:
                        dssp_array[i, DSSP_MAP[dssp_code]] = 1
            else:
                for i in range(dssp_structures.shape[0]):
                    dssp_code = dssp_structures[i]
                    if dssp_code in DSSP_MAP:
                        dssp_array[i, DSSP_MAP[dssp_code]] = 1
        else:         
            for i in range(n):
                dssp_code = dssp_structures[i]
                if dssp_code in DSSP_MAP:
                    dssp_array[i, DSSP_MAP[dssp_code]] = 1
    return dssp_array

def get_ppi_features(ids,feature_file,name_file):
    ppi_matrix = np.load(feature_file)
    mat = ppi_matrix
    feature_dim = mat.shape[1]  # 获取mat数组的第二个维度，即PPI矩阵的列数，也就是特征的维度

    feature_dict = {}
    # read protein name
    f = open(name_file, "r")
    alllines = f.readlines()
    f.close()
    for i, row in enumerate(alllines):
        protein = row[:-2]  # 获取蛋白名称
        feature_dict[protein] = mat[i, :]  # 为每个蛋白获取一行PPI嵌入
    ppi_features = []
    zero_vec = np.zeros(feature_dim)
    num_of_sites = 0
    for i in range(len(ids)):
        protein = ids[i]
        num_of_sites = num_of_sites + 1
        # 根据名称匹配相应PPI嵌入，缺失嵌入的分配0向量
        if protein in feature_dict.keys():
            ppi_features.append(feature_dict[protein])
        else:
            ppi_features.append(zero_vec)

    ppi_features = np.array(ppi_features)
    ppi_features = np.reshape(ppi_features, (num_of_sites, feature_dim))
    return ppi_features
