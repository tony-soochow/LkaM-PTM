import numpy as np
def get_sequence_feature(aa_list ,uniprot_ID  ,site, windowsize,PTM_name ):
    sequence_BLOSUMs=[]
    for sequence_index in range(len(aa_list)):
        sequence_BLOSUM=BLOSUM62(aa_list[sequence_index])#(31,20)
        sequence_one=one_hot(aa_list[sequence_index])
        seq_feature=np.hstack((sequence_BLOSUM,sequence_one))#(31,50)
        sequence_BLOSUMs.append(seq_feature)
    return sequence_BLOSUMs

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
