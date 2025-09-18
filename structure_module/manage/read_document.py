import pandas as pd
import numpy as np
def read_txt(address,train_and_test_structure_feature_input):
    sequences = []
    labels = []
    modify_site=[]
    uniprot_ID=[]
    with open(address, 'r') as file:
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list_used = line.split(' ')
            sequences.append(list_used[0])
            labels.append(int(list_used[1]))
            uniprot_ID.append(list_used[2])
            modify_site.append(int(list_used[3]) - 1)
    df1 = pd.DataFrame({
        'sequence': sequences,
        'label': labels,
        'uniprot_ID': uniprot_ID,
        'modify_site': modify_site})
    train_and_test_structure_feature_input.columns = ['uniprot_ID' if i == 0 else
                  'modify_site' if i == 1 else
                  'structure_feature_dim{}'.format(i - 1)
                  for i in range(len(train_and_test_structure_feature_input.columns))]
    train_and_test_structure_feature_input['modify_site'] = train_and_test_structure_feature_input['modify_site'].astype('int64')
    merged_df = pd.merge(df1, train_and_test_structure_feature_input, on=['uniprot_ID', 'modify_site'], how='left')#列匹配
    feature_cols = [col for col in merged_df.columns if col.startswith('structure')]#获取结构特征列，112维
    merged_df[feature_cols] = merged_df[feature_cols].fillna(0)
    structure_feature = pd.concat([df1, merged_df[feature_cols]], axis=1)
    return structure_feature