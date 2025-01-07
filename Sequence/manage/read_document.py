def read_txt(data_path_read_input,sequence_center,windowsize):
    sequences = []
    labels = []
    uniprot_ID=[]
    site=[]
    with open(data_path_read_input, 'r') as file:
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split(' ')
            sequences.append(list[0][(sequence_center-windowsize):(sequence_center+windowsize+1)])
            labels.append(int(list[1]))
            uniprot_ID.append(list[2])
            site.append(int(list[3]))
    return sequences, labels,uniprot_ID,site