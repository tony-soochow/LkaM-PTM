import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm


class Stru_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.structure_feature_learn_complete_block = nn.Sequential(
            nn.Linear(34, 34//2),
            nn.LeakyReLU(),
            nn.Linear(34//2, 34))
        self.structure_feature_learn_AA_block = nn.Sequential(
            nn.Linear(43, 43//2),
            nn.LeakyReLU(),
            nn.Linear(43//2, 43),)
        self.structure_feature_learn_atom_block = nn.Sequential(
            nn.Linear(35, 35//2),
            nn.LeakyReLU(),
            nn.Linear(35//2, 35),)
        self.feature_learned = nn.Sequential(
            nn.Linear(112, 112 // 2),
            nn.LeakyReLU(),
            nn.Linear(112 // 2, 112//4), )
        self.feature_final = nn.Sequential(
            nn.Linear(112//4, 2),)
        self.combine_block_MLP = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid())
    def forward(self, structure_feature):
        x_Complete = self.structure_feature_learn_complete_block(structure_feature[:, 0:(13 + 21)])
        x_AA = self.structure_feature_learn_AA_block(structure_feature[:, (13 + 21):(34 + 34 + 9)])
        x_ATOM = self.structure_feature_learn_atom_block(structure_feature[:, (34 + 34 + 9):])
        x1 = torch.cat((x_Complete, x_AA, x_ATOM), dim=1)
        x= self.feature_learned (x1)
        return x
    def trainModel(self, structure_feature):
        with torch.no_grad():
            output = self.forward(structure_feature)
        feature=self.feature_final(output)
        logit=self.combine_block_MLP(feature)
        return logit,feature,output

def test(net: torch.nn.Module, test_loader, loss_function, device,dim_feature):
    net.eval()
    return_feature_result = torch.empty((0, dim_feature), device=device)
    with torch.no_grad():
        for idx,(structure_feature,label) in tqdm(enumerate(test_loader), disable=False, total=len(test_loader)):
            structure_feature, y = structure_feature.to(device),label.to(device)
            class_fenlei,representation,feature = net.trainModel(structure_feature)
            return_feature_result = torch.cat((return_feature_result, feature), dim=0)
    return return_feature_result