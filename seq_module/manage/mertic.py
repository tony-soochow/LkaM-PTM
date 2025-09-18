import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc
import numpy as np
import torch
import matplotlib.pylab as plt



def caculate_metric(pred_y, labels, pred_prob):
    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    ACC = float(tp + tn) / test_num
    # precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)
    # SE
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)
    # SP
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)
    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)
    labels = labels.cpu()
    pred_prob = pred_prob.cpu()
    labels =labels.numpy().tolist()
    pred_prob = pred_prob.numpy().tolist()
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)
    AUC = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AUPR=auc(recall, precision)
    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC,AUPR])
    roc_data = [fpr, tpr, AUC]
    aupr_data = [recall, precision, AUPR]
    return metric,roc_data,aupr_data
