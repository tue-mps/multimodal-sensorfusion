import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
class FocalLoss(nn.Module):
    """
    Focal loss class. Stabilize training by reducing the weight of easily classified
    background sample and focussing on difficult foreground detections.
    """

    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, prediction, target):

        # get class probability
        pt = torch.where(target == 1.0, prediction, 1-prediction)

        # compute focal loss
        loss = -1 * (1-pt)**self.gamma * torch.log(pt+1e-6)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
    
def pixor_loss(batch_predictions, batch_labels,loss_param,mode_param):


    #########################
    #  classification loss  #
    #########################
    classification_prediction = batch_predictions[:, 0,:, :].contiguous().flatten()
    classification_label = batch_labels[:, 0,:, :].contiguous().flatten()
    # classification_label___ = classification_label.cpu().numpy()
    # are_ones = np.any(classification_label___ == 1.0)
    # are_values_other_than_zeros_and_ones = np.any((classification_label___ != 0) & (classification_label___ != 1))

    if(loss_param['classification']=='FocalLoss'):
        focal_loss = FocalLoss(gamma=2)
        classification_loss = focal_loss(classification_prediction, classification_label)
    else:
        classification_loss = F.binary_cross_entropy(classification_prediction.double(), classification_label.double(),reduction='sum')

    #####################
    #  Regression loss  #
    #####################

    if mode_param["view_birdseye"] == "True":
        regression_prediction = batch_predictions.permute([0, 2, 3, 1])[:, :, :, :-1]
        regression_prediction = regression_prediction.contiguous().view([regression_prediction.size(0) *
                                                                         regression_prediction.size(1) * regression_prediction.size(2),
                                                                         regression_prediction.size(3)])
        regression_label = batch_labels.permute([0, 2, 3, 1])[:, :, :, :-1]
        regression_label = regression_label.contiguous().view([regression_label.size(0) * regression_label.size(1) *
                                                               regression_label.size(2), regression_label.size(3)])

        positive_mask = torch.nonzero(torch.sum(torch.abs(regression_label), dim=1))
        pos_regression_label = regression_label[positive_mask.squeeze(), :]
        pos_regression_prediction = regression_prediction[positive_mask.squeeze(), :]

        T = batch_labels[:, 1:]
        P = batch_predictions[:, 1:]
        M = batch_labels[:, 0].unsqueeze(1)

        if (loss_param['regression'] == 'SmoothL1Loss'):
            reg_loss_fct = nn.SmoothL1Loss(reduction='sum')
        else:
            reg_loss_fct = nn.L1Loss(reduction='sum')

        regression_loss = reg_loss_fct(P * M, T)
        NbPts = M.sum()
        if (NbPts > 0):
            regression_loss /= NbPts
    else:
        regression_loss = 0

    return classification_loss, regression_loss