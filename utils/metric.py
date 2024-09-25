import sys
sys.path.append('..')

import torch

def metric_pixel_dice(predicts, tagerts,labelseq):
    predicts = torch.sigmoid(predicts)
    predicts = torch.where(predicts > 0.5, 1., 0)

    dsc_loss_list = []
    B = len(labelseq)
    for b in range(B):
        pred = predicts[b, labelseq[b].tolist(), :].contiguous().view(1, -1)
        targ = tagerts[b, labelseq[b].tolist(), :].contiguous().view(1, -1)

        num = torch.sum(torch.mul(pred, targ), dim=1)
        den = torch.sum(pred, dim=1) + torch.sum(targ, dim=1)

        dice_score = 2 * num / den
        dsc_loss_list.append(dice_score.tolist()[0])

    return dsc_loss_list


if __name__ == "__main__":
    target = torch.rand((5,12,224,224))
    predict = torch.rand((5,12,224,224))
    labelseq = torch.tensor([11,3,8,5,9])
    print(metric_pixel_dice(predict, target, labelseq))