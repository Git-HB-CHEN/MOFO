import torch
import torch.nn.functional as F
import torch.nn as nn

class SelectedDSCLoss(nn.Module):
    def __init__(self, smooth=1e-9):
        super(SelectedDSCLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target, labelseq):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        predict = torch.sigmoid(predict)

        dsc_loss_list = []
        B = len(labelseq)
        for b in range(B):

            pred = predict[b, labelseq[b].tolist(), :].contiguous().view(1, -1)
            targ = target[b, labelseq[b].tolist(), :].contiguous().view(1, -1)

            num = torch.sum(torch.mul(pred, targ), dim=1)
            den = torch.sum(pred, dim=1) + torch.sum(targ, dim=1) + self.smooth

            dice_score = 2 * num / den
            dsc_loss_list.append(1 - dice_score)

        dsc_loss_sum = torch.stack(dsc_loss_list).sum()

        return dsc_loss_sum


class SelectedBCELoss(nn.Module):
    def __init__(self):
        super(SelectedBCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target, labelseq):

        bce_loss_list = []
        B = len(labelseq)
        for b in range(B):
            bce_loss_list.append(self.criterion(predict[b, labelseq[b].tolist(), :], target[b, labelseq[b].tolist(),:]))

        bce_loss_sum = torch.stack(bce_loss_list).sum()
        return bce_loss_sum

class SelectedFLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2):
        super(SelectedFLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predict, target, labelseq):

        f_loss_list = []

        probab = predict.sigmoid()
        B = len(labelseq)

        for b in range(B):
            pred = predict[b, labelseq[b].tolist(), :]
            targ = target[b, labelseq[b].tolist(), :]
            prob = probab[b, labelseq[b].tolist(), :]

            f_loss = F.binary_cross_entropy_with_logits(pred, targ, reduction="none")
            p_t = prob * targ + (1 - prob) * (1 - targ)
            loss = f_loss * ((1 - p_t) ** self.gamma)

            alpha_t = self.alpha * targ + (1 - self.alpha) * (1 - targ)
            loss = alpha_t * loss
            f_loss_list.append(loss.mean())

        f_loss_sum = torch.stack(f_loss_list).sum()
        return f_loss_sum


class SelectedCLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5, beta=1.0):
        super(SelectedCLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta

    def forward(self, embeddings1, embeddings2):
        euclidean_distance = F.pairwise_distance(embeddings1, embeddings2, p=2)

        boundary_samples = torch.where(euclidean_distance < self.margin, torch.ones_like(euclidean_distance),
                                       torch.zeros_like(euclidean_distance))
        boundary_weights = self.alpha * boundary_samples + self.beta * (1 - boundary_samples)

        positive_pairs = torch.exp(-boundary_weights * (euclidean_distance - self.margin))
        negative_pairs = torch.exp(boundary_weights * (euclidean_distance + self.margin))

        loss_contrastive = torch.mean(torch.log(1 + positive_pairs * negative_pairs))

        return loss_contrastive


