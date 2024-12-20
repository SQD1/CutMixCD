import numpy as np
import torch
import torch.nn as nn

eps = 1e-10

class FourMetrics(nn.Module):
    def __init__(self, is_one=True, n_classes=2):
        super(FourMetrics, self).__init__()
        self.n_classes = n_classes
        self.is_one = is_one
        self.eps = 1e-8

    def get_cm(self, pred, label):
        mask = (label >= 0) & (label < self.n_classes)
        label = self.n_classes * label[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.n_classes ** 2)
        return count.reshape(self.n_classes, self.n_classes)

    def cal_one(self, y_pr, y_gt):
        if self.is_one:
            y_pr = torch.sigmoid(y_pr).squeeze(0)
            pr = (y_pr.detach().cpu().numpy() > 0.5).astype('uint8').flatten()
        else:
            pr = torch.argmax(y_pr, dim=0).detach().cpu().numpy().astype('uint8').flatten()
        gt = y_gt.cpu().numpy().astype('uint8').flatten()

        cm = self.get_cm(pr, gt)
        diag = np.diag(cm)
        sum0 = cm.sum(axis=0)
        sum1 = cm.sum(axis=1)

        precision = (diag + self.eps) / (sum0 + self.eps)
        recall = (diag + self.eps) / (sum1 + self.eps)
        f1 = (2 * precision * recall) / (precision + recall)
        iou = (diag + self.eps) / (sum0 + sum1 - diag + self.eps)

        return np.array([precision[1], recall[1], f1[1], iou[1]])

    def forward(self, y_pr, y_gt):
        res = np.array([0.0, 0.0, 0.0, 0.0])
        for i in range(y_pr.shape[0]):
            res += self.cal_one(y_pr[i], y_gt[i])
        res /= y_pr.shape[0]
        return res


class ChangeMetrics(torch.nn.Module):
    def __init__(self, is_one):
        super(ChangeMetrics, self).__init__()
        self.is_one = is_one

    def get_new_cm(self, pred, label):
        mask = (label >= 0) & (label < 2)
        label = 2 * label[mask].astype(int) + pred[mask]
        count = np.bincount(label, minlength=4)
        return count.reshape(2, 2)

    def cal_one(self, y_pr, y_gt):
        if isinstance(y_gt, list):
            y_gt = y_gt[0]
        if self.is_one:
            y_pr = torch.sigmoid(y_pr)
            pr = (y_pr.detach().cpu().numpy() > 0.5).astype('uint8').flatten()
        else:
            pr = torch.argmax(y_pr, dim=1).detach().cpu().numpy().astype('uint8').flatten()

        gt = y_gt.cpu().numpy().astype('uint8').flatten()

        return self.get_new_cm(pr, gt)

    def forward(self, pred, gt):
        return self.cal_one(pred, gt)


class CMMeter:
    def __init__(self):
        self.cm = np.zeros((2, 2))

    def add(self, cm):
        self.cm += cm

    def get_metrics(self):
        diag = np.diag(self.cm)[1]
        sum0 = self.cm.sum(axis=0)[1]
        sum1 = self.cm.sum(axis=1)[1]
        precision = diag / (sum0 + eps)
        recall = diag / (sum1 + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        iou = (diag + eps) / (sum0 + sum1 - diag + eps)
        oa = np.diag(self.cm).sum() / self.cm.sum()
        return [precision, recall, f1, iou, oa]