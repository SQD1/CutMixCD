import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Sampler
import itertools
import pickle
import random
from utils.mask_gen import BoxMaskGenerator
import torch.nn.functional as F
from torchvision.transforms import Resize, GaussianBlur


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path

    def write(self, txt):
        with open(self.log_path, 'a') as f:
            f.write(txt)
            f.write("\r\n")


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']

def get_mem():
    return '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

def get_next(dataloader, i):
    try:
        batch = next(i)
    except:
        trainloader_iter = iter(dataloader)
        batch = next(trainloader_iter)
    return batch

def list2device(x, device):
    if isinstance(x, list):
        y = []
        for i in x:
            y.append(i.to(device))
        return y
    else:
        return x.to(device)


class RepeatSampler(Sampler):
    r"""Repeated sampler

    Arguments:
        data_source (Dataset): dataset to sample from
        sampler (Sampler): sampler to draw from repeatedly
        repeats (int): number of repetitions or -1 for infinite
    """

    def __init__(self, sampler, repeats=-1):
        if repeats < 1 and repeats != -1:
            raise ValueError('repeats should be positive or -1')
        self.sampler = sampler
        self.repeats = repeats

    def __iter__(self):
        if self.repeats == -1:
            reps = itertools.repeat(self.sampler)
            return itertools.chain.from_iterable(reps)
        else:
            reps = itertools.repeat(self.sampler, self.repeats)
            return itertools.chain.from_iterable(reps)

    def __len__(self):
        if self.repeats == -1:
            return 2 ** 62
        else:
            return len(self.sampler) * self.repeats

def get_train_val_loader(train_dataset, val_dataset, train_bs, val_bs, labeled_ratio, train_split_path, work_dir):
    """
    Training data is divided into two parts：train_loader and train_loader_remain
    train_loader: with change labels
    train_loader_remain: without change labels
    """
    num_workers = 2
    train_dataset_size = len(train_dataset)
    print('dataset size: ', train_dataset_size)
    partial_size = int(labeled_ratio * train_dataset_size)
    print('partial size: ', partial_size)

    if train_split_path:
        train_ids = pickle.load(open(train_split_path, 'rb'))

    else:
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)
        pickle.dump(train_ids, open(os.path.join(work_dir, 'train_split.pkl'), 'wb'))

    train_sampler = RepeatSampler(SubsetRandomSampler(train_ids[:partial_size]))
    train_remain_sampler = RepeatSampler(SubsetRandomSampler(train_ids[partial_size:]))


    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_bs,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=train_bs,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True)

    train_loader_remain = DataLoader(train_dataset,
                                     batch_size=train_bs,
                                     sampler=train_remain_sampler,
                                     num_workers=num_workers,
                                     pin_memory=False)


    return train_loader, train_loader_remain, val_loader


def get_train_loader(train_dataset, train_unsup_dataset, mask_collate_fn, train_bs, labeled_ratio, train_split_path, work_dir):
    """
    Outputs:
    train_sup_loader: with change labels
    train_unsup_loader_0:  without change labels, with random mask parameter.)
    train_unsup_loader_1:  without change labels
    """
    num_workers = 2
    train_dataset_size = len(train_dataset)
    print('dataset size: ', train_dataset_size)
    partial_size = int(labeled_ratio * train_dataset_size)
    print('partial size: ', partial_size)

    if train_split_path:
        train_ids = pickle.load(open(train_split_path, 'rb'))

    else:
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)
        pickle.dump(train_ids, open(os.path.join(work_dir, 'train_split.pkl'), 'wb'))

    # train_sup_sampler = SubsetRandomSampler(train_ids[:partial_size])
    train_sup_sampler = RepeatSampler(SubsetRandomSampler(train_ids[:partial_size]))

    # train_remain_sampler = SubsetRandomSampler(train_ids[partial_size:])
    train_remain_sampler = RepeatSampler(SubsetRandomSampler(train_ids[partial_size:]))


    train_sup_loader = DataLoader(train_dataset,
                              batch_size=train_bs,
                              sampler=train_sup_sampler,
                              collate_fn=mask_collate_fn,  # 添加 mask参数
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True)

    train_unsup_loader_0 = DataLoader(train_unsup_dataset,
                                 batch_size=train_bs,
                                 sampler=train_remain_sampler,
                                 collate_fn=mask_collate_fn,  # 添加 mask参数
                                 num_workers=num_workers,
                                 pin_memory=False)

    train_unsup_loader_1 = DataLoader(train_unsup_dataset,
                                     batch_size=train_bs,
                                     sampler=train_remain_sampler,   # sampler传递的是函数，此步骤确保了unsup_loader_0和unsup_loader_1的输出顺序不同
                                     num_workers=num_workers,
                                     pin_memory=False)

    return train_sup_loader, train_unsup_loader_0, train_unsup_loader_1



def save_model(model, save_path, iteration, loss, metric):
    torch.save({
        'net': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'epoch': iteration,
        'loss': loss,
        'metric': metric
    }, save_path)

def format_logs(logs):
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s

def generate_mixed_images(sample1, sample2, mask):
    # [x1 y1], [x2, y2] -> [mix_x, mix_y]    x, y represent bi-temporal images
    mix_x = sample1[0] * mask + sample2[0] * (1-mask)
    mix_y = sample1[1] * mask + sample2[1] * (1-mask)
    return [mix_x, mix_y]

def generate_mixed_images_intra(sample, mask):
    # [x1 y1], [y1, x1] -> [mix_x, mix_y]    x, y represent bi-temporal images
    mix_x = sample[0] * mask + sample[1] * (1-mask)
    mix_y = sample[1] * mask + sample[0] * (1-mask)
    return [mix_x, mix_y]

def generate_mixed_images_withlabels(sample1, sample2, label, mask):
    # [x1 y1], [x2, y2] -> [mix_x, mix_y]    x, y represent bi-temporal images
    mix_x = sample1[0] * mask + sample2[0] * (1-mask)
    mix_y = sample1[1] * mask + sample2[1] * (1-mask)
    mix_label = label[0] * mask + label[1] * (1-mask)
    return mix_x, mix_y, mix_label


# masks = generate_salience_mask(logits_u1_tea)    # [B,1,H,W]
def generate_salience_mask(logits, mask_size):
    # mask size [64, 64]
    # logits [B, 2, H, W]  -> mask [B, 1, H, W]
    prob = F.softmax(logits, dim=1)
    prob_change = prob[:,1,:,:]
    torch_resize = Resize(mask_size)
    small_prob_change = torch_resize(prob_change)
    AddGaussianBlur = GaussianBlur((5,5),(0.1, 2.0))
    small_prob_change_blurred = AddGaussianBlur(small_prob_change) # [B, mask_size, mask_size]

    B, _, H, W = logits.shape

    mask = torch.ones((B, H, W))

    for batch in range(B):
        heatmap = small_prob_change_blurred[batch]
        max_index = torch.argmax((heatmap))
        # 根据索引计算点坐标
        row = torch.floor(max_index/mask_size[1])
        col = max_index % mask_size[1]
        # 放大后的行列号
        row = row / mask_size[0] * H
        col = col / mask_size[1] * W
        top_left = [i.to(torch.int) for i in [row - mask_size[0] / 2, col - mask_size[1] / 2]]
        bottom_right = [i.to(torch.int) for i in [row + mask_size[0] / 2, col + mask_size[1] / 2]]

        top_left[0], top_left[1] = max(top_left[0], 0), max(top_left[1], 0)
        bottom_right[0], bottom_right[1] = min(bottom_right[0], 256), min(bottom_right[1], 256)
        mask[batch, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 0

    mask = mask.unsqueeze(1)

    return mask

def generate_object_mask(logits, threshold=0.5):
    # mask size [64, 64]
    # logits [B, 2, H, W]  -> mask [B, 1, H, W]
    prob = F.softmax(logits, dim=1)
    prob_change = prob[:, 1, :, :]
    mask = (prob_change < threshold).float() # change的region为0

    mask = mask.unsqueeze(1)

    return mask


