import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.func import get_train_val_loader, Logger, get_learning_rate, save_model, format_logs, get_train_loader, list2device, generate_mixed_images
from utils.metrics import *
import torch
import torch.nn as nn
import time
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import models
import itertools
from loaders.datasets import S2lookingDataset_all
from utils.lr_schedules import make_lr_schedulers
from utils.ValEpoch import ValEpoch
from tqdm import tqdm


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="train process")
    parser.add_argument("--work_dirs", type=str, default='../semi_checkpoints/S2looking/supervised')
    parser.add_argument("--log", type=str, default='supervised_1.0')
    parser.add_argument('--ratio', '--labeled_ratio', type=float, default=1.0)
    parser.add_argument('--num_epochs', '--num_epochs', type=int, default=100)

    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument('--lr_sched', type=str, default='none') # ['none', 'stepped', 'cosine', 'poly']
    parser.add_argument('--lr_step_epochs', type=str, default='')
    parser.add_argument('--lr_step_gamma', type=float, default=0.1)
    parser.add_argument('--lr_poly_power', type=float, default=0.9)
    parser.add_argument('--aug_strong_colour', default=True)
    parser.add_argument('--freeze_bn', default=False)   # 待完善

    # parser.add_argument('--epoch_start_unsup', type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="train dataset batch size.")
    parser.add_argument("--val-batch-size", type=int, default=1,
                        help="val dataset batch size.")
    parser.add_argument("--opt_type", type=str, default='adam',
                        help="val dataset batch size.")
    # parser.add_argument("--data_parallel", action='store_true', default=True)

    return parser.parse_args()


def train():
    num_classes = 2
    torch_device = torch.device('cuda')
    args = get_arguments()

    # logger
    save_dir = os.path.join(args.work_dirs, args.log)
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, "train.log"))
    logger.write(str(args))

    # build network
    student_net = models.ResNet50_CD(num_classes, pretrained="/home/qidi/project/3x3resnet50-imagenet.pth").to(torch_device)

    # student_net = models.UNet(6).to(torch_device)

    # student_optim
    # if args.opt_type == 'adam':
    #     student_optim = torch.optim.Adam([
    #         dict(params=student_net.pretrained_parameters(), lr=args.learning_rate * 0.1),
    #         dict(params=student_net.new_parameters(), lr=args.learning_rate)])
    # elif args.opt_type == 'sgd':
    #     student_optim = torch.optim.SGD([
    #         dict(params=student_net.pretrained_parameters(), lr=args.learning_rate * 0.1),
    #         dict(params=student_net.new_parameters(), lr=args.learning_rate)],
    #         momentum=0.9, nesterov=False, weight_decay=5e-4)
    # else:
    #     raise ValueError('Unknown opt_type {}'.format(args.opt_type))
    student_optim = torch.optim.Adam(student_net.parameters(), lr=args.learning_rate)

    # CELoss
    supervised_loss = nn.CrossEntropyLoss()

    # val
    metric = ChangeMetrics(False)
    val_runner = ValEpoch(num_classes, student_net, supervised_loss, metric)

    print("Build network")

    # dataset
    train_dataset = S2lookingDataset_all("train", supervised_train=True, transforms_unsup=None)  # label [B, H, W]
    val_dataset = S2lookingDataset_all("val")

    # loader
    # loaders = get_train_val_loader(train_dataset, val_dataset, args.batch_size, args.val_batch_size,
    #                                args.ratio, train_split_path="/home/peter/sqd/semi_checkpoints/S2looking/train_split.pkl", work_dir=args.work_dirs)
    loaders = get_train_val_loader(train_dataset, val_dataset, args.batch_size, args.val_batch_size,
                                   args.ratio,
                                   train_split_path="/home/qidi/project/semi_checkpoints/S2looking/train_split.pkl",
                                   work_dir=args.work_dirs)


    train_sup_loader, _, val_loader = loaders

    # Create iterators
    train_sup_iter = iter(train_sup_loader)

    # scheduler
    label_size = len(train_dataset) * args.ratio
    iters_per_epoch = int(label_size // args.batch_size)  # 无标签的样本数除以batch size
    # print("fork",iters_per_epoch)
    total_iters = iters_per_epoch * args.num_epochs

    lr_epoch_scheduler, lr_iter_scheduler = make_lr_schedulers(
        optimizer=student_optim, total_iters=total_iters, schedule_type=args.lr_sched,
        step_epochs=args.lr_step_epochs, step_gamma=args.lr_step_gamma, poly_power=args.lr_poly_power
    )

    iter_i = 0
    print('Training...')
    best_val_metric = 0
    bms = 0
    for epoch_i in range(args.num_epochs):
        if lr_epoch_scheduler is not None:
            lr_epoch_scheduler.step(epoch_i)

        t1 = time.time()
        ramp_val = 1.0

        student_net.train()

        if args.freeze_bn:                            # 待完善
            student_net.freeze_batchnorm()

        sup_loss_acc = 0.0
        n_sup_batches = 0



        for sup_batch in tqdm(itertools.islice(train_sup_iter, iters_per_epoch)):  # 一个epoch包含的iteration数
            if lr_iter_scheduler is not None:
                lr_iter_scheduler.step(iter_i)
            student_optim.zero_grad()

            #
            # Supervised branch
            #

            batch_x = list2device(sup_batch['image'], torch_device)
            batch_y = sup_batch['labels'].to(torch_device)

            logits_sup = student_net(batch_x)
            sup_loss = supervised_loss(logits_sup, batch_y)
            sup_loss.backward()

            student_optim.step()

            sup_loss_val = float(sup_loss.detach())
            if np.isnan(sup_loss_val):
                print('NaN detected; network dead, bailing.')
                return

            sup_loss_acc += sup_loss_val
            n_sup_batches += 1
            iter_i += 1

        sup_loss_acc /= n_sup_batches

        t2 = time.time()

        # train results
        print('Epoch {}: took {:.3f}s, TRAIN clf loss={:.6f}'.format(
              epoch_i + 1, t2 - t1, sup_loss_acc))
        train_log = {'sup_loss': sup_loss_acc}

        # Eval this epoch
        val_log = val_runner.run(val_loader)
        val_metric = val_log['f1']

        # 保存最新模型
        save_model(student_net, os.path.join(save_dir, 'latest.pth'), epoch_i, val_log['loss'], val_metric)

        # 保存最好metric模型
        if val_log['f1'] > best_val_metric:
            best_val_metric = val_log['f1']
            bms = val_log
            save_model(student_net, os.path.join(save_dir, 'best.pth'), epoch_i, val_log['loss'], val_metric)

        logger.write('Epoch:\t' + str(epoch_i))
        logger.write('Train:\t' + format_logs(train_log))
        logger.write('Val:\t' + format_logs(val_log))
        logger.write("Best:\t" + format_logs(bms))
        logger.write("\n")

        print("train:", train_log)
        print("val:", val_log)
        print("best_metric:\t" + format_logs(bms))


if __name__ == "__main__":
    train()