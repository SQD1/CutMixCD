import argparse
from utils.func import Logger, save_model, format_logs, get_train_loader, list2device, generate_mixed_images, generate_salience_mask
from utils.metrics import *
import torch
import torch.nn as nn
import os
import time
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import models
import itertools
import utils.optim_weight_ema as optim_weight_ema
from loaders.datasets import S2lookingDataset_all
from utils.mask_gen import BoxMaskGenerator, AddMaskParamsToBatch
from utils.lr_schedules import make_lr_schedulers
import torchvision.transforms as tvt
from utils.ValEpoch import ValEpoch
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="train process")
    parser.add_argument("--work_dirs", type=str, default='../semi_checkpoints/S2looking/CutMixCD')
    parser.add_argument('--ratio', '--labeled_ratio', type=float, default=0.2)  # labeled data ratio
    parser.add_argument('--mask_size', type=list, default=[64,64])
    parser.add_argument("--log", type=str, default='cutmix_0.2')
    parser.add_argument("--data_root", type=str, default='/data0/qidi/S2looking')

    parser.add_argument('--train_split_path', type=str, default=None)
    # "train_split_path" is the location of a pkl file which records the split of labeled and unlabeled data
    # For the first run, you can set it to None, and the train_split.pkl will be generated automatically in work_dirs
    # After that, you can set the path of generated train_split.pkl to ensure each experiment is conducted with the same split for a dataset


    parser.add_argument('--num_epochs', '--num_epochs', type=int, default=100) # epoch of unsupervised training
    parser.add_argument('--epoch_start_unsup', type=int, default=50)           # epoch to start unsupervised training
                                                                                            # flexible for different datasets

    parser.add_argument('--cons_weight', type=float, default=1.0)
    parser.add_argument('--conf_thresh', type=float, default=0.97) # 0.97
    parser.add_argument('--conf_per_pixel', type=bool, default=False)
    parser.add_argument('--cons_loss_fn', type=str, default='var')

    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument('--lr_sched', type=str, default='cosine') # ['none', 'stepped', 'cosine', 'poly']
    parser.add_argument('--lr_step_epochs', type=str, default='')
    parser.add_argument('--lr_step_gamma', type=float, default=0.1)
    parser.add_argument('--lr_poly_power', type=float, default=0.9)
    parser.add_argument('--aug_strong_colour', default=True)
    parser.add_argument('--freeze_bn', default=False)

    parser.add_argument("--batch-size", type=int, default=8,
                        help="train dataset batch size.")
    parser.add_argument("--val-batch-size", type=int, default=1,
                        help="val dataset batch size.")
    parser.add_argument("--opt_type", type=str, default='adam',
                        help="val dataset batch size.")

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
    student_net = models.ResNet50_CD(num_classes, pretrained=True).to(torch_device)
    teacher_net = models.ResNet50_CD(num_classes, pretrained=True).to(torch_device)

    student_optim = torch.optim.Adam(student_net.parameters(), lr=args.learning_rate)

    # teacher_optim
    for p in teacher_net.parameters():
        p.requires_grad = False
    teacher_optim = optim_weight_ema.EMAWeightOptimizer(teacher_net, student_net, ema_alpha=0.99)
    eval_net = teacher_net

    # CELoss
    supervised_loss = nn.CrossEntropyLoss()

    # val
    metric = ChangeMetrics(False)

    print("Build network")

    if args.aug_strong_colour:
        train_unsup_transforms = tvt.Compose([
            tvt.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            tvt.ToTensor(),
            tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_unsup_transforms = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # dataset
    train_dataset = S2lookingDataset_all(args.data_root, "train", supervised_train=True, transforms_unsup=None)  # label [B, H, W]
    train_unsup_dataset = S2lookingDataset_all(args.data_root, "train", supervised_train=False, transforms_unsup=train_unsup_transforms)
    val_dataset = S2lookingDataset_all(args.data_root, "val")

    # random mask (not used)
    mask_generator = BoxMaskGenerator((0.25, 0.25), random_aspect_ratio=False)
    add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator)

    # loader
    loaders = get_train_loader(train_dataset, train_unsup_dataset, add_mask_params_to_batch,
                        args.batch_size, args.ratio, train_split_path=args.train_split_path, work_dir=args.work_dirs)

    train_sup_loader, train_unsup_loader_0, train_unsup_loader_1 = loaders
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.val_batch_size,
                            num_workers=1,
                            shuffle=False,
                            pin_memory=True)

    # Create iterators
    train_sup_iter = iter(train_sup_loader)
    train_unsup_iter_0 = iter(train_unsup_loader_0) if train_unsup_loader_0 is not None else None
    train_unsup_iter_1 = iter(train_unsup_loader_1) if train_unsup_loader_1 is not None else None

    # scheduler
    unlabel_size = len(train_dataset) * (1 - args.ratio)
    iters_per_epoch = int(unlabel_size // args.batch_size)  # 无标签的样本数除以batch size
    sup_iters_per_epoch = int(len(train_dataset) * args.ratio // args.batch_size)

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
        if teacher_net is not student_net:
            teacher_net.train()

        if args.freeze_bn:
            student_net.freeze_batchnorm()
            if teacher_net is not student_net:
                teacher_net.freeze_batchnorm()

        sup_loss_acc = 0.0
        consistency_loss_acc = 0.0
        feat_contras_loss_acc = 0.0
        conf_rate_acc = 0.0
        n_sup_batches = 0
        n_unsup_batches = 0

        if args.epoch_start_unsup > 0 and epoch_i < args.epoch_start_unsup:
            iters = sup_iters_per_epoch
            eval_net = student_net
        else:
            iters = iters_per_epoch
            eval_net = teacher_net

        val_runner = ValEpoch(num_classes, eval_net, supervised_loss, metric)

        # load best model of the supervised epoches.
        if epoch_i == args.epoch_start_unsup:
            print("load best model of the supervised epoches.")
            checkpoint = torch.load(os.path.join(save_dir, 'best.pth'))
            student_net.load_state_dict(checkpoint['net'])
            teacher_net.load_state_dict(checkpoint['net'])


        for sup_batch in tqdm(itertools.islice(train_sup_iter, iters)):  # 一个epoch包含的iteration数
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


            if args.cons_weight > 0.0 and epoch_i >= args.epoch_start_unsup:

                #
                #  Unsupervised branch  cutmix
                #
                unsup_batch0 = next(train_unsup_iter_0)
                unsup_batch1 = next(train_unsup_iter_1)

                # The teacher path should come from sample 0 that has weaker
                # augmentation (no colour augmentation), where the student should
                # use sample 1 that has stronger augmentation

                batch_ux0_tea = list2device(unsup_batch0['sample0']['image'], torch_device)
                batch_ux0_stu = list2device(unsup_batch0['sample1']['image'], torch_device)
                # batch_um0 = unsup_batch0['sample0']['mask'].to(torch_device)
                batch_ux1_tea = list2device(unsup_batch1['sample0']['image'], torch_device)
                batch_ux1_stu = list2device(unsup_batch1['sample1']['image'], torch_device)

                # batch_mix_masks = unsup_batch0['mask'].to(torch_device)   # (N,1,H,W)    中间的mask为0

                # Get teacher predictions for original images
                with torch.no_grad():
                    logits_u0_tea = teacher_net(batch_ux0_tea).detach()
                    logits_u1_tea = teacher_net(batch_ux1_tea).detach()

                # generate salience mask from logits_u1_tea  [B, 2, H, W]
                # change-aware masks
                batch_mix_masks = generate_salience_mask(logits_u1_tea, args.mask_size).to(torch_device)    # [B,1,H,W] 中间的mask为0
                # Mix images with masks
                batch_ux_stu_mixed = generate_mixed_images(batch_ux0_stu, batch_ux1_stu, batch_mix_masks)

                # Get student prediction and features for mixed image
                logits_cons_stu, feat_mixed = student_net(batch_ux_stu_mixed, return_features=True)   # output the Feature contrastive loss

                # Mix teacher predictions using same mask
                logits_cons_tea = logits_u0_tea * batch_mix_masks + logits_u1_tea * (1 - batch_mix_masks)

                # Logits -> probs
                prob_cons_tea = F.softmax(logits_cons_tea.detach(), dim=1)
                prob_cons_stu = F.softmax(logits_cons_stu, dim=1)


                # Confidence thresholding
                if args.conf_thresh > 0.0:
                    # Compute confidence of teacher predictions
                    conf_tea = prob_cons_tea.max(dim=1)[0]
                    # Compute confidence mask
                    conf_mask = (conf_tea >= args.conf_thresh).float()[:, None, :, :]
                    # Record rate for reporting
                    conf_rate_acc += float(conf_mask.mean())
                    # Average confidence mask if requested
                    if not args.conf_per_pixel:
                        conf_mask = conf_mask.mean()

                    loss_mask = conf_mask

                # Compute per-pixel consistency loss
                # Note that the way we aggregate the loss across the class/channel dimension (1)
                # depends on the loss function used. Generally, summing over the class dimension
                # keeps the magnitude of the gradient of the loss w.r.t. the logits
                # nearly constant w.r.t. the number of classes. When using logit-variance,
                # dividing by `sqrt(num_classes)` helps.

                if args.cons_loss_fn == 'var':
                    delta_prob = prob_cons_stu - prob_cons_tea
                    consistency_loss = delta_prob * delta_prob
                    consistency_loss = consistency_loss.sum(dim=1, keepdim=True)
                elif args.cons_loss_fn == 'kld':
                    consistency_loss = F.kl_div(F.log_softmax(logits_cons_stu, dim=1), prob_cons_tea, reduce=False)
                    consistency_loss = consistency_loss.sum(dim=1, keepdim=True)
                else:
                    raise ValueError('Unknown consistency loss function {}'.format(args.cons_loss_fn))

                # Apply consistency loss mask and take the mean over pixels and images
                if loss_mask:
                    consistency_loss = (consistency_loss * loss_mask).mean()

                # Feature contrastive loss
                feat_contras_loss = Alg_loss(prob_cons_tea, feat_mixed, args.conf_thresh)

                # unsupervised loss and back-prop
                unsup_loss = consistency_loss * args.cons_weight + feat_contras_loss.mean()
                # unsup_loss = consistency_loss * args.cons_weight
                unsup_loss.backward()

                consistency_loss_acc += float(consistency_loss.detach())
                feat_contras_loss_acc += float(feat_contras_loss.detach())

                n_unsup_batches += 1

            student_optim.step()
            if teacher_optim is not None:
                teacher_optim.step()

            sup_loss_val = float(sup_loss.detach())
            if np.isnan(sup_loss_val):
                print('NaN detected; network dead, bailing.')
                return

            sup_loss_acc += sup_loss_val
            n_sup_batches += 1
            iter_i += 1

        sup_loss_acc /= n_sup_batches
        if n_unsup_batches > 0:
            consistency_loss_acc /= n_unsup_batches
            feat_contras_loss_acc /= n_unsup_batches
            conf_rate_acc /= n_unsup_batches

        t2 = time.time()

        # train results
        print('Epoch {}: took {:.3f}s, TRAIN clf loss={:.6f}, consistency loss={:.6f}, contras loss={:.6f}, conf rate={:.3%}'.format(
              epoch_i + 1, t2 - t1, sup_loss_acc, consistency_loss_acc, feat_contras_loss_acc, conf_rate_acc))
        train_log = {'sup_loss': sup_loss_acc, 'consistency_loss':consistency_loss_acc, 'feat_contras_loss':feat_contras_loss_acc,'conf_rate':conf_rate_acc}

        # Eval this epoch
        val_log = val_runner.run(val_loader)
        val_metric = val_log['f1']


        # 保存最新模型
        save_model(eval_net, os.path.join(save_dir, 'latest.pth'), epoch_i, val_log['loss'], val_metric)

        # 保存最好metric模型
        if val_log['f1'] > best_val_metric:
            best_val_metric = val_log['f1']
            bms = val_log
            save_model(eval_net, os.path.join(save_dir, 'best.pth'), epoch_i, val_log['loss'], val_metric)
            # 保存相应的student模型
            torch.save({
                'net': student_net.module.state_dict() if hasattr(student_net, 'module') else student_net.state_dict(),
            }, os.path.join(save_dir, 'student.pth'))

        logger.write('Epoch:\t' + str(epoch_i))
        logger.write('Train:\t' + format_logs(train_log))
        logger.write('Val:\t' + format_logs(val_log))
        logger.write("Best:\t" + format_logs(bms))
        logger.write("\n")

        print("train:", train_log)
        print("val:", val_log)
        print("best_metric:\t" + format_logs(bms))


def Alg_loss(weak_prob_ul, strong_feat_ul, threshold):
    feat_size = strong_feat_ul.size()[-2:]
    weak_prob_ul = F.interpolate(weak_prob_ul, size=feat_size, mode='nearest')

    mask_unit = weak_prob_ul.ge(threshold).float()
    weight = (mask_unit.sum(dim=-1).sum(dim=-1) + 1e-5).unsqueeze(dim=-1)
    mask_unit = mask_unit.unsqueeze(dim=2)       # [B, 2, 1, H=32, W=32]
    feat_ul = strong_feat_ul.unsqueeze(dim=1)    # [B, 1, 512, H=32, W=32]

    class_centers = (mask_unit * feat_ul).sum(-1).sum(-1) / weight

    class_centers = F.normalize(class_centers, dim=-1)  # [B, 2, 512]

    loss_contras = - (class_centers[:, 0, :] - class_centers[:, 1, :]) ** 2
    loss_contras = loss_contras.mean()


    dist_pos = torch.bmm(class_centers.permute(1, 0, 2), class_centers.permute(1, 0, 2).permute(0, 2, 1))
    mask_pos = 1 - (dist_pos == 0).float()
    loss_pos = ((0.5 - dist_pos / 2) * mask_pos).mean()

    loss_pos += loss_contras

    return loss_pos

if __name__ == "__main__":
    train()