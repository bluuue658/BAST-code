
import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from networks.net_factory import BCP_net
from utils import losses, ramps, val_2d

# ---- 导入增强模块 (只用边界相关，ensemble 多类版写在本文件) ----
from BAST import (
    make_boundary_band,
    boundary_aux_loss,
    BoundaryAuxHead,
)



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/ACDC', help='ACDC data directory')
parser.add_argument('--exp', type=str, default='BCP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default=6.0, help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')


# Ensemble
parser.add_argument('--ensemble_alpha', type=float, default=0.7,
                    help='Teacher weight in ensemble. 1.0=off (original BCP), 0.7=recommended')
# Boundary weighting
parser.add_argument('--enable_boundary', type=int, default=1,
                    help='Enable boundary-weighted loss. 0=off, 1=on')
parser.add_argument('--lambda_boundary', type=float, default=1.0,
                    help='Extra weight at boundary regions')
parser.add_argument('--boundary_radius', type=int, default=1,
                    help='Boundary band width in pixels')
parser.add_argument('--boundary_rampup', type=int, default=10000,
                    help='Iterations before boundary loss reaches full strength')
# Auxiliary head
parser.add_argument('--enable_aux_head', type=int, default=1,
                    help='Enable boundary auxiliary head. 0=off, 1=on')
parser.add_argument('--lambda_aux', type=float, default=0.3,
                    help='Weight for auxiliary boundary detection loss')
parser.add_argument('--aux_mid_channels', type=int, default=16,
                    help='Mid channels in auxiliary head')
parser.add_argument('--run_phase', type=str, default='all',
                    choices=['all', 'pretrain_only', 'selftrain_only'],
                    help='all: pretrain+selftrain; pretrain_only: only pretrain; selftrain_only: only selftrain and reuse existing pretrain')

args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)



def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        assert(labels.max() != 0)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)
    return torch.Tensor(batch_list).cuda()

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs

def get_current_consistency_weight(epoch):
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y * 4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]



def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    return loss_dice, loss_ce



def mix_loss_enhanced(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5,
                      unlab=False, bweight=None):
 
   
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask


    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight

 
    ce_img = CE(output, img_l)
    ce_patch = CE(output, patch_l)

    if bweight is not None:
        ce_img = ce_img * bweight
        ce_patch = ce_patch * bweight

    loss_ce = image_weight * (ce_img * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (ce_patch * patch_mask).sum() / (patch_mask.sum() + 1e-16)

    return loss_dice, loss_ce



def ensemble_pseudo_label_acdc(teacher_logit, student_logit=None, alpha=0.7):
  
    with torch.no_grad():
        t_prob = F.softmax(teacher_logit, dim=1)  # [B, C, H, W]

        if student_logit is None or alpha >= 1.0:
            _, pseudo = torch.max(t_prob, dim=1)   # [B, H, W]
            pseudo = get_ACDC_2DLargestCC(pseudo)
            conf = t_prob.max(dim=1)[0]
            return pseudo.long(), conf, 1.0

        s_prob = F.softmax(student_logit, dim=1)
        ens_prob = alpha * t_prob + (1.0 - alpha) * s_prob
        _, pseudo = torch.max(ens_prob, dim=1)     # [B, H, W]
        pseudo = get_ACDC_2DLargestCC(pseudo)
        conf = ens_prob.max(dim=1)[0]

        t_pred = t_prob.argmax(dim=1)
        s_pred = s_prob.argmax(dim=1)
        agree = float((t_pred == s_pred).float().mean().item())

    return pseudo.long(), conf, agree

def generate_boundary_gt_multiclass(label_map, num_classes=4, radius=1):
  
    boundary = torch.zeros_like(label_map.float()).unsqueeze(1)  # [B, 1, H, W]
    for c in range(1, num_classes):
        class_mask = (label_map == c).float()  # [B, H, W]
        class_boundary = make_boundary_band(class_mask, radius=radius)  # [B, 1, H, W]
        boundary = torch.max(boundary, class_boundary)
    return boundary

def boundary_weight_map_multiclass(pseudo_label, num_classes=4,
                                    lambda_boundary=1.0, radius=1,
                                    iter_num=0, rampup=10000):
    
    ramp = min(float(iter_num) / max(float(rampup), 1.0), 1.0) if rampup > 0 else 1.0
    eff_lambda = lambda_boundary * ramp
    band = generate_boundary_gt_multiclass(pseudo_label, num_classes, radius)
    weight = 1.0 + eff_lambda * band.squeeze(1)  # [B, H, W]
    return weight, band


def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def self_train_orig_exact(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                img_mask, loss_mask = generate_mask(img_a)
                unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num//150)

            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
            out_unl = model(net_input_unl)
            out_l = model(net_input_l)
            unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice, l_ce = mix_loss(out_l, lab_b, plab_b, loss_mask, u_weight=args.u_weight)


            loss_ce = unl_ce + l_ce 
            loss_dice = unl_dice + l_dice

            loss = (loss_dice + loss_ce) / 2            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            if iter_num % 20 == 0:
                image = net_input_unl[1, 0:1, :, :]
                writer.add_image('train/Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = unl_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/Un_GroundTruth', labs, iter_num)

                image_l = net_input_l[1, 0:1, :, :]
                writer.add_image('train/L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_l, dim=1), dim=1, keepdim=True)
                writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = l_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/L_GroundTruth', labs_l, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()




def self_train(args, pre_snapshot_path, snapshot_path):
    use_plain_baseline = (
        args.ensemble_alpha == 1.0 and
        args.enable_boundary == 0 and
        args.enable_aux_head == 0
    )
    if use_plain_baseline:
        logging.info(">>> PLAIN BASELINE mode: dispatching to original exact self_train from ACDC_BCP_train.py")
        return self_train_orig_exact(args, pre_snapshot_path, snapshot_path)

    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

 
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))


    aux_head = None
    if args.enable_aux_head:
        aux_head = BoundaryAuxHead(
            in_channels=num_classes,  # ACDC: 4 classes
            mid_channels=args.aux_mid_channels,
            is_2d=True  # ACDC 是 2D
        ).cuda()
       
        optimizer.add_param_group({'params': aux_head.parameters()})
        logging.info("Boundary Auxiliary Head enabled (in_channels={}, mid_channels={})".format(
            num_classes, args.aux_mid_channels))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("=== Enhancement Config ===")
    logging.info("ensemble_alpha={}, enable_boundary={}, enable_aux_head={}".format(
        args.ensemble_alpha, args.enable_boundary, args.enable_aux_head))
    logging.info("lambda_boundary={}, lambda_aux={}".format(
        args.lambda_boundary, args.lambda_aux))
    logging.info("boundary_radius={}, boundary_rampup={}".format(
        args.boundary_radius, args.boundary_rampup))

    model.train()
    ema_model.train()
    if aux_head is not None:
        aux_head.train()

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)

  
    use_plain_baseline = (
        args.ensemble_alpha == 1.0 and
        args.enable_boundary == 0 and
        args.enable_aux_head == 0
    )
    if use_plain_baseline:
        logging.info(">>> PLAIN BASELINE mode: all enhancements OFF, following original ACDC_BCP_train.py exactly")
    else:
        logging.info(">>> ENHANCED mode: at least one enhancement is ON")

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]

            if use_plain_baseline:
              
                with torch.no_grad():
                    pre_a = ema_model(uimg_a)
                    pre_b = ema_model(uimg_b)
                    plab_a = get_ACDC_masks(pre_a, nms=1)
                    plab_b = get_ACDC_masks(pre_b, nms=1)
                    img_mask, loss_mask = generate_mask(img_a)
                  
                    unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                    l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
                consistency_weight = get_current_consistency_weight(iter_num // 150)

                net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
                net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
                out_unl = model(net_input_unl)
                out_l = model(net_input_l)
                unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
                l_dice, l_ce = mix_loss(out_l, lab_b, plab_b, loss_mask, u_weight=args.u_weight)

                loss_ce = unl_ce + l_ce
                loss_dice = unl_dice + l_dice
                loss = (loss_dice + loss_ce) / 2

                loss_aux = torch.tensor(0.0)
                agree_a, agree_b = 1.0, 1.0

            else:
               
                with torch.no_grad():
                    pre_a = ema_model(uimg_a)
                    pre_b = ema_model(uimg_b)

                    if args.ensemble_alpha < 1.0:
                        bn_backup = {}
                        for name, module in model.named_modules():
                            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                                bn_backup[name] = (module.running_mean.clone(),
                                                   module.running_var.clone(),
                                                   module.num_batches_tracked.clone())

                        student_out_a = model(uimg_a)
                        student_out_b = model(uimg_b)
                        for name, module in model.named_modules():
                            if name in bn_backup:
                                module.running_mean.copy_(bn_backup[name][0])
                                module.running_var.copy_(bn_backup[name][1])
                                module.num_batches_tracked.copy_(bn_backup[name][2])
                        plab_a, conf_a, agree_a = ensemble_pseudo_label_acdc(
                            pre_a, student_out_a, alpha=args.ensemble_alpha)
                        plab_b, conf_b, agree_b = ensemble_pseudo_label_acdc(
                            pre_b, student_out_b, alpha=args.ensemble_alpha)
                    else:
                        plab_a = get_ACDC_masks(pre_a, nms=1)
                        plab_b = get_ACDC_masks(pre_b, nms=1)
                        agree_a, agree_b = 1.0, 1.0

                    img_mask, loss_mask = generate_mask(img_a)
                 
                    unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                    l_label = lab_b * img_mask + ulab_b * (1 - img_mask)

                consistency_weight = get_current_consistency_weight(iter_num // 150)

               
                net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
                net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
                out_unl = model(net_input_unl)
                out_l = model(net_input_l)
                mixl_lab_unl = plab_a   # net_input_unl 的 background (unlabeled) 标签
                mixl_lab_l = lab_a      # net_input_unl 的 patch (labeled) 标签
                mixu_lab_l = lab_b      # net_input_l 的 background (labeled) 标签
                mixu_lab_u = plab_b     # net_input_l 的 patch (unlabeled) 标签

                if args.enable_boundary:  
                    combined_lab_unl = plab_a * img_mask + lab_a * (1 - img_mask)
                    combined_lab_l = lab_b * img_mask + plab_b * (1 - img_mask)
                    all_combined = torch.cat([combined_lab_unl, combined_lab_l], dim=0)
                    all_bw, all_band = boundary_weight_map_multiclass(
                        all_combined,
                        num_classes=num_classes,
                        lambda_boundary=args.lambda_boundary,
                        radius=args.boundary_radius,
                        iter_num=iter_num,
                        rampup=args.boundary_rampup)
                    n_unl = combined_lab_unl.shape[0]
                    bweight_unl = all_bw[:n_unl]
                    bweight_l = all_bw[n_unl:]
                else:
                    bweight_unl = None
                    bweight_l = None

                unl_dice, unl_ce = mix_loss_enhanced(
                    out_unl, plab_a, lab_a, loss_mask,
                    u_weight=args.u_weight, unlab=True, bweight=bweight_unl)
                l_dice, l_ce = mix_loss_enhanced(
                    out_l, lab_b, plab_b, loss_mask,
                    u_weight=args.u_weight, bweight=bweight_l)

                loss_ce = unl_ce + l_ce
                loss_dice = unl_dice + l_dice
                loss = (loss_dice + loss_ce) / 2

            
                loss_aux = torch.tensor(0.0, device=loss.device)
                if aux_head is not None and args.enable_aux_head:
                    ramp = min(float(iter_num) / max(float(args.boundary_rampup), 1.0), 1.0)

                    # net_input_unl: background=unlabeled, patch=labeled
                    # loss_mask=1 的区域对应 unlabeled (background), =0 对应 labeled (patch)
                    # 我们只在有 GT 的 labeled patch 区域监督 aux head
                    bpred_unl = aux_head(out_unl)
                    bgt_unl = generate_boundary_gt_multiclass(lab_a, num_classes, args.boundary_radius)
                    patch_mask_unl = (1 - loss_mask).unsqueeze(1).float()  # labeled patch 区域
                    loss_aux_unl = boundary_aux_loss(bpred_unl, bgt_unl, valid_mask=patch_mask_unl)

             
                    # loss_mask=1 的区域对应 labeled (background)
                    bpred_l = aux_head(out_l)
                    bgt_l = generate_boundary_gt_multiclass(lab_b, num_classes, args.boundary_radius)
                    bg_mask_l = loss_mask.unsqueeze(1).float()  # labeled background 区域
                    loss_aux_l = boundary_aux_loss(bpred_l, bgt_l, valid_mask=bg_mask_l)

                    loss_aux = (loss_aux_unl + loss_aux_l) * args.lambda_aux * ramp
                    loss = loss + loss_aux

              
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

          
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            if not use_plain_baseline:
                if args.enable_aux_head:
                    writer.add_scalar('info/loss_aux', loss_aux.item(), iter_num)
                if args.ensemble_alpha < 1.0:
                    writer.add_scalar('info/agree_ratio', (agree_a + agree_b) / 2, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))


            if iter_num % 20 == 0:
                image = net_input_unl[1, 0:1, :, :]
                writer.add_image('train/Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = unl_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/Un_GroundTruth', labs, iter_num)

                image_l = net_input_l[1, 0:1, :, :]
                writer.add_image('train/L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_l, dim=1), dim=1, keepdim=True)
                writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = l_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/L_GroundTruth', labs_l, iter_num)

          
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                   
                    if aux_head is not None:
                        torch.save(aux_head.state_dict(),
                                   os.path.join(snapshot_path, 'aux_head_best.pth'))

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()
                if aux_head is not None:
                    aux_head.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()



if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    exp_name = args.exp
    if args.ensemble_alpha < 1.0:
        exp_name += '_ens{}'.format(args.ensemble_alpha)
    if args.enable_boundary:
        exp_name += '_bd{}'.format(args.lambda_boundary)
    if args.enable_aux_head:
        exp_name += '_aux{}'.format(args.lambda_aux)
    exp_name += '_s{}'.format(args.seed)

    pre_snapshot_path = "./model/BCP/ACDC_{}_s{}_{}_labeled/pre_train".format(args.exp, args.seed, args.labelnum)
    self_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/self_train".format(exp_name, args.labelnum)

    print("=" * 60)
    print("ACDC BCP Enhanced Training")
    print("Pre-train path: {}".format(pre_snapshot_path))
    print("Self-train path: {}".format(self_snapshot_path))
    print("Exp name: {}".format(exp_name))
    print("Ensemble alpha: {}".format(args.ensemble_alpha))
    print("Enable boundary: {}".format(args.enable_boundary))
    print("Enable aux head: {}".format(args.enable_aux_head))
    print("Seed: {}".format(args.seed))
    print("=" * 60)

    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

   
    current_script = os.path.abspath(__file__)
    if os.path.exists(current_script):
        shutil.copy(current_script, self_snapshot_path)

  
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(stdout_handler)

    pre_model_path = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))

    if args.run_phase in ['all', 'pretrain_only']:
        pre_file_handler = logging.FileHandler(pre_snapshot_path + "/log.txt")
        pre_file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(pre_file_handler)
        logging.info(str(args))
        pre_train(args, pre_snapshot_path)
        logger.removeHandler(pre_file_handler)

        if args.run_phase == 'pretrain_only':
            logging.info("Finished pretrain_only. Exiting before self-train.")
            raise SystemExit(0)

    if args.run_phase == 'selftrain_only':
        if not os.path.exists(pre_model_path):
            raise FileNotFoundError("Expected frozen pretrain not found: {}".format(pre_model_path))
        logging.info("Pre-trained model found at {}, skipping pre-train.".format(pre_model_path))

    self_file_handler = logging.FileHandler(self_snapshot_path + "/log.txt")
    self_file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(self_file_handler)
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
