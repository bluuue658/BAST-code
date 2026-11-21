import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from skimage.measure import label
from torch.utils.data import DataLoader
from utils import losses, ramps, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask, update_ema_variables

from BAST import (
    ensemble_pseudo_label,
    boundary_weight_map,
    generate_boundary_gt,
    boundary_aux_loss,
    BoundaryAuxHead,
)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/LA', help='LA data directory')
parser.add_argument('--exp', type=str,  default='BCP', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default=10.0, help='magnitude')

parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')

parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')


parser.add_argument('--ensemble_alpha', type=float, default=0.7,
                    help='Teacher weight in ensemble. 1.0=off (original BCP), 0.7=recommended')

parser.add_argument('--enable_boundary', type=int, default=1,
                    help='Enable boundary-weighted loss. 0=off, 1=on')
parser.add_argument('--lambda_boundary', type=float, default=1.0,
                    help='Extra weight at boundary regions')
parser.add_argument('--boundary_radius', type=int, default=1,
                    help='Boundary band width in voxels')
parser.add_argument('--boundary_rampup', type=int, default=5000,
                    help='Iterations before boundary loss reaches full strength')

parser.add_argument('--enable_aux_head', type=int, default=1,
                    help='Enable boundary auxiliary head. 0=off, 1=on')
parser.add_argument('--lambda_aux', type=float, default=0.3,
                    help='Weight for auxiliary boundary detection loss')
parser.add_argument('--aux_mid_channels', type=int, default=16,
                    help='Mid channels in auxiliary head')

args = parser.parse_args()

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path, aux_head=None):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    if aux_head is not None:
        state['aux_head'] = aux_head.state_dict()
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2


def mix_loss_enhanced(output, img_l, patch_l, loss_mask, u_weight=0.5,
                      unlab=False, bweight=None):

    DICE = losses.mask_DiceLoss(nclass=2)
    CE_fn = nn.CrossEntropyLoss(reduction='none')

    image_weight, patch_weight = 1.0, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, 1.0

    patch_mask = 1 - loss_mask
    loss_dice = DICE(output_soft, img_l.unsqueeze(1), loss_mask.unsqueeze(1)) * image_weight
    loss_dice += DICE(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight

    ce_per_voxel_img = CE_fn(output, img_l.long())
    ce_per_voxel_patch = CE_fn(output, patch_l.long())

    if bweight is not None:
  
        ce_per_voxel_img = ce_per_voxel_img * bweight
        ce_per_voxel_patch = ce_per_voxel_patch * bweight

    loss_ce = image_weight * (ce_per_voxel_img * loss_mask).sum() / (loss_mask.sum() + 1e-16)
    loss_ce += patch_weight * (ce_per_voxel_patch * patch_mask).sum() / (patch_mask.sum() + 1e-16)

    return (loss_dice + loss_ce) / 2


def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs, _ = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break
        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
        param.detach_()

    aux_head = None
    if args.enable_aux_head:
        aux_head = BoundaryAuxHead(
            in_channels=num_classes,
            mid_channels=args.aux_mid_channels,
            is_2d=False  # LA 是 3D
        ).cuda()
        logging.info("Boundary Auxiliary Head enabled (mid_channels={})".format(args.aux_mid_channels))

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    params_to_optimize = list(model.parameters())
    if aux_head is not None:
        params_to_optimize += list(aux_head.parameters())
    optimizer = optim.SGD(params_to_optimize, lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)

    model.train()
    ema_model.train()
    if aux_head is not None:
        aux_head.train()

    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("=== Enhancement Config ===")
    logging.info("ensemble_alpha={}, enable_boundary={}, enable_aux_head={}".format(
        args.ensemble_alpha, args.enable_boundary, args.enable_aux_head))
    logging.info("lambda_boundary={}, lambda_aux={}".format(
        args.lambda_boundary, args.lambda_aux))
    logging.info("boundary_radius={}, boundary_rampup={}".format(
        args.boundary_radius, args.boundary_rampup))

    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)


    use_plain_baseline = (
        args.ensemble_alpha == 1.0 and
        args.enable_boundary == 0 and
        args.enable_aux_head == 0
    )

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch[args.labeled_bs + sub_bs:]

            if use_plain_baseline:
           
                with torch.no_grad():
                    unoutput_a, _ = ema_model(unimg_a)
                    unoutput_b, _ = ema_model(unimg_b)
                    plab_a = get_cut_mask(unoutput_a, nms=1)
                    plab_b = get_cut_mask(unoutput_b, nms=1)
                    img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

                consistency_weight = get_current_consistency_weight(iter_num // 150)

                mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
                mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
                mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
                mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)

                outputs_l, _ = model(mixl_img)
                outputs_u, _ = model(mixu_img)

                loss_l = mix_loss(outputs_l, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
                loss_u = mix_loss(outputs_u, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
                loss = loss_l + loss_u

                loss_aux = torch.tensor(0.0, device=loss.device)

            else:
 
                with torch.no_grad():
                    unoutput_a, _ = ema_model(unimg_a)
                    unoutput_b, _ = ema_model(unimg_b)

                    if args.ensemble_alpha < 1.0:
                        student_out_a, _ = model(unimg_a)
                        student_out_b, _ = model(unimg_b)
                        plab_a, conf_a, agree_a = ensemble_pseudo_label(
                            unoutput_a, student_out_a,
                            alpha=args.ensemble_alpha, nms_fn=LargestCC_pancreas)
                        plab_b, conf_b, agree_b = ensemble_pseudo_label(
                            unoutput_b, student_out_b,
                            alpha=args.ensemble_alpha, nms_fn=LargestCC_pancreas)
                    else:
                        plab_a = get_cut_mask(unoutput_a, nms=1)
                        plab_b = get_cut_mask(unoutput_b, nms=1)
                        agree_a, agree_b = 1.0, 1.0

                    img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

                consistency_weight = get_current_consistency_weight(iter_num // 150)

                mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
                mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
                mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
                mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)

                outputs_l, _ = model(mixl_img)
                outputs_u, _ = model(mixu_img)

              
                if args.enable_boundary:
                  
                    combined_lab = torch.cat([mixl_lab, mixu_lab], dim=0)
                    combined_bw, combined_band = boundary_weight_map(
                        combined_lab,
                        lambda_boundary=args.lambda_boundary,
                        radius=args.boundary_radius,
                        iter_num=iter_num,
                        rampup=args.boundary_rampup)
                    bweight_l = combined_bw[:sub_bs]
                    bweight_u = combined_bw[sub_bs:]
                else:
                    bweight_l = None
                    bweight_u = None

                loss_l = mix_loss_enhanced(outputs_l, lab_a, plab_a, loss_mask,
                                           u_weight=args.u_weight, bweight=bweight_l)
                loss_u = mix_loss_enhanced(outputs_u, plab_b, lab_b, loss_mask,
                                           u_weight=args.u_weight, unlab=True, bweight=bweight_u)

                loss = loss_l + loss_u

             
                loss_aux = torch.tensor(0.0, device=loss.device)
                if aux_head is not None and args.enable_aux_head:
                   
                    bpred_l = aux_head(outputs_l)
                    bgt_l_labeled = generate_boundary_gt(lab_a, radius=args.boundary_radius)
                  
                    loss_aux_l = boundary_aux_loss(bpred_l, bgt_l_labeled,
                                                   valid_mask=loss_mask.unsqueeze(1))

                   
                    bpred_u = aux_head(outputs_u)
                    bgt_u_labeled = generate_boundary_gt(lab_b, radius=args.boundary_radius)
                  
                    patch_mask = (1 - loss_mask).unsqueeze(1)
                    loss_aux_u = boundary_aux_loss(bpred_u, bgt_u_labeled,
                                                   valid_mask=patch_mask)

                    ramp = min(float(iter_num) / max(float(args.boundary_rampup), 1.0), 1.0)
                    loss_aux = (loss_aux_l + loss_aux_u) * args.lambda_aux * ramp
                    loss = loss + loss_aux


            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, 0.99)

         
            writer.add_scalar('Self/loss_all', loss, iter_num)
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)

            if args.enable_aux_head:
                writer.add_scalar('Self/loss_aux', loss_aux, iter_num)
            if args.ensemble_alpha < 1.0:
                writer.add_scalar('Self/agree_ratio', (agree_a + agree_b) / 2, iter_num)

            log_msg = 'iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f' % (iter_num, loss, loss_l, loss_u)
            if args.enable_aux_head:
                log_msg += ', loss_aux: %03f' % loss_aux
            logging.info(log_msg)

            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    # 也保存 aux_head
                    if aux_head is not None:
                        torch.save(aux_head.state_dict(),
                                   os.path.join(self_snapshot_path, 'aux_head_best.pth'))
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()
                if aux_head is not None:
                    aux_head.train()

            
            if iter_num % 200 == 1:
                ins_width = 2
                B, C, H, W, D = outputs_l.size()
                snapshot_img = torch.zeros(size=(D, 3, 3 * H + 3 * ins_width, W + ins_width), dtype=torch.float32)
                snapshot_img[:, :, H:H + ins_width, :] = 1
                snapshot_img[:, :, 2 * H + ins_width:2 * H + 2 * ins_width, :] = 1
                snapshot_img[:, :, 3 * H + 2 * ins_width:3 * H + 3 * ins_width, :] = 1
                snapshot_img[:, :, :, W:W + ins_width] = 1

                outputs_l_soft = F.softmax(outputs_l, dim=1)
                seg_out = outputs_l_soft[0, 1, ...].permute(2, 0, 1)
                target = mixl_lab[0, ...].permute(2, 0, 1)
                train_img = mixl_img[0, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                writer.add_images('Epoch_%d_Iter_%d_labeled' % (epoch, iter_num), snapshot_img)

                outputs_u_soft = F.softmax(outputs_u, dim=1)
                seg_out = outputs_u_soft[0, 1, ...].permute(2, 0, 1)
                target = mixu_lab[0, ...].permute(2, 0, 1)
                train_img = mixu_img[0, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                writer.add_images('Epoch_%d_Iter_%d_unlabel' % (epoch, iter_num), snapshot_img)

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    exp_name = args.exp
    if args.ensemble_alpha < 1.0:
        exp_name += '_ens{}'.format(args.ensemble_alpha)
    if args.enable_boundary:
        exp_name += '_bd{}'.format(args.lambda_boundary)
    if args.enable_aux_head:
        exp_name += '_aux{}'.format(args.lambda_aux)
    exp_name += '_s{}'.format(args.seed)

    pre_snapshot_path = "./model/BCP/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/LA_{}_{}_labeled/self_train".format(exp_name, args.labelnum)
    print("Starting BCP Enhanced training.")
    print("Exp: {}".format(exp_name))
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')


    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

   
    pre_model_path = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    if os.path.exists(pre_model_path):
        logging.info("Pre-trained model found at {}, skipping pre-train.".format(pre_model_path))
    else:
        pre_train(args, pre_snapshot_path)


    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
