"""BCP-repository U-Net baseline and BAST extension for PROMISE12.

The backbone, optimizer, two-stage schedule, EMA teacher, rectangular mask,
masked Dice+CE, and bidirectional copy-paste follow ``ACDC_BCP_train.py``.
PROMISE12-specific code is limited to PNG loading, case-level splits, binary
output, and complete-volume validation/test evaluation.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from skimage.measure import label as connected_components
import torch
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from BAST import (  # noqa: E402
    BoundaryAuxHead,
    boundary_aux_loss,
    boundary_weight_map,
    ensemble_pseudo_label,
    generate_boundary_gt,
)
from dataloaders.dataset import TwoStreamBatchSampler  # noqa: E402
from networks.unet import UNet_2d  # noqa: E402
from promise12_dataset import (  # noqa: E402
    Promise12BCPTrainDataset,
    Promise12VolumeDataset,
    parse_manifest,
)
from promise12_metrics import evaluate_dataset, save_results  # noqa: E402
from utils.losses import DiceLoss  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="PROMISE12 BCP/BAST with the official BCP 2D U-Net"
    )
    parser.add_argument(
        "--data-root",
        default="./data/promise12_ipnet_abd_256",
    )
    parser.add_argument(
        "--split-root",
        default="./splits/promise12/ipnet_seed42",
    )
    parser.add_argument("--ratio", default="20p")
    parser.add_argument("--method", choices=("bcp", "bast"), default="bast")
    parser.add_argument(
        "--stage", choices=("all", "pretrain", "selftrain", "test"), default="all"
    )
    parser.add_argument("--experiment", default="PROMISE12_IPNet20_seed42")
    parser.add_argument("--output-root", default="./outputs/PROMISE12")
    parser.add_argument(
        "--test-output-dir",
        default=None,
        help="Optional separate directory for test results; checkpoints remain in the method directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--labeled-bs", type=int, default=12)
    parser.add_argument("--base-lr", type=float, default=0.01)
    parser.add_argument("--pre-iterations", type=int, default=10000)
    parser.add_argument("--max-iterations", type=int, default=30000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--u-weight", type=float, default=0.5)
    parser.add_argument("--ema-alpha", type=float, default=0.99)
    parser.add_argument("--ensemble-alpha", type=float, default=0.7)
    parser.add_argument("--lambda-boundary", type=float, default=1.0)
    parser.add_argument("--lambda-aux", type=float, default=0.3)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--boundary-rampup", type=int, default=10000)
    parser.add_argument("--aux-mid-channels", type=int, default=16)
    parser.add_argument("--force-pretrain", action="store_true")
    parser.add_argument(
        "--resume-selftrain",
        action="store_true",
        help="Resume self-training from training_state_latest.pth, or the best state if no latest state exists.",
    )
    return parser.parse_args()


def configure_runtime(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def build_paths(args):
    split_root = Path(args.split_root)
    experiment = Path(args.output_root) / args.experiment
    return {
        "train": split_root / "train.txt",
        "labeled": split_root / args.ratio / "labeled.txt",
        "unlabeled": split_root / args.ratio / "unlabeled.txt",
        "val": split_root / "val.txt",
        "test": split_root / "test.txt",
        "experiment": experiment,
        "pretrain": experiment / "shared_pretrain",
        "method": experiment / args.method.upper(),
    }


def validate_protocol(paths):
    all_train = {entry[0] for entry in parse_manifest(paths["train"])}
    labeled = {entry[0] for entry in parse_manifest(paths["labeled"])}
    unlabeled = {entry[0] for entry in parse_manifest(paths["unlabeled"])}
    if labeled & unlabeled or labeled | unlabeled != all_train:
        raise ValueError("Labeled/unlabeled manifests do not partition train.txt")
    case_sets = {}
    for name in ("train", "labeled", "unlabeled", "val", "test"):
        case_sets[name] = {
            Path(entry[0]).parent.name for entry in parse_manifest(paths[name])
        }
    if len(case_sets["train"]) != 35 or len(case_sets["val"]) != 5 or len(case_sets["test"]) != 10:
        raise ValueError("PROMISE12 reference protocol requires a 35/5/10 case split")
    if len(case_sets["labeled"]) != 7 or len(case_sets["unlabeled"]) != 28:
        raise ValueError("PROMISE12 20% protocol requires 7 labeled and 28 unlabeled cases")
    if case_sets["train"] & (case_sets["val"] | case_sets["test"]):
        raise ValueError("Train cases overlap validation or test cases")
    if case_sets["val"] & case_sets["test"]:
        raise ValueError("Validation and test cases overlap")
    return len(all_train), len(labeled), len(unlabeled)


def setup_logger(path, name):
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"
    )
    file_handler = logging.FileHandler(path / "log.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def worker_init(seed):
    def initialize(worker_id):
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)

    return initialize


def make_datasets(args, paths):
    train = Promise12BCPTrainDataset(
        args.data_root,
        paths["labeled"],
        paths["unlabeled"],
        output_size=args.patch_size,
    )
    val = Promise12VolumeDataset(args.data_root, paths["val"])
    test = Promise12VolumeDataset(args.data_root, paths["test"])
    return train, val, test


def make_train_loader(args, dataset):
    if args.batch_size - args.labeled_bs != args.labeled_bs:
        raise ValueError("Official BCP pairing requires equal labeled/unlabeled batch sizes")
    if args.labeled_bs % 2:
        raise ValueError("--labeled-bs must be even")
    sampler = TwoStreamBatchSampler(
        dataset.labeled_indices,
        dataset.unlabeled_indices,
        args.batch_size,
        args.batch_size - args.labeled_bs,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=worker_init(args.seed),
    )


def build_model(device, ema=False):
    model = UNet_2d(in_chns=1, class_num=2).to(device)
    if ema:
        for parameter in model.parameters():
            parameter.detach_()
    return model


def save_pretrain(model, optimizer, path, iteration, best_dice, args):
    torch.save(
        {
            "net": model.state_dict(),
            "opt": optimizer.state_dict(),
            "iteration": iteration,
            "best_dice": best_dice,
            "args": vars(args),
            "backbone": "BCP repository UNet_2d",
        },
        path,
    )


def load_pretrain(model, optimizer, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["net"])
    optimizer.load_state_dict(state["opt"])
    return state


@torch.no_grad()
def update_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    blended = {
        key: alpha * ema_state[key] + (1.0 - alpha) * model_state[key]
        for key in model_state
    }
    ema_model.load_state_dict(blended)


def generate_mask(image):
    batch, _, height, width = image.shape
    loss_mask = torch.ones(batch, height, width, device=image.device)
    image_mask = torch.ones(height, width, device=image.device)
    patch_height, patch_width = int(height * 2 / 3), int(width * 2 / 3)
    row = np.random.randint(0, height - patch_height)
    column = np.random.randint(0, width - patch_width)
    image_mask[row : row + patch_height, column : column + patch_width] = 0
    loss_mask[:, row : row + patch_height, column : column + patch_width] = 0
    return image_mask.long(), loss_mask.long()


def largest_component_2d(segmentation):
    output = []
    device = segmentation.device
    for sample in segmentation:
        foreground = (sample > 0).detach().cpu().numpy().astype(np.uint8)
        components = connected_components(foreground)
        if components.max() == 0:
            largest = foreground
        else:
            largest = components == (np.bincount(components.flat)[1:].argmax() + 1)
        output.append(largest.astype(np.int64))
    return torch.from_numpy(np.stack(output)).long().to(device)


def teacher_pseudo(logits):
    pseudo = logits.softmax(dim=1).argmax(dim=1)
    return largest_component_2d(pseudo)


def backup_batch_norm(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            state[name] = (
                module.running_mean.clone(),
                module.running_var.clone(),
                module.num_batches_tracked.clone(),
            )
    return state


def restore_batch_norm(model, state):
    for name, module in model.named_modules():
        if name in state:
            module.running_mean.copy_(state[name][0])
            module.running_var.copy_(state[name][1])
            module.num_batches_tracked.copy_(state[name][2])


def mix_loss(output, image_label, patch_label, mask, dice_loss, u_weight=0.5, unlab=False, boundary_weight=None):
    image_label, patch_label = image_label.long(), patch_label.long()
    image_weight, patch_weight = (u_weight, 1.0) if unlab else (1.0, u_weight)
    patch_mask = 1 - mask
    probabilities = output.softmax(dim=1)
    dice = dice_loss(probabilities, image_label.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    dice = dice + dice_loss(
        probabilities, patch_label.unsqueeze(1), patch_mask.unsqueeze(1)
    ) * patch_weight
    ce_image = F.cross_entropy(output, image_label, reduction="none")
    ce_patch = F.cross_entropy(output, patch_label, reduction="none")
    if boundary_weight is not None:
        ce_image = ce_image * boundary_weight
        ce_patch = ce_patch * boundary_weight
    ce = image_weight * (ce_image * mask).sum() / mask.sum().clamp_min(1)
    ce = ce + patch_weight * (ce_patch * patch_mask).sum() / patch_mask.sum().clamp_min(1)
    return dice, ce


def validate(model, dataset, args, device):
    rows, summary = evaluate_dataset(
        model, dataset, tuple(args.patch_size), device=device, distances=False
    )
    return summary["dice_mean"], rows


def pretrain(args, paths, train_dataset, val_dataset, device):
    output = paths["pretrain"]
    checkpoint = output / "unet_best_model.pth"
    if checkpoint.is_file() and not args.force_pretrain:
        return checkpoint
    logger = setup_logger(output, "promise12_pretrain")
    writer = SummaryWriter(str(output / "tensorboard"))
    model = build_model(device)
    optimizer = SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    dice_loss = DiceLoss(n_classes=2)
    loader = make_train_loader(args, train_dataset)
    labeled_sub_batch = args.labeled_bs // 2
    max_epochs = args.pre_iterations // len(loader) + 1
    best_dice = -1.0
    iteration = 0
    logger.info("%s", json.dumps(vars(args), sort_keys=True))
    logger.info("train slices=%d labeled=%d unlabeled=%d", len(train_dataset), train_dataset.labeled_count, len(train_dataset) - train_dataset.labeled_count)
    model.train()
    for _ in tqdm(range(max_epochs), desc="BCP pretrain", ncols=90):
        for batch in loader:
            images = batch["image"][: args.labeled_bs].to(device, non_blocking=True)
            labels = batch["label"][: args.labeled_bs].to(device, non_blocking=True)
            image_a, image_b = images[:labeled_sub_batch], images[labeled_sub_batch:]
            label_a, label_b = labels[:labeled_sub_batch], labels[labeled_sub_batch:]
            image_mask, loss_mask = generate_mask(image_a)
            mixed = image_a * image_mask + image_b * (1 - image_mask)
            logits = model(mixed)
            loss_dice, loss_ce = mix_loss(
                logits, label_a, label_b, loss_mask, dice_loss, u_weight=1.0, unlab=True
            )
            loss = (loss_dice + loss_ce) / 2.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1
            writer.add_scalar("pretrain/loss", loss.item(), iteration)
            if iteration % 20 == 0:
                logger.info("iteration %d loss %.6f dice %.6f ce %.6f", iteration, loss.item(), loss_dice.item(), loss_ce.item())
            if iteration % args.eval_interval == 0:
                score, _ = validate(model, val_dataset, args, device)
                writer.add_scalar("pretrain/val_case_dice", score, iteration)
                logger.info("iteration %d validation case Dice %.6f", iteration, score)
                if score > best_dice:
                    best_dice = score
                    save_pretrain(model, optimizer, checkpoint, iteration, best_dice, args)
                model.train()
            if iteration >= args.pre_iterations:
                break
        if iteration >= args.pre_iterations:
            break
    if not checkpoint.is_file():
        save_pretrain(model, optimizer, checkpoint, iteration, best_dice, args)
    save_pretrain(model, optimizer, output / "unet_final_model.pth", iteration, best_dice, args)
    writer.close()
    return checkpoint


def save_selftrain_state(model, ema_model, optimizer, aux_head, path, iteration, best_dice, args):
    state = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "best_dice": best_dice,
        "method": args.method,
        "args": vars(args),
        "backbone": "BCP repository UNet_2d",
    }
    if aux_head is not None:
        state["aux_head"] = aux_head.state_dict()
    torch.save(state, path)


def selftrain(args, paths, train_dataset, val_dataset, device, pretrain_checkpoint):
    output = paths["method"]
    logger = setup_logger(output, f"promise12_{args.method}")
    writer = SummaryWriter(str(output / "tensorboard"))
    model = build_model(device)
    ema_model = build_model(device, ema=True)
    optimizer = SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    auxiliary_head = None
    resume_candidates = (
        output / "training_state_latest.pth",
        output / "training_state_best.pth",
    )
    resume_path = next(
        (path for path in resume_candidates if args.resume_selftrain and path.is_file()),
        None,
    )
    if resume_path is not None:
        if args.method == "bast":
            auxiliary_head = BoundaryAuxHead(
                in_channels=2, mid_channels=args.aux_mid_channels, is_2d=True
            ).to(device)
            optimizer.add_param_group({"params": auxiliary_head.parameters()})
        resume_state = torch.load(resume_path, map_location=device)
        if resume_state.get("method") != args.method:
            raise ValueError(
                f"Resume method mismatch: {resume_state.get('method')} != {args.method}"
            )
        model.load_state_dict(resume_state["model"])
        ema_model.load_state_dict(resume_state["ema_model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        if auxiliary_head is not None:
            auxiliary_head.load_state_dict(resume_state["aux_head"])
        best_dice = float(resume_state["best_dice"])
        iteration = int(resume_state["iteration"])
    else:
        load_pretrain(model, optimizer, pretrain_checkpoint, device)
        pretrain_state = torch.load(pretrain_checkpoint, map_location=device)
        ema_model.load_state_dict(pretrain_state["net"])
        if args.method == "bast":
            auxiliary_head = BoundaryAuxHead(
                in_channels=2, mid_channels=args.aux_mid_channels, is_2d=True
            ).to(device)
            optimizer.add_param_group({"params": auxiliary_head.parameters()})
        best_dice = -1.0
        iteration = 0

    loader = make_train_loader(args, train_dataset)
    dice_loss = DiceLoss(n_classes=2)
    labeled_sub_batch = args.labeled_bs // 2
    unlabeled_sub_batch = (args.batch_size - args.labeled_bs) // 2
    max_epochs = args.max_iterations // len(loader) + 1
    logger.info("%s", json.dumps(vars(args), sort_keys=True))
    logger.info("shared pretraining checkpoint: %s", pretrain_checkpoint)
    if resume_path is not None:
        logger.info(
            "resumed self-training from %s at iteration %d with best Dice %.6f",
            resume_path,
            iteration,
            best_dice,
        )
    model.train()
    ema_model.train()
    if auxiliary_head is not None:
        auxiliary_head.train()

    for _ in tqdm(range(max_epochs), desc=args.method.upper(), ncols=90):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            image_a = images[:labeled_sub_batch]
            image_b = images[labeled_sub_batch : args.labeled_bs]
            unlabeled_a = images[args.labeled_bs : args.labeled_bs + unlabeled_sub_batch]
            unlabeled_b = images[args.labeled_bs + unlabeled_sub_batch :]
            label_a = labels[:labeled_sub_batch]
            label_b = labels[labeled_sub_batch : args.labeled_bs]

            with torch.no_grad():
                teacher_a = ema_model(unlabeled_a)
                teacher_b = ema_model(unlabeled_b)
                if args.method == "bast":
                    bn_state = backup_batch_norm(model)
                    student_a = model(unlabeled_a)
                    student_b = model(unlabeled_b)
                    restore_batch_norm(model, bn_state)
                    pseudo_a, _, agreement_a = ensemble_pseudo_label(
                        teacher_a,
                        student_a,
                        alpha=args.ensemble_alpha,
                        nms_fn=largest_component_2d,
                    )
                    pseudo_b, _, agreement_b = ensemble_pseudo_label(
                        teacher_b,
                        student_b,
                        alpha=args.ensemble_alpha,
                        nms_fn=largest_component_2d,
                    )
                else:
                    pseudo_a, pseudo_b = teacher_pseudo(teacher_a), teacher_pseudo(teacher_b)
                    agreement_a = agreement_b = 1.0
                image_mask, loss_mask = generate_mask(image_a)

            mixed_unlabeled = unlabeled_a * image_mask + image_a * (1 - image_mask)
            mixed_labeled = image_b * image_mask + unlabeled_b * (1 - image_mask)
            output_unlabeled = model(mixed_unlabeled)
            output_labeled = model(mixed_labeled)

            boundary_unlabeled = boundary_labeled = None
            if args.method == "bast":
                mixed_target_unlabeled = pseudo_a * image_mask + label_a * (1 - image_mask)
                mixed_target_labeled = label_b * image_mask + pseudo_b * (1 - image_mask)
                all_targets = torch.cat((mixed_target_unlabeled, mixed_target_labeled))
                all_weights, _ = boundary_weight_map(
                    all_targets,
                    lambda_boundary=args.lambda_boundary,
                    radius=args.boundary_radius,
                    iter_num=iteration,
                    rampup=args.boundary_rampup,
                )
                boundary_unlabeled = all_weights[:labeled_sub_batch]
                boundary_labeled = all_weights[labeled_sub_batch:]

            unlabeled_dice, unlabeled_ce = mix_loss(
                output_unlabeled,
                pseudo_a,
                label_a,
                loss_mask,
                dice_loss,
                u_weight=args.u_weight,
                unlab=True,
                boundary_weight=boundary_unlabeled,
            )
            labeled_dice, labeled_ce = mix_loss(
                output_labeled,
                label_b,
                pseudo_b,
                loss_mask,
                dice_loss,
                u_weight=args.u_weight,
                boundary_weight=boundary_labeled,
            )
            loss_dice = unlabeled_dice + labeled_dice
            loss_ce = unlabeled_ce + labeled_ce
            loss = (loss_dice + loss_ce) / 2.0

            auxiliary_loss = torch.zeros((), device=device)
            if auxiliary_head is not None:
                aux_unlabeled = auxiliary_head(output_unlabeled)
                gt_unlabeled = generate_boundary_gt(label_a, args.boundary_radius)
                aux_loss_unlabeled = boundary_aux_loss(
                    aux_unlabeled,
                    gt_unlabeled,
                    valid_mask=(1 - loss_mask).unsqueeze(1),
                )
                aux_labeled = auxiliary_head(output_labeled)
                gt_labeled = generate_boundary_gt(label_b, args.boundary_radius)
                aux_loss_labeled = boundary_aux_loss(
                    aux_labeled,
                    gt_labeled,
                    valid_mask=loss_mask.unsqueeze(1),
                )
                ramp = min(iteration / max(float(args.boundary_rampup), 1.0), 1.0)
                auxiliary_loss = (
                    aux_loss_unlabeled + aux_loss_labeled
                ) * args.lambda_aux * ramp
                loss = loss + auxiliary_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1
            update_ema(model, ema_model, args.ema_alpha)

            writer.add_scalar("selftrain/loss", loss.item(), iteration)
            writer.add_scalar("selftrain/mix_dice", loss_dice.item(), iteration)
            writer.add_scalar("selftrain/mix_ce", loss_ce.item(), iteration)
            if args.method == "bast":
                writer.add_scalar("selftrain/aux", auxiliary_loss.item(), iteration)
                writer.add_scalar(
                    "selftrain/agreement", (float(agreement_a) + float(agreement_b)) / 2, iteration
                )
            if iteration % 20 == 0:
                logger.info(
                    "iteration %d loss %.6f dice %.6f ce %.6f aux %.6f",
                    iteration,
                    loss.item(),
                    loss_dice.item(),
                    loss_ce.item(),
                    auxiliary_loss.item(),
                )
            if iteration % args.eval_interval == 0:
                score, _ = validate(model, val_dataset, args, device)
                writer.add_scalar("selftrain/val_case_dice", score, iteration)
                logger.info("iteration %d validation case Dice %.6f", iteration, score)
                if score > best_dice:
                    best_dice = score
                    torch.save(model.state_dict(), output / "unet_best_model.pth")
                    save_selftrain_state(
                        model,
                        ema_model,
                        optimizer,
                        auxiliary_head,
                        output / "training_state_best.pth",
                        iteration,
                        best_dice,
                        args,
                    )
                save_selftrain_state(
                    model,
                    ema_model,
                    optimizer,
                    auxiliary_head,
                    output / "training_state_latest.pth",
                    iteration,
                    best_dice,
                    args,
                )
                model.train()
                ema_model.train()
                if auxiliary_head is not None:
                    auxiliary_head.train()
            if iteration >= args.max_iterations:
                break
        if iteration >= args.max_iterations:
            break

    torch.save(model.state_dict(), output / "unet_final_model.pth")
    save_selftrain_state(
        model,
        ema_model,
        optimizer,
        auxiliary_head,
        output / "training_state_final.pth",
        iteration,
        best_dice,
        args,
    )
    if not (output / "unet_best_model.pth").is_file():
        torch.save(model.state_dict(), output / "unet_best_model.pth")
    writer.close()
    return output / "unet_best_model.pth"


def test(args, paths, test_dataset, device, checkpoint=None):
    checkpoint_dir = paths["method"]
    output = Path(args.test_output_dir) if args.test_output_dir else checkpoint_dir
    checkpoint = Path(checkpoint or checkpoint_dir / "unet_best_model.pth")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    model = build_model(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    rows, summary = evaluate_dataset(
        model,
        test_dataset,
        tuple(args.patch_size),
        device=device,
        distances=True,
        postprocess="largest_3d",
    )
    summary.update(
        {
            "dataset": "PROMISE12",
            "method": args.method.upper(),
            "backbone": "BCP repository UNet_2d",
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "labeled_ratio": args.ratio,
            "postprocessing": "largest 26-connected 3D foreground component",
        }
    )
    save_results(rows, summary, output, prefix="test")
    return summary


def main():
    args = parse_args()
    configure_runtime(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal BCP/BAST training")
    device = torch.device("cuda")
    paths = build_paths(args)
    paths["experiment"].mkdir(parents=True, exist_ok=True)
    paths["pretrain"].mkdir(parents=True, exist_ok=True)
    paths["method"].mkdir(parents=True, exist_ok=True)
    total, labeled, unlabeled = validate_protocol(paths)
    train_dataset, val_dataset, test_dataset = make_datasets(args, paths)
    metadata = {
        "dataset": "PROMISE12",
        "backbone": "BCP repository UNet_2d",
        "train_slices": total,
        "labeled_slices": labeled,
        "unlabeled_slices": unlabeled,
        "labeled_cases": sorted(
            {Path(entry[0]).parent.name for entry in parse_manifest(paths["labeled"])}
        ),
        "seed": args.seed,
        "method": args.method,
        "args": vars(args),
    }
    (paths["experiment"] / "protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(__file__, paths["method"] / Path(__file__).name)

    pretrain_checkpoint = paths["pretrain"] / "unet_best_model.pth"
    if args.stage in ("all", "pretrain"):
        pretrain_checkpoint = pretrain(
            args, paths, train_dataset, val_dataset, device
        )
    if args.stage == "pretrain":
        return
    if not pretrain_checkpoint.is_file() and args.stage != "test":
        raise FileNotFoundError(
            f"Shared BCP pretraining checkpoint not found: {pretrain_checkpoint}"
        )
    if args.stage in ("all", "selftrain"):
        selftrain(
            args,
            paths,
            train_dataset,
            val_dataset,
            device,
            pretrain_checkpoint,
        )
    if args.stage in ("all", "test"):
        summary = test(args, paths, test_dataset, device)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
