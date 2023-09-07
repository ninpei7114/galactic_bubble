import argparse
import itertools
import os
from itertools import product as product
from math import sqrt as sqrt

import numpy as np
import torch
import torch.optim as optim
import wandb
from PIL import ImageFile

import l18_infer
from train_model import train_model
from training_sub import print_and_log, weights_init
from utils.ssd_model import SSD, MultiBoxLoss


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of SSD")
    parser.add_argument("spitzer_path", metavar="DIR", help="spitzer_path", default="/dataset/spitzer_data/")
    parser.add_argument(
        "--validation_data_path",
        metavar="DIR",
        help="validation data path",
        default="/workspace/cut_val_png/region_val_png",
    )
    parser.add_argument(
        "--NonRing_data_path",
        metavar="DIR",
        help="NonRing data path",
        default="/workspace/NonRing_png/region_NonRing_png",
    )
    parser.add_argument("--savedir_path", metavar="DIR", default="/workspace/weights/", help="savedire path")
    # minibatch
    parser.add_argument("--num_epoch", type=int, default=300, help="number of total epochs to run (default: 300)")
    parser.add_argument("--Ring_mini_batch", default=32, type=int, help="mini-batch size (default: 32)")
    parser.add_argument("--NonRing_mini_batch", default=32, type=int, help="mini-batch size (default: 32)")
    parser.add_argument("--Val_mini_batch", default=16, type=int, help="Validation mini-batch size (default: 16)")
    parser.add_argument("--True_iou", default=0.5, type=float, help="True IoU in MultiBoxLoss(default: 0.5)")
    parser.add_argument("--region_suffle", "-s", action="store_true")
    parser.add_argument("--fits_index", "-i", type=int)  # , required=True)
    parser.add_argument("--n_splits", "-n", type=int, default=8)
    # random seed
    parser.add_argument("--fits_random_state", "-r", type=int, default=123)
    parser.add_argument("--data_random_state", "-d", type=int, default=123)
    # 学習率
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    # option
    parser.add_argument("--l18_infer", action="store_true")
    parser.add_argument("--ring_select", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="リングの選定")
    # NonRing
    parser.add_argument("--NonRing_class_num", type=int, default=8)
    parser.add_argument("--NonRing_remove_class_list", nargs="*", type=int, default=[6])
    parser.add_argument("--NonRing_aug_num", nargs="*", type=int, default=[5, 2, 1, 0, 0, 0, 3, 1])
    parser.add_argument("--Val_remove_size_list", nargs="*", type=int, default=[])

    return parser.parse_args()


# SSDの学習
def main(args):
    """SSDの学習を行う。

    :Example command:
    >>> python train_main.py /dataset/spitzer_data --savedir_path /workspace/webdataset_weights/Ring_selection_compare/ \
        --NonRing_data_path /workspace/NonRing_png/region_NonRing_png/ \
        --validation_data_path /workspace/cut_val_png/region_val_png/ \
        -s -i 0 --NonRing_remove_class_list 3 --Ring_mini_batch 16 --NonRing_mini_batch 2 --Val_mini_batch 64 \
        --l18_infer --ring_select

    """
    torch.manual_seed(args.fits_random_state)
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    os.makedirs(args.savedir_path, exist_ok=True)
    default_val_size = np.array([150, 300, 600, 900, 1200, 1800, 2500, 3000])
    for remove_size in args.Val_remove_size_list:
        default_val_size = default_val_size[default_val_size != remove_size]

    ## 上下反転、回転、縮小、平行移動の4パターンの組み合わせでaugmentationをする。
    flip_list = [True]  # , False]
    rotate_list = [True]  # , False]
    scale_list = [False]
    translation_list = [True]

    for flip, rotate, scale, translation in itertools.product(flip_list, rotate_list, scale_list, translation_list):
        train_cfg = {"flip": flip, "rotate": rotate, "scale": scale, "translation": translation}
        name_ = []
        [name_.append(k + "_" + str(v) + "__") for k, v in zip(list(train_cfg.keys()), list(train_cfg.values()))]
        name = args.savedir_path + "".join(name_)
        os.makedirs(name, exist_ok=True)

        ############
        ## logger ##
        ############
        f_log = open(name + "/log.txt", "w")
        log_list = [
            "#######################",
            "augmentation parameter",
            "#######################",
            f"flip: {flip}, rotate: {rotate}, scale: {scale}, translation: {translation}",
            " ",
            "#######################",
            "   args parameters",
            "#######################",
            f"region shuffle: {args.region_suffle}",
            f"fits_index: {args.fits_index}",
            f"n_splits: {args.n_splits}",
            f"fits_random_state: {args.fits_random_state}",
            " ",
            "====================================",
        ]
        print_and_log(f_log, log_list)
        run = wandb.init(
            project=args.wandb_project,
            name=f"リング選定 : {args.ring_select}",
            config={
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "fits_index": args.fits_index,
                "n_splits": args.n_splits,
                "fits_random_state": args.fits_random_state,
            },
        )
        wandb.watch(net, log_freq=100)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print_and_log(f_log, f"使用デバイス： {device}")

        net = SSD()
        ## パラメータを初期化
        for net_sub in [net.vgg, net.extras, net.loc, net.conf]:
            net_sub.apply(weights_init)
        net.to(device)

        train_model_params = {
            "net": net,
            "criterion": MultiBoxLoss(jaccard_thresh=args.True_iou, neg_pos=3, device=device),
            "optimizer": optim.AdamW(
                net.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.weight_decay,
                amsgrad=False,
            ),
            "num_epochs": args.num_epoch,
            "f_log": f_log,
            "augmentation_name": name,
            "args": args,
            "train_cfg": train_cfg,
            "device": device,
            "run": run,
            "val_size": default_val_size,
        }

        ####################
        ## Training Model ##
        ####################
        val_best_confthre = train_model(**train_model_params)

        artifact = wandb.Artifact("model", type="model")
        artifact.add_dir(name)
        run.log_artifact(artifact, aliases=["latest", "best"])
        f_log.close()

        # l18領域の推論
        if args.l18_infer:
            f1_score, pre, re, conf_thre = l18_infer.infer_l18(name, args, default_val_size, val_best_confthre)
            print_and_log(
                f_log,
                [f"l18 F1 score: {f1_score}", f"precision: {pre}", f"recall: {re}", f"conf_threshold: {conf_thre}"],
            )
            wandb.run.summary["l18_f1_score"] = f1_score
            wandb.run.summary["l18_precision"] = pre
            wandb.run.summary["l18_recall"] = re
            wandb.run.summary["l18_conf_threshold"] = conf_thre
        else:
            pass

        wandb.alert(title="学習が終了しました", text="学習が終了しました")
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
