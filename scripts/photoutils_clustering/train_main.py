import argparse
import itertools
import os
import shutil
from itertools import product as product
from math import sqrt as sqrt

import numpy as np
import torch
import torch.optim as optim
import wandb
from PIL import ImageFile

import test_infer
from train_model import train_model
from training_sub import print_and_log, weights_init
from utils.ssd_model import SSD, MultiBoxLoss


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of SSD")
    parser.add_argument("--spitzer_path", metavar="DIR", help="spitzer_path", default="/dataset/spitzer_data/")
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
    parser.add_argument("--savedir_path", metavar="DIR", default="/workspace/weights/search/", help="savedire path")
    # minibatch
    parser.add_argument("--num_epoch", type=int, default=300, help="number of total epochs to run (default: 300)")
    parser.add_argument("--Ring_mini_batch", default=32, type=int, help="mini-batch size (default: 32)")
    parser.add_argument("--NonRing_mini_batch", default=32, type=int, help="mini-batch size (default: 32)")
    parser.add_argument("--Val_mini_batch", default=128, type=int, help="Validation mini-batch size (default: 128)")
    parser.add_argument("--True_iou", default=0.5, type=float, help="True IoU in MultiBoxLoss(default: 0.5)")
    # fits index
    parser.add_argument("--fits_index", "-i", type=int, default=0)  # , required=True)
    parser.add_argument("--n_splits", "-n", type=int, default=8)
    # random seed
    parser.add_argument("--fits_random_state", "-r", type=int, default=123)
    parser.add_argument("--data_random_state", "-d", type=int, default=123)
    parser.add_argument("--init_random_state", type=int, default=123)
    # 学習率
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    # option
    parser.add_argument("--test_infer_false", action="store_false")
    parser.add_argument("--ring_select_false", action="store_false")
    parser.add_argument("--training_ring_catalogue", type=str, default="MWP")
    parser.add_argument("--val_ring_catalogue", type=str, default="MWP")
    parser.add_argument("--wandb_project", type=str, default="リングの選定")
    parser.add_argument("--wandb_name", type=str, default="search_validation_size")
    parser.add_argument("--fscore", type=str, default="f2_score")
    # NonRing
    parser.add_argument("--NonRing_class_num", type=int, default=8)
    parser.add_argument("--NonRing_remove_class_list", nargs="*", type=int, default=[3, 4])
    parser.add_argument("--NonRing_aug_num", nargs="*", type=int, default=[5, 0, 0, 0, 0, 0, 2, 3])
    # Valiation
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
    torch.manual_seed(args.init_random_state)
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if os.path.exists(args.savedir_path):
        print("REMOVE FILES...")
        shutil.rmtree(args.savedir_path)
    os.makedirs(args.savedir_path, exist_ok=True)
    default_val_size = np.array([150, 300, 600, 900, 1200, 1800, 2400, 3000])
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
        name = args.savedir_path + "/" + "".join(name_)
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
            f"fits_index: {args.fits_index}",
            f"n_splits: {args.n_splits}",
            f"fits_random_state: {args.fits_random_state}",
            f"data_random_state: {args.data_random_state}",
            f"training_ring_catalogue: {args.training_ring_catalogue}",
            f"val_ring_catalogue: {args.val_ring_catalogue}",
            f"NoRing_class_num: {args.NonRing_class_num}",
            f"NoRing_remove_class_list: {args.NonRing_remove_class_list}",
            f"NoRing_aug_num: {args.NonRing_aug_num}",
            " ",
            "====================================",
        ]
        print_and_log(f_log, log_list)
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "fits_index": args.fits_index,
                "n_splits": args.n_splits,
                "fits_random_state": args.fits_random_state,
                "data_random_state": args.data_random_state,
                "Ring_mini_batch": args.Ring_mini_batch,
                "NonRing_mini_batch": args.NonRing_mini_batch,
                "NonRing_remove_class_list": args.NonRing_remove_class_list,
            },
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print_and_log(f_log, f"使用デバイス： {device}")

        net = SSD()
        ## パラメータを初期化
        for net_sub in [net.vgg, net.extras, net.loc, net.conf]:
            net_sub.apply(weights_init)
        net.to(device)
        wandb.watch(net, log_freq=100)

        optimizer = optim.AdamW(
            net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False
        )
        train_model_params = {
            "net": net,
            "criterion": MultiBoxLoss(jaccard_thresh=args.True_iou, neg_pos=3, device=device),
            "optimizer": optimizer,
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

        # l18領域の推論
        if args.test_infer_false:
            f_score, pre, re, conf_thre = test_infer.infer_test(name, args, default_val_size, val_best_confthre)
            print_and_log(
                f_log,
                [f"test {args.fscore}: {f_score}", f"precision: {pre}", f"recall: {re}", f"conf_threshold: {conf_thre}"],
            )
            wandb.run.summary[f"test_{args.fscore}"] = f_score
            wandb.run.summary["test_precision"] = pre
            wandb.run.summary["test_recall"] = re
            wandb.run.summary["test_conf_threshold"] = conf_thre

        artifact = wandb.Artifact("training_log", type="dir")
        artifact.add_dir(name)
        run.log_artifact(artifact, aliases=["latest", "best"])

        f_log.close()
        run.alert(title="学習が終了しました", text="学習が終了しました")
        run.finish()

        shutil.rmtree(args.savedir_path)
        os.makedirs(args.savedir_path, exist_ok=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
