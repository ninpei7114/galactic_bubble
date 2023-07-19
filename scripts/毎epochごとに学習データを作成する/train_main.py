import argparse
import itertools
import os
from itertools import product as product
from math import sqrt as sqrt

import torch
import torch.optim as optim

from make_figure import make_figure
from sub import print_and_log
from train_model import train_model
from utils.ssd_model import SSD, MultiBoxLoss


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of SSD")
    parser.add_argument("spitzer_path", metavar="DIR", help="spitzer_path", default="/dataset/spitzer_data/")
    parser.add_argument("--validation_data_path", metavar="DIR", help="validation data path", default="/workspace/val")
    parser.add_argument(
        "--savedir_path",
        metavar="DIR",
        default="/workspace/weights/",
        help="savedire path  (default: /workspace/weights/)",
    )
    parser.add_argument("--num_epoch", type=int, default=300, help="number of total epochs to run (default: 300)")
    parser.add_argument("--batch_size", default=16, type=int, help="mini-batch size (default: 16)")
    parser.add_argument("--NonRing_ratio", default=3, type=int, help="Ring / NonRing ratio (default: 3)")
    parser.add_argument("--augmentation_ratio", default=4, type=int, help="1 Ring augmentation ratio (default: 4)")
    parser.add_argument(
        "--True_iou", default=0.5, type=float, help="True IoU in MultiBoxLoss &  calc F1 score (default: 0.5)"
    )

    parser.add_argument("--region_suffle", "-s", action="store_true")
    parser.add_argument("--fits_index", "-i", type=int)  # , required=True)
    parser.add_argument("--n_splits", "-n", type=int, default=8)
    parser.add_argument("--fits_random_state", "-r", type=int, default=123)
    parser.add_argument("--NonRing_mini_batch", type=int, default=16)

    return parser.parse_args()


# SSDの学習
def main(args):
    """SSDの学習を行う。

    :Example command:
    >>> python /workspace/galactic_bubble/scripts/毎epochごとに学習データを作成する/train_main.py /dataset/spitzer_data/ \
        --savedir_path /workspace/webdataset_weights//augmentationの数を決める/Change_region_123/augmentation_ratio_${i}/ \
        --NonRing_ratio 1 --augmentation_ratio 1 -s -i 123 -n 3 -r 123

    """
    torch.manual_seed(123)

    os.makedirs(args.savedir_path, exist_ok=True)
    ssd_cfg = {
        "num_classes": 2,  # 背景クラスを含めた合計クラス数
        "input_size": 300,  # 画像の入力サイズ
        "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        "feature_maps": [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
        "steps": [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        "min_sizes": [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
        "max_sizes": [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    ## 上下反転、回転、縮小、平行移動の4パターンの組み合わせでaugmentationをする。
    flip_list = [True, False]
    rotate_list = [True, False]
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
            f"flip: {flip}, rotate: {rotate}, scale: {scale}, translation: {translation}",
            "###################",
            "  args parameters",
            "###################",
            f"augmentation_ratio: {args.augmentation_ratio}",
            f"region shuffle: {args.region_suffle}",
            f"fits_index: {args.fits_index}",
            f"n_splits: {args.n_splits}",
            f"fits_random_state: {args.fits_random_state}",
            " ",
            "====================================",
        ]
        print_and_log(f_log, log_list)

        ##############
        ## Training ##
        ##############
        net = SSD(cfg=ssd_cfg)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        train_model_params = {
            "net": net,
            "criterion": MultiBoxLoss(jaccard_thresh=args.True_iou, neg_pos=3, device=device),
            "optimizer": optim.AdamW(
                net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False
            ),
            "num_epochs": args.num_epoch,
            "f_log": f_log,
            "augmentation_name": name,
            "args": args,
            "train_cfg": train_cfg,
        }

        (
            loss_l_list_val,
            loss_c_list_val,
            loss_l_list_train,
            loss_c_list_train,
            train_f1_score,
            val_f1_score,
        ) = train_model(**train_model_params)
        f_log.close()

        ## lossの推移を描画する
        make_figure(
            name, loss_l_list_train, loss_c_list_train, loss_l_list_val, loss_c_list_val, train_f1_score, val_f1_score
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
