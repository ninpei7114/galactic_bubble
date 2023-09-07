import glob
import os
import shutil
import time
from itertools import product as product

import numpy as np
from numpy.random import default_rng
import pandas as pd
import torch
import torch.nn as nn

from data import make_training_dataloader, make_validatoin_dataloader
from make_data import make_training_val_data
from make_figure import make_figure
from nonring_augmentation import nonring_augmentation
from training_sub import EarlyStopping_f1_score, calc_f1score_val, management_loss, print_and_log, write_train_log
from utils.ssd_model import Detect


def train_model(
    net, criterion, optimizer, num_epochs, f_log, augmentation_name, args, train_cfg, device, run, val_size
):
    """モデルの学習を実行する関数

    Args:
        net (pytorch Modulelist): SSDネットワーク
        criterion (MultiBoxLoss): 損失関数
        optimizer (AdamW)       : 最適化手法
        num_epochs (int)        : 最大epoch数
        f_log (txt file)        : logファイル
        augmentation_name (int) : どのaugmentationを使用したかの名前
        args (args)             : argparseの引数
        train_cfg (dictionary)  : augmentationのパラメータ
        device (torch.device)   : GPU or CPU
    """
    NonRing_class = np.delete(np.arange(args.NonRing_class_num), args.NonRing_remove_class_list)
    NonRing_rg = default_rng(args.fits_random_state)
    early_stopping = EarlyStopping_f1_score(
        patience=10, verbose=True, path=augmentation_name + "/earlystopping.pth", flog=f_log
    )
    detect = Detect(nms_thresh=0.45, top_k=500, conf_thresh=0.3)  # F1 scoreのconfの計算が0.3からなので、ここも0.3
    save_training_val_loss = management_loss()
    logs, f1_score_val_l = [], []

    ##########################
    ## Validation dataの作成 ##
    ##########################
    Make_data = make_training_val_data(augmentation_name, f_log, args)
    Validation_data_path, Val_num = Make_data.make_validation_data(val_size)
    dl_val = make_validatoin_dataloader(Validation_data_path, args)
    NonRing_num_l = Make_data.make_training_nonring_data()
    all_iter_val = int(int(Val_num) / args.Val_mini_batch)

    for epoch in range(num_epochs):
        start_time = time.time()
        iteration_train, iteration_val = 0, 0
        save_training_val_loss()  # lossの初期化
        print_and_log(f_log, ["-------------", "Epoch {}/{}".format(epoch + 1, num_epochs), "-------------"])

        ########################
        ## Training dataの作成 ##
        ########################
        # png形式のRing画像とjson形式のlabelを作成
        Training_data_path = Make_data.make_training_ring_data(train_cfg, epoch)
        train_Ring_num = Make_data.data_logger()
        # Training Ring の Dataloader を作成
        dl_ring_train, dl_nonring = make_training_dataloader(
            Training_data_path, train_Ring_num, args, NonRing_num_l, NonRing_class
        )
        dataloaders_dict = {"train": dl_ring_train, "val": dl_val}
        all_iter = int(int(train_Ring_num) / args.Ring_mini_batch)

        #############
        ## 学習開始 ##
        #############
        for phase in ["train", "val"]:
            if phase == "train":
                print_and_log(f_log, f" ({phase}) ")
                net.train()
            else:
                print_and_log(f_log, f" \n ({phase}) ")
                net.eval()
                result, position, regions = [], [], []

            ############################
            ## データの整形とモデルに入力 ##
            ############################
            for _ in dataloaders_dict[phase]:
                if phase == "train":
                    images, targets = _[0], _[1]
                    noring_image, noring_target = nonring_augmentation(dl_nonring, NonRing_class, NonRing_rg, args)
                    images = np.concatenate((images, noring_image))
                    targets = targets + noring_target
                else:
                    images, targets, offset, region_info = _[0], _[1], _[2], _[3]

                images = torch.from_numpy(images).permute(0, 3, 1, 2)[:, :2, :, :]
                images = images.to(device, dtype=torch.float)
                targets = [ann.to(device, dtype=torch.float) for ann in targets]  # リストの各要素のテンソルをGPUへ

                # optimizerを初期化
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs, decoded_box = net(images)
                    loss_dic = criterion(outputs, targets)
                    loss = loss_dic["loc_loss"] + loss_dic["conf_loss"]

                    if phase == "train":
                        loss.backward()  # 勾配の計算
                        # 勾配が大きくなりすぎると計算が不安定になるため、clipで最大でも勾配10.0に留める
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
                        optimizer.step()  # パラメータ更新
                        print("\r" + str(iteration_train) + "/" + str(all_iter) + " ", end="")
                        iteration_train += 1
                        save_training_val_loss.sum_iter_loss(loss_dic, "train")
                    else:
                        print("\r" + str(iteration_val) + "/" + str(all_iter_val) + " ", end="")
                        iteration_val += 1
                        result.append(detect(*outputs).to("cpu").detach().numpy().copy())
                        position.extend(offset)
                        regions.extend(region_info)
                        save_training_val_loss.sum_iter_loss(loss_dic, "val")

        ###############
        ## Lossの管理 ##
        ###############
        # loc, confのlossを出力
        loss_train = save_training_val_loss.output_each_loss("train", iteration_train)
        loss_val = save_training_val_loss.output_each_loss("val", iteration_val)

        ##############################
        ## Validation F1 scoreの計算 ##
        ##############################
        f1_score_val, precision, recall, conf_threshold_val = calc_f1score_val(
            np.concatenate(result), np.array(position), regions, args
        )
        f1_score_val_l.append(f1_score_val)

        log_epoch = write_train_log(
            f_log, epoch, loss_train, loss_val, f1_score_val, precision, recall, conf_threshold_val, start_time
        )
        run.log(log_epoch)
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(augmentation_name + "/log_output.csv")

        # early_stopping(epoch_val_loss, net)
        early_stopping(f1_score_val, net, epoch, optimizer, loss_train, loss_val)
        if early_stopping.early_stop:
            print_and_log(f_log, "Early_Stopping")
            break

        # データの削除
        os.remove(f"{Training_data_path}/bubble_dataset_train_ring.tar")
        shutil.rmtree(args.savedir_path + "".join("dataset") + "/" + augmentation_name.split("/")[-1] + "/train")

    ## lossの推移を描画する
    loc_l_val_s, conf_l_val_s, loc_l_train_s, conf_l_train_s = save_training_val_loss.output_all_epoch_loss()
    make_figure(augmentation_name, loc_l_val_s, conf_l_val_s, loc_l_train_s, conf_l_train_s, f1_score_val)
