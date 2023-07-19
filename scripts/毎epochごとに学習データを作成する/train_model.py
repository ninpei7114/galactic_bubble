import os
import pickle
import shutil
import time
from itertools import product as product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import webdataset

from data import od_collate_fn, preprocess
from make_data import make_training_val_data
from sub import (EarlyStopping_f1_score, calc_f1score, print_and_log,
                 weights_init)


def train_model(net, criterion, optimizer, num_epochs, f_log, augmentation_name, args, train_cfg):
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    NonRing_mini_batch = args.NonRing_mini_batch * args.NonRing_ratio
    early_stopping = EarlyStopping_f1_score(
        patience=10, verbose=True, path=augmentation_name + "/earlystopping.pth", flog=f_log
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print_and_log(f_log, f"使用デバイス： {device}")
    ## パラメータを初期化
    for net_sub in [net.vgg, net.extras, net.loc, net.conf]:
        net_sub.apply(weights_init)

    logs = []
    loss_l_list_val, loss_c_list_val, loss_c_nega_list_val, loss_c_posi_list_val = [], [], [], []
    loss_l_list_train, loss_c_list_train, loss_c_nega_list_train, loss_c_posi_list_train = [], [], [], []

    Make_data = make_training_val_data(augmentation_name, f_log, args)
    Validation_data_path = Make_data.make_validation_data()
    Dataset_val = webdataset.WebDataset(Validation_data_path).decode("pil").to_tuple("png", "json").map(preprocess)
    dl_val = torch.utils.data.DataLoader(
        Dataset_val, collate_fn=od_collate_fn, batch_size=32, num_workers=2, pin_memory=True
    )

    #############
    ## 学習開始 ##
    #############
    for epoch in range(num_epochs):
        ########################
        ## Training dataの作成 ##
        ########################
        ## png形式のRing画像とjson形式のlabelを作成
        Ring_path, NonRing_path = Make_data.make_training_data(train_cfg, epoch)
        train_Ring_num = Make_data.data_logger()

        Train_Ring_path = (
            webdataset.WebDataset(Ring_path).shuffle(10000000).decode("pil").to_tuple("png", "json").map(preprocess)
        )
        Train_NonRing_path = (
            webdataset.WebDataset(NonRing_path)
            .rsample(0.1)
            .shuffle(10000000)
            .decode("pil")
            .to_tuple("png", "json")
            .map(preprocess)
        )
        dl_ring_train = torch.utils.data.DataLoader(
            Train_Ring_path, collate_fn=od_collate_fn, batch_size=args.batch_size, num_workers=2, pin_memory=True
        )
        dl_noring_train = torch.utils.data.DataLoader(
            Train_NonRing_path, collate_fn=od_collate_fn, batch_size=NonRing_mini_batch, num_workers=2, pin_memory=True
        )
        dataloaders_dict = {"train": dl_ring_train, "val": dl_val}

        t_epoch_start = time.time()
        print_and_log(f_log, ["-------------", "Epoch {}/{}".format(epoch + 1, num_epochs), "-------------"])

        train_bbbb_loc, train_bbbb_conf, train_bbbb_b, train_seikai = [], [], [], []
        val_bbbb_loc, val_bbbb_conf, val_bbbb_b, val_seikai = [], [], [], []
        train_f1_score_l, train_f1_score_l_non_ring, val_f1_score_l, val_f1_score_l_non_ring = [], [], [], []

        iteration, val_iter, epoch_train_loss, epoch_val_loss = 0, 0.0, 0.0, 0.0
        loss_ll_val, loss_cc_val, loss_c_posii_val, loss_c_negaa_val = 0.0, 0.0, 0.0, 0.0
        loss_ll_train, loss_cc_train, loss_c_posii_train, loss_c_negaa_train = 0.0, 0.0, 0.0, 0.0

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()  # モデルを訓練モードに
                iter_noring = dl_noring_train.__iter__()
                print_and_log(f_log, f" ({phase}) ")
            else:
                print_and_log(f_log, f" ({phase}) ")
                net.eval()

            for images, targets in dataloaders_dict[phase]:
                if phase == "train":
                    # images = torch.from_numpy(train_rng.uniform(0.5, 1.8, size=(images.shape[0],1,1,1))) * images
                    noring = next(iter_noring, None)
                    if noring is None:
                        iter_noring = dl_noring_train.__iter__()  # 最後まで行っていたら最初に戻して
                        noring = next(iter_noring)
                    images = np.concatenate((images, noring[0]))
                    targets = targets + noring[1]
                    images = torch.from_numpy(images)
                else:
                    images = torch.from_numpy(images)

                images = images.permute(0, 3, 1, 2)[:, :2, :, :]
                images = images.to(device, dtype=torch.float)
                targets = [ann.to(device, dtype=torch.float) for ann in targets]  # リストの各要素のテンソルをGPUへ
                ## optimizerを初期化
                optimizer.zero_grad()
                ## 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == "train"):
                    # 順伝搬（forward）計算
                    outputs, decoded_box = net(images)
                    loss_l, loss_c, loss_c_posi, loss_c_nega = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        train_seikai.extend([ann.to("cpu").detach().numpy() for ann in targets])

                        loss_ll_train += loss_l.to("cpu").item()
                        loss_cc_train += loss_c.to("cpu").item()
                        loss_c_posii_train += loss_c_posi.to("cpu").item()
                        loss_c_negaa_train += loss_c_nega.to("cpu").item()
                        train_bbbb_loc.append(outputs[0].to("cpu"))
                        train_bbbb_conf.append(outputs[1].to("cpu"))
                        train_bbbb_b.append(outputs[2].to("cpu"))

                        loss.backward()
                        epoch_train_loss += loss.item()
                        ## 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配10.0に留める
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)

                        optimizer.step()  # パラメータ更新
                        print(
                            "\r"
                            + str(iteration)
                            + "/"
                            + str(
                                int(
                                    (int(train_Ring_num) + int(train_Ring_num * args.NonRing_ratio))
                                    / (args.batch_size + NonRing_mini_batch)
                                )
                            )
                            + "       ",
                            end="",
                        )

                        iteration += 1

                    else:
                        val_seikai.extend([ann.to("cpu").detach().numpy() for ann in targets])
                        val_bbbb_loc.append(outputs[0].to("cpu"))
                        val_bbbb_conf.append(outputs[1].to("cpu"))
                        val_bbbb_b.append(outputs[2].to("cpu"))

                        loss_ll_val += loss_l.to("cpu").item()
                        loss_cc_val += loss_c.to("cpu").item()
                        loss_c_posii_val += loss_c_posi.to("cpu").item()
                        loss_c_negaa_val += loss_c_nega.to("cpu").item()

                        epoch_val_loss += loss.to("cpu").item()
                        val_iter += 1

        ## Lossの計算
        avg_train_loss = epoch_train_loss / iteration
        avg_val_loss = epoch_val_loss / val_iter
        t_epoch_finish = time.time()

        print_and_log(
            f_log,
            [
                "\nepoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f} ".format(
                    epoch + 1, avg_train_loss, avg_val_loss
                ),
                "time:  {:.4f} sec.".format(t_epoch_finish - t_epoch_start),
                "avarage_loss_l:{:.4f} ||avarage_loss_c:{:.4f} ||avarage_loss_c_posi:{:.4f} ||avarage_loss_c_nega:{:.4f}".format(
                    loss_ll_val / val_iter,
                    loss_cc_val / val_iter,
                    loss_c_posii_val / val_iter,
                    loss_c_negaa_val / val_iter,
                ),
            ],
        )

        loss_c_list_val.append(loss_cc_val / val_iter)
        loss_l_list_val.append(loss_ll_val / val_iter)
        loss_c_posi_list_val.append(loss_c_posii_val / val_iter)
        loss_c_nega_list_val.append(loss_c_negaa_val / val_iter)

        loss_c_list_train.append(loss_cc_train / iteration)
        loss_l_list_train.append(loss_ll_train / iteration)
        loss_c_posi_list_train.append(loss_c_posii_train / iteration)
        loss_c_nega_list_train.append(loss_c_negaa_train / iteration)

        val_bbbb = [torch.cat(val_bbbb_loc), torch.cat(val_bbbb_conf), val_bbbb_b[0]]
        train_bbbb = [torch.cat(train_bbbb_loc), torch.cat(train_bbbb_conf), train_bbbb_b[0]]
        train_f1_score, train_threthre, train_f1_score_non_ring, train_threthre_noring = calc_f1score(
            train_seikai, train_bbbb, mode="train", iou=args.True_iou, top_k=50
        )
        val_f1_score, val_threthre, val_f1_score_non_ring, val_threthre_noring = calc_f1score(
            val_seikai, val_bbbb, mode="val", iou=args.True_iou
        )

        print_and_log(
            f_log,
            [
                "train_f1_score : {:.4f}, threshold : {:.4f}".format(train_f1_score, train_threthre),
                "train_f1_score_add_non_ring : {:.4f}, threshold : {:.4f}\n".format(
                    train_f1_score_non_ring, train_threthre_noring
                ),
                "val_f1_score : {:.4f}, threshold : {:.4f}".format(val_f1_score, val_threthre),
                "val_f1_score_add_non_ring : {:.4f}, threshold_add_non_ring : {:.4f}\n".format(
                    val_f1_score_non_ring, val_threthre_noring
                ),
            ],
        )

        train_f1_score_l.append(train_f1_score)
        train_f1_score_l_non_ring.append(train_f1_score_non_ring)
        val_f1_score_l.append(val_f1_score)
        val_f1_score_l_non_ring.append(val_f1_score_non_ring)

        t_epoch_start = time.time()

        ## ログを保存
        log_epoch = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "avarage_loss_l": loss_ll_val / val_iter,
            "avarage_loss_c": loss_cc_val / val_iter,
            "avarage_loss_c_posi": loss_c_posii_val / val_iter,
            "avarage_loss_c_nega": loss_c_negaa_val / val_iter,
            "val_f1_score": val_f1_score,
            "val_f1_score_non_ring": val_f1_score_non_ring,
            "train_f1_score": train_f1_score,
            "train_f1_score_non_ring": train_f1_score_non_ring,
        }

        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(augmentation_name + "/log_output.csv")

        early_stopping(val_f1_score, net)
        # early_stopping(epoch_val_loss, net)

        if early_stopping.counter == 0:
            f_early_ = open(augmentation_name + "/train_bbbb.txt", "wb")
            pickle.dump(train_bbbb, f_early_)
            f_early_ = open(augmentation_name + "/val_bbbb.txt", "wb")
            pickle.dump(val_bbbb, f_early_)
            f_early_ = open(augmentation_name + "/train_seikai.txt", "wb")
            pickle.dump(train_seikai, f_early_)
            f_early_ = open(augmentation_name + "/val_seikai.txt", "wb")
            pickle.dump(val_seikai, f_early_)

        if early_stopping.early_stop:
            print_and_log(f_log, "Early_Stopping")
            break

        ## データの削除
        os.remove(Ring_path)
        os.remove(NonRing_path)
        shutil.rmtree(args.savedir_path + "".join("dataset") + "/" + augmentation_name.split("/")[-1] + "/train")

        # ネットワークを保存する
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), augmentation_name + "/ssd300_" + str(epoch + 1) + ".pth")

    return loss_l_list_val, loss_c_list_val, loss_l_list_train, loss_c_list_train, train_f1_score_l, val_f1_score_l
