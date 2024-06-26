import os
import shutil
import time
from itertools import product as product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.random import default_rng

from data import make_training_dataloader, make_validatoin_dataloader
from make_data import make_training_val_data
from make_figure import make_figure
from nonring_augmentation import nonring_augmentation
from training_sub import (
    EarlyStopping_f1_score,
    EarlyStopping_loss,
    calc_fscore_val,
    management_loss,
    print_and_log,
    write_train_log,
)
from utils.ssd_model import Detect


def train_model(
    net, criterion, optimizer, num_epochs, f_log, augmentation_name, args, train_cfg, device, run, val_size
):
    """Function to perform model training

    Args:
        net (pytorch Modulelist): SSD network
        criterion (MultiBoxLoss): Loss function
        optimizer (AdamW)       : Optimization method
        num_epochs (int)        : Maximum number of epochs
        f_log (txt file)        : log file
        augmentation_name (int) : Name of which augmentation was used
        args (args)             : argparse
        train_cfg (dictionary)  : Parameters for augmentation
        device (torch.device)   : GPU or CPU
    """
    NonRing_class = np.delete(np.arange(args.NonRing_class_num), args.NonRing_remove_class_list)
    NonRing_rg = default_rng(args.fits_random_state)
    # early_stopping = EarlyStopping_loss(
    #     patience=15, verbose=True, path=augmentation_name + "/earlystopping.pth", flog=f_log
    # )
    early_stopping = EarlyStopping_f1_score(
        patience=15, verbose=True, path=augmentation_name + "/earlystopping.pth", flog=f_log
    )
    detect = Detect(nms_thresh=0.45, top_k=500, conf_thresh=0.5)  # F1 scoreのconfの計算が0.3からなので、ここも0.3
    save_training_val_loss = management_loss()
    logs, f_score_val_l = [], []

    ##############################
    ## Creating Validation Data ##
    ##############################
    Make_data = make_training_val_data(augmentation_name, f_log, args)
    Validation_data_path, Val_num = Make_data.make_validation_data(val_size)
    dl_val = make_validatoin_dataloader(Validation_data_path, args)
    NonRing_num_l = Make_data.make_training_nonring_data()
    all_iter_val = int(int(Val_num) / args.Val_mini_batch)

    for epoch in range(num_epochs):
        start_time = time.time()
        iteration_train, iteration_val = 0, 0
        save_training_val_loss()  # Initialise loss
        print_and_log(f_log, ["-------------", "Epoch {}/{}".format(epoch + 1, num_epochs), "-------------"])

        ############################
        ## Creating Training Data ##
        ############################
        # Create Ring images in png format and labels in json format
        Training_data_path = Make_data.make_training_ring_data(train_cfg, epoch)
        train_Ring_num = Make_data.data_logger()
        # Create Dataloader for Training Ring
        dl_ring_train, dl_nonring = make_training_dataloader(
            Training_data_path, train_Ring_num, args, NonRing_num_l, NonRing_class
        )
        dataloaders_dict = {"train": dl_ring_train, "val": dl_val}
        all_iter = int(int(train_Ring_num) / args.Ring_mini_batch)

        ####################
        ## Start Training ##
        ####################
        for phase in ["train", "val"]:
            if phase == "train":
                print_and_log(f_log, f" ({phase}) ")
                net.train()
            else:
                print_and_log(f_log, f" \n ({phase})")
                net.eval()
                result, position, regions = [], [], []

            #####################################
            ## Data Shaping and Input to Model ##
            #####################################
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

                # Initialize the optimizer
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs, decoded_box = net(images)
                    loss_dic = criterion(outputs, targets)
                    loss = loss_dic["loc_loss"] + loss_dic["conf_loss"]

                    if phase == "train":
                        loss.backward()  # Calculate the gradient
                        # To prevent the calculation from becoming unstable when the gradient becomes too large, clip it to a maximum gradient of 10.0
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

        ###################
        ## Managing Loss ##
        ###################
        # Output the loss of loc, conf
        loss_train = save_training_val_loss.output_each_loss("train", iteration_train)
        loss_val = save_training_val_loss.output_each_loss("val", iteration_val)

        ########################################
        ## Calculation of Validation F1 score ##
        ########################################
        f_score_val, precision, recall, conf_threshold = calc_fscore_val(
            np.concatenate(result), np.array(position), regions, args
        )

        f_score_val_l.append(f_score_val)

        log_epoch = write_train_log(
            f_log, epoch, loss_train, loss_val, f_score_val, precision, recall, conf_threshold, start_time, args
        )
        run.log(log_epoch)
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(augmentation_name + "/log_output.csv")

        # early_stopping(loss_val["loc_loss"] + loss_val["conf_loss"], net, epoch, optimizer, loss_train, f_score_val)
        early_stopping(f_score_val, net, epoch, optimizer, loss_train, loss_val)
        if early_stopping.early_stop:
            print_and_log(f_log, "Early_Stopping")
            break

        # Delete data
        os.remove(f"{Training_data_path}/bubble_dataset_train_ring.tar")
        shutil.rmtree(args.savedir_path + "".join("dataset") + "/" + augmentation_name.split("/")[-1] + "/train")

    ## Plotting the transition of loss
    loc_l_val_s, conf_l_val_s, loc_l_train_s, conf_l_train_s = save_training_val_loss.output_all_epoch_loss()
    make_figure(augmentation_name, loc_l_val_s, conf_l_val_s, loc_l_train_s, conf_l_train_s, f_score_val_l)

    return df.iloc[df[f"val_{args.fscore}"].idxmax()][
        "val_conf_threshold"
    ]  # Return the conf_threshold when the f1_score of Val is maximum
