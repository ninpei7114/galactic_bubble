import collections
import copy
import time
from itertools import product as product
from math import sqrt as sqrt

import astropy.io.fits
import astropy.wcs
import numpy as np
import pandas as pd
import ring_augmentation
import torch
import torch.nn as nn
import torch.nn.init as init
from processing import conv, data_view_rectangl, norm_res, remove_nan
from utils.ssd_model import nm_suppression


class EarlyStopping_f1_score:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, path, flog, patience=10, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.f1_score_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.flog = flog

    def __call__(self, f1_score, model, epoch, optimizer, loss_train, loss_eval):
        score = f1_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(f1_score, model, epoch, optimizer, loss_train, loss_eval)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            self.flog.write(f"EarlyStopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(f1_score, model, epoch, optimizer, loss_train, loss_eval)
            self.counter = 0

    def save_checkpoint(self, f1_score, model, epoch, optimizer, loss_train, loss_eval):
        """Saves model when f1_score increase."""
        if self.verbose:
            self.trace_func(f"f1_score increase ({self.f1_score_max:.6f} --> {f1_score:.6f}).  Saving model ...")
            self.flog.write(f"f1_score increase ({self.f1_score_max:.6f} --> {f1_score:.6f}).  Saving model ...\n")
        # torch.save(model.state_dict(), self.path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1_score": f1_score,
                "loss_train": loss_train,
                "loss_eval": loss_eval,
            },
            self.path,
        )
        self.f1_score_max = f1_score


class EarlyStopping_loss:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, path, flog, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.flog = flog

    def __call__(self, loss_val, model, epoch, optimizer, loss_train, f1_score):
        score = -loss_val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss_val, model, epoch, optimizer, loss_train, f1_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            self.flog.write(f"EarlyStopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss_val, model, epoch, optimizer, loss_train, f1_score)
            self.counter = 0

    def save_checkpoint(self, loss_val, model, epoch, optimizer, loss_train, f1_score):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {loss_val:.6f}).  Saving model ..."
            )
            self.flog.write(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {loss_val:.6f}).  Saving model ...\n"
            )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1_score": f1_score,
                "loss_train": loss_train,
                "loss_eval": loss_val,
            },
            self.path,
        )
        self.val_loss_min = loss_val


def weights_init(m):
    """Initialize the parameters of the model

    Args:
        m (_type_): _description_
    """
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def calc_location_each_region(detections, position, regions, conf_thre):
    """Convert bbox to coordinates on the overall map

    Args:
        detections (numpy array) : The output of the model to which the Detect function has been applied
        position (numpy array)   : Pixel coordinates when the image was cropped
        regions (list)           : List of strings indicating which fits file it is
        conf_thre (float)        : A number between 0.3 and 0.8

    Returns:
        predict_bbox (list) : Detected bubble coordinates on the map corrected from bbox and cropping coordinates
        scores (list)       : Score of the detected bubble
        wcs_regions (list)  : List of strings indicating which fits file it is
    """
    predict_bbox = []
    scores = []
    wcs_regions = []

    for d, p, w in zip(detections, position, regions):
        conf_mask = d[1, :, 0] >= conf_thre
        detection_mask = d[1, :][conf_mask]
        if np.sum(conf_mask) >= 1:
            bbox = detection_mask[:, 1:] * np.array(int(p[2]))
            bbox = bbox + np.array([int(p[1]), int(p[0]), int(p[1]), int(p[0])])

            predict_bbox.append(bbox)
            scores.append(detection_mask[:, 0])
            wcs_regions.append(w)

    return predict_bbox, scores, wcs_regions


def make_catalogue(region_dict, Ring_CATALOGUE, args):
    """_summary_

    Args:
        region_dict (dictionary): A dictionary storing ring position information and score on the map for each region
        Ring_CATALOGUE (pandas dataframe): MWP Catalog
        args (args): argparse

    Returns:
        mwp (pandas dataframe): Catalog of MWP rings within the validation region
        catalogue (pandas dataframe): Position coordinates of detected rings converted to wcs coordinate system
    """
    target_catalogue = []
    target_catalogue_fits_path = []
    infer_catalogue = []

    for key, value in region_dict.items():
        bbox = torch.Tensor(np.concatenate(value[0]))
        scores = torch.Tensor(np.concatenate(value[1]))
        keep, count = nm_suppression(bbox, scores, overlap=0.3, top_k=5000)
        keep = keep[:count]
        bbox = bbox[keep]
        scores = scores[keep]

        spitzer_g = astropy.io.fits.open(f"{args.spitzer_path}/spitzer_{key}_rgb/g.fits")[0]
        w = astropy.wcs.WCS(spitzer_g.header)
        a = spitzer_g.data.shape[0]
        b = spitzer_g.data.shape[1]
        GLON_min, GLAT_min = w.all_pix2world(b, 0, 0)
        GLON_max, GLAT_max = w.all_pix2world(0, a, 0)

        MWP_ = Ring_CATALOGUE.query("@GLON_min < GLON <= @GLON_max")
        target_catalogue.append(MWP_)
        target_catalogue_fits_path.extend([f"spitzer_{key}_rgb"] * len(MWP_))

        for i in bbox:
            GLONmax, GLATmin = w.all_pix2world(i[0], i[1], 0)
            GLONmin, GLATmax = w.all_pix2world(i[2], i[3], 0)
            temp = pd.DataFrame(
                columns=["dec_min", "ra_min", "dec_max", "ra_max", "fits_path"],
                data=[[GLATmin, GLONmin, GLATmax, GLONmax, f"spitzer_{key}_rgb"]],
            )
            infer_catalogue.append(temp)

    infer_catalogue = pd.concat(infer_catalogue)
    target_catalogue = pd.concat(target_catalogue)  # .reset_index()
    mask = ~target_catalogue.duplicated()
    target_catalogue = target_catalogue[mask]
    target_catalogue["fits_path"] = list(np.array(target_catalogue_fits_path)[mask])

    return target_catalogue.reset_index(), infer_catalogue


def calc_TP_FP_FN(target_catalogue, infer_catalogue, Rout, val_ring_catalogue, world="Galactic"):
    """Calculate TP, FP, FN

    Args:
        mwp (pandas dataframe): MWP catalog within the validation region
        infer (pandas dataframe): Position coordinates of detected bubbles converted to wcs coordinate system

    Returns:
        TP (list): True Positive
        FP (list): False Positive
        mwp_mask (list): List where detected items from the MWP catalog are marked as True
    """
    TP = []
    FP = []
    target_mask = [False] * len(target_catalogue)

    if val_ring_catalogue == "MWP":
        rout_num = 1.3
    elif val_ring_catalogue == "CH":
        rout_num = 1
    elif val_ring_catalogue == "SUM":
        rout_num = 1.3

    if world == "Galactic":
        x_axis = "GLON"
        y_axis = "GLAT"
    else:
        x_axis = "_RA_icrs"
        y_axis = "_DE.icrs"

    for _, infer_row in infer_catalogue.iterrows():
        judge = []
        for t_i, t_row in target_catalogue.iterrows():
            GLON_min = t_row[x_axis] - rout_num * t_row[Rout] / 60
            GLON_max = t_row[x_axis] + rout_num * t_row[Rout] / 60
            GLAT_min = t_row[y_axis] - rout_num * t_row[Rout] / 60
            GLAT_max = t_row[y_axis] + rout_num * t_row[Rout] / 60
            star_area = (GLON_max - GLON_min) * (GLAT_max - GLAT_min)

            clip_GLON = np.clip([infer_row["ra_min"], infer_row["ra_max"]], GLON_min, GLON_max)
            clip_GLAT = np.clip([infer_row["dec_min"], infer_row["dec_max"]], GLAT_min, GLAT_max)
            clip_width = clip_GLON[1] - clip_GLON[0] + 1e-9
            clip_height = clip_GLAT[1] - clip_GLAT[0] + 1e-9
            clip_area = clip_width * clip_height

            if clip_area >= star_area * 1 / 3:
                target_mask[t_i] = True
                judge.append(True)
            else:
                judge.append(False)
        if np.sum(judge) >= 1:
            TP.append(infer_row)
        else:
            FP.append(infer_row)
    return TP, FP, target_mask


def imaging_infer_result(args, frame, save_name, Rout, infer_result=False):
    """推論結果を保存する関数

    Args:
        args (args)                   : Arguments from argparse
        frame (pandas dataframe)      : Inference results
        save_name (str)               : Filename to save
        infer_result (bool, optional) : If it's an inference result, set to True. Defaults to False.

    """
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
    test_region = [
        "spitzer_01200+0000_rgb",
        "spitzer_01500+0000_rgb",
        "spitzer_01800+0000_rgb",
        "spitzer_02100+0000_rgb",
    ]
    fits_data_dict = {}
    for fits in test_region:
        data_fits_R = args.spitzer_path + f"/{fits}/r.fits"  ##2D
        data_fits_G = args.spitzer_path + f"/{fits}/g.fits"  ##2D
        data_fits_B = args.spitzer_path + f"/{fits}/b.fits"

        spitzer_g = astropy.io.fits.open(data_fits_G)[0]
        w = astropy.wcs.WCS(spitzer_g.header)
        data = np.concatenate(
            [
                remove_nan(astropy.io.fits.getdata(data_fits_R)[:, :, None]),
                remove_nan(astropy.io.fits.getdata(data_fits_G)[:, :, None]),
                remove_nan(astropy.io.fits.getdata(data_fits_B)[:, :, None]),
            ],
            axis=2,
        )
        fits_data_dict[fits] = [data, w]

    data_list = []
    for _, row in frame.iterrows():
        data, w = fits_data_dict[row["fits_path"]]
        if infer_result:
            x_min, y_min = w.all_world2pix(row["ra_max"], row["dec_min"], 0)
            x_max, y_max = w.all_world2pix(row["ra_min"], row["dec_max"], 0)
        else:
            l_center = row["GLON"]
            b_center = row["GLAT"]
            x_center, y_center = w.all_world2pix(l_center, b_center, 0)
            w_rout = 1.3 * row[Rout] / 60 / spitzer_g.header["CD2_2"]
            x_min = int(x_center) - w_rout
            x_max = int(x_center) + w_rout
            y_min = int(y_center) - w_rout
            y_max = int(y_center) + w_rout

        width = x_max - x_min
        height = y_max - y_min
        x_pix_min = x_min - width / 50
        y_pix_min = y_min - height / 50
        x_pix_max = x_max + width / 50
        y_pix_max = y_max + height / 50

        if x_pix_min <= 0 or y_pix_min <= 0:
            pass
        else:
            c_data = data[int(y_pix_min) : int(y_pix_max), int(x_pix_min) : int(x_pix_max)].view()
            cut_data = copy.deepcopy(c_data)
            pi = conv(300, sig1, cut_data)
            r_shape_y = pi.shape[0]
            r_shape_x = pi.shape[1]
            res_data = pi[
                int(r_shape_y / 52) : int(r_shape_y * 51 / 52), int(r_shape_x / 52) : int(r_shape_x * 51 / 52)
            ]
            if np.isnan(res_data.sum()) or np.std(res_data[:, :, 0]) < 1e-9:
                pass
            else:
                res_data = norm_res(
                    res_data, astropy.io.fits.open(data_fits_R)[0].header["PIXSCAL1"], spitzer_g.header["PIXSCAL1"]
                )
                data_list.append(res_data)

    if len(data_list) >= 1:
        data_list = np.uint8(np.array(data_list) * 255)
        data_list[:, :, :, 2] = 0
        data_view_rectangl(10, data_list).save(save_name)


## Compare with the ring catalog of the Milky Way Project and calculate the F1 score
def calc_fscore_val(detections, position, regions, args, threshold=None, save=False, save_path=None):
    """Calculate the f1 score

    Args:
        detections (numpy array)    : The result of applying the Detect function to the model output
        position (numpy array)      : Pixel coordinates when cropping the image
        regions (list)              : A list of strings indicating which fits file
        args (args)                 : Arguments from argparse
        threshold (float, optional) : A number between 0.3 and 0.8. Defaults to None.
        save (bool, optional)       : If True, it saves the inference results. Defaults to False.
        save_path (str, optional)   : The path to save the inference results. Defaults to None.

    Returns:
        F_score (float)    : f1score
        Precision (float)  : Precision
        Recall (float)     : Recall
        conf_thre (float)  : A number between 0.3 and 0.8
    """
    if threshold is None:
        thresholds = np.arange(0.5, 0.96, 0.01)
    else:
        thresholds = [threshold]

    if args.val_ring_catalogue == "MWP":
        Rout = "MajAxis"
    else:
        Rout = "Rout"
    Ring_CATALOGUE = ring_augmentation.catalogue(args.val_ring_catalogue, ring_select=True)
    F_score = -10000

    for conf_thre_ in thresholds:
        predict_bbox, scores, wcs_regions = calc_location_each_region(detections, position, regions, conf_thre_)
        if len(predict_bbox) > 0:
            ## Create a dictionary to store the location information and score for each region
            region_dict = {}
            for i in list(collections.Counter(wcs_regions).keys()):
                region_dict[i] = [[], []]

            ## Store the bbox and score for each region
            for p, s, w in zip(predict_bbox, scores, wcs_regions):
                region_dict[w][0].append(p)
                region_dict[w][1].append(s)

            target_catalogue_, infer_catalogue_ = make_catalogue(region_dict, Ring_CATALOGUE, args)
            _, FP_c_, target_mask_ = calc_TP_FP_FN(target_catalogue_, infer_catalogue_, Rout, args.val_ring_catalogue)

            TP = target_mask_.count(True)
            FN = target_mask_.count(False)
            FP = len(FP_c_)
            Precision_ = TP / (TP + FP)
            Recall_ = TP / (TP + FN)
            if args.fscore == "f2_score":
                F_score_ = 5 * Precision_ * Recall_ / (4 * Precision_ + Recall_ + 1e-9)
            elif args.fscore == "f1_score":
                F_score_ = 2 * Precision_ * Recall_ / (Precision_ + Recall_ + 1e-9)
            elif args.fscore == "f05_score":
                F_score_ = 1.25 * Precision_ * Recall_ / (0.25 * Precision_ + Recall_ + 1e-9)

            if F_score_ > F_score:
                F_score = F_score_
                conf_thre = conf_thre_
                Precision = Precision_
                Recall = Recall_
                infer_catalogue = infer_catalogue_
                target_catalogue = target_catalogue_
                target_mask = target_mask_
                FP_c = FP_c_

    if save:
        infer_catalogue.to_csv(save_path + "/infer_catalogue_test.csv")
        target_catalogue.to_csv(save_path + "/target_catalogue_test.csv")
        imaging_infer_result(args, target_catalogue[target_mask], save_path + "/test_TP.jpeg", Rout)
        imaging_infer_result(args, target_catalogue[~np.array(target_mask)], save_path + "/test_FN.jpeg", Rout)
        imaging_infer_result(args, pd.DataFrame(FP_c), save_path + "/test_FP.jpeg", Rout, infer_result=True)
    return F_score, Precision, Recall, conf_thre


def print_and_log(f, moji):
    """Performs print and log simultaneously

    Args:
        f (txt file) : File to save the log
        moji (str)   : String to output to print and log
    """
    if isinstance(moji, list):
        for i in moji:
            print(i)
            f.write(i + "\n")
    else:
        print(moji)
        f.write(moji + "\n")


class management_loss:
    """This class manages loss"""

    def __init__(self):
        self.loc_l_val_s, self.conf_l_val_s = [], []
        self.loc_l_train_s, self.conf_l_train_s = [], []

    def __call__(self):
        self.loc_l_val, self.conf_l_val, self.conf_l_pos_val, self.conf_l_neg_val = 0.0, 0.0, 0.0, 0.0
        self.loc_l_train, self.conf_l_train, self.conf_l_pos_train, self.conf_l_neg_train = 0.0, 0.0, 0.0, 0.0

    def sum_iter_loss(self, loss_dic, mode):
        if mode == "train":
            self.loc_l_train += loss_dic["loc_loss"].to("cpu").item()
            self.conf_l_train += loss_dic["conf_loss"].to("cpu").item()
            self.conf_l_pos_train += loss_dic["conf_loss_positive"].to("cpu").item()
            self.conf_l_neg_train += loss_dic["conf_loss_negative"].to("cpu").item()

        elif mode == "val":
            self.loc_l_val += loss_dic["loc_loss"].to("cpu").item()
            self.conf_l_val += loss_dic["conf_loss"].to("cpu").item()
            self.conf_l_pos_val += loss_dic["conf_loss_positive"].to("cpu").item()
            self.conf_l_neg_val += loss_dic["conf_loss_negative"].to("cpu").item()

    def output_each_loss(self, mode, iteration):
        if mode == "train":
            self.loc_l_train_s.append(self.loc_l_train / iteration)
            self.conf_l_train_s.append(self.conf_l_train / iteration)

            return {
                "loc_loss": self.loc_l_train / iteration,
                "conf_loss": self.conf_l_train / iteration,
                "conf_loss_positive": self.conf_l_pos_train / iteration,
                "conf_loss_negative": self.conf_l_neg_train / iteration,
            }

        elif mode == "val":
            self.loc_l_val_s.append(self.loc_l_val / iteration)
            self.conf_l_val_s.append(self.conf_l_val / iteration)

            return {
                "loc_loss": self.loc_l_val / iteration,
                "conf_loss": self.conf_l_val / iteration,
                "conf_loss_positive": self.conf_l_pos_val / iteration,
                "conf_loss_negative": self.conf_l_neg_val / iteration,
            }

    def output_all_epoch_loss(self):
        return self.loc_l_val_s, self.conf_l_val_s, self.loc_l_train_s, self.conf_l_train_s


def write_train_log(
    f_log,
    epoch,
    each_loss_train,
    each_loss_val,
    val_f_score,
    Precision,
    Recall,
    conf_threshold,
    epoch_start_time,
    args,
):
    """Output the log for each epoch
    Args:
        f_log (txt file)             : File to save the log
        epoch (int)                  : Number of epochs
        each_loss_train (dictionary) : loc_loss and conf_loss of training
        each_loss_val (dictionary)   : loc_loss and conf_loss of validation
        val_f1_score (float)         : f1_score of validation
        Precision (float)            : Precision of validation
        Recall (float)               : Recall of validation
        val_conf_threshold (float)   : conf_threshold of validation
        epoch_start_time (float)     : Start time of the epoch

    Returns:
        log_epoch (dictionary): Log for each epoch
    """
    epoch_finish_time = time.time()
    avg_train_loss = each_loss_train["loc_loss"] + each_loss_train["conf_loss"]
    avg_val_loss = each_loss_val["loc_loss"] + each_loss_val["conf_loss"]
    print_and_log(
        f_log,
        [
            "\nepoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f} ".format(
                epoch + 1, avg_train_loss, avg_val_loss
            ),
            "avarage_loc_loss:{:.4f} ||avarage_conf_loss:{:.4f} ||avarage_conf_loss_positive:{:.4f} ||avarage_conf_loss_negative:{:.4f}".format(
                each_loss_val["loc_loss"],
                each_loss_val["conf_loss"],
                each_loss_val["conf_loss_positive"],
                each_loss_val["conf_loss_negative"],
            ),
            "val_{} : {:.4f}, Precision : {:.4f}, Recall : {:.4f}, threshold : {:.4f}".format(
                args.fscore, val_f_score, Precision, Recall, conf_threshold
            ),
            "time:  {:.4f} sec.".format(epoch_finish_time - epoch_start_time),
        ],
    )

    ## ログを保存
    log_epoch = {
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "avarage_loc_loss": each_loss_val["loc_loss"],
        "avarage_conf_loss": each_loss_val["conf_loss"],
        "avarage_conf_loss_positive": each_loss_val["conf_loss_positive"],
        "avarage_conf_loss_negative": each_loss_val["conf_loss_negative"],
        "val_f05_score": 1.25 * Precision * Recall / (0.25 * Precision + Recall + 1e-9),
        "val_f1_score": 2 * Precision * Recall / (Precision + Recall + 1e-9),
        "val_f2_score": 5 * Precision * Recall / (4 * Precision + Recall + 1e-9),
        "val_precision": Precision,
        "val_recall": Recall,
        "val_conf_threshold": conf_threshold,
        "training_time": epoch_finish_time - epoch_start_time,
    }

    return log_epoch
