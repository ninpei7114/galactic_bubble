import collections
import copy
import time
from itertools import product as product
from math import sqrt as sqrt

import astropy.io.fits
import astropy.wcs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init

import ring_augmentation
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


def weights_init(m):
    """モデルのパラメーターを初期化する

    Args:
        m (_type_): _description_
    """
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def calc_location_each_region(detections, position, regions, conf_thre):
    """bboxを全体マップでの座標に変換する

    Args:
        detections (numpy array): モデルの出力にDetect関数を適用したもの
        position (numpy array): 画像を切り出した際のpixel座標
        regions (list): どのfitsファイルかを示すstrのlist
        conf_thre (float): 0.3~0.8の数字

    Returns:
        predict_bbox (list): bboxと切り出し座標から補正したマップ上の検出リング座標
        scores (list): 検出リングのscore
        wcs_regions (list): どのfitsファイルかを示すstrのlist
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
        region_dict (dictionary): 領域ごとのマップ上のリング位置情報とscoreを格納した辞書
        Ring_CATALOGUE (pandas dataframe): MWPリングのカタログ
        args (args): argparseの引数

    Returns:
        mwp (pandas dataframe): Validation領域内のMWPリングカタログ
        catalogue (pandas dataframe): wcs座標系に変換した検出リングの位置座標
    """
    target_MWP_catalogue = []
    catalogue = pd.DataFrame(columns=["dec_min", "ra_min", "dec_max", "ra_max"])

    for key, value in region_dict.items():
        bbox = torch.Tensor(np.concatenate(value[0]))
        scores = torch.Tensor(np.concatenate(value[1]))
        keep, count = nm_suppression(bbox, scores)
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
        target_MWP_catalogue.append(MWP_)

        for i in bbox:
            GLONmax, GLATmin = w.all_pix2world(i[0], i[1], 0)
            GLONmin, GLATmax = w.all_pix2world(i[2], i[3], 0)
            temp = pd.DataFrame(
                columns=["dec_min", "ra_min", "dec_max", "ra_max"],
                data=[[GLATmin, GLONmin, GLATmax, GLONmax]],
                dtype="float64",
            )
            catalogue = pd.concat([catalogue, temp])

    mwp = pd.concat(target_MWP_catalogue).reset_index()

    return mwp, catalogue


def calc_TP_FP_FN(mwp, infer):
    """TP, FP, FNを計算する

    Args:
        mwp (pandas dataframe): Validation領域内のMWPリングカタログ
        infer (pandas dataframe): wcs座標系に変換した検出リングの位置座標

    Returns:
        TP (list): True Positive
        FP (list): False Positive
        mwp_mask (list): MWPリングカタログのうち、検出されたものをTrueとしたリスト
    """
    TP = []
    FP = []
    mwp_mask = [False] * len(mwp)
    for _, infer_row in infer.iterrows():
        judge = []
        for mwp_i, mwp_row in mwp.iterrows():
            mwp_GLON_min = mwp_row["GLON"] - mwp_row["Reff"] / 60
            mwp_GLON_max = mwp_row["GLON"] + mwp_row["Reff"] / 60
            mwp_GLAT_min = mwp_row["GLAT"] - mwp_row["Reff"] / 60
            mwp_GLAT_max = mwp_row["GLAT"] + mwp_row["Reff"] / 60
            star_area = (mwp_GLON_max - mwp_GLON_min) * (mwp_GLAT_max - mwp_GLAT_min)

            clip_GLON = np.clip([infer_row["ra_min"], infer_row["ra_max"]], mwp_GLON_min, mwp_GLON_max)
            clip_GLAT = np.clip([infer_row["dec_min"], infer_row["dec_max"]], mwp_GLAT_min, mwp_GLAT_max)
            clip_width = clip_GLON[1] - clip_GLON[0] + 1e-9
            clip_height = clip_GLAT[1] - clip_GLAT[0] + 1e-9
            clip_area = clip_width * clip_height

            if clip_area >= star_area * 1 / 3:
                mwp_mask[mwp_i] = True
                judge.append(True)
            else:
                judge.append(False)
        if np.sum(judge) >= 1:
            TP.append(infer_row)
        else:
            FP.append(infer_row)
    return TP, FP, mwp_mask


def imaging_infer_result(args, frame, save_name, infer_result=False):
    """推論結果を保存する関数

    Args:
        args (args): argparseの引数
        frame (pandas dataframe): 推論結果
        save_name (str): 保存するファイル名
        infer_result (bool, optional): 推論結果の場合はTrue. Defaults to False.

    """
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
    data_list = []
    data_fits_R = args.spitzer_path + "/spitzer_01800+0000_rgb/r.fits"  ##2D
    data_fits_G = args.spitzer_path + "/spitzer_01800+0000_rgb/g.fits"  ##2D
    data_fits_B = args.spitzer_path + "/spitzer_01800+0000_rgb/b.fits"

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

    for _, row in frame.iterrows():
        if infer_result:
            x_min, y_min = w.all_world2pix(row["ra_max"], row["dec_min"], 0)
            x_max, y_max = w.all_world2pix(row["ra_min"], row["dec_max"], 0)
        else:
            l_center = row["GLON"]
            b_center = row["GLAT"]
            x_center, y_center = w.all_world2pix(l_center, b_center, 0)
            x_min = int(x_center) - row["MajAxis"] / 60 / spitzer_g.header["CD2_2"]
            x_max = int(x_center) + row["MajAxis"] / 60 / spitzer_g.header["CD2_2"]
            y_min = int(y_center) - row["MajAxis"] / 60 / spitzer_g.header["CD2_2"]
            y_max = int(y_center) + row["MajAxis"] / 60 / spitzer_g.header["CD2_2"]

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
            res_data = norm_res(res_data)
            data_list.append(res_data)

    data_list = np.uint8(np.array(data_list) * 255)
    data_list[:, :, :, 2] = 0
    data_view_rectangl(10, data_list).save(save_name)


## Milky Way Projectのリングカタログと比較し、F1scoreを算出する
def calc_f1score_val(detections, position, regions, args, threshold=None, save=False, save_path=None):
    """f1scoreを計算する

    Args:
        detections (numpy array): モデルの出力にDetect関数を適用したもの
        position (numpy array): 画像を切り出した際のpixel座標
        regions (list): どのfitsファイルかを示すstrのlist
        args (args): argparseの引数
        threshold (float, optional): 0.3~0.8の数字. Defaults to None.
        save (bool, optional): Trueにすると推論結果を保存する. Defaults to False.
        save_path (str, optional): 推論結果を保存するパス. Defaults to None.

    Returns:
        F1_score (float): f1score
        Precision (float): Precision
        Recall (float): Recall
        threthre (float): 0.3~0.8の数字
    """
    if threshold is None:
        thresholds = [i / 20 for i in range(6, 16, 1)]
    else:
        thresholds = [threshold]
    Ring_CATALOGUE = ring_augmentation.catalogue("MWP", args)
    F1_score = -10000

    for conf_thre in thresholds:
        predict_bbox, scores, wcs_regions = calc_location_each_region(detections, position, regions, conf_thre)

        ## 領域ごとの位置情報とscoreを格納する辞書を作成
        region_dict = {}
        for i in list(collections.Counter(wcs_regions).keys()):
            region_dict[i] = [[], []]

        ## 領域ごとのbboxとscoreを格納
        for p, s, w in zip(predict_bbox, scores, wcs_regions):
            region_dict[w][0].append(p)
            region_dict[w][1].append(s)

        mwp, catalogue = make_catalogue(region_dict, Ring_CATALOGUE, args)
        _, FP_, mwp_mask = calc_TP_FP_FN(mwp, catalogue)

        TP = mwp_mask.count(True)
        FN = mwp_mask.count(False)
        FP = len(FP_)
        Precision_ = TP / (TP + FP)
        Recall_ = TP / (TP + FN)
        F1_score_ = 2 * Precision_ * Recall_ / (Precision_ + Recall_)

        if F1_score_ > F1_score:
            F1_score = F1_score_
            threthre = conf_thre
            Precision = Precision_
            Recall = Recall_

    if save:
        catalogue.to_csv(save_path + "/infer_catalogue_l18.csv")
        imaging_infer_result(args, mwp[mwp_mask], save_path + "/l18_TP.png")
        imaging_infer_result(args, mwp[list(map(lambda x: not x, mwp_mask))], save_path + "/l18_FN.png")
        imaging_infer_result(args, pd.DataFrame(FP_), save_path + "/l18_FP.png", infer_result=True)
    return F1_score, Precision, Recall, threthre


def print_and_log(f, moji):
    """printとlogを同時に行う

    Args:
        f (txtファイル): logを保存するファイル
        moji (str): printとlogに出力する文字列
    """
    if isinstance(moji, list):
        for i in moji:
            print(i)
            f.write(i + "\n")
    else:
        print(moji)
        f.write(moji + "\n")


class management_loss:
    """lossの管理を行うクラス"""

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
    f_log, epoch, each_loss_train, each_loss_val, val_f1_score, Precision, Recall, val_conf_threshold, epoch_start_time
):
    """epochごとのlogを出力する
    Args:
        f_log (txtファイル): logを保存するファイル
        epoch (int): epoch数
        each_loss_train (dictionary): trainのloc_lossとconf_loss
        each_loss_val (dictionary): valのloc_lossとconf_loss
        val_f1_score (float): valのf1_score
        Precision (float): valのPrecision
        Recall (float): valのRecall
        val_conf_threshold (float): valのconf_threshold
        epoch_start_time (float): epochの開始時間

    Returns:
        log_epoch (dictionary): epochごとのlog
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
            "val_f1_score : {:.4f}, Precision : {:.4f}, Recall : {:.4f}, threshold : {:.4f}".format(
                val_f1_score, Precision, Recall, val_conf_threshold
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
        "val_f1_score": val_f1_score,
        "val_precision": Precision,
        "val_recall": Recall,
        "val_conf_threshold": val_conf_threshold,
        "training_time": epoch_finish_time - epoch_start_time,
    }

    return log_epoch