import copy
import glob
import json
import os

import astropy.io.fits
import astropy.wcs
import label_caliculator
import numpy as np
import pandas as pd
import processing
import ring_augmentation
import tqdm
from PIL import Image


def make_ring(savedir_name, train_cfg, args, train_l, trans_rng, epoch, save_data_path):
    """trainingに使用するRingを作成する関数

    Args:
        savedir_name (str): labelとring_pdfを保存するpath
        train_cfg (dictionary): augmentationの種類
        args (args): args
        train_l (list): trainingに使用するfitsのリスト
        trans_rng (numpy default_rng): numpy default_rng
        epoch (int): epochの上限
        save_data_path (str): 学習データを保存するpath

    Returns:
        None
    """
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
    choice_catalogue = args.training_ring_catalogue  # choice catalogue from 'CH' or 'MWP'
    Ring_CATALOGUE = ring_augmentation.catalogue(choice_catalogue, args.ring_select_false)
    pbar = tqdm.tqdm(range(len(train_l)))
    flip = train_cfg["flip"]
    rot = train_cfg["rotate"]
    translation = train_cfg["translation"]
    frame_mwp_train, count = [], 0

    for i in pbar:
        pbar.set_description(train_l[i])
        fits_path = train_l[i]
        spitzer_rfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "r.fits")[0]
        spitzer_gfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "g.fits")[0]
        spitzer_bfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "b.fits")[0]

        data = np.concatenate(
            [
                processing.remove_nan(spitzer_rfits.data[:, :, None]),
                processing.remove_nan(spitzer_gfits.data[:, :, None]),
                processing.remove_nan(spitzer_bfits.data[:, :, None]),
            ],
            axis=2,
        )

        #####################################
        ## fits範囲のChurchwellカタログを取得 ##
        #####################################
        a = data.shape[0]
        b = data.shape[1]
        w = astropy.wcs.WCS(spitzer_rfits.header)
        GLON_min, GLAT_min = w.all_pix2world(b, 0, 0)
        GLON_max, GLAT_max = w.all_pix2world(0, a, 0)
        Ring_catalogue = Ring_CATALOGUE.query("@GLON_min <= GLON <= @GLON_max")

        label_cal = label_caliculator.label_caliculator(choice_catalogue, w)
        label_cal.all_star(Ring_catalogue)

        for _, row in Ring_catalogue.iterrows():
            x_pix_min, y_pix_min, x_pix_max, y_pix_max, flag = label_cal.calc_pix(
                row, GLON_min, GLON_max, GLAT_min, GLAT_max, 1.7
            )
            if flag and x_pix_min >= 0 and y_pix_min >= 0:  # calc_pix時に100回試行してもできなかった場合の場合分け
                label_cal.find_cover()
                label_cal.make_label(Ring_catalogue)

                c_data = data[int(y_pix_min) : int(y_pix_max), int(x_pix_min) : int(x_pix_max)].view()
                cut_data = copy.deepcopy(c_data)
                pi = processing.conv(300, sig1, cut_data)
                r_shape_y = pi.shape[0]
                r_shape_x = pi.shape[1]
                res_data = pi[
                    int(r_shape_y / 52) : int(r_shape_y * 51 / 52), int(r_shape_x / 52) : int(r_shape_x * 51 / 52)
                ]
                if np.isnan(res_data.sum()) or np.std(res_data[:, :, 0]) < 1e-9:
                    pass
                else:
                    ########################
                    ## 普通に切り出したリング ##
                    ########################
                    info = label_cal.check_list()
                    info["fits"] = fits_path
                    count = make_png_and_json(save_data_path, count, processing.norm_res(res_data), info)
                    frame_mwp_train.append(info)
                    #######################
                    ## Ring augmentation ##
                    #######################
                    # for _ in range(args.augmentation_ratio):
                    label_cal_for_trans = label_caliculator.label_caliculator(choice_catalogue, w)
                    label_cal_for_trans.all_star(Ring_catalogue)
                    trans_params = {
                        "row": row,
                        "fits_path": fits_path,
                        "GLON_min": GLON_min,
                        "GLON_max": GLON_max,
                        "GLAT_min": GLAT_min,
                        "GLAT_max": GLAT_max,
                        "Ring_catalogue": Ring_catalogue,
                        "data": data,
                        "label_cal": label_cal_for_trans,
                        "trans_rg": trans_rng,
                    }

                    ###### 並行移動 ######
                    if translation:
                        fl, trans_data, trans_info = ring_augmentation.translation(**trans_params)
                        ## データやlabelの作成に不備があれば、fl=False(例えば、xmin<0や、xmin=xmaxなど)
                        ## 問題がなければ、fl=True
                        if fl:
                            trans_data_ = trans_data.copy()
                            count = make_png_and_json(
                                save_data_path, count, processing.norm_res(trans_data_), trans_info
                            )
                            frame_mwp_train.append(trans_info)
                    ###### 回転 ######
                    if rot:
                        if translation:
                            if fl:
                                for deg in [90, 180, 270]:
                                    rot_data, rotate_info = ring_augmentation.rotate_data(deg, trans_data, trans_info)
                                    count = make_png_and_json(
                                        save_data_path, count, processing.norm_res(rot_data), rotate_info
                                    )
                                    frame_mwp_train.append(rotate_info)
                            else:
                                pass
                        else:
                            for deg in [90, 180, 270]:
                                rot_data, rotate_info = ring_augmentation.rotate_data(deg, res_data, info)
                                count = make_png_and_json(
                                    save_data_path, count, processing.norm_res(rot_data), rotate_info
                                )
                                frame_mwp_train.append(rotate_info)
                    ###### 上下反転 ######
                    if flip:
                        if translation:
                            if fl:
                                ud_res_data, lr_res_data, ud_info, lr_info = ring_augmentation.flip_data(
                                    trans_data, trans_info
                                )
                                count = make_png_and_json(
                                    save_data_path, count, processing.norm_res(ud_res_data), ud_info
                                )
                                count = make_png_and_json(
                                    save_data_path, count, processing.norm_res(lr_res_data), lr_info
                                )
                                frame_mwp_train.append(ud_info)
                                frame_mwp_train.append(lr_info)
                            else:
                                pass
                        else:
                            ud_res_data, lr_res_data, ud_info, lr_info = ring_augmentation.flip_data(res_data, info)
                            count = make_png_and_json(save_data_path, count, processing.norm_res(ud_res_data), ud_info)
                            count = make_png_and_json(save_data_path, count, processing.norm_res(lr_res_data), lr_info)
                            frame_mwp_train.append(lr_info)
                            frame_mwp_train.append(ud_info)

    ##################################
    ## 学習に用いるリングのpdf画像を作成 ##
    ##################################
    frame_mwp_train = pd.DataFrame(frame_mwp_train)
    frame_mwp_train["id"] = [i for i in range(len(frame_mwp_train))]
    choice_ind = np.sort(trans_rng.choice(count, size=90, replace=False))
    mwp_ring_list_train = []
    for i in choice_ind:
        mwp_ring_list_train.append(np.array(Image.open(save_data_path + f"/train/ring/Ring_{i}.png")))

    os.makedirs(savedir_name + "/train_ring_pdf", exist_ok=True)
    processing.data_view_rectangl(25, np.array(mwp_ring_list_train), frame_mwp_train.iloc[choice_ind]).save(
        savedir_name + "/train_ring_pdf" + f"/train_ring_{epoch}.pdf"
    )
    frame_mwp_train.to_csv(savedir_name + "/train_label.csv")


def make_png_and_json(save_data_path, count, ring_data, info):
    pil_image = Image.fromarray(np.uint8(ring_data * 255))
    pil_image.save(f"{save_data_path}/train/ring/Ring_{count}.png")

    ll = []
    if len(info["xmin"]) >= 1:
        for la in range(len(info["xmin"])):
            ll.append(
                {
                    "Confidence": str(0),
                    "XMin": str(info["xmin"][la]),
                    "XMax": str(info["xmax"][la]),
                    "YMin": str(info["ymin"][la]),
                    "YMax": str(info["ymax"][la]),
                }
            )
    else:
        pass
    with open(f"{save_data_path}/train/ring/Ring_{count}.json", "w") as f:
        json.dump(ll, f, indent=4)

    count += 1

    return count
