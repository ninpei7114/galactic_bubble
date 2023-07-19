import copy
import os

import astropy.io.fits
import astropy.wcs
import numpy as np
import pandas as pd
import tqdm

import label_caliculator
import processing
import ring_augmentation


def make_ring(savedir_name, train_cfg, args, train_l, trans_rng, epoch):
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))

    ## choice catalogue from 'CH' or 'MWP'
    choice_catalogue = "CH"
    Ring_CATALOGUE = ring_augmentation.catalogue(choice_catalogue)
    pbar = tqdm.tqdm(range(len(train_l)))
    flip = train_cfg["flip"]
    rot = train_cfg["rotate"]
    scale = train_cfg["scale"]
    translation = train_cfg["translation"]

    frame_mwp_train, mwp_ring_list_train = [], []

    for i in pbar:
        pbar.set_description(train_l[i])
        fits_path = train_l[i]
        spitzer_rfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "r.fits")[0]
        spitzer_gfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "g.fits")[0]
        spitzer_bfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "b.fits")[0]

        ## RGBにしたいため、fitsのdataを重ねる
        data = np.concatenate(
            [
                processing.remove_nan(spitzer_rfits.data[:, :, None]),
                processing.remove_nan(spitzer_gfits.data[:, :, None]),
                processing.remove_nan(spitzer_bfits.data[:, :, None]),
            ],
            axis=2,
        )

        a = data.shape[0]
        b = data.shape[1]
        w = astropy.wcs.WCS(spitzer_rfits.header)

        GLON_min, GLAT_min = w.all_pix2world(b, 0, 0)
        GLON_max, GLAT_max = w.all_pix2world(0, a, 0)
        GLON_center = (GLON_min + GLON_max) / 2
        GLON_new_min = GLON_center - 1.5
        GLON_new_max = GLON_center + 1.5

        Ring_catalogue = Ring_CATALOGUE.query("@GLON_new_min < GLON <= @GLON_new_max")

        ## star_listは辞書
        label_cal = label_caliculator.label_caliculator(choice_catalogue, w)
        label_cal.all_star(Ring_catalogue)

        for _, row in Ring_catalogue.iterrows():
            x_pix_min, y_pix_min, x_pix_max, y_pix_max, flag = label_cal.calc_pix(
                row, GLON_new_min, GLON_new_max, GLAT_min, GLAT_max, 1
            )
            if flag:  # calc_pix時に100回試行してもできなかった場合の場合分け
                label_cal.find_cover()
                if x_pix_min < 0 or y_pix_min < 0:
                    pass

                else:
                    c_data = data[int(y_pix_min) : int(y_pix_max), int(x_pix_min) : int(x_pix_max)].view()
                    cut_data = copy.deepcopy(c_data)
                    if np.isnan(cut_data.sum()):
                        pass

                    else:
                        ########################
                        ## 普通に切り出したリング ##
                        ########################
                        pi = processing.conv(300, sig1, cut_data)
                        pi_ = copy.deepcopy(pi)
                        label_cal.make_label(Ring_catalogue)
                        r_shape_y = pi_.shape[0]
                        r_shape_x = pi_.shape[1]
                        res_data = pi_[
                            int(r_shape_y / 7) : int(r_shape_y * 6 / 7), int(r_shape_x / 7) : int(r_shape_x * 6 / 7)
                        ]
                        res_data = processing.norm_res(res_data)

                        xmin_list, ymin_list, xmax_list, ymax_list, name_list = label_cal.check_list()
                        info = {
                            "fits": fits_path,
                            "name": name_list,
                            "xmin": xmin_list,
                            "xmax": xmax_list,
                            "ymin": ymin_list,
                            "ymax": ymax_list,
                        }

                        #######################
                        ## Ring augmentation ##
                        #######################
                        for _ in range(args.augmentation_ratio):
                            label_cal_for_trans = label_caliculator.label_caliculator(choice_catalogue, w)
                            label_cal_for_trans.all_star(Ring_catalogue)
                            trans_params = {
                                "row": row,
                                "fits_path": fits_path,
                                "GLON_new_min": GLON_new_min,
                                "GLON_new_max": GLON_new_max,
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
                                    append_data(
                                        processing.norm_res(trans_data_),
                                        trans_info,
                                        mwp_ring_list_train,
                                        frame_mwp_train,
                                    )

                            ###### 回転 ######
                            if rot:
                                if translation:
                                    if fl:
                                        for deg in [90, 180, 270]:
                                            rot_data, rotate_info = ring_augmentation.rotate_data(
                                                deg, trans_data, trans_info
                                            )
                                            append_data(
                                                processing.norm_res(rot_data),
                                                rotate_info,
                                                mwp_ring_list_train,
                                                frame_mwp_train,
                                            )
                                    else:
                                        pass
                                else:
                                    for deg in [90, 180, 270]:
                                        rot_data, rotate_info = ring_augmentation.rotate_data(deg, res_data, info)
                                        append_data(
                                            processing.norm_res(rot_data),
                                            rotate_info,
                                            mwp_ring_list_train,
                                            frame_mwp_train,
                                        )

                            ###### 上下反転 ######
                            if flip:
                                if translation:
                                    if fl:
                                        ud_res_data, lr_res_data, ud_info, lr_info = ring_augmentation.flip_data(
                                            trans_data, trans_info
                                        )
                                        append_data(
                                            processing.norm_res(ud_res_data),
                                            ud_info,
                                            mwp_ring_list_train,
                                            frame_mwp_train,
                                        )
                                        append_data(
                                            processing.norm_res(lr_res_data),
                                            lr_info,
                                            mwp_ring_list_train,
                                            frame_mwp_train,
                                        )
                                    else:
                                        pass
                                else:
                                    ud_res_data, lr_res_data, ud_info, lr_info = ring_augmentation.flip_data(
                                        res_data, info
                                    )
                                    append_data(
                                        processing.norm_res(ud_res_data),
                                        ud_info,
                                        mwp_ring_list_train,
                                        frame_mwp_train,
                                    )
                                    append_data(
                                        processing.norm_res(lr_res_data),
                                        lr_info,
                                        mwp_ring_list_train,
                                        frame_mwp_train,
                                    )

    frame_mwp_train = pd.DataFrame(frame_mwp_train)
    frame_mwp_train["id"] = [i for i in range(len(frame_mwp_train))]
    mwp_ring_list_train = np.array(mwp_ring_list_train).astype(np.float32)
    os.makedirs(savedir_name, exist_ok=True)

    mwp_ring_list_train_ = np.uint8(mwp_ring_list_train * 255)
    if mwp_ring_list_train_.shape[0] > 3000:
        slice = 10
    else:
        slice = 1
    os.makedirs(savedir_name + "/train_ring_pdf", exist_ok=True)
    processing.data_view_rectangl(25, mwp_ring_list_train_[::slice], frame_mwp_train[::slice]).save(
        savedir_name + "/train_ring_pdf" + f"/train_ring_{epoch}.pdf"
    )
    frame_mwp_train.to_csv(savedir_name + "/train_label.csv")

    return mwp_ring_list_train, frame_mwp_train


def append_data(data, info, data_list, frame):
    if not np.isnan(data.sum()):
        data_list.append(data)
        frame.append(info)
