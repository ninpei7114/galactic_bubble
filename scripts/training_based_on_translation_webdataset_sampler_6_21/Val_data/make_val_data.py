import argparse
import copy
import json
import os
import sys

import astropy.io.fits
import astropy.wcs
import numpy as np
import pandas as pd
from numpy.random import default_rng
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

sys.path.append("../")
import processing
import val_label_calculator
import val_ring_sub

"""

example command:
python make_val_data.py /dataset/spitzer_data/ -r True
"""


def parse_args():
    parser = argparse.ArgumentParser(description="make validataion data for SSD")
    parser.add_argument("spitzer_path", metavar="DIR", help="spitzer_path")
    # parser.add_argument('savedir_name', metavar='DIR', help='savedir_name')
    parser.add_argument("--each_region", "-r", action="store_true")

    return parser.parse_args()


def main(args):
    ################################################
    ## 領域ごとに作るのか、デフォルトの領域で作るのか選択 ##
    ################################################

    if args.each_region:
        ## 各領域ごとにVal-Ringを作成する
        ## 'spitzer_29400+0000_rgb'は、8µmのデータが全然ないため使用しない
        # fmt: off
        val_l = [
        'spitzer_00300+0000_rgb','spitzer_00600+0000_rgb','spitzer_00900+0000_rgb','spitzer_01200+0000_rgb',
        'spitzer_01500+0000_rgb','spitzer_01800+0000_rgb','spitzer_02100+0000_rgb','spitzer_02400+0000_rgb',
        'spitzer_02700+0000_rgb','spitzer_03000+0000_rgb','spitzer_03300+0000_rgb','spitzer_03600+0000_rgb',
        'spitzer_03900+0000_rgb','spitzer_04200+0000_rgb','spitzer_04500+0000_rgb','spitzer_04800+0000_rgb',
        'spitzer_05100+0000_rgb','spitzer_05400+0000_rgb','spitzer_05700+0000_rgb','spitzer_06000+0000_rgb',
        'spitzer_29700+0000_rgb','spitzer_30000+0000_rgb','spitzer_30300+0000_rgb','spitzer_30600+0000_rgb',
        'spitzer_30900+0000_rgb','spitzer_31200+0000_rgb','spitzer_31500+0000_rgb','spitzer_31800+0000_rgb',
        'spitzer_32100+0000_rgb','spitzer_32400+0000_rgb','spitzer_32700+0000_rgb','spitzer_33000+0000_rgb',
        'spitzer_33300+0000_rgb','spitzer_33600+0000_rgb','spitzer_33900+0000_rgb','spitzer_34200+0000_rgb',
        'spitzer_34500+0000_rgb','spitzer_34800+0000_rgb','spitzer_35100+0000_rgb','spitzer_35400+0000_rgb',
        'spitzer_35700+0000_rgb']
        # fmt: on
        val_l = sorted(val_l)

        if os.path.exists("/workspace/val_png/region_val_png"):
            pass
        else:
            os.mkdir("/workspace/val_png/region_val_png")
    else:
        ## デフォルトの領域を用いて、Val-Ringを作成する
        val_l = [
            "spitzer_00900+0000_rgb",
            "spitzer_03900+0000_rgb",
            "spitzer_31200+0000_rgb",
            "spitzer_34200+0000_rgb",
            "spitzer_33900+0000_rgb",
        ]
        val_l = sorted(val_l)

    # choice catalogue from 'CH' or 'MWP'
    choice = "CH"
    Ring_CATA = val_ring_sub.catalogue(choice)

    random_uni = default_rng(123)
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))

    frame_mwp_val = []
    mwp_ring_list_val = []
    val_count = 0

    #####################
    ## Val-Ring作成開始 ##
    #####################

    pbar = tqdm(range(len(val_l)))
    for i in pbar:
        if args.each_region:
            region_frame_mwp_val = []
            region_mwp_ring_list_val = []

        fits_path = val_l[i]
        pbar.set_description(fits_path)
        spitzer_rfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "r.fits")[0]
        spitzer_gfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "g.fits")[0]
        spitzer_bfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "b.fits")[0]

        # RGBにしたいため、fitsのdataを重ねる
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

        Ring_cata = Ring_CATA.query("@GLON_new_min < GLON <= @GLON_new_max")
        Ring_cata = Ring_cata.reset_index()
        val_count += len(Ring_cata)

        # star_listは辞書
        label_cal = val_label_calculator.label_caliculator(choice, "val", w)
        star_dic = label_cal.all_star(Ring_cata)

        if choice == "CH":
            Rout = "Rout"
        else:
            Rout = "Reff"

        for _, row in Ring_cata.iterrows():
            ccc = 0
            ok = True

            while ok:
                random_num = 1 / random_uni.uniform(0.125, 0.8)
                lmax = row["GLON"] + random_num * row[Rout] / 60
                bmin = row["GLAT"] - random_num * row[Rout] / 60
                # 右端
                lmin = row["GLON"] - random_num * row[Rout] / 60
                bmax = row["GLAT"] + random_num * row[Rout] / 60
                ccc += 1
                if GLON_min <= lmin and lmax <= GLON_max and GLAT_min <= bmin and bmax <= GLAT_max:
                    ok = False
                    flag = True
                if ccc >= 100:
                    ok = False
                    flag = False
            if flag:
                x_min, y_min = w.all_world2pix(lmax, bmin, 0)
                x_max, y_max = w.all_world2pix(lmin, bmax, 0)
                r = int((x_max - x_min) / (2 * random_num))  # ringの半径pixel

                width = x_max - x_min
                height = y_max - y_min

                x_pix_min = x_min - width / 2
                y_pix_min = y_min - height / 2
                x_pix_max = x_max + width / 2
                y_pix_max = y_max + height / 2

                x_offset = random_uni.uniform(-(random_num - 0.5) * r, (random_num - 0.5) * r)
                y_offset = random_uni.uniform(-(random_num - 0.5) * r, (random_num - 0.5) * r)
                x_pix_min = x_pix_min + int(x_offset)
                x_pix_max = x_pix_max + int(x_offset)
                y_pix_min = y_pix_min + int(y_offset)
                y_pix_max = y_pix_max + int(y_offset)
                width = x_pix_max - x_pix_min
                height = y_pix_max - y_pix_min

                # calc_pix時に100回試行してもできなかった場合の場合分け
                if x_pix_min < 0 or y_pix_min < 0:
                    pass
                else:
                    # print(x_pix_min, y_pix_min, x_pix_max, y_pix_max)
                    cover_star_position, cover_star_name = label_cal.find_cover(
                        star_dic, x_pix_min, y_pix_min, x_pix_max, y_pix_max
                    )
                    c_data = data[int(y_pix_min) : int(y_pix_max), int(x_pix_min) : int(x_pix_max)].view()
                    cut_data = copy.deepcopy(c_data)
                    if np.isnan(cut_data.sum()):
                        pass

                    else:
                        sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
                        pi = processing.conv(300, sig1, cut_data)
                        label_cal.make_label(
                            x_pix_min,
                            y_pix_min,
                            x_pix_max,
                            y_pix_max,
                            cover_star_position,
                            cover_star_name,
                            width,
                            height,
                            Ring_CATA,
                        )
                        r_shape_y = pi.shape[0]
                        r_shape_x = pi.shape[1]
                        res_data = pi[
                            int(r_shape_y / 4) : int(r_shape_y * 3 / 4), int(r_shape_x / 4) : int(r_shape_x * 3 / 4)
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
                        ## 完成したデータを格納 ###
                        #######################
                        if not np.isnan(res_data.sum()):
                            if args.each_region:
                                region_mwp_ring_list_val.append(res_data)
                                region_frame_mwp_val.append(info)
                            else:
                                mwp_ring_list_val.append(res_data)
                                frame_mwp_val.append(info)

        ####################################
        ## 領域ごとにRingデータを保存していく ###
        ####################################
        if args.each_region:
            region_frame_mwp_val = pd.DataFrame(region_frame_mwp_val)
            region_frame_mwp_val["id"] = [_ for _ in range(len(region_frame_mwp_val))]
            savedir_name = "/workspace/val_png/region_val_png/%s" % fits_path
            if os.path.exists(savedir_name):
                pass
            else:
                os.mkdir(savedir_name)

            ## dataの作成
            region_mwp_ring_list_val = np.array(region_mwp_ring_list_val).astype(np.float32)
            for ind in range(region_mwp_ring_list_val.shape[0]):
                pil_image = Image.fromarray(np.uint8(region_mwp_ring_list_val[ind] * 255))
                pil_image.save("%s/Ring_%s.png" % (savedir_name, ind))

            ## labelの作成
            for ind, row in region_frame_mwp_val.iterrows():
                ll = []
                if len(row["xmin"]) >= 1:
                    for la in range(len(row["xmin"])):
                        ll.append(
                            {
                                "Confidence": str(0),
                                "XMin": str(row["xmin"][la]),
                                "XMax": str(row["xmax"][la]),
                                "YMin": str(row["ymin"][la]),
                                "YMax": str(row["ymax"][la]),
                            }
                        )
                else:
                    ll.append({"Confidence": str(0)})

                with open("%s/Ring_%s.json" % (savedir_name, ind), "w") as f:
                    json.dump(ll, f, indent=4)

    #########################################
    ## デフォルト領域のRingデータを保存していく ###
    #########################################
    if not args.each_region:
        frame_mwp_val = pd.DataFrame(frame_mwp_val)
        frame_mwp_val["id"] = [i for i in range(len(frame_mwp_val))]
        savedir_name = "/workspace/val_png/default_val/"
        if os.path.exists(savedir_name):
            pass
        else:
            os.mkdir(savedir_name)

        mwp_ring_list_val = np.array(mwp_ring_list_val).astype(np.float32)
        for i in range(mwp_ring_list_val.shape[0]):
            pil_image = Image.fromarray(np.uint8(mwp_ring_list_val[i] * 255))
            pil_image.save("%s/Ring_%s.png" % (savedir_name, i))

        for i, row in frame_mwp_val.iterrows():
            ll = []
            if len(row["xmin"]) >= 1:
                for la in range(len(row["xmin"])):
                    ll.append(
                        {
                            "Confidence": str(0),
                            "XMin": str(row["xmin"][la]),
                            "XMax": str(row["xmax"][la]),
                            "YMin": str(row["ymin"][la]),
                            "YMax": str(row["ymax"][la]),
                        }
                    )
            else:
                ll.append({"Confidence": str(0)})

            with open("%s/Ring_%s.json" % (savedir_name, i), "w") as f:
                json.dump(ll, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
