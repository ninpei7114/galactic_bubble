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
from tqdm import tqdm

sys.path.append("../")
import processing
import label_caliculator
import ring_augmentation

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


def append_data(data, info, data_list, frame):
    if not np.isnan(data.sum()):
        data_list.append(data)
        frame.append(info)


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

    ## choice catalogue from 'CH' or 'MWP'
    choice = "CH"
    Ring_CATALOGUE = ring_augmentation.catalogue(choice)
    trans_rng = default_rng(123)
    val_count = 0

    #####################
    ## Val-Ring作成開始 ##
    #####################

    pbar = tqdm(range(len(val_l)))
    for i in pbar:
        frame_mwp_val = []
        mwp_ring_list_val = []

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

        Ring_catalogue = Ring_CATALOGUE.query("@GLON_new_min < GLON <= @GLON_new_max")
        val_count += len(Ring_catalogue)

        # star_listは辞書
        label_cal = label_caliculator.label_caliculator(choice, w)
        label_cal.all_star(Ring_catalogue)

        for _, row in Ring_catalogue.iterrows():
            trans_params = {
                "row": row,
                "fits_path": fits_path,
                "GLON_new_min": GLON_new_min,
                "GLON_new_max": GLON_new_max,
                "GLAT_min": GLAT_min,
                "GLAT_max": GLAT_max,
                "Ring_catalogue": Ring_catalogue,
                "data": data,
                "label_cal": label_cal,
                "trans_rg": trans_rng,
            }
            fl, trans_data, trans_info = ring_augmentation.translation(**trans_params)
            ## データやlabelの作成に不備があれば、fl=False(例えば、xmin<0や、xmin=xmaxなど)
            ## 問題がなければ、fl=True
            if fl:
                trans_data_ = trans_data.copy()
                append_data(
                    processing.norm_res(trans_data_),
                    trans_info,
                    mwp_ring_list_val,
                    frame_mwp_val,
                )

        ####################################
        ## 領域ごとにRingデータを保存していく ###
        ####################################
        if args.each_region:
            savedir_name = "/workspace/val_png/region_val_png/%s" % fits_path
        else:
            savedir_name = "/workspace/val_png/default_val/"
        os.makedirs(savedir_name, exist_ok=True)

        mwp_ring_list_val = np.array(mwp_ring_list_val).astype(np.float32)
        for i in range(mwp_ring_list_val.shape[0]):
            pil_image = Image.fromarray(np.uint8(mwp_ring_list_val[i] * 255))
            pil_image.save("%s/Ring_%s.png" % (savedir_name, i))

        frame_mwp_val = pd.DataFrame(frame_mwp_val)
        frame_mwp_val["id"] = [i for i in range(len(frame_mwp_val))]
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
                # ll.append({"Confidence": str(0)})
                pass

            with open("%s/Ring_%s.json" % (savedir_name, i), "w") as f:
                json.dump(ll, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
