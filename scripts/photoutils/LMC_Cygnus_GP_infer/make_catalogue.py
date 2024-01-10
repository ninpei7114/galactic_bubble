import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append("../")

from processing import data_view_rectangl
from training_sub import calc_TP_FP_FN
from make_catalogue_sub import (
    calc_bbox,
    make_cut_ring,
    make_data,
    make_infer_catalogue,
    make_map,
    make_MWP_catalogue,
    make_TP_FN,
    make_FP,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of SSD")
    parser.add_argument("result_path", type=str, help="model's path to infer")
    parser.add_argument(
        "save_dir", help="Infer Result Save Directory", default="/home/cygnus/jupyter/research/each_model_region_infer"
    )
    parser.add_argument(
        "LMC_data_path", help="LMC data path", default="/home/cygnus/jupyter/fits_data/LMC_data/spitzer_lmc_rgb"
    )
    parser.add_argument("Cygnus_data_path", help="Cyg data path", default="/home/cygnus/jupyter/fits_data/cygnus_fits")
    parser.add_argument("--val_ring_catalogue", type=str, default="MWP")

    return parser.parse_args()


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    conf_thre = 0.95

    for region in ["LMC", "Cygnus"]:
        print(region)
        os.makedirs(f"{args.save_dir}/{region}", exist_ok=True)
        if region == "LMC":
            r_fits_path = args.LMC_data_path + "/r.fits"
            g_fits_path = args.LMC_data_path + "/g.fits"
            save_png_name = "/home/cygnus/jupyter/fits_data/LMC_data/lmc_RG_-1.0_10.0_-1.0_8.0_0.0_1.0.png"
            fits_path = [[r_fits_path, g_fits_path, region, save_png_name]]
        elif region == "Cygnus":
            r_fits_path = args.Cygnus_data_path + "/M1_fits_file/M1_cygnus_2.4.fits"
            g_fits_path = args.Cygnus_data_path + "/I4_fits_file/I4_2.4_reg.fits"
            save_png_name = "/home/cygnus/jupyter/fits_data/cygnus_fits/RG_30.0_100.0_10.0_150.0_30.0_100.0.png"
            fits_path = [[r_fits_path, g_fits_path, region, save_png_name]]

        for fp in fits_path:
            ##################
            ## 推論結果の解析 ##
            ##################
            save_png_name = fp[3]
            data_, hdu_r, a, b, w, region_ = make_data(fp)
            bbox = calc_bbox(args, region, conf_thre)
            catalogue = make_infer_catalogue(bbox, w)
            np.save(f"{args.save_dir}/{region}/bbox.npy", bbox)
            catalogue.to_csv(f"{args.save_dir}/{region}/cygnus_infer_catalogue.csv")

            if region == "Cygnus":
                GLON_min, GLAT_min = w.all_pix2world(b, 0, 0)
                GLON_max, GLAT_max = w.all_pix2world(0, a, 0)
                MWP = make_MWP_catalogue(region)
                MWP_catalogue = MWP.query("@GLON_min <= _RA_icrs <= @GLON_max")
                MWP_catalogue.to_csv(f"{args.save_dir}/{region}/MWP_catalogue.csv")
                make_map(save_png_name, region, catalogue, hdu_r, args, g_fits_path, MWP_catalogue, region_)
            else:
                make_map(save_png_name, region, catalogue, hdu_r, args, g_fits_path)
            make_cut_ring(bbox, data_, args, region, region_)

            ######################
            ## TP, FP, FN の解析 ##
            ######################

            if region == "Cygnus" or region == "Spitzer":
                if region == "Cygnus":
                    world = "RA"
                elif region == "Spitzer":
                    world = "Galactic"
                TP_c, FP_c, target_mask = calc_TP_FP_FN(
                    MWP_catalogue.reset_index(), catalogue, "Reff", args.val_ring_catalogue, world=world
                )
                TP = target_mask.count(True)
                FN = target_mask.count(False)
                FP = len(FP_c)
                Precision_ = TP / (TP + FP)
                Recall_ = TP / (TP + FN)
                F1_score_ = 2 * Precision_ * Recall_ / (Precision_ + Recall_ + 1e-9)
                print(region)
                print(f"TP: {TP}/{len(target_mask)}, FN: {FN}/{len(target_mask)}, FP: {FP}")
                print(f"Precision: {Precision_}, Recall: {Recall_}, F1_score: {F1_score_}")

                target_TP = make_TP_FN(MWP_catalogue, target_mask, data_, w, hdu_r, region)
                target_FN = make_TP_FN(MWP_catalogue, ~np.array(target_mask), data_, w, hdu_r, region)
                target_FP = make_FP(pd.DataFrame(FP_c), data_, w)
                if region == "Cygnus":
                    save_png = f"{args.save_dir}/{region}"
                    data_view_rectangl(10, np.uint8(np.array(target_TP) * 255)).save(f"{save_png}/TP.jpg")
                    data_view_rectangl(10, np.uint8(np.array(target_FN) * 255)).save(f"{save_png}/FN.jpg")
                    data_view_rectangl(10, np.uint8(np.array(target_FP) * 255)).save(f"{save_png}/FP.jpg")


if __name__ == "__main__":
    args = parse_args()
    main(args)