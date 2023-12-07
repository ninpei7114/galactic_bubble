import argparse
import sys
import time
import os

import astropy.io.fits
import numpy as np
import torch
import wandb

sys.path.append("/home/cygnus/jupyter/galactic_bubble/scripts/wandb_training")
from processing import remove_nan
from utils.ssd_model import SSD, Detect

from infer_sub import calc_ind, infer


"""Example command line:

python LMC_Cygnus_GP_infer.py galactic_bubble/clustering_NewNorm/training_log:v0
"""


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of SSD")
    parser.add_argument("model_ver", type=str, help="model's path to infer")
    parser.add_argument(
        "--model_download_dir",
        metavar="DIR",
        help="model Download Directory",
        default="/home/cygnus/jupyter/research/each_model_region_infer",
    )
    parser.add_argument(
        "--result_save_dir",
        metavar="DIR",
        help="Infer Resuly Save Directory",
        default="/home/cygnus/jupyter/research/each_model_region_infer",
    )
    parser.add_argument(
        "--spitzer_path", metavar="DIR", help="spitzer_path", default="/home/cygnus/jupyter/fits_data/spitzer_data"
    )
    parser.add_argument(
        "--LMC_data_path",
        metavar="DIR",
        help="LMC data path",
        default="/home/cygnus/jupyter/fits_data/LMC_data/spitzer_lmc_rgb",
    )
    parser.add_argument(
        "--Cygnus_data_path", metavar="DIR", help="Cyg data path", default="/home/cygnus/jupyter/fits_data/cygnus_fits"
    )

    return parser.parse_args()


def main(args):
    start = time.time()
    torch.backends.cudnn.benchmark = True
    device = torch.device(torch.device("cuda:0") if torch.cuda.is_available() else "cpu")

    api = wandb.Api()
    artifact = api.artifact(f"{args.model_ver}")
    artifact.download(f"{args.model_download_dir}" + "/artifacts/" + "/".join(args.model_ver.split("/")[-2:]))
    net_w = SSD()
    net_weights = torch.load(
        args.model_download_dir + "/artifacts/" + "/".join(args.model_ver.split("/")[-2:]) + "/earlystopping.pth"
    )
    net_w.load_state_dict(net_weights["model_state_dict"])
    del net_weights
    net_w.to(device)

    size_list = [150, 300, 600, 1200, 1800, 2400, 3000]
    batch_list = [5000, 2000, 1000, 600, 300, 50, 30]

    detect = Detect(nms_thresh=0.3, top_k=2000)
    model_ver = "/".join(args.model_ver.split("/")[-2:])
    os.makedirs(f"{args.result_save_dir}/{model_ver}", exist_ok=True)
    f_log = open(f"{args.result_save_dir}/{model_ver}/" + "/log.txt", "w")
    f_log.write("使用モデル: " + args.model_ver + "\n")
    f_log.close()

    for region in ["LMC", "Cygnus", "Spitzer"]:
        print(f"{region=}")
        if region == "LMC":
            r_fits_path = args.LMC_data_path + "/r.fits"
            g_fits_path = args.LMC_data_path + "/g.fits"
        elif region == "Cygnus":
            r_fits_path = args.Cygnus_data_path + "/M1_fits_file/M1_cygnus_2.4.fits"
            g_fits_path = args.Cygnus_data_path + "/I4_fits_file/I4_2.4_reg.fits"

        if region == "LMC" or region == "Cygnus":
            data_ = np.concatenate(
                [
                    remove_nan(astropy.io.fits.open(r_fits_path)[0].data[:, :, None]),
                    remove_nan(astropy.io.fits.open(g_fits_path)[0].data[:, :, None]),
                ],
                axis=2,
            )
            for size, batch_size in zip(size_list, batch_list):
                ### indexの計算 ###
                cut_shape = (size, size)
                if size <= 300:
                    fragment = 3
                else:
                    fragment = 3
                ind = calc_ind(cut_shape, fragment, data_)
                ### infer ###
                infer(ind, batch_size, cut_shape, data_, net_w, detect, args, region, device, model_ver)

        else:
            spitzer_regions = [
                "spitzer_01800+0000_rgb",
                "spitzer_03300+0000_rgb",
                "spitzer_03900+0000_rgb",
                "spitzer_04800+0000_rgb",
                "spitzer_05400+0000_rgb",
            ]
            for sp_r in spitzer_regions:
                print(f"{sp_r=}")
                r_fits_path = args.spitzer_path + "/" + sp_r + "/" + "/r.fits"
                g_fits_path = args.spitzer_path + "/" + sp_r + "/" + "/g.fits"
                data_ = np.concatenate(
                    [
                        remove_nan(astropy.io.fits.open(r_fits_path)[0].data[:, :, None]),
                        remove_nan(astropy.io.fits.open(g_fits_path)[0].data[:, :, None]),
                    ],
                    axis=2,
                )

                for size, batch_size in zip(size_list, batch_list):
                    ### indexの計算 ###
                    cut_shape = (size, size)
                    if size <= 300:
                        fragment = 8
                    else:
                        fragment = 8
                    ind = calc_ind(cut_shape, fragment, data_)
                    ### infer ###
                    infer(ind, batch_size, cut_shape, data_, net_w, detect, args, sp_r, device, model_ver)
    print(f"elapsed_time:{time.time() - start}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
