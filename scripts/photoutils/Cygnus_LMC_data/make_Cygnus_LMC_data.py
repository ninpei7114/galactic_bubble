import argparse
import os
import copy

import astropy.io.fits
import astropy.wcs
import numpy as np
from tqdm import tqdm
from PIL import Image

from processing import remove_nan, norm_res, conv


"""
example command:

python make_Cygnus_LMC_data.py /home/cygnus/jupyter/fits_data/cygnus_fits/ /home/cygnus/jupyter/fits_data/LMC_data/spitzer_lmc_rgb/ /home/cygnus/jupyter/fits_data/
"""


def parse_args():
    parser = argparse.ArgumentParser(description="make validataion data for SSD")
    parser.add_argument("Cygnus_path", type=str, metavar="DIR", help="Cygnus_path")
    parser.add_argument("LMC_path", type=str, metavar="DIR", help="LMC_path")
    parser.add_argument("save_dir", type=str, metavar="DIR", help="save_dir")

    return parser.parse_args()


def cut_data(data_, many_ind, cut_shape, r_resolution, g_resolution, savedir_name, region):
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
    for i in many_ind:
        xmin = int(i[1])
        ymin = int(i[0])
        extra_xmin = xmin - cut_shape / 50
        extra_xmax = xmin + cut_shape + cut_shape / 50
        extra_ymin = ymin - cut_shape / 50
        extra_ymax = ymin + cut_shape + cut_shape / 50
        data_c = data_[int(extra_ymin) : int(extra_ymax), int(extra_xmin) : int(extra_xmax)].view()
        if not np.isnan(data_c.sum()):
            d = copy.deepcopy(data_c)
            d = conv(300, sig1, d)
            d = d[int(cut_shape / 52) : int(cut_shape * 51 / 52), int(cut_shape / 52) : int(cut_shape * 51 / 52)]

            flag = True
            for dim in range(d.shape[2] - 1):
                non_zero_count = np.count_nonzero(d[:, :, dim])
                if non_zero_count >= d.shape[0] * d.shape[1] * 3 / 4:
                    pass
                else:
                    flag = False
            if flag:
                cut_data = norm_res(d, r_resolution * 3600, g_resolution * 3600)
                pil_image = Image.fromarray(np.uint8(cut_data * 255))
                pil_image.save(f"{savedir_name}/{ymin}_{xmin}_{cut_shape}_{region}.png")
        else:
            pass


def main(args):
    """Validationデータを作成する

    Args:
        args (argparser): argparser

    """
    Cygnus_LMC_l = ["Cygnus", "LMC"]
    size_list = [150, 300, 600, 900, 1200, 1800, 2400, 3000]
    fragment = 3

    ##############################
    ## Cygnus LMC Data 作成開始 ##
    ##############################
    for region in Cygnus_LMC_l:
        print(f"{region=}")
        print("MAKING DATA ...")
        os.makedirs(f"{args.save_dir}/Cygnus_LMC_png/", exist_ok=True)
        os.makedirs(f"{args.save_dir}/Cygnus_LMC_png/{region}", exist_ok=True)

        if region == "Cygnus":
            hdu_r = astropy.io.fits.open(args.Cygnus_path + "/" + "M1_fits_file/M1_cygnus_2.4.fits")[0]
            hdu_g = astropy.io.fits.open(args.Cygnus_path + "/" + "I4_fits_file/I4_2.4_reg.fits")[0]
            hdu_b = astropy.io.fits.open(args.Cygnus_path + "/" + "I1_fits_file/I1_2.4_reg.fits")[0].data
            savedir_name = f"{args.save_dir}/Cygnus_LMC_png/{region}"
        elif region == "LMC":
            hdu_r = astropy.io.fits.open(args.LMC_path + "/" + "r.fits")[0]
            hdu_g = astropy.io.fits.open(args.LMC_path + "/" + "g.fits")[0]
            hdu_b = np.zeros(hdu_g.data.shape)
            savedir_name = f"{args.save_dir}/Cygnus_LMC_png/{region}"

        data = np.concatenate(
            [remove_nan(hdu_r.data[:, :, None]), remove_nan(hdu_g.data[:, :, None]), remove_nan(hdu_b[:, :, None])],
            axis=2,
        )

        pbar = tqdm(range(len(size_list)))
        for i in pbar:
            size = size_list[i]
            pbar.set_description(str(size))
            ################
            ## indexの計算 ##
            ################
            cut_shape = (size, size)
            slide_pix = (int(round(cut_shape[0] / fragment)), int(round(cut_shape[1] / fragment)))
            shape = data.shape
            x_num = int(shape[1] / slide_pix[1])
            y_num = int(shape[0] / slide_pix[0])
            x_idx = np.arange(cut_shape[1] / 5, slide_pix[1] * x_num, slide_pix[1])
            y_idx = np.arange(cut_shape[0] / 5, slide_pix[0] * y_num, slide_pix[0])
            x_ind, y_ind = np.meshgrid(x_idx, y_idx)

            ind_array = []
            for x, y in zip(x_ind.ravel(), y_ind.ravel()):
                ind_array.append([y, x])
            ind_array = np.array(ind_array)
            if region == "Cygnus":
                cut_data(
                    data, ind_array, cut_shape[0], hdu_r.header["CDELT2"], hdu_g.header["CDELT2"], savedir_name, region
                )
            elif region == "LMC":
                cut_data(
                    data, ind_array, cut_shape[0], hdu_r.header["CD2_2"], hdu_g.header["CD2_2"], savedir_name, region
                )


if __name__ == "__main__":
    args = parse_args()
    main(args)
