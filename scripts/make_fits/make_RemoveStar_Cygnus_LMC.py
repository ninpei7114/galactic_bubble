import argparse
import os
import tqdm

import astropy
import astropy.io.fits
import astropy.wcs
import cv2
import numpy as np
from scipy import interpolate

from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


def parse_args():
    parser = argparse.ArgumentParser(description="make data for deepcluster")

    parser.add_argument("cygnus_path", metavar="DIR", help="path to dataset")
    parser.add_argument("lmc_path", metavar="DIR", help="path to dataset")
    parser.add_argument("save_dir", metavar="DIR", help="path to save directory")
    # parser.add_argument('ring_sentei_path', metavar='DIR', help='path to ring setntei file')

    return parser.parse_args()


def main(args):
    """リングの箇所をnanにする

    Args:
        args (argparse): argparse

    >>> example command
    python make_circle_nan_fits.py /dataset/spitzer_data/ /workspace/fits_data/ring_to_circle_nan_fits/
    """
    l = ["Cygnus", "LMC"]
    l = sorted(l)
    pbar = tqdm.tqdm(range(len(l)))

    for i in pbar:
        pbar.set_description(l[i])
        if l[i] == "Cygnus":
            data_fits_G = f"{args.cygnus_path}/I4_fits_file/I4_2.4_reg.fits"
            os.makedirs(f"{args.save_dir}/RemoveStar_Cygnus_fits/{l[i]}", exist_ok=True)
        elif l[i] == "LMC":
            data_fits_G = f"{args.lmc_path}/g.fits"
            os.makedirs(f"{args.save_dir}/RemoveStar_LMC_fits/{l[i]}", exist_ok=True)

        hdu = astropy.io.fits.open(data_fits_G)[0]
        data = hdu.data.copy()
        mean, median, std = sigma_clipped_stats(data, sigma=3)
        daofind = DAOStarFinder(fwhm=1.98, threshold=20 * mean)
        sources = daofind(data)
        for col in sources.colnames:
            if col not in ("id", "npix"):
                sources[col].info.format = "%.2f"
        positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))

        if l[i] == "Cygnus":
            fits_type_list = ["I4_fits_file/I4_2.4_reg.fits", "/M1_fits_file/M1_cygnus_2.4.fits"]
        elif l[i] == "LMC":
            fits_type_list = ["g.fits", "r.fits"]

        for fits_type in fits_type_list:
            if l[i] == "Cygnus":
                hdu_type = astropy.io.fits.open(f"{args.cygnus_path}/{fits_type}")[0]
            elif l[i] == "LMC":
                hdu_type = astropy.io.fits.open(f"{args.lmc_path}/{fits_type}")[0]
            data = hdu_type.data.copy()
            data[data != data] = 0
            same_shape_zero = np.zeros_like(data)
            for y, x in positions:
                same_shape_zero = cv2.circle(same_shape_zero, (int(y), int(x)), int(5), (255, 255, 255), -1)
            data[same_shape_zero == same_shape_zero.max()] = np.nan

            x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij")
            # 欠損箇所のマスクを作成
            ma_data = np.ma.masked_invalid(data)
            # 欠損していないデータのみを取得
            val_x = x[~ma_data.mask]
            val_y = y[~ma_data.mask]
            val_data = data[~ma_data.mask]
            inp_near = interpolate.NearestNDInterpolator((val_x, val_y), val_data.ravel())
            inp_near_data = inp_near((x, y))
            inp_near_data[inp_near_data == 0] = np.nan

            new_hdu = astropy.io.fits.PrimaryHDU(inp_near_data, hdu_type.header)
            new_hdu_list = astropy.io.fits.HDUList([new_hdu])
            if l[i] == "Cygnus":
                new_hdu_list.writeto(
                    f"{args.save_dir}/RemoveStar_Cygnus_fits/{fits_type.split('/')[-1]}", overwrite=True
                )
            elif l[i] == "LMC":
                new_hdu_list.writeto(f"{args.save_dir}/RemoveStar_LMC_fits/{fits_type}", overwrite=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
