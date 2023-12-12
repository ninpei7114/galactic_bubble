import argparse
import os
import pathlib
import tqdm

import astropy
import astropy.io.fits
import astropy.wcs
import astroquery.vizier
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage


def parse_args():
    parser = argparse.ArgumentParser(description="make data for deepcluster")

    parser.add_argument("fits_path", metavar="DIR", help="path to dataset")
    parser.add_argument("save_dir", metavar="DIR", help="path to save directory")
    # parser.add_argument('ring_sentei_path', metavar='DIR', help='path to ring setntei file')

    return parser.parse_args()


def circle_nan(fits_data, same_shape_zero, pandas_catalog1, w):  # , pandas_catalog2):
    for _, series_i in pandas_catalog1.iterrows():
        l = series_i["GLON"]
        b = series_i["GLAT"]
        lpix, bpix = w.all_world2pix(l, b, 0)
        # 左端
        lmax = l + series_i["Reff"] / 60
        bmin = b - series_i["Reff"] / 60
        # 右端
        lmin = l - series_i["Reff"] / 60
        bmax = b + series_i["Reff"] / 60
        x_pix_min, y_pix_min = w.all_world2pix(lmax, bmin, 0)
        x_pix_max, y_pix_max = w.all_world2pix(lmin, bmax, 0)

        rout = (x_pix_max - x_pix_min) / 2
        same_shape_zero = cv2.circle(
            same_shape_zero, (int(np.round(lpix)), int(np.round(bpix))), int(rout), (255, 255, 255), -1
        )

    fits_data[same_shape_zero == same_shape_zero.max()] = np.nan

    return fits_data


def main(args):
    """リングの箇所をnanにする

    Args:
        args (argparse): argparse

    >>> example command
    python make_circle_nan_fits.py /dataset/spitzer_data/ /workspace/fits_data/ring_to_circle_nan_fits/
    """
    # fmt: off
    l = [
        'spitzer_02100+0000_rgb','spitzer_04200+0000_rgb','spitzer_33300+0000_rgb','spitzer_35400+0000_rgb',
        'spitzer_00300+0000_rgb','spitzer_02400+0000_rgb','spitzer_04500+0000_rgb','spitzer_31500+0000_rgb',
        'spitzer_33600+0000_rgb','spitzer_35700+0000_rgb','spitzer_00600+0000_rgb','spitzer_02700+0000_rgb',
        'spitzer_04800+0000_rgb','spitzer_29700+0000_rgb','spitzer_31800+0000_rgb','spitzer_03000+0000_rgb',
        'spitzer_05100+0000_rgb','spitzer_30000+0000_rgb','spitzer_32100+0000_rgb','spitzer_01200+0000_rgb',
        'spitzer_03300+0000_rgb','spitzer_05400+0000_rgb','spitzer_30300+0000_rgb','spitzer_32400+0000_rgb',
        'spitzer_34500+0000_rgb','spitzer_01500+0000_rgb','spitzer_03600+0000_rgb','spitzer_05700+0000_rgb',
        'spitzer_30600+0000_rgb','spitzer_32700+0000_rgb','spitzer_34800+0000_rgb','spitzer_01800+0000_rgb',
        'spitzer_06000+0000_rgb','spitzer_30900+0000_rgb','spitzer_33000+0000_rgb','spitzer_35100+0000_rgb',
        'spitzer_00900+0000_rgb','spitzer_03900+0000_rgb','spitzer_31200+0000_rgb','spitzer_34200+0000_rgb',
        'spitzer_33900+0000_rgb','spitzer_29400+0000_rgb','spitzer_06300+0000_rgb','spitzer_00000+0000_rgb']
    # fmt: on
    l = sorted(l)
    pbar = tqdm.tqdm(range(len(l)))

    viz = astroquery.vizier.Vizier(columns=["*"])
    viz.ROW_LIMIT = -1
    MWP = viz.query_constraints(catalog="2019yCat..74881141J ")[0].to_pandas()
    MWP.loc[MWP["GLON"] >= 358.446500015535, "GLON"] -= 360
    MWP.index = MWP["MWP"].tolist()
    rank_3 = np.load("../MWP_rank3_name.npy")
    MWP = MWP.loc[rank_3]

    for i in pbar:
        pbar.set_description(l[i])
        save_name = pathlib.Path(args.save_dir) / l[i]
        os.makedirs(save_name, exist_ok=True)

        for rgb in ["r.fits", "g.fits", "b.fits"]:
            hdu = astropy.io.fits.open(args.fits_path + f"{l[i]}/{rgb}")[0]
            header = hdu.header
            fits = hdu.data

            w = astropy.wcs.WCS(header)
            a = fits.shape[0]
            b = fits.shape[1]
            GLON_min, GLAT_min = w.all_pix2world(b, 0, 0)
            GLON_max, GLAT_max = w.all_pix2world(0, a, 0)

            if GLON_min >= 358.446500015535:
                GLON_min -= 360
            else:
                pass

            cut_catalogue = MWP.query("@GLON_min < GLON < @GLON_max")
            if len(cut_catalogue) == 0:
                data = fits
            else:
                c = np.zeros_like(fits)
                data = circle_nan(fits, c, cut_catalogue, w)

            new_hdu = astropy.io.fits.PrimaryHDU(data, header)
            new_hdu_list = astropy.io.fits.HDUList([new_hdu])
            new_hdu_list.writeto(save_name / rgb, overwrite=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
