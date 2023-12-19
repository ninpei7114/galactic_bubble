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

    parser.add_argument("fits_path", metavar="DIR", help="path to dataset")
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

    for i in pbar:
        pbar.set_description(l[i])
        data_fits_G = f"{args.fits_path}/{l[i]}/g.fits"
        os.makedirs(f"{args.save_dir}/remove_star_fits/{l[i]}")

        hdu = astropy.io.fits.open(data_fits_G)[0]
        data = hdu.data.copy()
        mean, median, std = sigma_clipped_stats(data, sigma=3)
        daofind = DAOStarFinder(fwhm=1.98, threshold=20 * mean)
        sources = daofind(data)

        for col in sources.colnames:
            if col not in ("id", "npix"):
                sources[col].info.format = "%.2f"
        positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))

        for fits_type in ["r.fits", "g.fits", "b.fits"]:
            hdu_type = astropy.io.fits.open(f"{args.fits_path}/{l[i]}/{fits_type}")[0]
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
            new_hdu_list.writeto(f"{args.save_dir}/remove_star_fits/{l[i]}/{fits_type}", overwrite=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
