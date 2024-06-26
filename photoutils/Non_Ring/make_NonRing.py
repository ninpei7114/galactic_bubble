import argparse
import json
import os
import pathlib
import time
import sys

import astropy.io.fits
import NonRing_sub
import numpy as np
from numpy.random import default_rng
from PIL import Image
from tqdm import tqdm

sys.path.append("../")
import processing

"""
Create Non-Ring png images for use with webdataset
Also, create labels in json format

example command:
python make_NonRing.py /home/cygnus/jupyter/リング研究/fits_data/ring_to_circle_nan_fits_CHrank3/ /home/cygnus/jupyter/workspase/
"""


def parse_args():
    parser = argparse.ArgumentParser(description="make data for SSD")
    parser.add_argument("fits_path", metavar="DIR", help="path to Ring_to_circle_nan_fits")
    parser.add_argument("savedir_name", metavar="DIR", help="savedir_name")
    # parser.add_argument("--each_region", "-r", action="store_true")
    # parser.add_argument('ring_sentei_path', metavar='DIR', help='path to ring setntei file')

    return parser.parse_args()


def main(args):
    """Create Non-Ring png images

    Args:
        args (argparse): argparse

    example command:
    >>> python make_NonRing.py /workspace/fits_data/ring_to_circle_nan_fits /workspace/
    """

    ## Create Non-Ring for each region
    ## 'spitzer_29400+0000_rgb' is not used because there is hardly any 8µm data
    # fmt: off
    all_l = [
        "spitzer_00300+0000_rgb", "spitzer_00600+0000_rgb", "spitzer_00900+0000_rgb", "spitzer_01200+0000_rgb",
        "spitzer_01500+0000_rgb", "spitzer_01800+0000_rgb", "spitzer_02100+0000_rgb", "spitzer_02400+0000_rgb",
        "spitzer_02700+0000_rgb", "spitzer_03000+0000_rgb", "spitzer_03300+0000_rgb", "spitzer_03600+0000_rgb",
        "spitzer_03900+0000_rgb", "spitzer_04200+0000_rgb", "spitzer_04500+0000_rgb", "spitzer_04800+0000_rgb",
        "spitzer_05100+0000_rgb", "spitzer_05400+0000_rgb", "spitzer_05700+0000_rgb", "spitzer_06000+0000_rgb",
        "spitzer_29700+0000_rgb", "spitzer_30000+0000_rgb", "spitzer_30300+0000_rgb", "spitzer_30600+0000_rgb",
        "spitzer_30900+0000_rgb", "spitzer_31200+0000_rgb", "spitzer_31500+0000_rgb", "spitzer_31800+0000_rgb",
        "spitzer_32100+0000_rgb", "spitzer_32700+0000_rgb", "spitzer_33000+0000_rgb", "spitzer_33300+0000_rgb",
        "spitzer_33600+0000_rgb", "spitzer_33900+0000_rgb", "spitzer_34200+0000_rgb", "spitzer_34500+0000_rgb",
        "spitzer_34800+0000_rgb", "spitzer_35100+0000_rgb", "spitzer_35400+0000_rgb", "spitzer_35700+0000_rgb",
    ]
    # fmt: on
    savedir_name = f"{args.savedir_name}/NonRing_png/region_NonRing_png/"
    os.makedirs(savedir_name, exist_ok=True)

    random_uni = default_rng(123)
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
    iter = 3000
    fits_path = pathlib.Path(args.fits_path)
    pbar = tqdm(range(len(all_l)))
    #############################
    ## Start creating Non-Ring ##
    #############################
    start = time.time()
    for k in pbar:
        path = all_l[k]
        pbar.set_description(path)
        os.makedirs(f"{savedir_name}/{path}", exist_ok=True)

        spitzer_rfits = astropy.io.fits.open(fits_path / path / "r.fits")[0]
        spitzer_gfits = astropy.io.fits.open(fits_path / path / "g.fits")[0]
        spitzer_bfits = astropy.io.fits.open(fits_path / path / "b.fits")[0]
        header = spitzer_rfits.header
        data = np.concatenate(
            [spitzer_rfits.data[:, :, None], spitzer_gfits.data[:, :, None], spitzer_bfits.data[:, :, None]], axis=2
        )

        NonRing_sub_c = NonRing_sub.NonRing_sub(header, data, random_uni)

        for i in range(iter):
            cut_data = NonRing_sub_c.no_nan_ring()
            pi = processing.conv(300, sig1, cut_data)
            r_shape_y = pi.shape[0]
            r_shape_x = pi.shape[1]
            res_data = pi[
                int(r_shape_y / 52) : int(r_shape_y * 51 / 52), int(r_shape_x / 52) : int(r_shape_x * 51 / 52)
            ]
            res_data = processing.norm_res(
                res_data, spitzer_rfits.header["PIXSCAL1"], spitzer_gfits.header["PIXSCAL1"]
            )
            pil_image = Image.fromarray(np.uint8(res_data * 255))

            ############################################
            ## Save images and labels for each region ##
            ############################################
            pil_image.save(f"{savedir_name}/{path}/NonRing_{k * iter + i}.png")
            with open(f"{savedir_name}/{path}/NonRing_{k * iter + i}.json", "w") as f:
                json.dump([], f, indent=4)

    print((time.time() - start) / 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
