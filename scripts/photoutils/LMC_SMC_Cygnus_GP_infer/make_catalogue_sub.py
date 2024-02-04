import copy
import sys

import aplpy
import astropy
import astroquery.vizier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image

sys.path.append("../")
from processing import conv, data_view_rectangl, norm_res, remove_nan
from utils.ssd_model import nm_suppression


def make_MWP_catalogue(region):
    viz = astroquery.vizier.Vizier(columns=["*"])
    viz.ROW_LIMIT = -1
    MWP = viz.query_constraints(catalog="2019yCat..74881141J ")[0].to_pandas()
    MWP.loc[MWP["GLON"] >= 358.446500015535, "GLON"] -= 360
    MWP.index = MWP["MWP"].tolist()
    if region == "Cygnus":
        MWP = MWP.rename({"_RA.icrs": "_RA_icrs"}, axis="columns")
    elif region == "Spitzer":
        rank_3 = np.load("../MWP_rank3_name.npy")
        MWP = MWP.loc[rank_3]

    return MWP


def make_data(fp):
    r_fits_path = fp[0]
    g_fits_path = fp[1]
    region_ = fp[2]
    hdu_r = astropy.io.fits.open(r_fits_path)[0]
    a = hdu_r.data.shape[0]
    b = hdu_r.data.shape[1]
    w = astropy.wcs.WCS(hdu_r.header)
    data_ = np.concatenate(
        [
            remove_nan(hdu_r.data[:, :, None]),
            remove_nan(astropy.io.fits.open(f"{g_fits_path}")[0].data[:, :, None]),
            np.zeros(hdu_r.data.shape)[:, :, None],
        ],
        axis=2,
    )

    return data_, hdu_r, a, b, w, region_


def calc_bbox(args, region, conf_thre):
    predict_bbox, scores = [], []
    detections = np.load(f"{args.result_path}/{region}/result.npy")
    position = np.load(f"{args.result_path}/{region}/position.npy")

    for d, p in zip(detections, position):
        conf_mask = d[1, :, 0] >= conf_thre
        detection_mask = d[1, :][conf_mask]
        if np.sum(conf_mask) >= 1:
            bbox = detection_mask[:, 1:] * np.array(int(p[2]))
            bbox = bbox + np.array([int(p[1]), int(p[0]), int(p[1]), int(p[0])])
            predict_bbox.append(bbox)
            scores.append(detection_mask[:, 0])

    bbox = torch.Tensor(np.concatenate(predict_bbox))
    scores = torch.Tensor(np.concatenate(scores))
    keep, count = nm_suppression(bbox, scores, overlap=0.3, top_k=5000)
    keep = keep[:count]
    bbox = bbox[keep]

    return bbox


def make_infer_catalogue(bbox, w):
    catalogue = pd.DataFrame(columns=["dec_min", "ra_min", "dec_max", "ra_max", "width_pix", "height_pix"])
    for i in bbox:
        width = int(i[2]) - int(i[0])
        height = int(i[3]) - int(i[1])
        GLONmax, GLATmin = w.all_pix2world(i[0], i[1], 0)
        GLONmin, GLATmax = w.all_pix2world(i[2], i[3], 0)
        temp = pd.DataFrame(
            columns=["dec_min", "ra_min", "dec_max", "ra_max", "width_pix", "height_pix"],
            data=[[GLATmin, GLONmin, GLATmax, GLONmax, width, height]],
            dtype="float64",
        )
        catalogue = pd.concat([catalogue, temp])

    return catalogue


def make_map(save_png_name, region, catalogue, hdu, g_fits_path, save_dir, MWP_catalogue=None, region_=None):
    Image.MAX_IMAGE_PIXELS = 1000000000
    fig = plt.figure(figsize=(16, 16))
    if region == "LMC" or region == "SMC" or region == "Spitzer":
        f = aplpy.FITSFigure(g_fits_path, slices=[0], figure=fig, convention="wells")
    elif region == "Cygnus":
        f = aplpy.FITSFigure(g_fits_path, slices=[0], figure=fig, convention="wells", north=True)
    f.show_rgb(save_png_name)
    f.ticks.set_color("w")
    f.ticks.set_linewidth(1.5)
    f.ticks.set_minor_frequency(2)
    f.tick_labels.set_font(size=40, family="serif")
    f.axis_labels.set_font(size=40, family="serif")

    if region == "Cygnus":
        header_d1 = hdu.header["CDELT1"]
        header_d2 = hdu.header["CDELT2"]
    else:
        header_d1 = hdu.header["CD1_1"]
        header_d2 = hdu.header["CD2_2"]

    f.show_rectangles(
        xw=(catalogue["ra_min"] + catalogue["ra_max"]) / 2,
        yw=(catalogue["dec_min"] + catalogue["dec_max"]) / 2,
        width=abs(header_d1) * catalogue["width_pix"],
        height=abs(header_d2) * catalogue["height_pix"],
        edgecolor=None,
        color="w",
        linewidth=1,
    )

    if region == "Cygnus":
        f.show_circles(
            xw=MWP_catalogue["_RA_icrs"],
            yw=MWP_catalogue["_DE.icrs"],
            radius=MWP_catalogue["Reff"] / 60,
            linewidth=2,
            edgecolor="b",
        )
        f.recenter(307.7, 40.23, width=6.5, height=6.3)
    elif region == "Spitzer":
        f.show_circles(
            xw=MWP_catalogue["GLON"],
            yw=MWP_catalogue["GLAT"],
            radius=MWP_catalogue["Reff"] / 60,
            linewidth=2,
            edgecolor="b",
        )
    plt.title(region, fontsize=60)
    plt.tight_layout()
    if region == "Spitzer":
        f.save(f"{save_dir}/{region_}/{region}_predict.png", dpi=300)
    else:
        f.save(f"{save_dir}/{region}_predict.png", dpi=300)


def make_cut_ring(bbox, data, save_dir, region, r_header, g_header, region_=None):
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
    d_cut = []
    width_list = []
    if region == "Spitzer":
        save_png_name = f"{save_dir}/{region_}/{region}_predict_ring.png"
        r_resolution = r_header["PIXSCAL1"]
        g_resolution = g_header["PIXSCAL1"]
    else:
        save_png_name = f"{save_dir}/{region}_predict_ring.png"
        if region == "LMC":
            r_resolution = r_header["CD2_2"] * 3600
            g_resolution = g_header["CD2_2"] * 3600
        elif region == "SMC":
            r_resolution = r_header["CD2_2"] * 3600
            g_resolution = g_header["CDELT2"] * 3600
        elif region == "Cygnus":
            r_resolution = r_header["CDELT2"] * 3600
            g_resolution = g_header["CDELT2"] * 3600
    for i in bbox:
        height = int(i[3]) - int(i[1])
        width = int(i[2]) - int(i[0])
        width_list.append(width)
        ymin = int(i[1]) - height / 50
        ymax = int(i[3]) + height / 50
        xmin = int(i[0]) - width / 50
        xmax = int(i[2]) + width / 50

        cut = data[int(ymin) : int(ymax), int(xmin) : int(xmax), :].view()
        cut_ = copy.deepcopy(cut)
        cut_ = conv(300, sig1, cut_)
        cut_ = cut_[
            int(cut_.shape[0] / 52) : int(cut_.shape[0] * 51 / 52),
            int(cut_.shape[1] / 52) : int(cut_.shape[1] * 51 / 52),
        ]
        cut_ = norm_res(cut_, r_resolution, g_resolution)
        d_cut.append(cut_)

    d_cut_ = np.array(d_cut)
    d_cut_ = d_cut_ * 255
    d_cut_ = np.uint8(d_cut_)
    d_cut_[:, :, :, 2] = 0
    data_view_rectangl(20, d_cut_).save(save_png_name)


def make_TP_FN(target_catalogue, target_mask, data, w, hdu, region):
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
    target_ring_list = []
    if region == "Cygnus":
        coordinate_x = "_RA_icrs"
        coordinate_y = "_DE.icrs"
        CDELT = hdu.header["CDELT2"]
        r_resolution = hdu.header["CDELT2"] * 3600
        g_resolution = hdu.header["CDELT2"] * 3600
    elif region == "Spitzer":
        coordinate_x = "GLON"
        coordinate_y = "GLAT"
        CDELT = hdu.header["CD2_2"]
        r_resolution = hdu.header["PIXSCAL1"]
        g_resolution = hdu.header["PIXSCAL1"]

    for _, row in tqdm.tqdm(target_catalogue[target_mask].sort_values("Reff").iterrows()):
        x_center, y_center = w.all_world2pix(row[coordinate_x], row[coordinate_y], 0)
        r = 2 * row["MajAxis"] / 60 / CDELT

        x_pix_min = x_center - r - r / 50
        y_pix_min = y_center - r - r / 50
        x_pix_max = x_center + r + r / 50
        y_pix_max = y_center + r + r / 50

        if x_pix_min <= 0 or y_pix_min <= 0:
            pass
        else:
            c_data = data[int(y_pix_min) : int(y_pix_max), int(x_pix_min) : int(x_pix_max)].view()
            cut_data = copy.deepcopy(c_data)
            pi = conv(300, sig1, cut_data)
            r_shape_y = pi.shape[0]
            r_shape_x = pi.shape[1]
            res_data = pi[
                int(r_shape_y / 52) : int(r_shape_y * 51 / 52), int(r_shape_x / 52) : int(r_shape_x * 51 / 52)
            ]
            res_data = norm_res(res_data, r_resolution, g_resolution)
            target_ring_list.append(res_data)

    return np.array(target_ring_list)


def make_FP(FP_catalogue, data, w, hdu, region):
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
    target_ring_list = []
    if region == "Cygnus":
        r_resolution = hdu.header["CDELT2"] * 3600
        g_resolution = hdu.header["CDELT2"] * 3600
    elif region == "Spitzer":
        r_resolution = hdu.header["PIXSCAL1"]
        g_resolution = hdu.header["PIXSCAL1"]

    for _, row in tqdm.tqdm(FP_catalogue.sort_values("width_pix").iterrows()):
        x_min, y_min = w.all_world2pix(row["ra_max"], row["dec_min"], 0)
        x_max, y_max = w.all_world2pix(row["ra_min"], row["dec_max"], 0)
        r = x_max - x_min

        x_pix_min = x_min - r / 50
        y_pix_min = y_min - r / 50
        x_pix_max = x_max + r / 50
        y_pix_max = y_max + r / 50

        if x_pix_min <= 0 or y_pix_min <= 0:
            pass

        else:
            c_data = data[int(y_pix_min) : int(y_pix_max), int(x_pix_min) : int(x_pix_max)].view()
            cut_data = copy.deepcopy(c_data)
            pi = conv(300, sig1, cut_data)
            r_shape_y = pi.shape[0]
            r_shape_x = pi.shape[1]
            res_data = pi[
                int(r_shape_y / 52) : int(r_shape_y * 51 / 52), int(r_shape_x / 52) : int(r_shape_x * 51 / 52)
            ]
            res_data = norm_res(res_data, r_resolution, g_resolution)
            target_ring_list.append(res_data)

    return np.array(target_ring_list)
