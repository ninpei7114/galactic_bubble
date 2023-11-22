import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import aplpy
import sys
from PIL import Image
import copy
import tqdm
import astroquery.vizier
import astropy

sys.path.append("../")
from utils.ssd_model import nm_suppression
from processing import norm_res, conv, data_view_rectangl, remove_nan


def make_MWP_catalogue():
    viz = astroquery.vizier.Vizier(columns=["*"])
    viz.ROW_LIMIT = -1
    MWP = viz.query_constraints(catalog="2019yCat..74881141J ")[0].to_pandas()
    MWP.loc[MWP["GLON"] >= 358.446500015535, "GLON"] -= 360
    MWP.index = MWP["MWP"].tolist()
    rank_3 = np.load("../MWP_rank3_name.npy")
    MWP = MWP.loc[rank_3]
    MWP = MWP.rename({"_RA.icrs": "_RA_icrs"}, axis="columns")

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


def calc_bbox(args, region):
    detection_list = []
    position_list = []
    size_list = [150, 300, 600, 1200, 1800, 2400, 3000]
    conf_list = [0.8] * len(size_list)
    for size in size_list:
        detection_list.append(np.load(f"{args.result_path}/{region}/result_ring_select_csize{size}.npy"))
        position_list.append(np.load(f"{args.result_path}/{region}/position_ring_select_csize{size}.npy"))

    predict_bbox = []
    scores = []
    for s in range(len(size_list)):
        for d, p in zip(detection_list[s], position_list[s]):
            # dのshapeは[2, 200, 5]
            find_index = np.where(
                d[1, :, 0] >= conf_list[s]
            )  # d[1, :, 0]ringのconf、最初の１は、[ring:1, no_ring:0]の１、最後の０はconfを指定

            # fing_indexは、ringのconfが0.99以上のindex
            # dのshapeは、いろいろ、[0, 5]だったり、[ringの数, 5]
            d = d[1][find_index]
            if len(d) == 0:
                pass
            else:
                for i in range(len(find_index)):  # 抽出した物体数分ループを回す
                    sc = d[i][0]  # 確信度
                    bbox = d[i][1:] * [size_list[s], size_list[s], size_list[s], size_list[s]]
                    # 返り値のリストに追加
                    bbox = bbox + np.array([p[1], p[0], p[1], p[0]])
                    predict_bbox.append(bbox)
                    scores.append(sc)
    bbox = torch.Tensor(np.array(predict_bbox))
    scores = torch.Tensor(scores)
    del detection_list
    keep, count = nm_suppression(bbox, scores, overlap=0.3, top_k=5000)
    keep = keep[:count]
    bbox = bbox[keep]

    return bbox


def make_infer_catalogue(bbox, w):
    catalogue = pd.DataFrame(columns=["dec_min", "ra_min", "dec_max", "ra_max"])
    for i in bbox:
        #     print(i)GLONmin
        GLONmax, GLATmin = w.all_pix2world(i[0], i[1], 0)
        GLONmin, GLATmax = w.all_pix2world(i[2], i[3], 0)
        temp = pd.DataFrame(
            columns=["dec_min", "ra_min", "dec_max", "ra_max"],
            data=[[GLATmin, GLONmin, GLATmax, GLONmax]],
            dtype="float64",
        )
        catalogue = pd.concat([catalogue, temp])

    x_width = []
    y_width = []
    for i in bbox:
        x_width.append(i[2] - i[0])
        y_width.append(i[3] - i[1])
    x_width = np.array(x_width, dtype=np.float64)
    y_width = np.array(y_width, dtype=np.float64)

    catalogue["width_pix"] = x_width
    catalogue["height_pix"] = x_width

    return catalogue


def make_map(save_png_name, region, catalogue, hdu, args, g_fits_path, MWP_catalogue=None, region_=None):
    Image.MAX_IMAGE_PIXELS = 1000000000
    fig = plt.figure(figsize=(16, 16))
    if region == "LMC" or region == "Spitzer":
        f = aplpy.FITSFigure(g_fits_path, slices=[0], figure=fig, convention="wells")
    elif region == "Cygnus":
        f = aplpy.FITSFigure(g_fits_path, slices=[0], figure=fig, convention="wells", north=True)
    f.show_rgb(save_png_name)
    f.ticks.set_color("w")
    f.ticks.set_linewidth(1.5)
    f.ticks.set_minor_frequency(2)
    f.tick_labels.set_font(size=40, family="serif")
    f.axis_labels.set_font(size=40, family="serif")

    f.show_rectangles(
        xw=(catalogue["ra_min"] + catalogue["ra_max"]) / 2,
        yw=(catalogue["dec_min"] + catalogue["dec_max"]) / 2,
        width=abs(hdu.header["CDELT1"]) * catalogue["width_pix"],
        height=abs(hdu.header["CDELT2"]) * catalogue["height_pix"],
        edgecolor=None,
        color="c",
        linewidth=1,
    )

    if region == "Cygnus":
        f.show_circles(
            xw=MWP_catalogue["_RA_icrs"],
            yw=MWP_catalogue["_DE.icrs"],
            radius=MWP_catalogue["Reff"] / 60,
            linewidth=2,
            edgecolor="#F5F5F5",
        )
        f.recenter(307.7, 40.23, width=6.5, height=6.3)
    elif region == "Spitzer":
        f.show_circles(
            xw=MWP_catalogue["GLON"],
            yw=MWP_catalogue["GLAT"],
            radius=MWP_catalogue["Reff"] / 60,
            linewidth=2,
            edgecolor="#F5F5F5",
        )
    plt.title(region, fontsize=60)
    plt.tight_layout()
    if region == "Spitzer":
        f.save(f"{args.save_dir}/{region}/{region_}/{region}_predict.png", dpi=300)
    else:
        f.save(f"{args.save_dir}/{region}/{region}_predict.png", dpi=300)


def make_cut_ring(bbox, data, args, region, region_=None):
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
    d_cut = []
    width_list = []
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
        cut_ = norm_res(cut_)
        d_cut.append(cut_)

    d_cut_ = np.array(d_cut)
    d_cut_ = d_cut_ * 255
    d_cut_ = np.uint8(d_cut_)
    d_cut_[:, :, :, 2] = 0
    if region == "Spitzer":
        data_view_rectangl(20, d_cut_).save(f"{args.save_dir}/{region}/{region_}/{region}_predict_ring.png")
    else:
        data_view_rectangl(20, d_cut_).save(f"{args.save_dir}/{region}/{region}_predict_ring.png")


def make_TP_FN(target_catalogue, target_mask, data, w, hdu):
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
    target_ring_list = []

    for _, row in tqdm.tqdm(target_catalogue[target_mask].iterrows()):
        lmax = row["_RA_icrs"] + row["MajAxis"] / 60
        bmin = row["_DE.icrs"] - row["MajAxis"] / 60
        lmin = row["_RA_icrs"] - row["MajAxis"] / 60
        bmax = row["_DE.icrs"] + row["MajAxis"] / 60
        l_center = (lmax + lmin) / 2
        b_center = (bmax + bmin) / 2

        x_center, y_center = w.all_world2pix(l_center, b_center, 0)
        r = (lmax - lmin) / hdu.header["CDELT2"]

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
            res_data = norm_res(res_data)
            target_ring_list.append(res_data)

    return np.array(target_ring_list)
