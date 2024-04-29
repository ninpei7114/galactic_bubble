import copy

import astroquery.vizier
import numpy as np
import pandas as pd
import processing
from skimage import transform

"""
Script to create ring augmentation patterns
"""


def translation(row, fits_path, GLON_min, GLON_max, GLAT_min, GLAT_max, Ring_catalogue, data, label_cal, trans_rg):
    """
    Function used for parallel translation augmentation.
    Cuts out so that the Ring is in a random position (seed value fixed) within the image.

    Parameters
    ----------
    row : row of the Ring catalogue
    fits_path : path to the fits file
    GLON_min, GLON_max, GLAT_min, GLAT_max : Galactic longitude and latitude coordinates at both ends of the fits to be used
    Ring_catalogue : Ring catalogue in fits (either Milky Way Project or Churchwell)
    data : fits data
    label_cal : function to calculate the label (pre-instantiated)
    trans_rg : default_rng

    Returns
    -------
    True, pi_conv, info : if the Ring is successfully cut out
    False, 0, 0 : if the Ring is not successfully cut out

    """

    ###################################################################################################################
    ## Randomly calculate how many times the radius from the center coordinates registered in the catalog to cut out ##
    ###################################################################################################################
    ## The information registered in the catalog is in galactic longitude and latitude,
    ## so it needs to be converted to pix information.
    ## ↓ In this state, the bubble is still located in the center of the image.

    random_num = 1 / trans_rg.uniform(0.17, 0.58)
    x_pix_min, y_pix_min, x_pix_max, y_pix_max, flag = label_cal.calc_pix(
        row, GLON_min, GLON_max, GLAT_min, GLAT_max, random_num
    )

    #####################################################
    ## Move the Ring randomly within a 300 x 300 image ##
    #####################################################
    ## Case differentiation when it was not possible to do it even after 400 trials at the time of calc_pix
    ## For example, if the Ring is located at the edge of the fits, it may not be able to cut out well.

    if flag:
        ## Because ears can be created when doing image processing convolution,
        ## it cuts out an extra half the size of width, height on the left, right, top, and bottom
        extra_width = (x_pix_max - x_pix_min) * 2 / 52
        ## r is finding the radius pix of the Ring
        r = int(((x_pix_max - extra_width) - (x_pix_min + extra_width)) / (2 * random_num))

        ## So that the Ring does not protrude from the image, shift the position to cut out,
        ## Change the position of the Ring in the image.
        x_offset = trans_rg.uniform(-(random_num - 1) * r, (random_num - 1) * r)
        y_offset = trans_rg.uniform(-(random_num - 1) * r, (random_num - 1) * r)
        x_pix_min = x_pix_min + int(x_offset)
        x_pix_max = x_pix_max + int(x_offset)
        y_pix_min = y_pix_min + int(y_offset)
        y_pix_max = y_pix_max + int(y_offset)

        if x_pix_min < 0 or y_pix_min < 0:
            return False, 0, 0

        else:
            ##################################################
            ## Image processing of other bubble's label and data in the range to cut out ##
            ##################################################

            ## Find other Rings in the range to cut out
            pix_info = {
                "x_pix_min": x_pix_min,
                "x_pix_max": x_pix_max,
                "y_pix_min": y_pix_min,
                "y_pix_max": y_pix_max,
            }
            label_cal.find_cover(pix_info)
            sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))

            ## Cut out the data, conv → normalize → resize
            c_data = data[int(y_pix_min) : int(y_pix_max), int(x_pix_min) : int(x_pix_max)].view()
            cut_data = c_data.copy()
            pi = processing.conv(300, sig1, cut_data)
            pi_ = copy.deepcopy(pi)
            r_shape_y = pi_.shape[0]
            r_shape_x = pi_.shape[1]
            pi_conv = pi_[
                int(r_shape_y / 52) : int(r_shape_y * 51 / 52), int(r_shape_x / 52) : int(r_shape_x * 51 / 52)
            ]

            if np.isnan(pi_conv.sum()) or np.std(pi_conv[:, :, 0]) < 1e-9:
                return False, 0, 0

            else:
                ## Create a label to be used for training data.
                label_cal.make_label(Ring_catalogue)
                info = label_cal.check_list()
                info["fits"] = fits_path
                return True, pi_conv, info

    else:
        return False, 0, 0


def rotate_data(deg, trans_data, trans_info):
    """
    リングを回転させる。
    それに伴いlabelも変更させる。
    """
    tempo = copy.deepcopy(trans_data)
    rotate_cut_data = transform.rotate(tempo, deg)
    xmin_list_, ymin_list_, xmax_list_, ymax_list_ = [], [], [], []

    for xy_num in range(len(trans_info["xmin"])):
        width = trans_info["xmax"][xy_num] - trans_info["xmin"][xy_num]
        height = trans_info["ymax"][xy_num] - trans_info["ymin"][xy_num]
        center_x = ((trans_info["xmin"][xy_num] - 0.5) + (trans_info["xmax"][xy_num] - 0.5)) / 2
        center_y = ((trans_info["ymin"][xy_num] - 0.5) + (trans_info["ymax"][xy_num] - 0.5)) / 2

        new_center_x = center_x * np.cos(np.deg2rad(-deg)) - center_y * np.sin(np.deg2rad(-deg)) + 0.5
        new_center_y = center_x * np.sin(np.deg2rad(-deg)) + center_y * np.cos(np.deg2rad(-deg)) + 0.5

        xmin_list_.append(np.clip(new_center_x - width / 2, 0, 1))
        ymin_list_.append(np.clip(new_center_y - height / 2, 0, 1))
        xmax_list_.append(np.clip(new_center_x + width / 2, 0, 1))
        ymax_list_.append(np.clip(new_center_y + height / 2, 0, 1))

    info = {
        "fits": trans_info["fits"],
        "name": trans_info["name"],
        "xmin": xmin_list_,
        "xmax": xmax_list_,
        "ymin": ymin_list_,
        "ymax": ymax_list_,
    }

    return rotate_cut_data, info


def flip_data(trans_data, trans_info):
    """
    Flip the rings vertically and horizontally.
    While the labels remain unchanged for the target ring when flipped vertically,
    there are also interfering rings other than the target ring,
    so label adjustments are necessary for flips.
    """

    ## Vertical flip
    tempo = copy.deepcopy(trans_data)
    ud = np.flipud(tempo)

    xmin_list = trans_info["xmin"]
    ymin_list = trans_info["ymin"]
    xmax_list = trans_info["xmax"]
    ymax_list = trans_info["ymax"]
    name_list = trans_info["name"]

    ud_xmin_list, ud_xmax_list, ud_ymin_list, ud_ymax_list = [], [], [], []
    for i in range(len(xmin_list)):
        y_max = 1 - ymin_list[i]
        x_min = xmin_list[i]
        y_min = 1 - ymax_list[i]
        x_max = xmax_list[i]

        assert x_max >= x_min and y_max >= y_min

        ud_xmin_list.append(x_min)
        ud_xmax_list.append(x_max)
        ud_ymin_list.append(y_min)
        ud_ymax_list.append(y_max)

    ud_info = {
        "fits": trans_info["fits"],
        "name": name_list,
        "xmin": ud_xmin_list,
        "xmax": ud_xmax_list,
        "ymin": ud_ymin_list,
        "ymax": ud_ymax_list,
    }

    ## Horizontal flip
    tempo = copy.deepcopy(trans_data)
    lr = np.fliplr(tempo)

    lr_xmin_list, lr_xmax_list, lr_ymin_list, lr_ymax_list = [], [], [], []
    for i in range(len(xmin_list)):
        x_max = 1 - xmin_list[i]
        y_min = ymin_list[i]
        x_min = 1 - xmax_list[i]
        y_max = ymax_list[i]

        assert x_max >= x_min and y_max >= y_min

        lr_xmin_list.append(x_min)
        lr_xmax_list.append(x_max)
        lr_ymin_list.append(y_min)
        lr_ymax_list.append(y_max)

    lr_info = {
        "fits": trans_info["fits"],
        "name": name_list,
        "xmin": lr_xmin_list,
        "xmax": lr_xmax_list,
        "ymin": lr_ymin_list,
        "ymax": lr_ymax_list,
    }

    return ud, lr, ud_info, lr_info


def catalogue(choice, ring_select=False, rank_path="rank_3.npy"):
    if choice == "CH":
        viz = astroquery.vizier.Vizier(columns=["*"])
        viz.ROW_LIMIT = -1
        bub_2006 = viz.query_constraints(catalog="J/ApJ/649/759/bubbles")[0].to_pandas()
        bub_2007 = viz.query_constraints(catalog="J/ApJ/670/428/bubble")[0].to_pandas()
        bub_2006_change = bub_2006.set_index("__CPA2006_")
        bub_2007_change = bub_2007.set_index("__CWP2007_")
        CH = pd.concat([bub_2006_change, bub_2007_change])
        CH["CH"] = CH.index
        if ring_select:
            print("\n#######################")
            print("   Ring selection")
            print("#######################")
            rank_2_3 = np.load(rank_path)
            CH = CH.loc[rank_2_3]
        return CH

    elif choice == "MWP":
        viz = astroquery.vizier.Vizier(columns=["*"])
        viz.ROW_LIMIT = -1
        MWP = viz.query_constraints(catalog="2019yCat..74881141J ")[0].to_pandas()
        MWP.loc[MWP["GLON"] >= 358.446500015535, "GLON"] -= 360
        MWP.index = MWP["MWP"].tolist()
        if ring_select:
            print("\n#######################")
            print("   Ring selection")
            print("#######################")
            rank_3 = np.load("MWP_rank3_name.npy")
            MWP = MWP.loc[rank_3]
        return MWP

    elif choice == "SUM":
        viz = astroquery.vizier.Vizier(columns=["*"])
        viz.ROW_LIMIT = -1
        bub_2006 = viz.query_constraints(catalog="J/ApJ/649/759/bubbles")[0].to_pandas()
        bub_2007 = viz.query_constraints(catalog="J/ApJ/670/428/bubble")[0].to_pandas()
        bub_2006_change = bub_2006.set_index("__CPA2006_")
        bub_2007_change = bub_2007.set_index("__CWP2007_")
        CH = pd.concat([bub_2006_change, bub_2007_change])
        CH["SUM"] = CH.index

        viz = astroquery.vizier.Vizier(columns=["*"])
        viz.ROW_LIMIT = -1
        MWP = viz.query_constraints(catalog="2019yCat..74881141J ")[0].to_pandas()
        MWP.loc[MWP["GLON"] >= 358.446500015535, "GLON"] -= 360
        MWP.index = MWP["MWP"].tolist()
        MWP = MWP.rename({"MajAxis": "Rout"}, axis="columns")
        MWP = MWP.rename({"MWP": "SUM"}, axis="columns")
        CH_MWP = pd.concat([CH, MWP])

        if ring_select:
            print("\n#######################")
            print("   Ring selection")
            print("#######################")
            CH_MWP_name = np.load("CH_MWP_SUM.npy")
            CH_MWP = CH_MWP.loc[CH_MWP_name]

        return CH_MWP

    else:
        print("this choice catalogue does not exist")
