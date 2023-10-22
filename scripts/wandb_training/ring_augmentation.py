import copy

import astroquery.vizier
import numpy as np
import pandas as pd
from skimage import transform

import processing

"""
リングのaugmentationパターンを作成するスクリプト
"""


def translation(row, fits_path, GLON_min, GLON_max, GLAT_min, GLAT_max, Ring_catalogue, data, label_cal, trans_rg):
    """
    並行移動augmentationに用いる関数
    画像内でランダムな位置（シード値固定）にRingが入るように切り出す。

    (引数)
    GLON_new_min, GLON_new_max, GLAT_min, GLAT_max : 使用するfitsの両端の銀径銀緯座標
    Ring_catalogue       : fits内 の Ringのカタログ (Milky Way Project か Chuchwell のどちらか)
    data      : fitsのデータ
    label_cal : labelを求めるための関数（事前にインスタンス化している）
    m2_size   : カタログに登録されている Ringの半径 の 何倍で切り出すかのランダム値
    trans_rg  : default_rng
    """

    ####################################################################
    ## カタログに登録されている中心座標から半径の何倍で切り出すかをランダムに計算 ##
    ####################################################################
    ## カタログに登録されている情報は銀径銀緯なため、
    ## pix情報に変換する必要がある。
    ## ↓ この状態では、Ringは画像の中心に位置したまま。

    random_num = 1 / trans_rg.uniform(0.125, 0.8)
    x_pix_min, y_pix_min, x_pix_max, y_pix_max, flag = label_cal.calc_pix(
        row, GLON_min, GLON_max, GLAT_min, GLAT_max, random_num
    )

    ################################################
    ## Ringを300 x 300の画像内に、ランダムに移動させる ##
    ################################################
    ## calc_pix時に400回試行してもできなかった場合の場合分け
    ## 例えば、Ringがfitsの端に位置する場合は、上手く切り取れない場合がある。

    if flag:
        ## 画像処理のconvolutionをする際に耳ができるため、
        ## 左右上下にwidth, heightの半分の大きさ分を余分に切り出している
        half_width = (x_pix_max - x_pix_min) * 2 / 52
        ## rはRingの半径pixを求めている
        r = int(((x_pix_max - half_width) - (x_pix_min + half_width)) / (2 * random_num))

        ## Ringが画像にはみ出さないように、切り出す位置をずらし、
        ## 画像内のRingの位置を変える。
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
            ## 切り出す範囲にある他のRingのlabelとデータの画像処理 ##
            ##################################################

            ## 切り出す範囲にある他のRingを見つける
            pix_info = {
                "x_pix_min": x_pix_min,
                "x_pix_max": x_pix_max,
                "y_pix_min": y_pix_min,
                "y_pix_max": y_pix_max,
            }
            label_cal.find_cover(pix_info)
            sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))

            ## データを切り出し、conv → normalize → resize
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
                ## 学習データに用いるlabelを作成する。
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
    リングを上下左右反転させる。
    ターゲットのリング以外の割り込みリングもあるため、
    flipもラベルの修正が必要
    """

    ## 上下
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

    ## 左右
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
