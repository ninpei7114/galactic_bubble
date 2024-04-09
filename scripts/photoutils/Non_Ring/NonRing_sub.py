import astropy.wcs
import astroquery.vizier
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class NonRing_sub(object):
    def __init__(self, header, fits_data, random_uni):
        self.w = astropy.wcs.WCS(header)
        self.fits_data = fits_data
        self.fits_data_shape_x = fits_data.shape[1]
        self.fits_data_shape_y = fits_data.shape[0]
        self.random_uni = random_uni

        viz = astroquery.vizier.Vizier(columns=["*"])
        viz.ROW_LIMIT = -1
        bub_2006 = viz.query_constraints(catalog="J/ApJ/649/759/bubbles")[0].to_pandas()
        bub_2007 = viz.query_constraints(catalog="J/ApJ/670/428/bubble")[0].to_pandas()
        bub_2006_change = bub_2006.set_index("__CPA2006_")
        bub_2007_change = bub_2007.set_index("__CWP2007_")
        CH = pd.concat([bub_2006_change, bub_2007_change])
        CH["CH"] = CH.index
        CH.loc[CH["GLON"] >= 358.446500015535, "GLON"] -= 360

        num, value = np.histogram(CH["Rout"] * 60 / header["PIXSCAL1"] + 145, bins=200)
        bin = []
        for i, d in enumerate(value):
            if i + 1 >= len(value):
                break
            bin.append((value[i + 1] + value[i]) / 2)
        popt, pocv = curve_fit(gauss_func, np.array(bin), num, p0=[100, 150, 10, 0])
        x_ = np.arange(100, 2500)
        y_ = gauss_func(x_, popt[0], popt[1], popt[2], popt[3])
        self.distiribution = y_ / np.sum(y_)

    def calc_cut_pix(self):
        center_x = self.random_uni.integers(150, self.fits_data_shape_x - 150)
        center_y = self.random_uni.integers(150, self.fits_data_shape_y - 150)
        random_Rout = self.random_uni.choice(np.arange(150, 2500), p=self.distiribution)
        x_random_min = center_x - random_Rout - random_Rout / 50
        x_random_max = center_x + random_Rout + random_Rout / 50
        y_random_min = center_y - random_Rout - random_Rout / 50
        y_random_max = center_y + random_Rout + random_Rout / 50

        return x_random_min, x_random_max, y_random_min, y_random_max

    def cut_no_ring(self):
        while True:
            x_random_min, x_random_max, y_random_min, y_random_max = self.calc_cut_pix()

            if not (
                x_random_min <= 0
                or x_random_max >= self.fits_data_shape_x
                or y_random_min <= 0
                or y_random_max >= self.fits_data_shape_y
            ):
                break

        cut_data_random = self.fits_data[
            int(y_random_min) : int(y_random_max), int(x_random_min) : int(x_random_max)
        ].view()
        cut_data_random_ = cut_data_random.copy()

        return cut_data_random_

    # fitsの場所をnanにしたfitsを使う
    def no_nan_ring(self):
        """
        このすぐ下にcut_no_ring関数を使っている
        というのも、この関数はnan判定するための物だから
        """

        while True:
            cut_data_random = self.cut_no_ring()
            if np.isnan(cut_data_random.sum()) or np.std(cut_data_random[:, :, 0]) < 1e-9:
                pass
            else:
                break

        return cut_data_random

    ### nan処理をする
    ### cut_no_ring関数と併用するため、importする時はno_nan関数とcut_no_ring関数を同時にする必要がある
    ### no_nan関数で出てくるarrayは一つだけだから、for 文の中にこの関数を入れて使用する

    # def GLON_LAT(self):
    #     self.GLON_min, self.GLAT_min = self.w.all_pix2world(self.fits_data_shape_x, 0, 0)
    #     self.GLON_max, self.GLAT_max = self.w.all_pix2world(0, self.fits_data_shape_y, 0)

    #     # 主に銀中のfitsを扱う為の処理
    #     # cut_no_ring関数でrandom uniformをするのだが、その時に大小関係をつける為に必要な処理
    #     if self.GLON_min >= 358.0:
    #         self.GLON_min = self.GLON_min - 360
    #     else:
    #         pass


def gauss_func(x, a, mu, sigma, d):
    """ガウシアン（正規分布）"""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + d
