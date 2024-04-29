import copy
import json
import warnings
import sys

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw

sys.path.append("../")
import processing

warnings.resetwarnings()
warnings.simplefilter("ignore")


class imaging_validation:
    def __init__(
        self, data, ring_count, non_ring_count, obj_sig, fits_path, savedir_name, label_cal, r_header, g_header
    ):
        self.data = data
        self.ring_count = ring_count
        self.non_ring_count = non_ring_count
        self.obj_sig = obj_sig
        self.fits_path = fits_path
        self.savedir_name = savedir_name
        self.label_cal = label_cal
        self.r_resolution = r_header["PIXSCAL1"]
        self.g_resolution = g_header["PIXSCAL1"]

    def cut_data(self, many_ind, cut_shape):
        self.cut_shape = int(cut_shape)
        Ring_data = []
        Ring_info = []

        for i in many_ind:
            self.offset_xmin = int(i[1])
            self.offset_ymin = int(i[0])
            extra_x_min = self.offset_xmin - self.cut_shape / 50
            extra_x_max = self.offset_xmin + self.cut_shape + self.cut_shape / 50
            extra_y_min = self.offset_ymin - self.cut_shape / 50
            extra_y_max = self.offset_ymin + self.cut_shape + self.cut_shape / 50
            data_c = self.data[int(extra_y_min) : int(extra_y_max), int(extra_x_min) : int(extra_x_max)].view()
            d = copy.deepcopy(data_c)
            d = processing.conv(300, self.obj_sig, d)
            d = d[
                int(self.cut_shape / 52) : int(self.cut_shape * 51 / 52),
                int(self.cut_shape / 52) : int(self.cut_shape * 51 / 52),
            ]

            if np.isnan(d.sum()) or np.std(d[:, :, 0]) < 1e-9:
                pass
            else:
                self.cut_region = processing.norm_res(d, self.r_resolution, self.g_resolution).astype(np.float32)

                self.label_cal.make_label(self.offset_xmin, self.offset_ymin, self.cut_shape)
                xmin_list, ymin_list, xmax_list, ymax_list, name_list = self.label_cal.check_list()
                self.info = {
                    "fits": self.fits_path,
                    "name": name_list,
                    "xmin": xmin_list,
                    "xmax": xmax_list,
                    "ymin": ymin_list,
                    "ymax": ymax_list,
                }
                Ring_or_NonRing = self.make_Validation_png()
                if Ring_or_NonRing == "Ring":
                    Ring_data.append(self.cut_region)
                    Ring_info.append(self.info)

        Ring_info = pd.DataFrame(Ring_info)
        return np.array(Ring_data), Ring_info

    def make_Validation_png(self):
        ll = []
        if len(self.info["xmin"]) >= 1:
            for la in range(len(self.info["xmin"])):
                ll.append(
                    {
                        "Confidence": str(0),
                        "XMin": str(self.info["xmin"][la]),
                        "XMax": str(self.info["xmax"][la]),
                        "YMin": str(self.info["ymin"][la]),
                        "YMax": str(self.info["ymax"][la]),
                    }
                )
            Ring_or_NonRing = "Ring"
            self.ring_count += 1
            cut_count = self.ring_count
        else:
            Ring_or_NonRing = "NonRing"
            self.non_ring_count += 1
            cut_count = self.non_ring_count

        with open(
            f"{self.savedir_name}/{Ring_or_NonRing}/{Ring_or_NonRing}_{cut_count}_{self.offset_ymin}_{self.offset_xmin}_{self.cut_shape}_{self.fits_path.split('_')[1]}_.json",
            "w",
        ) as f:
            json.dump(ll, f, indent=4)
        pil_image = Image.fromarray(np.uint8(self.cut_region * 255))
        pil_image.save(
            f"{self.savedir_name}/{Ring_or_NonRing}/{Ring_or_NonRing}_{cut_count}_{self.offset_ymin}_{self.offset_xmin}_{self.cut_shape}_{self.fits_path.split('_')[1]}_.png"
        )
        cut_count += 1

        return Ring_or_NonRing


def data_view_rectangl(col, imgs, infos=None, moji_size=100):
    """
    col: number of columns
    imgs: tensor or nparray with a shape of (?, y, x, 1) or (?, y, x, 3)
    infos: dictonary from CutTable
    """
    imgs = np.uint8(imgs[:, ::-1, :, 0]) if imgs.shape[3] == 1 else np.uint8(imgs[:, ::-1])
    row = (lambda x, y: x // y if x / y - x // y == 0.0 else x // y + 1)(imgs.shape[0], col)
    dst = Image.new("RGB", (imgs.shape[1] * col, imgs.shape[2] * row))

    # font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', moji_size)
    for i, arr in enumerate(imgs):
        img = Image.fromarray(arr)
        img = img.point(lambda x: x * 1.5)
        if infos is not None:
            draw = ImageDraw.Draw(img)
            # draw.text((10, 10), '%s'%infos['id'].tolist()[i], font=font)
            for j in range(len(infos["xmin"].tolist()[i])):
                draw.rectangle(
                    (
                        infos["xmin"].tolist()[i][j] * 300,
                        (1 - infos["ymax"].tolist()[i][j]) * 300,
                        infos["xmax"].tolist()[i][j] * 300,
                        (1 - infos["ymin"].tolist()[i][j]) * 300,
                    ),
                    width=2,
                )

        quo, rem = i // col, i % col
        dst.paste(img, (arr.shape[0] * rem, arr.shape[1] * quo))

    return dst
