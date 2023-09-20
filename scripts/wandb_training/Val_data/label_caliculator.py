import numpy as np


class label_caliculator(object):
    def __init__(self, choice, world):
        if choice == "MWP":
            self.Rout = "Reff"
        else:
            self.Rout = "Rout"

        self.world = world
        self.choice = choice

    def all_star(self, dataframe):
        """
        使用fits内にあるリングのpix情報を取得する。
        """

        self.star_dic = {}
        for _, row in dataframe.iterrows():
            lmax = row["GLON"] + row[self.Rout] / 60
            bmin = row["GLAT"] - row[self.Rout] / 60
            ## 右端
            lmin = row["GLON"] - row[self.Rout] / 60
            bmax = row["GLAT"] + row[self.Rout] / 60
            ## これは、リングを切り取る範囲　　切り取る範囲はRoutの3倍
            x_pix_min, y_pix_min = self.world.all_world2pix(lmax, bmin, 0)
            x_pix_max, y_pix_max = self.world.all_world2pix(lmin, bmax, 0)
            self.star_dic[row[self.choice]] = [x_pix_min, y_pix_min, x_pix_max, y_pix_max]

    def judge_01(self, number):
        """
        label付けする際に、位置labelの範囲が0-1になっていないといけない
        この関数は、位置labelを0-1の範囲に収めるための関数
        """
        if number > 1:
            return 1
        elif number < 0:
            return 0
        else:
            return number

    def make_label(self, x_min, y_min, cut_shape):
        """
        sは、主体となるringの位置情報
        x_pix_min, y_pix_min,x_pix_max, y_pix_maxは、切り出す画像のサイズ
        主体となるringに重なっているringのindex情報、重なったringの情報はstar_listの中にある。
        """

        self.xmin_list = []
        self.ymin_list = []
        self.xmax_list = []
        self.ymax_list = []
        self.named_list = []

        for ring_name, star_pix_list in self.star_dic.items():
            star_xmin = star_pix_list[0]
            star_ymin = star_pix_list[1]
            star_xmax = star_pix_list[2]
            star_ymax = star_pix_list[3]
            xx = np.array([star_xmin, star_xmax])
            yy = np.array([star_ymin, star_ymax])

            ## リングの本当の面積
            star_area = (xx[1] - xx[0]) * (yy[1] - yy[0])

            ## 切り出す範囲内での対象リングの面積
            ## リングが切り出す範囲外なら、0になる
            clip_xx = np.clip(xx, x_min, x_min + cut_shape)
            clip_yy = np.clip(yy, y_min, y_min + cut_shape)
            clip_width = clip_xx[1] - clip_xx[0] + 1e-9
            clip_height = clip_yy[1] - clip_yy[0] + 1e-9
            clip_area = clip_width * clip_height
            ## 場合分け、全体に対してringが1/3以上入っていないといけない
            ## width/height比が1/3以上でないとlabel付けしない
            if (
                clip_area >= star_area * 1 / 3
                and clip_height / (clip_width + 1e-9) > 1 / 3
                and clip_width / (clip_height + 1e-9) > 1 / 3
            ):
                xmin_c = star_xmin - x_min
                ymin_c = star_ymin - y_min
                xmax_c = star_xmax - x_min
                ymax_c = star_ymax - y_min
                self.xmin_list.append(self.judge_01(xmin_c / cut_shape))
                self.xmax_list.append(self.judge_01(xmax_c / cut_shape))
                self.ymin_list.append(self.judge_01(ymin_c / cut_shape))
                self.ymax_list.append(self.judge_01(ymax_c / cut_shape))
                self.named_list.append(ring_name)

    def check_list(self):
        """
        回転や反転などaugmentationのバグなどで、
        xminとxmax、yminとymaxの大小が反転してしまっている時がある。
        その場合、モデル学習時にエラーが発生するため、その対処として行う。
        """
        xmin_list_, ymin_list_, xmax_list_, ymax_list_, name_list_ = [], [], [], [], []
        for xy_num in range(len(self.xmin_list)):
            #####################################
            ## xminとxmax, yminとymaxの大小を調査 ##
            #####################################

            assert self.xmax_list[xy_num] > self.xmin_list[xy_num] and self.ymax_list[xy_num] > self.ymin_list[xy_num]
            xmin_list_.append(self.xmin_list[xy_num])
            ymin_list_.append(self.ymin_list[xy_num])
            xmax_list_.append(self.xmax_list[xy_num])
            ymax_list_.append(self.ymax_list[xy_num])
            name_list_.append(self.named_list[xy_num])

        return xmin_list_, ymin_list_, xmax_list_, ymax_list_, name_list_  # , self.flag
