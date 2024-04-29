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
        Get the pix information of the bubble in the used fits.
        """

        self.star_dic = {}
        if self.choice == "MWP":
            rout_num = 1.3
        elif self.choice == "CH":
            rout_num = 1
        elif self.choice == "SUM":
            rout_num = 1.3

        for _, row in dataframe.iterrows():
            lmax = row["GLON"] + rout_num * row[self.Rout] / 60
            bmin = row["GLAT"] - rout_num * row[self.Rout] / 60
            ## Right end
            lmin = row["GLON"] - rout_num * row[self.Rout] / 60
            bmax = row["GLAT"] + rout_num * row[self.Rout] / 60
            ## This is the range to cut out the ring. The range to cut out is three times the Rout
            x_pix_min, y_pix_min = self.world.all_world2pix(lmax, bmin, 0)
            x_pix_max, y_pix_max = self.world.all_world2pix(lmin, bmax, 0)

            self.star_dic[row[self.choice]] = [x_pix_min, y_pix_min, x_pix_max, y_pix_max]

    def judge_01(self, number):
        """
        When labeling, the range of the position label must be 0-1.
        This function is for confining the position label to the range of 0-1.
        """
        if number > 1:
            return 1
        elif number < 0:
            return 0
        else:
            return number

    def make_label(self, x_min, y_min, cut_shape):
        """
        The variable 's' contains the position information of the main ring.
        'x_pix_min', 'y_pix_min', 'x_pix_max', and 'y_pix_max' represent the dimensions of the cropped image.
        The index information of rings overlapping with the main ring, along with their details, is stored in the 'star_list'.
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

            ## The actual area of the ring
            star_area = (xx[1] - xx[0]) * (yy[1] - yy[0])

            ## The area of the target ring within the cut-out range
            ## If the ring is outside the cut-out range, it becomes 0
            clip_xx = np.clip(xx, x_min, x_min + cut_shape)
            clip_yy = np.clip(yy, y_min, y_min + cut_shape)
            clip_width = clip_xx[1] - clip_xx[0] + 1e-9
            clip_height = clip_yy[1] - clip_yy[0] + 1e-9
            clip_area = clip_width * clip_height
            picture_area = cut_shape**2
            ## the ring must be more than 1/3 in the whole
            ## Do not label if the width/height ratio is not more than 1/3
            if (
                clip_area >= star_area * 3 / 5
                and clip_height / (clip_width + 1e-9) > 1 / 3
                and clip_width / (clip_height + 1e-9) > 1 / 3
                and clip_area / picture_area >= 1 / 36
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
        There are times when the sizes of xmin and xmax, ymin and ymax are reversed due to bugs in augmentations such as rotation and inversion.
        In that case, an error occurs during model training, so this is done as a countermeasure.
        """
        xmin_list_, ymin_list_, xmax_list_, ymax_list_, name_list_ = [], [], [], [], []
        for xy_num in range(len(self.xmin_list)):
            ##########################################################
            ## Investigate the size of xmin and xmax, ymin and ymax ##
            ##########################################################

            assert self.xmax_list[xy_num] > self.xmin_list[xy_num] and self.ymax_list[xy_num] > self.ymin_list[xy_num]
            xmin_list_.append(self.xmin_list[xy_num])
            ymin_list_.append(self.ymin_list[xy_num])
            xmax_list_.append(self.xmax_list[xy_num])
            ymax_list_.append(self.ymax_list[xy_num])
            name_list_.append(self.named_list[xy_num])

        return xmin_list_, ymin_list_, xmax_list_, ymax_list_, name_list_  # , self.flag
