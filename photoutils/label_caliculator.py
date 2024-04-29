import numpy as np


class label_caliculator(object):
    def __init__(self, choice, world):
        if choice == "MWP":
            self.Rout = "MajAxis"
        elif choice == "CH":
            self.Rout = "Rout"
        elif choice == "SUM":
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

    def calc_pix(self, row, GLON_min, GLON_max, GLAT_min, GLAT_max, scale):
        """
        Decide the range to cut out the image to be used as training data
        randomly (with a fixed seed value) from the fits data.
        """

        ## ccc, ok are used as conditions to break the loop
        ## when the range to cut out could not be decided well

        ccc = 0
        ok = True

        while ok:
            random_num = scale

            lmax = row["GLON"] + random_num * row[self.Rout] / 60
            bmin = row["GLAT"] - random_num * row[self.Rout] / 60
            # Right end
            lmin = row["GLON"] - random_num * row[self.Rout] / 60
            bmax = row["GLAT"] + random_num * row[self.Rout] / 60
            ccc += 1
            if GLON_min <= lmin and lmax <= GLON_max and GLAT_min <= bmin and bmax <= GLAT_max:
                ok = False
                flag = True
            if ccc >= 400:
                ok = False
                flag = False

        # This is the range to cut out the ring
        x_min, y_min = self.world.all_world2pix(lmax, bmin, 0)
        x_max, y_max = self.world.all_world2pix(lmin, bmax, 0)
        width = x_max - x_min
        height = y_max - y_min

        self.x_pix_min = x_min - width / 50
        self.y_pix_min = y_min - height / 50
        self.x_pix_max = x_max + width / 50
        self.y_pix_max = y_max + height / 50

        self.width = self.x_pix_max - self.x_pix_min
        self.height = self.y_pix_max - self.y_pix_min

        return self.x_pix_min, self.y_pix_min, self.x_pix_max, self.y_pix_max, flag

    def find_cover(self, trans_pix_info=None):
        """
        Checks if other rings are present within the cropped image.
        If present, labels them.
        star_dic is a dictionary containing the following keys: x_pix_min, y_pix_min, x_pix_max, y_pix_max.
        """
        ## The x_pix_max values, etc., are set slightly larger to account for convolution.
        ## Calculate the precise cropping range based on the width.
        if trans_pix_info is not None:
            self.x_pix_min = trans_pix_info["x_pix_min"]
            self.x_pix_max = trans_pix_info["x_pix_max"]
            self.y_pix_min = trans_pix_info["y_pix_min"]
            self.y_pix_max = trans_pix_info["y_pix_max"]
            self.width = self.x_pix_max - self.x_pix_min
            self.height = self.y_pix_max - self.y_pix_min

        extra_width = self.width * 1 / 52
        extra_height = self.height * 1 / 52
        self.overlapp_list = []
        self.overlapp_name = []

        for d in self.star_dic.items():
            # Position information for each ring
            star_xmin = d[1][0]
            star_xmax = d[1][2]
            star_ymin = d[1][1]
            star_ymax = d[1][3]
            xx = np.array([star_xmin, star_xmax])
            yy = np.array([star_ymin, star_ymax])
            ## Actual area of the ring
            star_area = (xx[1] - xx[0]) * (yy[1] - yy[0])

            # Area of the target ring within the cropped region
            # If the ring is outside the cropping range, the area is 0
            clip_xx = np.clip(xx, self.x_pix_min + extra_width, self.x_pix_max - extra_width)
            clip_yy = np.clip(yy, self.y_pix_min + extra_height, self.y_pix_max - extra_height)
            clip_width = clip_xx[1] - clip_xx[0] + 1e-9
            clip_height = clip_yy[1] - clip_yy[0] + 1e-9
            clip_area = clip_width * clip_height

            picture_area = (self.x_pix_max - self.x_pix_min - 2 * extra_width) * (
                self.y_pix_max - self.y_pix_min - 2 * extra_height
            )

            # Conditions: The entire ring must be at least 1/3 inside the overall area,
            # and the width/height ratio must be at least 1/3 for labeling.
            if (
                clip_area >= star_area * 3 / 5
                and clip_height / (clip_width + 1e-9) > 1 / 3
                and clip_width / (clip_height + 1e-9) > 1 / 3
                and clip_area / picture_area >= 1 / 36
            ):
                self.overlapp_list.append(d)
                self.overlapp_name.append(d[0])
            else:
                pass

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

    def make_label(self, Ring_catalogue):
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
        Ring_catalogue_name_select = Ring_catalogue.index.tolist()

        #############################################
        ## Labeling for the Main Bubble and Others ##
        #############################################
        ## oThe 'overlapp_list' contains both the main ring and other rings.
        if len(self.overlapp_list) == 0:
            # self.flag = False
            pass
        else:
            # assert len(self.overlapp_list) == 0
            # self.flag = True
            for p, n in zip(self.overlapp_list, self.overlapp_name):
                ## The contents of 'p' include the celestial body name and [xmin, ymin, xmax, ymax].
                ## ('2G0020120-0068213',
                ## [array(7573.50002914), array(4663.19997904), array(7673.50003014), array(4763.19998004)])
                if p[0] in Ring_catalogue_name_select:
                    ##########################################################################
                    ## Convert pixel information to labels in the range 0~1 for model input ##
                    ##########################################################################

                    ## 'p[1][0]' represents the celestial body position, and 'x_pix_min' is the pixel information cropped from the FITS file.
                    ## Adding 'width/4' accounts for convolution, which can create artifacts.

                    xmin_c = p[1][0] - (self.x_pix_min + self.width / 52)
                    ymin_c = p[1][1] - (self.y_pix_min + self.height / 52)
                    xmax_c = p[1][2] - (self.x_pix_min + self.width / 52)
                    ymax_c = p[1][3] - (self.y_pix_min + self.height / 52)
                    self.xmin_list.append(self.judge_01(xmin_c / (self.width * 51 / 52)))
                    self.xmax_list.append(self.judge_01(xmax_c / (self.width * 51 / 52)))
                    self.ymin_list.append(self.judge_01(ymin_c / (self.height * 51 / 52)))
                    self.ymax_list.append(self.judge_01(ymax_c / (self.height * 51 / 52)))
                    self.named_list.append(n)

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

        return {"xmin": xmin_list_, "ymin": ymin_list_, "xmax": xmax_list_, "ymax": ymax_list_, "name": name_list_}
