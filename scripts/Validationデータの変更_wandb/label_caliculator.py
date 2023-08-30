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

    def calc_pix(self, row, GLON_min, GLON_max, GLAT_min, GLAT_max, scale):
        """
        fitsデータから、学習データとして用いる画像の切り出す範囲を
        ランダム（シード値固定）で決めていく。
        """

        ## ccc, okは、切り出す範囲をうまく決められなかった時の
        ## ループを抜ける条件として用いる

        ccc = 0
        ok = True

        while ok:
            random_num = scale

            lmax = row["GLON"] + random_num * row[self.Rout] / 60
            bmin = row["GLAT"] - random_num * row[self.Rout] / 60
            # 右端
            lmin = row["GLON"] - random_num * row[self.Rout] / 60
            bmax = row["GLAT"] + random_num * row[self.Rout] / 60
            ccc += 1
            if GLON_min <= lmin and lmax <= GLON_max and GLAT_min <= bmin and bmax <= GLAT_max:
                ok = False
                flag = True
            if ccc >= 400:
                ok = False
                flag = False

        # これは、リングを切り取る範囲
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
        切り出した画像の中に、他のリングが入っていないか確かめる。
        入っていたら、ラベル付けする
        star_dicはdictionaryで、中身は、x_pix_min, y_pix_min, x_pix_max, y_pix_maxという順になっている
        """
        ## x_pix_maxなどは、convolutionを考えて幅の1/4大きめに設定しているため
        ## widthを計算して、正確な切り出し範囲を算出する
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
            ## 各リングの位置情報
            star_xmin = d[1][0]
            star_xmax = d[1][2]
            star_ymin = d[1][1]
            star_ymax = d[1][3]
            xx = np.array([star_xmin, star_xmax])
            yy = np.array([star_ymin, star_ymax])
            ## リングの本当の面積
            star_area = (xx[1] - xx[0]) * (yy[1] - yy[0])

            ## 切り出す範囲内での対象リングの面積
            ## リングが切り出す範囲外なら、0になる
            clip_xx = np.clip(xx, self.x_pix_min + extra_width, self.x_pix_max - extra_width)
            clip_yy = np.clip(yy, self.y_pix_min + extra_height, self.y_pix_max - extra_height)
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
                self.overlapp_list.append(d)
                self.overlapp_name.append(d[0])
            else:
                pass

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

    def make_label(self, Ring_catalogue):
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
        Ring_catalogue_name_select = Ring_catalogue.index.tolist()

        #############################################
        ## 主体となるRing と それ以外のRing のlabel付け ##
        #############################################
        ## overlapp_listには主体となるRingとそれ以外のRingが入っている。
        if len(self.overlapp_list) == 0:
            # self.flag = False
            pass
        else:
            # assert len(self.overlapp_list) == 0
            # self.flag = True
            for p, n in zip(self.overlapp_list, self.overlapp_name):
                ## pの中身は、以下のような天体名と[xmin, ymin, xmax, ymax]が入っている
                ## ('2G0020120-0068213',
                ## [array(7573.50002914), array(4663.19997904), array(7673.50003014), array(4763.19998004)])
                if p[0] in Ring_catalogue_name_select:
                    ######################################################
                    ## モデルに入力するために、pix情報を0~1のlabelに変換させる ##
                    ######################################################

                    ## p[1][0]は天体の位置であり、x_pix_minはfitsから切り出すpix情報
                    ## width/4を足しているのは、画像処理の際に行うconvolutionにより耳ができるため
                    ## 余分に大きく切り出しているため

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

        return {"xmin": xmin_list_, "ymin": ymin_list_, "xmax": xmax_list_, "ymax": ymax_list_, "name": name_list_}
